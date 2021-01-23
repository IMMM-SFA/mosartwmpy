import numpy as np
import pandas as pd

def regulation(state, grid, parameters, delta_t):
    # regulation of the flow from the reservoirs, applied to flow entering the grid cell, i.e. subnetwork storage downstream of reservoir
    
    base_condition = (
        (grid.mosart_mask > 0) &
        state.euler_mask &
        (state.tracer == parameters.LIQUID_TRACER) &
        np.isfinite(grid.reservoir_id)
    )
    
    flow_volume = -state.channel_outflow_downstream * delta_t
    
    flow_reservoir = state.reservoir_release * delta_t
    
    evaporation = 1e6 * state.reservoir_potential_evaporation * delta_t * grid.reservoir_surface_area
    
    minimum_flow = parameters.reservoir_runoff_capacity_condition * state.reservoir_streamflow * delta_t
    minimum_storage = parameters.reservoir_runoff_capacity_condition * grid.reservoir_storage_capacity
    maximum_storage = 1 * grid.reservoir_storage_capacity
    
    condition_max = flow_volume + state.reservoir_storage - flow_reservoir - evaporation >= maximum_storage
    condition_min = flow_volume + state.reservoir_storage - flow_reservoir - evaporation < minimum_storage
    condition_min_one = flow_reservoir <= flow_volume - evaporation
    condition_min_two = flow_volume - evaporation >= minimum_flow
    
    flow_reservoir = np.where(
        condition_max,
        flow_volume + state.reservoir_storage - maximum_storage - evaporation,
        np.where(
            condition_min,
            np.where(
                condition_min_one,
                flow_reservoir,
                np.where(
                    condition_min_two,
                    flow_volume - evaporation,
                    flow_volume
                )
            ),
            flow_reservoir
        )
    )
    
    state.reservoir_storage = np.where(
        base_condition,
        np.where(
            condition_max,
            maximum_storage,
            np.where(
                condition_min,
                np.where(
                    condition_min_one,
                    state.reservoir_storage + flow_volume - flow_reservoir - evaporation,
                    np.where(
                        condition_min_two,
                        state.reservoir_storage,
                        np.maximum(0, state.reservoir_storage - flow_reservoir + flow_volume - evaporation)
                    )
                ),
                state.reservoir_storage + flow_volume - flow_reservoir - evaporation
            )
        ),
        state.reservoir_storage
    )
    
    state.channel_outflow_downstream = np.where(
        base_condition,
        -flow_reservoir / delta_t,
        state.channel_outflow_downstream
    )

def extraction_regulated_flow(state, grid, parameters, config, delta_t):
    # extract water from the reservoir release
    # the extraction needs to be distributed across the dependent cells demand
    
    # notes from fortran mosart:
    # This is an iterative algorithm that converts main channel flow
    # at each dam into gridcell supply based on the demand of each
    # gridcell.
    # The basic algorithm is as follows
    # - Compute flow_vol at each dam based on the main channel flow at the gridcell
    # - Compute the demand at each dam based on the demand at each gridcell and the
    #   gridcell/dam dependency.  This dependency is stored in the sparse matrix
    #   SMatP_g2d.  The demand on each dam is the sum of the demand of all the gridcells
    #   that depend on that dam.
    # - Covert dam flow_vol to gridcell supply.  In doing so, reduce the flow_vol
    #   at the dam, reduce the demand at the gridcell, and increase the supply at
    #   the gridcell by the same amount.  There are three conditions for this conversion
    #   to occur and these are carried out in the following order.  dam fraction
    #   is the ratio of the dam flow_vol over the total dam demand.
    #   1. if any dam fraction >= 1.0 for a gridcell, then provide full demand to gridcell
    #      prorated by the number of dams that can provide all the water.
    #   2. if any sum of dam fraction >= 1.0 for a gridcell, then provide full demand to
    #      gridcell prorated by the dam fraction of each dam.
    #   3. if any sum of dam fraction < 1.0 for a gridcell, then provide fraction of 
    #      demand to gridcell prorated by the dam fraction of each dam.
    # - Once the iterative solution has converged, convert the residual flow_vol
    #   back into main channel flow.
    #
    # This implementation assumes several things
    # - Each dam is associated with a particular gridcell and each gridcell has
    #   either 0 or 1 dam associated with it.
    # - The local dam decomposition
    
    base_condition = np.isfinite(grid.reservoir_id)
    
    flow_volume = np.where(
        base_condition,
        parameters.reservoir_flow_volume_ratio * delta_t * -state.channel_outflow_downstream,
        np.nan
    )
    
    state.channel_outflow_downstream = np.where(
        base_condition,
        state.channel_outflow_downstream + flow_volume / delta_t,
        state.channel_outflow_downstream
    )
    
    # TODO need a real strategy here
    # filter reservoir mapping down to the cells local to this processor
    # if config.get('multiprocessing.enabled', False) and config.get('muliprocessing.cores', 1) > 1:
    #     # TODO need a more efficient way to do this
    #     mapping = grid.reservoir_to_grid_mapping[grid.reservoir_to_grid_mapping.reservoir_id.isin(grid.reservoir_id[np.isfinite(grid.reservoir_id)])]
    # else:
    #     mapping = grid.reservoir_to_grid_mapping
    
    # iterate through the supply code
    for _ in np.arange(parameters.reservoir_supply_iterations):
        # join grid cell demand, then drop where no demand
        local_mapping = grid.reservoir_to_grid_mapping.merge(
            pd.DataFrame(state.reservoir_demand, columns=['reservoir_demand']), how='left', left_on='grid_cell_id', right_index=True
        ).rename(columns={'reservoir_demand': 'grid_cell_demand'})
        local_mapping = local_mapping[local_mapping.grid_cell_demand.gt(0)]
        # aggregate to the total demand potential on each reservoir
        local_mapping = local_mapping.merge(local_mapping[['reservoir_id', 'grid_cell_demand']].groupby('reservoir_id').sum().rename(columns={'grid_cell_demand': 'reservoir_demand'}), how='left', left_on='reservoir_id', right_index=True)
        # assign the flow volume to each reservoir in the mapping
        local_mapping = local_mapping.merge(pd.DataFrame(grid.reservoir_id, columns=['reservoir_id']).join(pd.DataFrame(flow_volume, columns=['flow_volume'])), how='left', left_on='reservoir_id', right_on='reservoir_id')
        # for each reservoir calculate the ratio of flow volume over demand; if greater than 1 it means there is more water than demanded
        local_mapping = local_mapping.join(pd.DataFrame(local_mapping.flow_volume.values / local_mapping.reservoir_demand.values, columns=['demand_fraction']))
        # initialize supply to 0 (supply provided to grid cell during this iteration)
        local_mapping = local_mapping.join(pd.DataFrame(0.0 * local_mapping.flow_volume.values, columns=['grid_cell_supply']))

        # convert reservoir flow volume to grid cell supply
        # notes from fortran mosart:
        # Covert dam flow_vol to gridcell supply.  In doing so, reduce the flow_vol
        # at the dam, reduce the demand at the gridcell, and increase the supply at
        # the gridcell by the same amount.  There are three conditions for this conversion
        # to occur and these are carried out in the following order.  dam fraction
        # is the ratio of the dam flow_vol over the total dam demand.
        # 1. if any dam fraction >= 1.0 for a gridcell, then provide full demand to gridcell
        #    prorated by the number of dams that can provide all the water.
        # 2. if any sum of dam fraction >= 1.0 for a gridcell, then provide full demand to
        #    gridcell prorated by the dam fraction of each dam.
        # 3. if any sum of dam fraction < 1.0 for a gridcell, then provide fraction of 
        #    demand to gridcell prorated by the dam fraction of each dam.
        #
        # dam_uptake is the amount of water removed from the dam, it's a global array.
        # Gridcells from different tasks will accumluate the amount of water removed
        # from each dam in this array.
        
        # case 1 - reservoirs capable of supplying all demand
        # TODO the code is written with sctrictly greater than one, but comment says greater or equal...
        if np.any(local_mapping.demand_fraction.values > 1):
            # logging.info('case1')
            # reduce to the set and get the count of reservoirs that satisfy this condition
            local_mapping = local_mapping[local_mapping.demand_fraction.gt(1)]
            local_mapping = local_mapping.merge(
                local_mapping[['grid_cell_id', 'reservoir_id']].groupby('grid_cell_id').count().rename(columns={'reservoir_id': 'condition_count'}),
                how='left',
                left_on='grid_cell_id',
                right_index=True
            )
            local_mapping.loc[:, local_mapping.grid_cell_supply.name] = local_mapping.grid_cell_demand.values / local_mapping.condition_count.values

        else: 
            # sum the demand_fraction for each grid_cell; if greater than 1 it means all demand can be met, otherwise it can't
            local_mapping = local_mapping.merge(local_mapping[['grid_cell_id', 'demand_fraction']].groupby('grid_cell_id').sum().rename(columns={'demand_fraction': 'demand_fraction_sum'}), how='left', left_on='grid_cell_id', right_index=True)
            # case 2 - grid cells capable of fulfilling all demand from their reservoirs
            if np.any(local_mapping.demand_fraction_sum.values >= 1):
                # logging.info('case2')
                local_mapping = local_mapping[local_mapping.demand_fraction_sum.ge(1)]
                # prorate by demand_fraction
                local_mapping.loc[:, local_mapping.grid_cell_supply.name] = local_mapping.grid_cell_demand.values * local_mapping.demand_fraction.values / local_mapping.demand_fraction_sum.values

            # case 3 - grid cells incapable of fulfilling all demand from their reservoirs
            else:
                # logging.info('case3')
                local_mapping = local_mapping[local_mapping.demand_fraction_sum.gt(0)]
                # prorate by demand_fraction
                local_mapping.loc[:, local_mapping.grid_cell_supply.name] = local_mapping.grid_cell_demand.values * local_mapping.demand_fraction.values

        # update the overall flow/supply/demand based on this iteration
        supplied_to_cell = pd.DataFrame(grid.id, columns=['id']).merge(local_mapping[['grid_cell_id', 'grid_cell_supply']].groupby('grid_cell_id').sum(), how='left', left_on='id', right_index=True).grid_cell_supply.fillna(0).values
        taken_from_reservoir = pd.DataFrame(grid.reservoir_id, columns=['reservoir_id']).merge(local_mapping[['reservoir_id', 'grid_cell_supply']].groupby('reservoir_id').sum(), how='left', left_on='reservoir_id', right_index=True).grid_cell_supply.fillna(0).values
        # remove the supplied water from the flow at the reservoirs
        flow_volume = flow_volume - taken_from_reservoir
        # if the supply is within tiny value of demand, set them equal so that the loop works properly
        supplied_to_cell = np.where(
            np.abs(supplied_to_cell - state.reservoir_demand) < parameters.small_value,
            state.reservoir_demand,
            supplied_to_cell
        )

        # remove the supplied water from the demand at the grid cells
        state.reservoir_demand = state.reservoir_demand - supplied_to_cell
        # accumulate the supplied water to the supply at the grid cells
        state.reservoir_supply = state.reservoir_supply + supplied_to_cell
    
    # accumulate the deficit to the grid cell
    state.reservoir_deficit = state.reservoir_deficit + state.reservoir_demand
    
    # add residual flow volume back into the main channel flow
    state.channel_outflow_downstream = np.where(
        base_condition,
        state.channel_outflow_downstream - flow_volume / delta_t,
        state.channel_outflow_downstream
    )