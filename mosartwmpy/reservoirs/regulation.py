import numpy as np
import numexpr as ne
import pandas as pd

from mosartwmpy.utilities.timing import timing

#@timing
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

#@timing
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
    
    # join grid cell demand, then drop where no demand
    demand = grid.reservoir_to_grid_mapping.join(pd.DataFrame(state.reservoir_demand, columns=['grid_cell_demand']))
    demand = demand[demand.grid_cell_demand.gt(0)]
    # aggregate to the total demand potential on each reservoir
    demand = demand.merge(
        demand.groupby('reservoir_id')[['grid_cell_demand']].sum().rename(columns={'grid_cell_demand': 'reservoir_demand'}),
        how='left', left_on='reservoir_id', right_index=True
    )
    # assign the flow volume to each reservoir in the mapping
    demand = demand.merge(pd.DataFrame({'flow_volume': flow_volume, 'reservoir_id': grid.reservoir_id}).dropna().set_index('reservoir_id'), how='left', left_on='reservoir_id', right_index=True)
    
    for _ in np.arange(parameters.reservoir_supply_iterations):
        # for each reservoir calculate the ratio of flow volume over demand; if greater than 1 it means there is more water than demanded
        demand.eval('demand_fraction = flow_volume / reservoir_demand', inplace=True)
        case = demand[demand.grid_cell_demand.gt(0)]

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
        # TODO the code is written with strictly greater than one, but comment says greater or equal...
        if case.demand_fraction.gt(1).any():
            # logging.info('case1')
            # reduce to the set and get the count of reservoirs that satisfy this condition
            case = case[case.demand_fraction.gt(1)]
            case['condition_count'] = case.groupby(case.index)['reservoir_id'].transform('count')
            case.eval('supply = grid_cell_demand / condition_count', inplace=True)

        else: 
            # sum the demand_fraction for each grid_cell; if greater than 1 it means all demand can be met, otherwise it can't
            case['demand_fraction_sum'] = case.groupby(case.index)['demand_fraction'].transform('sum')
            # case 2 - grid cells capable of fulfilling all demand from their reservoirs
            if case.demand_fraction_sum.ge(1).any():
                # logging.info('case2')
                case = case[case.demand_fraction_sum.ge(1)]
                # prorate by demand_fraction
                case.eval('supply = grid_cell_demand * demand_fraction / demand_fraction_sum', inplace=True)

            # case 3 - grid cells incapable of fulfilling all demand from their reservoirs
            else:
                # logging.info('case3')
                case = case[case.demand_fraction_sum.gt(0)]
                # prorate by demand_fraction
                case.eval('supply = grid_cell_demand * demand_fraction', inplace=True)
            
            del case['demand_fraction_sum']

        demand['supplied_to_grid'] = demand.join(case.groupby(case.index)[['supply']].sum().rename(columns={'supply': 'supplied_to_grid'})).supplied_to_grid.fillna(0)
        demand['taken_from_reservoir'] = demand.merge(case.groupby('reservoir_id')[['supply']].sum().rename(columns={'supply': 'taken_from_reservoir'}), how='left', left_on='reservoir_id', right_index=True).taken_from_reservoir.fillna(0)
        demand.eval('grid_cell_demand = grid_cell_demand - supplied_to_grid', inplace=True)
        demand.eval('reservoir_demand = reservoir_demand - taken_from_reservoir', inplace=True)
        demand.eval('flow_volume = flow_volume - taken_from_reservoir', inplace=True)
        del demand['supplied_to_grid']
        del demand['taken_from_reservoir']
        
        # set small values to zero
        demand.loc[demand.grid_cell_demand.abs().lt(parameters.small_value), 'grid_cell_demand'] = 0
    
    remaining_demand = pd.DataFrame(grid.id).join(demand.groupby(demand.index)[['grid_cell_demand']].first()).grid_cell_demand.values
    flow_volume = pd.DataFrame(grid.reservoir_id, columns=['reservoir_id']).merge(demand.groupby('reservoir_id')[['flow_volume']].first(), how='left', left_on='reservoir_id', right_index=True).flow_volume.fillna(0).values
    
    # accumulate the supplied water to the supply at the grid cells
    state.reservoir_supply[:] = np.where(
        np.isfinite(remaining_demand),
        state.reservoir_supply + state.reservoir_demand - remaining_demand,
        state.reservoir_supply
    )
    # remove the supplied water from the demand at the grid cells
    state.reservoir_demand[:] = np.where(
        np.isfinite(remaining_demand),
        remaining_demand,
        state.reservoir_demand
    )
    
    # accumulate the deficit to the grid cell
    
    state.reservoir_deficit = np.where(
        np.isfinite(state.reservoir_demand),
        state.reservoir_deficit + state.reservoir_demand,
        state.reservoir_deficit
    )
    
    # add residual flow volume back into the main channel flow
    state.channel_outflow_downstream = np.where(
        base_condition,
        state.channel_outflow_downstream - flow_volume / delta_t,
        state.channel_outflow_downstream
    )