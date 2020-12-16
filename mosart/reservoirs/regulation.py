import numpy as np

def regulation(state, grid, parameters, config, delta_t):
    # regulation of the flow from the reservoirs, applied to flow entering the grid cell, i.e. subnetwork storage downstream of reservoir
    
    base_condition = (
        (grid.mosart_mask.values > 0) &
        state.euler_mask.values &
        (state.tracer.values == parameters.LIQUID_TRACER) &
        np.isfinite(grid.reservoir_id.values)
    )
    
    flow_volume = -state.channel_outflow_downstream.values * delta_t
    
    flow_reservoir = state.reservoir_release.values * delta_t
    
    evaporation = 1e6 * state.reservoir_potential_evaporation.values * delta_t * grid.reservoir_surface_area.values
    
    minimum_flow = parameters.reservoir_runoff_capacity_condition * state.reservoir_streamflow.values * delta_t
    minimum_storage = parameters.reservoir_runoff_capacity_condition * grid.reservoir_storage_capacity.values
    maximum_storage = 1 * grid.reservoir_storage_capacity.values
    
    condition_max = flow_volume + state.reservoir_storage.values - flow_reservoir - evaporation >= maximum_storage
    condition_min = flow_volume + state.reservoir_storage.values - flow_reservoir - evaporation < minimum_storage
    condition_min_one = flow_reservoir <= flow_volume - evaporation
    condition_min_two = flow_volume - evaporation >= minimum_flow
    
    flow_reservoir = np.where(
        condition_max,
        flow_volume + state.reservoir_storage.values - maximum_storage - evaporation,
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
    
    state.reservoir_storage[:] = np.where(
        base_condition,
        np.where(
            condition_max,
            maximum_storage,
            np.where(
                condition_min,
                np.where(
                    condition_min_one,
                    state.reservoir_storage.values + flow_volume - flow_reservoir - evaporation,
                    np.where(
                        condition_min_two,
                        state.reservoir_storage.values,
                        np.maximum(0, state.reservoir_storage.values - flow_reservoir + flow_volume - evaporation)
                    )
                ),
                state.reservoir_storage.values + flow_volume - flow_reservoir - evaporation
            )
        ),
        state.reservoir_storage.values
    )
    
    state.channel_outflow_downstream[:] = np.where(
        base_condition,
        -flow_reservoir / delta_t,
        state.channel_outflow_downstream.values
    )
    
    return state

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
    
    condition = np.isfinite(grid.reservoir_id)
    
    flow_volume = np.where(
        condition,
        parameters.reservoir_flow_volume_ratio * delta_t * -state.channel_outflow_downstream.values,
        np.nan
    )
    
    state.channel_outflow_downstream[:] = np.where(
        condition,
        state.channel_outflow_downstream.values + flow_volume / delta_t,
        state.channel_outflow_downstream.values
    )
    
    # find total demand of grid cells dependent on each reservoir
    # merge demand to the reservoir to grid mapping, group by reservoir, and sum
    # then merege back to domain
    local_mapping = parameters.reservoir_to_grid_mapping[parameters.reservoir_to_grid_mapping.reservoir_id.isin(grid.reservoir_id.notna())]
    local_mapping = local_mapping.merge(state.reservoir_monthly_demand, how='left', left_on='grid_cell_id', right_index=True)
    reservoir_demand = grid[['reservoir_id']].merge(
        local_mapping[['reservoir_id', 'reservoir_monthly_demand']].groupby('reservoir_id').sum(),
        how='left',
        left_on='reservoir_id',
        right_index=True
    ).reservoir_monthly_demand.values
    # ratio of flow volume over demand; if greater than 1 it means there is more water than demanded
    reservoir_demand_ratio = np.where(
        reservoir_demand > 0,
        flow_volume / reservoir_demand,
        0
    )
    
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
    # TODO