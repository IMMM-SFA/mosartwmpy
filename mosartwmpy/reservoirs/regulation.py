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

# @timing
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
    #
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
    
    cells = pd.DataFrame({'id': grid.id[state.reservoir_demand > 0]}).set_index('id')
    cells['supply'] = 0
    
    # join grid cell demand, then drop where no demand
    demand = grid.reservoir_to_grid_mapping.join(pd.DataFrame(state.reservoir_demand, columns=['grid_cell_demand']))
    demand = demand[demand.grid_cell_demand.gt(0)]
    
    # aggregate demand to each reservoir and join to flow volume
    reservoir_demand_flow = demand.groupby('reservoir_id')[['grid_cell_demand']].sum().rename(columns={'grid_cell_demand': 'reservoir_demand'}).join(pd.DataFrame({'flow_volume': flow_volume, 'reservoir_id': grid.reservoir_id}).dropna().set_index('reservoir_id'))
    
    for _ in np.arange(parameters.reservoir_supply_iterations):
        
        if _ == 0:
            case = reservoir_demand_flow
        else:
            # subset reservoir list to speed up calculation
            case = reservoir_demand_flow[reservoir_demand_flow.index.isin(demand.reservoir_id.unique())]
            case.loc[:, 'reservoir_demand'] = case.join(demand.groupby('reservoir_id')[['grid_cell_demand']].sum()).grid_cell_demand.fillna(0)
        
        # ratio of flow to total demand
        case.loc[:, 'demand_fraction'] = case.flow_volume.values / case.reservoir_demand.values
        
        # case 1
        if case.demand_fraction.gt(1).any():
            case = demand[demand.reservoir_id.isin(case[case.demand_fraction.gt(1)].index)]
            case.loc[:, 'condition_count'] = case.groupby(case.index)['reservoir_id'].transform('count')
            case.loc[:, 'supply'] = case.grid_cell_demand / case.condition_count
            taken_from_reservoir = reservoir_demand_flow.join(case.groupby('reservoir_id').supply.sum()).supply.fillna(0).values
            reservoir_demand_flow.loc[:, 'reservoir_demand'] -= taken_from_reservoir
            reservoir_demand_flow.loc[:, 'flow_volume'] -= taken_from_reservoir
            # all demand was supplied to these cells
            cells.loc[:, 'supply'] += cells.join(case.groupby(case.index)[['grid_cell_demand']].first()).grid_cell_demand.fillna(0)
            demand = demand[~demand.index.isin(case.index.unique())]
        
        else:
            # sum demand fraction
            case = demand.merge(case, how='left', left_on='reservoir_id', right_index=True)
            case.loc[:, 'demand_fraction_sum'] = case.groupby(case.index).demand_fraction.transform('sum').fillna(0).values
            
            # case 2
            if case.demand_fraction_sum.ge(1).any():
                case = case[case.demand_fraction_sum.ge(1)]
                case.loc[:, 'supply'] = case.grid_cell_demand.values  * case.demand_fraction.values / case.demand_fraction_sum.values
                taken_from_reservoir = reservoir_demand_flow.join(case.groupby('reservoir_id')['supply'].sum()).supply.fillna(0).values
                reservoir_demand_flow.loc[:, 'reservoir_demand'] -= taken_from_reservoir
                reservoir_demand_flow.loc[:, 'flow_volume'] -= taken_from_reservoir
                # all demand was supplied to these cells
                cells.loc[:, 'supply'] += cells.join(case.groupby(case.index)[['grid_cell_demand']].first()).grid_cell_demand.fillna(0)
                demand = demand[~demand.index.isin(case.index.unique())]
                
            else:
                case = case[case.demand_fraction_sum.gt(0)]
                case.loc[:, 'supply'] = case.grid_cell_demand.values * case.demand_fraction.values
                taken_from_reservoir = reservoir_demand_flow.join(case.groupby('reservoir_id')['supply'].sum()).supply.fillna(0).values
                reservoir_demand_flow.loc[:, 'reservoir_demand'] -= taken_from_reservoir
                reservoir_demand_flow.loc[:, 'flow_volume'] -= taken_from_reservoir
                # not all demand was supplied to these cells
                supplied = cells[[]].join(case.groupby(case.index)[['supply']].sum()).supply.fillna(0)
                cells.loc[:, 'supply'] += supplied
                demand.loc[:, 'grid_cell_demand'] -= demand[[]].join(supplied).fillna(0).supply.values
    
    # merge the supply back in and update demand
    supplied = pd.DataFrame(grid.id).join(cells).supply.fillna(0).values
    state.reservoir_supply[:] += supplied
    state.reservoir_demand[:] -= supplied
    
    # add the residual flow volume back
    state.channel_outflow_downstream[:] -= pd.DataFrame(grid.reservoir_id, columns=['reservoir_id']).merge(reservoir_demand_flow.flow_volume, how='left', left_on='reservoir_id', right_index=True).flow_volume.fillna(0).values / delta_t