import numpy as np

from mosart.subnetwork.state import update_subnetwork_state

def subnetwork_irrigation(state, grid, parameters, config):
    # subnetwork channel routing irrigation extraction
    
    base_condition = (grid.mosart_mask.values > 0) & (
        state.euler_mask.values &
        (state.tracer.values == parameters.LIQUID_TRACER) &
        (state.subnetwork_depth.values >= parameters.irrigation_extraction_condition)
    )
    
    flow_volume = 1 * state.subnetwork_storage
    
    volume_condition = flow_volume >= state.reservoir_demand.values
    
    state.reservoir_supply[:] = np.where(
        base_condition,
        np.where(
            volume_condition,
            state.reservoir_supply.values + state.reservoir_demand.values,
            state.reservoir_supply.values + flow_volume
        ),
        state.reservoir_supply.values
    )
    
    flow_volume = np.where(
        base_condition & volume_condition,
        flow_volume - state.reservoir_demand.values,
        flow_volume
    )
    
    state.reservoir_demand[:] = np.where(
        base_condition,
        np.where(
            volume_condition,
            0,
            state.reservoir_demand.values - flow_volume
        ),
        state.reservoir_demand
    )
    
    flow_volume = np.where(
        base_condition & np.logical_not(volume_condition),
        0,
        flow_volume
    )
    
    state.subnetwork_storage[:] = np.where(
        base_condition,
        flow_volume,
        state.subnetwork_storage.values
    )
    
    state = update_subnetwork_state(state, grid, parameters, config, base_condition)
    
    return state