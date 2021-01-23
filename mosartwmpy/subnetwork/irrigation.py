import numpy as np

from mosartwmpy.subnetwork.state import update_subnetwork_state

def subnetwork_irrigation(state, grid, parameters):
    # subnetwork channel routing irrigation extraction
    
    base_condition = (grid.mosart_mask > 0) & (
        state.euler_mask &
        (state.tracer == parameters.LIQUID_TRACER) &
        (state.subnetwork_depth >= parameters.irrigation_extraction_condition)
    )
    
    flow_volume = 1.0 * state.subnetwork_storage
    
    volume_condition = (flow_volume >= state.reservoir_demand)
    
    state.reservoir_supply = np.where(
        base_condition,
        np.where(
            volume_condition,
            state.reservoir_supply + state.reservoir_demand,
            state.reservoir_supply + flow_volume
        ),
        state.reservoir_supply
    )
    
    flow_volume = np.where(
        base_condition & volume_condition,
        flow_volume - state.reservoir_demand,
        flow_volume
    )
    
    state.reservoir_demand = np.where(
        base_condition,
        np.where(
            volume_condition,
            0,
            state.reservoir_demand - flow_volume
        ),
        state.reservoir_demand
    )
    
    flow_volume = np.where(
        base_condition & np.logical_not(volume_condition),
        0,
        flow_volume
    )
    
    state.subnetwork_storage = np.where(
        base_condition,
        flow_volume,
        state.subnetwork_storage
    )
    
    update_subnetwork_state(state, grid, parameters, base_condition)