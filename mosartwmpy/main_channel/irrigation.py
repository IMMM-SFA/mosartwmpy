import numpy as np

from mosartwmpy.main_channel.state import update_main_channel_state

def main_channel_irrigation(state, grid, parameters):
    # main channel routing irrigation extraction
    
    base_condition = (
        (grid.mosart_mask > 0) &
        state.euler_mask &
        (state.tracer == parameters.LIQUID_TRACER) &
        (state.channel_depth >= parameters.irrigation_extraction_condition)
    )
    
    sub_condition = (
        (state.channel_storage > parameters.tinier_value) &
        (state.reservoir_demand > parameters.tinier_value) &
        (grid.channel_length > parameters.tinier_value)
    )
    
    flow_volume = 1.0 * state.channel_storage
    
    volume_condition = parameters.irrigation_extraction_maximum_fraction * flow_volume >= state.reservoir_demand
    
    state.reservoir_supply = np.where(
        base_condition & sub_condition,
        np.where(
            volume_condition,
            state.reservoir_supply + state.reservoir_demand,
            state.reservoir_supply + parameters.irrigation_extraction_maximum_fraction  * flow_volume
        ),
        state.reservoir_supply
    )
    
    flow_volume = np.where(
        base_condition & sub_condition & volume_condition,
        flow_volume - state.reservoir_demand,
        flow_volume
    )
    
    state.reservoir_demand = np.where(
        base_condition & sub_condition,
        np.where(
            volume_condition,
            0,
            state.reservoir_demand - parameters.irrigation_extraction_maximum_fraction * flow_volume
        ),
        state.reservoir_demand
    )
    
    flow_volume = np.where(
        base_condition & sub_condition & np.logical_not(volume_condition),
        (1.0 - parameters.irrigation_extraction_maximum_fraction) * flow_volume,
        flow_volume
    )
    
    state.channel_storage = np.where(
        base_condition & sub_condition,
        flow_volume,
        state.channel_storage
    )
    
    # TODO ? fortran mosart appears to do some more math with temp_erout and TRunoff%erout that looks to me to always be zero
    
    update_main_channel_state(state, grid, parameters, base_condition)