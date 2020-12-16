import numpy as np

from mosart.main_channel.state import update_main_channel_state

def main_channel_irrigation(state, grid, parameters, config, delta_t):
    # main channel routing irrigation extraction
    
    base_condition = (
        (grid.mosart_mask.values > 0) &
        state.euler_mask.values &
        (state.tracer.values == parameters.LIQUID_TRACER) &
        (state.channel_depth.values >= parameters.irrigation_extraction_condition)
    )
    
    sub_condition = (
        (state.channel_storage.values > parameters.tinier_value) &
        (state.reservoir_demand > parameters.tinier_value) &
        (grid.main_channel_length > parameters.tinier_value)
    )
    
    flow_volume = 1 * state.channel_storage
    
    extraction_condition = parameters.irrigation_extraction_maximum_fraction * flow_volume >= state.reservoir_demand.values
    
    state.reservoir_supply[:] = np.where(
        base_condition & sub_condition,
        np.where(
            extraction_condition,
            state.reservoir_supply.values + state.reservoir_demand.values,
            state.reservoir_supply.values + parameters.irrigation_extraction_maximum_fraction  * flow_volume
        ),
        state.reservoir_supply.values
    )
    
    flow_volume = np.where(
        extraction_condition,
        flow_volume - state.reservoir_demand.values,
        flow_volume
    )
    
    state.reservoir_demand[:] = np.where(
        base_condition & sub_condition,
        np.where(
            extraction_condition,
            0,
            state.reservoir_demand.values - parameters.irrigation_extraction_maximum_fraction * flow_volume
        ),
        state.reservoir_demand.values
    )
    
    flow_volume = np.where(
        np.logical_not(extraction_condition),
        (1 - parameters.irrigation_extraction_maximum_fraction) * flow_volume,
        flow_volume
    )
    
    state.channel_storage[:] = np.where(
        base_condition & sub_condition,
        flow_volume,
        state.channel_storage.values
    )
    
    # TODO ? fortran mosart appears to do some more math with temp_erout and TRunoff%erout that looks to me to always be zero
    
    state = update_main_channel_state(state, grid, parameters, config, base_condition)
    
    return state