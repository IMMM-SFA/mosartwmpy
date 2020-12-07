import numpy as np

from mosart.main_channel.kinematic_wave import kinematic_wave_routing
from mosart.main_channel.state import update_main_channel_state

def main_channel_routing(state, grid, parameters, config, delta_t):
    # perform the main channel routing
    # TODO describe what is happening here
    
    tmp_outflow_downstream = state.zeros
    local_delta_t = (delta_t / config.get('simulation.routing_iterations') / grid.iterations_main_channel)
    
    # step through max iterations, masking out the unnecessary cells each time
    base_condition = (grid.mosart_mask.gt(0) & state.euler_mask)
    for _ in np.arange(grid.iterations_main_channel.max()):
        iteration_condition = base_condition & grid.iterations_main_channel.gt(_)
    
        # routing
        routing_method = config.get('simulation.routing_method', 1)
        if routing_method == 1:
            state = kinematic_wave_routing(state, grid, parameters, config, local_delta_t, iteration_condition)
        else:
            raise Exception(f"Error - Routing method {routing_method} not implemented.")
        
        # update storage
        state.channel_storage_previous_timestep = state.channel_storage_previous_timestep.mask(
            iteration_condition,
            state.channel_storage
        )
        state.channel_storage = state.channel_storage.mask(
            iteration_condition,
            state.channel_storage + state.channel_delta_storage * local_delta_t
        )
        
        state = update_main_channel_state(state, grid, parameters, config, iteration_condition)
        
        # update outflow tracking
        tmp_outflow_downstream = tmp_outflow_downstream.mask(
            iteration_condition,
            tmp_outflow_downstream + state.channel_outflow_downstream
        )
    
    # update outflow
    state.channel_outflow_downstream = state.channel_outflow_downstream.mask(
        base_condition,
        tmp_outflow_downstream / grid.iterations_main_channel
    )
    state.channel_outflow_downstream_current_timestep = state.channel_outflow_downstream_current_timestep.mask(
        base_condition,
        state.channel_outflow_downstream_current_timestep - state.channel_outflow_downstream
    )
    state.channel_flow = state.channel_flow.mask(
        base_condition,
        state.channel_flow - state.channel_outflow_downstream
    )
    
    return state