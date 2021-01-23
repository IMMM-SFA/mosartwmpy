import numpy as np

from mosartwmpy.main_channel.kinematic_wave import kinematic_wave_routing
from mosartwmpy.main_channel.state import update_main_channel_state

def main_channel_routing(state, grid, parameters, config, delta_t):
    # perform the main channel routing
    # TODO describe what is happening here
    
    tmp_outflow_downstream = 0.0 * state.zeros
    local_delta_t = (delta_t / config.get('simulation.routing_iterations') / grid.iterations_main_channel)
    
    # step through max iterations, masking out the unnecessary cells each time
    base_condition = (grid.mosart_mask > 0) & state.euler_mask
    for _ in np.arange(np.nanmax(grid.iterations_main_channel)):
        iteration_condition = base_condition & (grid.iterations_main_channel > _)
    
        # routing
        routing_method = config.get('simulation.routing_method', 1)
        if routing_method == 1:
            kinematic_wave_routing(state, grid, parameters, local_delta_t, iteration_condition)
        else:
            raise Exception(f"Error - Routing method {routing_method} not implemented.")
        
        # update storage
        state.channel_storage_previous_timestep = np.where(
            iteration_condition,
            state.channel_storage,
            state.channel_storage_previous_timestep
        )
        state.channel_storage = np.where(
            iteration_condition,
            state.channel_storage + state.channel_delta_storage * local_delta_t,
            state.channel_storage
        )
        
        update_main_channel_state(state, grid, parameters, iteration_condition)
        
        # update outflow tracking
        tmp_outflow_downstream = np.where(
            iteration_condition,
            tmp_outflow_downstream + state.channel_outflow_downstream,
            tmp_outflow_downstream
        )
    
    # update outflow
    state.channel_outflow_downstream = np.where(
        base_condition,
        tmp_outflow_downstream / grid.iterations_main_channel,
        state.channel_outflow_downstream
    )