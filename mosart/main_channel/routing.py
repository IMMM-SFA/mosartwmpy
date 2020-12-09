import numpy as np
import pandas as pd

from mosart.main_channel.kinematic_wave import kinematic_wave_routing
from mosart.main_channel.state import update_main_channel_state

def main_channel_routing(state, grid, parameters, config, delta_t):
    # perform the main channel routing
    # TODO describe what is happening here
    
    tmp_outflow_downstream = state.zeros.values
    local_delta_t = (delta_t / config.get('simulation.routing_iterations') / grid.iterations_main_channel)
    
    # step through max iterations, masking out the unnecessary cells each time
    base_condition = (grid.mosart_mask.values > 0) & state.euler_mask.values
    for _ in np.arange(grid.iterations_main_channel.max()):
        iteration_condition = base_condition & (grid.iterations_main_channel.values > _)
    
        # routing
        routing_method = config.get('simulation.routing_method', 1)
        if routing_method == 1:
            state = kinematic_wave_routing(state, grid, parameters, config, local_delta_t, iteration_condition)
        else:
            raise Exception(f"Error - Routing method {routing_method} not implemented.")
        
        # update storage
        state.channel_storage_previous_timestep = pd.DataFrame(np.where(
            iteration_condition,
            state.channel_storage.values,
            state.channel_storage_previous_timestep.values
        ))
        state.channel_storage = pd.DataFrame(np.where(
            iteration_condition,
            state.channel_storage.values + state.channel_delta_storage.values * local_delta_t,
            state.channel_storage.values
        ))
        
        state = update_main_channel_state(state, grid, parameters, config, iteration_condition)
        
        # update outflow tracking
        tmp_outflow_downstream = np.where(
            iteration_condition,
            tmp_outflow_downstream + state.channel_outflow_downstream.values,
            tmp_outflow_downstream
        )
    
    # update outflow
    state.channel_outflow_downstream = pd.DataFrame(np.where(
        base_condition,
        tmp_outflow_downstream / grid.iterations_main_channel.values,
        state.channel_outflow_downstream
    ))
    state.channel_outflow_downstream_current_timestep = pd.DataFrame(np.where(
        base_condition,
        state.channel_outflow_downstream_current_timestep.values - state.channel_outflow_downstream.values,
        state.channel_outflow_downstream_current_timestep
    ))
    
    state.channel_flow = pd.DataFrame(np.where(
        base_condition,
        state.channel_flow.values - state.channel_outflow_downstream.values,
        state.channel_flow 
    ))
    
    return state