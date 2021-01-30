import numpy as np
import numexpr as ne

from mosartwmpy.main_channel.kinematic_wave import kinematic_wave_routing
from mosartwmpy.main_channel.state import update_main_channel_state
from mosartwmpy.utilities.timing import timing

#@timing
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
        state.channel_storage_previous_timestep = calculate_channel_storage_previous_timestep(iteration_condition, state.channel_storage, state.channel_storage_previous_timestep)
        state.channel_storage = calculate_channel_storage(iteration_condition, state.channel_storage, state.channel_delta_storage, local_delta_t)
        
        # update channel state
        update_main_channel_state(state, grid, parameters, iteration_condition)
        
        # update outflow tracking
        tmp_outflow_downstream = calculate_tmp_outflow_downstream(iteration_condition, tmp_outflow_downstream, state.channel_outflow_downstream)
    
    # update outflow
    state.channel_outflow_downstream = calculate_channel_outflow_downstream(base_condition, tmp_outflow_downstream, grid.iterations_main_channel, state.channel_outflow_downstream)


calculate_channel_storage_previous_timestep = ne.NumExpr(
    'where('
        'iteration_condition,'
        'channel_storage,'
        'channel_storage_previous_timestep'
    ')',
    (('iteration_condition', np.bool), ('channel_storage', np.float64), ('channel_storage_previous_timestep', np.float64))
)

calculate_channel_storage = ne.NumExpr(
    'where('
        'iteration_condition,'
        'channel_storage + channel_delta_storage * local_delta_t,'
        'channel_storage'
    ')',
    (('iteration_condition', np.bool), ('channel_storage', np.float64), ('channel_delta_storage', np.float64), ('local_delta_t', np.float64))
)

calculate_tmp_outflow_downstream = ne.NumExpr(
    'where('
        'iteration_condition,'
        'tmp_outflow_downstream + channel_outflow_downstream,'
        'tmp_outflow_downstream'
    ')',
    (('iteration_condition', np.bool), ('tmp_outflow_downstream', np.float64), ('channel_outflow_downstream', np.float64))
)

calculate_channel_outflow_downstream = ne.NumExpr(
    'where('
        'base_condition,'
        'tmp_outflow_downstream / iterations_main_channel,'
        'channel_outflow_downstream'
    ')',
    (('base_condition', np.bool), ('tmp_outflow_downstream', np.float64), ('iterations_main_channel', np.float64), ('channel_outflow_downstream', np.float64))
)