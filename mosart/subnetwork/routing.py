import numpy as np
import pandas as pd

from mosart.subnetwork.state import update_subnetwork_state

def subnetwork_routing(state, grid, parameters, config, delta_t):
    # perform the subnetwork (tributary) routing
    # TODO describe what is happening here
    
    state.channel_lateral_flow_hillslope = state.zeros
    local_delta_t = (delta_t / config.get('simulation.routing_iterations') / grid.iterations_subnetwork)
    
    # step through max iterations, masking out the unnecessary cells each time
    base_condition = (grid.mosart_mask.values > 0) & state.euler_mask.values
    sub_condition = grid.subnetwork_length.values > grid.hillslope_length.values # has tributaries
    
    for _ in np.arange(grid.iterations_subnetwork.max()):
        iteration_condition = base_condition & (grid.iterations_subnetwork.values > _)

        state.subnetwork_flow_velocity = pd.DataFrame(np.where(
            iteration_condition & sub_condition,
            np.where(
                state.subnetwork_hydraulic_radii.values > 0,
                (state.subnetwork_hydraulic_radii.values ** (2/3)) * (grid.subnetwork_slope.values ** (1/2)) / grid.subnetwork_manning.values,
                0
            ),
            state.subnetwork_flow_velocity.values
        ))
        
        state.subnetwork_discharge = pd.DataFrame(np.where(
            iteration_condition,
            np.where(
                sub_condition,
                -state.subnetwork_flow_velocity.values * state.subnetwork_cross_section_area.values,
                -state.subnetwork_lateral_inflow.values
            ),
            state.subnetwork_discharge.values
        ))
        
        condition = (
            iteration_condition &
            sub_condition &
            ((state.subnetwork_storage.values + (state.subnetwork_lateral_inflow.values + state.subnetwork_discharge.values) * local_delta_t) < parameters.tiny_value)
        )
        
        state.subnetwork_discharge = pd.DataFrame(np.where(
            condition,
            -(state.subnetwork_lateral_inflow.values + state.subnetwork_storage.values / local_delta_t),
            state.subnetwork_discharge.values
        ))
        
        state.subnetwork_flow_velocity = pd.DataFrame(np.where(
            condition & (state.subnetwork_cross_section_area.values > 0),
            -state.subnetwork_discharge.values / state.subnetwork_cross_section_area.values,
            state.subnetwork_flow_velocity.values
        ))
        
        state.subnetwork_delta_storage = pd.DataFrame(np.where(
            iteration_condition,
            state.subnetwork_lateral_inflow.values + state.subnetwork_discharge.values,
            state.subnetwork_delta_storage.values
        ))
        
        # update storage
        state.subnetwork_storage_previous_timestep = pd.DataFrame(np.where(
            iteration_condition,
            state.subnetwork_storage.values,
            state.subnetwork_storage_previous_timestep.values
        ))
        state.subnetwork_storage = pd.DataFrame(np.where(
            iteration_condition,
            state.subnetwork_storage.values + state.subnetwork_delta_storage.values * local_delta_t,
            state.subnetwork_storage
        ))
        
        state = update_subnetwork_state(state, grid, parameters, config, iteration_condition)
        
        state.channel_lateral_flow_hillslope = pd.DataFrame(np.where(
            iteration_condition,
            state.channel_lateral_flow_hillslope.values - state.subnetwork_discharge.values,
            state.channel_lateral_flow_hillslope
        ))
    
    # average lateral flow over substeps
    state.channel_lateral_flow_hillslope = pd.DataFrame(np.where(
        base_condition,
        state.channel_lateral_flow_hillslope.values / grid.iterations_subnetwork.values,
        state.channel_lateral_flow_hillslope.values
    ))
    
    return state