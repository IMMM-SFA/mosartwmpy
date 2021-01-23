import numpy as np

from mosartwmpy.subnetwork.state import update_subnetwork_state

def subnetwork_routing(state, grid, parameters, config, delta_t):
    # perform the subnetwork (tributary) routing
    # TODO describe what is happening here
    
    state.channel_lateral_flow_hillslope = state.zeros
    local_delta_t = (delta_t / config.get('simulation.routing_iterations') / grid.iterations_subnetwork)
    
    # step through max iterations, masking out the unnecessary cells each time
    base_condition = (grid.mosart_mask > 0) & state.euler_mask
    sub_condition = grid.subnetwork_length > grid.hillslope_length # has tributaries
    
    for _ in np.arange(np.nanmax(grid.iterations_subnetwork)):
        iteration_condition = base_condition & (grid.iterations_subnetwork > _)

        state.subnetwork_flow_velocity = np.where(
            iteration_condition & sub_condition,
            np.where(
                state.subnetwork_hydraulic_radii > 0,
                (state.subnetwork_hydraulic_radii ** (2/3)) * np.sqrt(grid.subnetwork_slope) / grid.subnetwork_manning,
                0
            ),
            state.subnetwork_flow_velocity
        )
        
        state.subnetwork_discharge = np.where(
            iteration_condition,
            np.where(
                sub_condition,
                -state.subnetwork_flow_velocity * state.subnetwork_cross_section_area,
                -state.subnetwork_lateral_inflow
            ),
            state.subnetwork_discharge
        )
        
        condition = (
            iteration_condition &
            sub_condition &
            ((state.subnetwork_storage + (state.subnetwork_lateral_inflow + state.subnetwork_discharge) * local_delta_t) < parameters.tiny_value)
        )
        
        state.subnetwork_discharge = np.where(
            condition,
            -(state.subnetwork_lateral_inflow + state.subnetwork_storage / local_delta_t),
            state.subnetwork_discharge
        )
        
        state.subnetwork_flow_velocity = np.where(
            condition & (state.subnetwork_cross_section_area > 0),
            -state.subnetwork_discharge / state.subnetwork_cross_section_area,
            state.subnetwork_flow_velocity
        )
        
        state.subnetwork_delta_storage = np.where(
            iteration_condition,
            state.subnetwork_lateral_inflow + state.subnetwork_discharge,
            state.subnetwork_delta_storage
        )
        
        # update storage
        state.subnetwork_storage_previous_timestep = np.where(
            iteration_condition,
            state.subnetwork_storage,
            state.subnetwork_storage_previous_timestep
        )
        state.subnetwork_storage = np.where(
            iteration_condition,
            state.subnetwork_storage + state.subnetwork_delta_storage * local_delta_t,
            state.subnetwork_storage
        )
        
        update_subnetwork_state(state, grid, parameters, iteration_condition)
        
        state.channel_lateral_flow_hillslope = np.where(
            iteration_condition,
            state.channel_lateral_flow_hillslope - state.subnetwork_discharge,
            state.channel_lateral_flow_hillslope
        )
    
    # average lateral flow over substeps
    state.channel_lateral_flow_hillslope = np.where(
        base_condition,
        state.channel_lateral_flow_hillslope / grid.iterations_subnetwork,
        state.channel_lateral_flow_hillslope
    )