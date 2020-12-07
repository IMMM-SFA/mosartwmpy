import numpy as np

from mosart.subnetwork.state import update_subnetwork_state

def subnetwork_routing(state, grid, parameters, config, delta_t):
    # perform the subnetwork (tributary) routing
    # TODO describe what is happening here
    
    state.channel_lateral_flow_hillslope = state.zeros
    local_delta_t = (delta_t / config.get('simulation.routing_iterations') / grid.iterations_subnetwork)
    
    # step through max iterations, masking out the unnecessary cells each time
    base_condition = grid.mosart_mask.gt(0) & state.euler_mask
    sub_condition = grid.subnetwork_length.gt(grid.hillslope_length) # has tributaries
    
    for _ in np.arange(grid.iterations_subnetwork.max()):
        iteration_condition = base_condition & grid.iterations_subnetwork.gt(_)

        state.subnetwork_flow_velocity = state.subnetwork_flow_velocity.mask(
            iteration_condition & sub_condition,
            state.zeros.mask(
                state.subnetwork_hydraulic_radii.gt(0),
                (state.subnetwork_hydraulic_radii ** (2/3)) * (grid.subnetwork_slope ** (1/2)) / grid.subnetwork_manning
            )
        )
        
        state.subnetwork_discharge = state.subnetwork_discharge.mask(
            iteration_condition,
            (-state.subnetwork_lateral_inflow).mask(
                sub_condition,
                -state.subnetwork_flow_velocity * state.subnetwork_cross_section_area
            )
        )
        
        condition = (
            iteration_condition &
            sub_condition &
            (state.subnetwork_storage + (state.subnetwork_lateral_inflow + state.subnetwork_discharge) * local_delta_t).lt(parameters.tiny_value)
        )
        
        state.subnetwork_discharge = state.subnetwork_discharge.mask(
            condition,
            -(state.subnetwork_lateral_inflow + state.subnetwork_storage / local_delta_t)
        )
        
        state.subnetwork_flow_velocity = state.subnetwork_flow_velocity.mask(
            condition & state.subnetwork_cross_section_area.gt(0),
            -state.subnetwork_discharge / state.subnetwork_cross_section_area
        )
        
        state.subnetwork_delta_storage = state.subnetwork_delta_storage.mask(
            iteration_condition,
            state.subnetwork_lateral_inflow + state.subnetwork_discharge
        )
        
        # update storage
        state.subnetwork_storage_previous_timestep = state.subnetwork_storage_previous_timestep.mask(
            iteration_condition,
            state.subnetwork_storage
        )
        state.subnetwork_storage = state.subnetwork_storage.mask(
            iteration_condition,
            state.subnetwork_storage + state.subnetwork_delta_storage * local_delta_t
        )
        
        state = update_subnetwork_state(state, grid, parameters, config, iteration_condition)
        
        state.channel_lateral_flow_hillslope = state.channel_lateral_flow_hillslope.mask(
            iteration_condition,
            state.channel_lateral_flow_hillslope - state.subnetwork_discharge
        )
    
    # average lateral flow over substeps
    state.channel_lateral_flow_hillslope = state.channel_lateral_flow_hillslope.mask(
        base_condition,
        state.channel_lateral_flow_hillslope / grid.iterations_subnetwork
    )
    
    return state