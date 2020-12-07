from mosart.hillslope.state import update_hillslope_state

def hillslope_routing(state, grid, parameters, config, delta_t):
    # perform the hillslope routing for the whole grid
    # TODO describe what is happening heres
    
    base_condition = (grid.mosart_mask.gt(0) & state.euler_mask)
    
    velocity_hillslope = state.zeros.mask(
        base_condition & state.hillslope_depth.gt(0),
        (state.hillslope_depth ** (2/3)) * (grid.hillslope ** (1/2)) / grid.hillslope_manning
    )
    
    state.hillslope_overland_flow = state.hillslope_overland_flow.mask(
        base_condition,
        -state.hillslope_depth * velocity_hillslope * grid.drainage_density
    )
    state.hillslope_overland_flow = state.hillslope_overland_flow.mask(
        base_condition &
        state.hillslope_overland_flow.lt(0) &
        (state.hillslope_storage + delta_t * (state.hillslope_surface_runoff + state.hillslope_overland_flow)).lt(parameters.tiny_value),
        -(state.hillslope_surface_runoff + state.hillslope_storage / delta_t)
    )
    
    state.hillslope_delta_storage = state.hillslope_delta_storage.mask(
        base_condition,
        state.hillslope_surface_runoff + state.hillslope_overland_flow
    )
    
    state.hillslope_storage = state.hillslope_storage.mask(
        base_condition,
        state.hillslope_storage + delta_t * state.hillslope_delta_storage
    )
    
    state = update_hillslope_state(state, grid, parameters, config, base_condition)
    
    state.subnetwork_lateral_inflow = state.subnetwork_lateral_inflow.mask(
        base_condition,
        (state.hillslope_subsurface_runoff - state.hillslope_overland_flow) * grid.drainage_fraction * grid.area
    )
    
    return state