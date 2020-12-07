def update_subnetwork_state(state, grid, parameters, config, base_condition):
    # update the physical properties of the subnetwork
        
    # update state variables
    condition = grid.subnetwork_length.gt(0) & state.subnetwork_storage.gt(0)
    state.subnetwork_cross_section_area = state.subnetwork_cross_section_area.mask(
        base_condition,
        state.zeros.mask(
            condition,
            state.subnetwork_storage / grid.subnetwork_length
        )
    )
    state.subnetwork_depth =  state.subnetwork_depth.mask(
        base_condition,
        state.zeros.mask(
            condition & state.subnetwork_cross_section_area.gt(parameters.tiny_value),
            state.subnetwork_cross_section_area / grid.subnetwork_width
        )
    )
    state.subnetwork_wetness_perimeter = state.subnetwork_wetness_perimeter.mask(
        base_condition,
        state.zeros.mask(
            condition & state.subnetwork_depth.gt(parameters.tiny_value),
            grid.subnetwork_width + 2 * state.subnetwork_depth
        )
    )
    state.subnetwork_hydraulic_radii = state.subnetwork_hydraulic_radii.mask(
        base_condition,
        state.zeros.mask(
            condition & state.subnetwork_wetness_perimeter.gt(parameters.tiny_value),
            state.subnetwork_cross_section_area / state.subnetwork_wetness_perimeter
        )
    )
    
    return state