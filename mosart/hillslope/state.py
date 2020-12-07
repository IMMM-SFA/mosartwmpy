def update_hillslope_state(state, grid, parameters, config, base_condition):
    # update hillslope water depth
    state.hillslope_depth = state.hillslope_depth.mask(
        base_condition,
        state.hillslope_storage
    )
    
    return state