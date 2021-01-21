import numpy as np

def update_hillslope_state(state, grid, parameters, config, base_condition):
    # update hillslope water depth
    state.hillslope_depth[:] = np.where(
        base_condition,
        state.hillslope_storage.values,
        state.hillslope_depth.values
    )
    
    return state