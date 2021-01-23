import numpy as np

def update_hillslope_state(state, base_condition):
    # update hillslope water depth
    state.hillslope_depth = np.where(
        base_condition,
        state.hillslope_storage,
        state.hillslope_depth
    )