import numpy as np
import pandas as pd

def update_hillslope_state(state, grid, parameters, config, base_condition):
    # update hillslope water depth
    state.hillslope_depth = pd.DataFrame(np.where(
        base_condition,
        state.hillslope_storage.values,
        state.hillslope_depth.values
    ))
    
    return state