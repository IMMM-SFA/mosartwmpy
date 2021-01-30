import numpy as np
import numexpr as ne

def update_hillslope_state(state, base_condition):
    # update hillslope water depth
    state.hillslope_depth = calculate_hillslope_depth(base_condition, state.hillslope_storage, state.hillslope_depth)

calculate_hillslope_depth = ne.NumExpr(
    'where('
        'base_condition,'
        'hillslope_storage,'
        'hillslope_depth'
    ')',
    (('base_condition', np.bool), ('hillslope_storage', np.float64), ('hillslope_depth', np.float64))
)