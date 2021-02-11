import numpy as np
import numexpr as ne

from mosartwmpy.state.state import State

def update_hillslope_state(state: State, base_condition: np.ndarray) -> None:
    """Updates the depth of water remaining in the hillslope.

    Args:
        state (State): the current model state; will be mutated
        base_condition (np.ndarray): a boolean array representing where the update should occur in the state
    """
    
    state.hillslope_depth = calculate_hillslope_depth(base_condition, state.hillslope_storage, state.hillslope_depth)

calculate_hillslope_depth = ne.NumExpr(
    'where('
        'base_condition,'
        'hillslope_storage,'
        'hillslope_depth'
    ')',
    (('base_condition', np.bool), ('hillslope_storage', np.float64), ('hillslope_depth', np.float64))
)