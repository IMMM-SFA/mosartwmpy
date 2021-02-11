import numpy as np
import numexpr as ne

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State

def update_subnetwork_state(state: State, grid: Grid, parameters: Parameters, base_condition: np.ndarray) -> None:
    """Updates the physical properties of the subnetwork river channels based on current state.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        parameters (Parameters): the model parameters
        base_condition (np.ndarray): a boolean array representing where the update should occur in the state
    """
        
    # update state variables
    length_condition = calculate_length_condition(grid.subnetwork_length, state.subnetwork_storage)
    state.subnetwork_cross_section_area = calculate_subnetwork_cross_section_area(base_condition, length_condition, state.subnetwork_storage, grid.subnetwork_length, state.subnetwork_cross_section_area)
    state.subnetwork_depth = calculate_subnetwork_depth(base_condition, length_condition, state.subnetwork_cross_section_area, parameters.tiny_value, grid.subnetwork_width, state.subnetwork_depth)
    state.subnetwork_wetness_perimeter = calculate_subnetwork_wetness_perimeter(base_condition, length_condition, state.subnetwork_depth, parameters.tiny_value, grid.subnetwork_width, state.subnetwork_wetness_perimeter)
    state.subnetwork_hydraulic_radii = calculate_subnetwork_hydraulic_radii(base_condition, length_condition, state.subnetwork_wetness_perimeter, parameters.tiny_value, state.subnetwork_cross_section_area, state.subnetwork_hydraulic_radii)


calculate_length_condition = ne.NumExpr(
    '(subnetwork_length > 0) & (subnetwork_storage > 0)',
    (('subnetwork_length', np.float64), ('subnetwork_storage', np.float64))
)

calculate_subnetwork_cross_section_area = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            'length_condition,'
            'subnetwork_storage / subnetwork_length,'
            '0'
        '),'
        'subnetwork_cross_section_area'
    ')',
    (('base_condition', np.bool), ('length_condition', np.bool), ('subnetwork_storage', np.float64), ('subnetwork_length', np.float64), ('subnetwork_cross_section_area', np.float64))
)

calculate_subnetwork_depth = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            'length_condition & (subnetwork_cross_section_area > tiny_value),'
            'subnetwork_cross_section_area / subnetwork_width,'
            '0'
        '),'
        'subnetwork_depth'
    ')',
    (('base_condition', np.bool), ('length_condition', np.bool), ('subnetwork_cross_section_area', np.float64), ('tiny_value', np.float64), ('subnetwork_width', np.float64), ('subnetwork_depth', np.float64))
)

calculate_subnetwork_wetness_perimeter = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            'length_condition & (subnetwork_depth > tiny_value),'
            'subnetwork_width + 2 * subnetwork_depth,'
            '0'
        '),'
        'subnetwork_wetness_perimeter'
    ')',
    (('base_condition', np.bool), ('length_condition', np.bool), ('subnetwork_depth', np.float64), ('tiny_value', np.float64), ('subnetwork_width', np.float64), ('subnetwork_wetness_perimeter', np.float64))
)

calculate_subnetwork_hydraulic_radii = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            'length_condition & (subnetwork_wetness_perimeter > tiny_value),'
            'subnetwork_cross_section_area / subnetwork_wetness_perimeter,'
            '0'
        '),'
        'subnetwork_hydraulic_radii'
    ')',
    (('base_condition', np.bool), ('length_condition', np.bool),  ('subnetwork_wetness_perimeter', np.float64), ('tiny_value', np.float64), ('subnetwork_cross_section_area', np.float64), ('subnetwork_hydraulic_radii', np.float64))
)