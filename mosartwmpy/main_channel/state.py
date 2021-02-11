import numpy as np
import numexpr as ne

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State

def update_main_channel_state(state: State, grid: Grid, parameters: Parameters, base_condition: np.ndarray) -> None:
    """Updates the physical properties of the main river channel based on current state.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        parameters (Parameters): the model parameters
        base_condition (np.ndarray): a boolean array representing where the update should occur in the state
    """
    
    condition = calculate_storage_condition(grid.channel_length, state.channel_storage)
    state.channel_cross_section_area = calculate_channel_cross_section_area(base_condition, condition, state.channel_storage, grid.channel_length, state.channel_cross_section_area)

    # Function for estimating maximum water depth assuming rectangular channel and tropezoidal flood plain
    # here assuming the channel cross-section consists of three parts, from bottom to up,
    # part 1 is a rectangular with bankfull depth (rdep) and bankfull width (rwid)
    # part 2 is a tropezoidal, bottom width rwid and top width rwid0, height 0.1*((rwid0-rwid)/2), assuming slope is 0.1
    # part 3 is a rectagular with the width rwid0
    not_flooded = calculate_not_flooded_depth_condition(state.channel_cross_section_area, grid.grid_channel_depth, grid.channel_width, parameters.tiny_value)
    delta_area = calculate_delta_area(state.channel_cross_section_area, grid.grid_channel_depth, grid.channel_width, grid.channel_floodplain_width, parameters.slope_1_def)
    state.channel_depth = calculate_channel_depth(base_condition, condition, state.channel_cross_section_area, parameters.tiny_value, not_flooded, grid.channel_width, delta_area, grid.grid_channel_depth, parameters.slope_1_def, grid.channel_floodplain_width, state.channel_depth)

    # Function for estimating wetness perimeter based on same assumptions as above
    not_flooded = calculate_not_flooded_wetness_condition(state.channel_depth, grid.grid_channel_depth, parameters.tiny_value)
    delta_depth = calculate_delta_depth(state.channel_depth, grid.grid_channel_depth, grid.channel_floodplain_width, grid.channel_width, parameters.slope_1_def)
    state.channel_wetness_perimeter = calculate_channel_wetness_perimeter(base_condition, condition, state.channel_depth, parameters.tiny_value, not_flooded, grid.channel_width, delta_depth, grid.grid_channel_depth, grid.channel_floodplain_width, parameters.slope_1_def, parameters.inverse_sin_atan_slope_1_def, state.channel_wetness_perimeter)

    state.channel_hydraulic_radii = calculate_channel_hydraulic_radii(base_condition, condition, state.channel_wetness_perimeter, parameters.tiny_value, state.channel_cross_section_area, state.channel_hydraulic_radii)


calculate_storage_condition = ne.NumExpr(
    '(channel_length > 0) & (channel_storage > 0)',
    (('channel_length', np.float64), ('channel_storage', np.float64))
)

calculate_channel_cross_section_area = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            'storage_condition,'
            'channel_storage / channel_length,'
            '0'
        '),'
        'channel_cross_section_area'
    ')',
    (('base_condition', np.bool), ('storage_condition', np.bool), ('channel_storage', np.float64), ('channel_length', np.float64), ('channel_cross_section_area', np.float64))
)

calculate_not_flooded_depth_condition = ne.NumExpr(
    '(channel_cross_section_area - (grid_channel_depth * channel_width)) <= tiny_value',
    (('channel_cross_section_area', np.float64), ('grid_channel_depth', np.float64), ('channel_width', np.float64), ('tiny_value', np.float64))
)

calculate_delta_area = ne.NumExpr(
    'channel_cross_section_area - grid_channel_depth  * channel_width - (channel_width + channel_floodplain_width) * slope_1_def * ((channel_floodplain_width - channel_width) / 2.0) / 2.0',
    (('channel_cross_section_area', np.float64), ('grid_channel_depth', np.float64), ('channel_width', np.float64), ('channel_floodplain_width', np.float64), ('slope_1_def', np.float64))
)

calculate_channel_depth = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            'storage_condition & (channel_cross_section_area > tiny_value),'
            'where('
                'not_flooded_condition,'
                'channel_cross_section_area / channel_width,'
                'where('
                    'delta_area > tiny_value,'
                    'grid_channel_depth + slope_1_def * ((channel_floodplain_width  - channel_width) / 2.0) + delta_area / channel_floodplain_width,'
                    'grid_channel_depth + (-channel_width + (((channel_width ** 2) + 4 * (channel_cross_section_area  - grid_channel_depth * channel_width) / slope_1_def) ** (1/2))) * slope_1_def / 2.0'
                ')'
            '),'
            '0'
        '),'
        'channel_depth'
    ')',
    (('base_condition', np.bool), ('storage_condition', np.bool), ('channel_cross_section_area', np.float64), ('tiny_value', np.float64), ('not_flooded_condition', np.bool), ('channel_width', np.float64), ('delta_area', np.float64), ('grid_channel_depth', np.float64), ('slope_1_def', np.float64), ('channel_floodplain_width', np.float64), ('channel_depth', np.float64))
)

calculate_not_flooded_wetness_condition = ne.NumExpr(
    'channel_depth <= (grid_channel_depth + tiny_value)',
    (('channel_depth', np.float64), ('grid_channel_depth', np.float64), ('tiny_value', np.float64))
)

calculate_delta_depth = ne.NumExpr(
    'channel_depth - grid_channel_depth - ((channel_floodplain_width -  channel_width) / 2.0) * slope_1_def',
    (('channel_depth', np.float64), ('grid_channel_depth', np.float64), ('channel_floodplain_width', np.float64), ('channel_width', np.float64), ('slope_1_def', np.float64))
)

calculate_channel_wetness_perimeter = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            'storage_condition & (channel_depth >= tiny_value),'
            'where('
                'not_flooded_condition,'
                'channel_width + 2 * channel_depth,'
                'where('
                    'delta_depth > tiny_value,'
                    'channel_width + 2 * (grid_channel_depth + ((channel_floodplain_width - channel_width) / 2.0) * slope_1_def * inverse_sin_atan_slope_1_def + delta_depth),'
                    'channel_width + 2 * (grid_channel_depth + (channel_depth - grid_channel_depth) * inverse_sin_atan_slope_1_def)'
                ')'
            '),'
            '0'
        '),'
        'channel_wetness_perimeter'
    ')',
    (('base_condition', np.bool), ('storage_condition', np.bool), ('channel_depth', np.float64), ('tiny_value', np.float64), ('not_flooded_condition', np.bool), ('channel_width', np.float64), ('delta_depth', np.float64), ('grid_channel_depth', np.float64), ('channel_floodplain_width', np.float64), ('slope_1_def', np.float64), ('inverse_sin_atan_slope_1_def', np.float64), ('channel_wetness_perimeter', np.float64))
)

calculate_channel_hydraulic_radii = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            'storage_condition & (channel_wetness_perimeter > tiny_value),'
            'channel_cross_section_area / channel_wetness_perimeter,'
            '0'
        '),'
        'channel_hydraulic_radii'
    ')',
    (('base_condition', np.bool), ('storage_condition', np.bool), ('channel_wetness_perimeter', np.float64), ('tiny_value', np.float64), ('channel_cross_section_area', np.float64), ('channel_hydraulic_radii', np.float64))
)