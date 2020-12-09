import numpy as np
import pandas as pd

def update_main_channel_state(state, grid, parameters, config, base_condition):
    # update the physical properties of the main channel
    
    condition = (grid.channel_length.values > 0) & (state.channel_storage > 0)
    state.channel_cross_section_area = pd.DataFrame(np.where(
        base_condition,
        np.where(
            condition,
            state.channel_storage.values / grid.channel_length.values,
            0
        ),
        state.channel_cross_section_area.values
    ))
    # Function for estimating maximum water depth assuming rectangular channel and tropezoidal flood plain
    # here assuming the channel cross-section consists of three parts, from bottom to up,
    # part 1 is a rectangular with bankfull depth (rdep) and bankfull width (rwid)
    # part 2 is a tropezoidal, bottom width rwid and top width rwid0, height 0.1*((rwid0-rwid)/2), assuming slope is 0.1
    # part 3 is a rectagular with the width rwid0
    not_flooded = (state.channel_cross_section_area.values - (grid.channel_depth.values * grid.channel_width.values)) <= parameters.tiny_value
    delta_area = state.channel_cross_section_area.values - grid.channel_depth.values  * grid.channel_width.values - (grid.channel_width.values + grid.channel_floodplain_width.values) * parameters.slope_1_def * ((grid.channel_floodplain_width.values - grid.channel_width.values) / 2.0) / 2.0
    equivalent_depth_condition =  delta_area > parameters.tiny_value
    state.channel_depth = pd.DataFrame(np.where(
        base_condition,
        np.where(
            condition & (state.channel_cross_section_area.values > parameters.tiny_value),
            np.where(
                not_flooded,
                state.channel_cross_section_area.values / grid.channel_width.values,
                np.where(
                    equivalent_depth_condition,
                    grid.channel_depth.values + parameters.slope_1_def * ((grid.channel_floodplain_width.values  - grid.channel_width.values) / 2.0) + delta_area / grid.channel_floodplain_width.values,
                    grid.channel_depth.values + (-grid.channel_width.values + (((grid.channel_width.values ** 2) + 4 * (state.channel_cross_section_area.values  - grid.channel_depth.values * grid.channel_width.values) / parameters.slope_1_def) ** (1/2))) * parameters.slope_1_def / 2.0
                )
            ),
            0
        ),
        state.channel_depth.values
    ))
    # Function for estimating wetness perimeter based on same assumptions as above
    not_flooded = state.channel_depth.values <= (grid.channel_depth.values + parameters.tiny_value)
    delta_depth = state.channel_depth.values - grid.channel_depth.values - ((grid.channel_floodplain_width.values -  grid.channel_width.values) / 2.0) * parameters.slope_1_def
    equivalent_depth_condition = delta_depth > parameters.tiny_value
    state.channel_wetness_perimeter = pd.DataFrame(np.where(
        base_condition,
        np.where(
            condition & (state.channel_depth.values >= parameters.tiny_value),
            np.where(
                not_flooded,
                grid.channel_width.values + 2 * state.channel_depth.values,
                np.where(
                    equivalent_depth_condition,
                    grid.channel_width.values + 2 * (grid.channel_depth.values + ((grid.channel_floodplain_width.values - grid.channel_width.values) / 2.0) * parameters.slope_1_def * parameters.inverse_sin_atan_slope_1_def + delta_depth),
                    grid.channel_width.values + 2 * (grid.channel_depth.values + (state.channel_depth.values - grid.channel_depth.values) * parameters.inverse_sin_atan_slope_1_def)
                )
            ),
            0
        ),
        state.channel_wetness_perimeter.values
    ))
    state.channel_hydraulic_radii = pd.DataFrame(np.where(
        base_condition,
        np.where(
            condition & (state.channel_wetness_perimeter.values > parameters.tiny_value),
            state.channel_cross_section_area.values / state.channel_wetness_perimeter.values,
            0
        ),
        state.channel_hydraulic_radii
    ))
    
    return state