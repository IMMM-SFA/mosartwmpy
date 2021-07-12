import numba as nb


@nb.jit(
    "void("
        "int64, float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64, float64, float64"
    ")",
    nopython=True,
    nogil=True,
    cache=True,
)
def update_main_channel_state(
    i,
    channel_length,
    grid_channel_depth,
    channel_width,
    channel_floodplain_width,
    channel_storage,
    channel_cross_section_area,
    channel_depth,
    channel_wetness_perimeter,
    channel_hydraulic_radii,
    tiny_value,
    slope_1_def,
    inverse_sin_atan_slope_1_def,
):
    """Updates the physical properties of the main river channel based on current state."""

    storage_condition = (channel_length[i] > 0.0) and (channel_storage[i] > 0.0)

    if storage_condition:
        channel_cross_section_area[i] = channel_storage[i] / channel_length[i]
    else:
        channel_cross_section_area[i] = 0.0

    not_flooded = (channel_cross_section_area[i] - (grid_channel_depth[i] * channel_width[i])) <= tiny_value

    delta_area = channel_cross_section_area[i] - grid_channel_depth[i] * channel_width[i] - (channel_width[i] + channel_floodplain_width[i]) * slope_1_def * ((channel_floodplain_width[i] - channel_width[i]) / 2.0) / 2.0

    # Function for estimating maximum water depth assuming rectangular channel and trapezoidal flood plain
    # here assuming the channel cross-section consists of three parts, from bottom to up,
    # part 1 is a rectangular with bankfull depth (rdep) and bankfull width (rwid)
    # part 2 is a trapezoidal, bottom width rwid and top width rwid0, height slope*((rwid0-rwid)/2)
    # part 3 is a rectangular with the width rwid0
    if storage_condition and (channel_cross_section_area[i] > tiny_value):
        if not_flooded:
            channel_depth[i] = channel_cross_section_area[i] / channel_width[i]
        else:
            if delta_area > tiny_value:
                channel_depth[i] = grid_channel_depth[i] + slope_1_def * ((channel_floodplain_width[i] - channel_width[i]) / 2.0) + delta_area / channel_floodplain_width[i]
            else:
                channel_depth[i] = grid_channel_depth[i] + (-channel_width[i] + (((channel_width[i] ** 2.0) + 4.0 * (channel_cross_section_area[i] - grid_channel_depth[i] * channel_width[i]) / slope_1_def) ** (1.0/2.0))) * slope_1_def / 2.0
    else:
        channel_depth[i] = 0.0

    not_flooded = (channel_depth[i] <= (grid_channel_depth[i] + tiny_value))

    delta_depth = channel_depth[i] - grid_channel_depth[i] - ((channel_floodplain_width[i] - channel_width[i]) / 2.0) * slope_1_def

    # Function for estimating wetness perimeter based on same assumptions as above
    if storage_condition and (channel_depth[i] >= tiny_value):
        if not_flooded:
            channel_wetness_perimeter[i] = channel_width[i] + 2.0 * channel_depth[i]
        else:
            if delta_depth > tiny_value:
                channel_wetness_perimeter[i] = channel_width[i] + 2.0 * (grid_channel_depth[i] + ((channel_floodplain_width[i] - channel_width[i]) / 2.0) * slope_1_def * inverse_sin_atan_slope_1_def + delta_depth)
            else:
                channel_wetness_perimeter[i] = channel_width[i] + 2.0 * (grid_channel_depth[i] + (channel_depth[i] - grid_channel_depth[i]) * inverse_sin_atan_slope_1_def)
    else:
        channel_wetness_perimeter[i] = 0.0

    if storage_condition and (channel_wetness_perimeter[i] > tiny_value):
        channel_hydraulic_radii[i] = channel_cross_section_area[i] / channel_wetness_perimeter[i]
    else:
        channel_hydraulic_radii[i] = 0.0
