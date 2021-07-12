import numba as nb

from mosartwmpy.main_channel.state import update_main_channel_state


@nb.jit(
    "void("
        "int64, int64[:], float64[:], float64[:], float64[:], float64[:], boolean[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64, float64, float64, float64, float64, float64"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def main_channel_irrigation(
    n,
    mosart_mask,
    channel_length,
    grid_channel_depth,
    channel_width,
    channel_floodplain_width,
    euler_mask,
    channel_depth,
    channel_storage,
    grid_cell_unmet_demand,
    grid_cell_supply,
    channel_cross_section_area,
    channel_wetness_perimeter,
    channel_hydraulic_radii,
    tiny_value,
    tinier_value,
    slope_1_def,
    inverse_sin_atan_slope_1_def,
    irrigation_extraction_parameter,
    irrigation_extraction_maximum_fraction,
):
    """Tracks the supply of water from the main river channel extracted into the grid cells."""

    for i in nb.prange(n):

        # is the channel deep enough to extract from
        depth_condition = (mosart_mask[i] > 0) and euler_mask[i] and (channel_depth[i] >= irrigation_extraction_parameter)

        # skip if certain parameters are very close to zero
        tiny_condition = (channel_storage[i] > tinier_value) and (grid_cell_unmet_demand[i] > tinier_value) and (channel_length[i] > tinier_value)

        flow_volume = 1.0 * channel_storage[i]

        # is there enough water to allow full extraction
        volume_condition = (irrigation_extraction_maximum_fraction * flow_volume) >= grid_cell_unmet_demand[i]

        if depth_condition and tiny_condition:
            if volume_condition:
                grid_cell_supply[i] = grid_cell_supply[i] + grid_cell_unmet_demand[i]
                flow_volume = flow_volume - grid_cell_unmet_demand[i]
                grid_cell_unmet_demand[i] = 0.0
            else:
                grid_cell_supply[i] = grid_cell_supply[i] + irrigation_extraction_maximum_fraction * flow_volume
                grid_cell_unmet_demand[i] = grid_cell_unmet_demand[i] - irrigation_extraction_maximum_fraction * flow_volume
                flow_volume = flow_volume - irrigation_extraction_maximum_fraction * flow_volume

            channel_storage[i] = 1.0 * flow_volume
            update_main_channel_state(
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
                inverse_sin_atan_slope_1_def
            )
