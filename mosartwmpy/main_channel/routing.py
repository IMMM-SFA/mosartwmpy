import numba as nb

from mosartwmpy.main_channel.kinematic_wave import kinematic_wave_routing
from mosartwmpy.main_channel.state import update_main_channel_state


@nb.jit(
    "void("
        "int64, float64, int64, int64,"
        "int64[:], int64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], boolean[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64, float64, float64, float64"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def main_channel_routing(
    n,
    delta_t,
    routing_iterations,
    max_iterations_main_channel,
    iterations_main_channel,
    mosart_mask,
    channel_length,
    channel_slope,
    channel_manning,
    total_drainage_area_single,
    channel_width,
    area,
    drainage_fraction,
    grid_channel_depth,
    channel_floodplain_width,
    euler_mask,
    channel_inflow_upstream,
    channel_outflow_sum_upstream_instant,
    channel_hydraulic_radii,
    channel_flow_velocity,
    channel_outflow_downstream,
    channel_cross_section_area,
    channel_lateral_flow_hillslope,
    channel_storage,
    channel_storage_previous_timestep,
    hillslope_wetland_runoff,
    channel_delta_storage,
    channel_depth,
    channel_wetness_perimeter,
    slope_1_def,
    inverse_sin_atan_slope_1_def,
    tiny_value,
    kinematic_wave_parameter,
):
    """Tracks the storage and flow of water in the main river channels."""

    for i in nb.prange(n):

        if ~euler_mask[i] or ~(mosart_mask[i] > 0):
            continue

        local_delta_t = (delta_t / routing_iterations) / iterations_main_channel[i]
        outflow = 0.0

        # step through max iterations
        for _ in nb.prange(max_iterations_main_channel):
            if ~(iterations_main_channel[i] > _):
                continue

            # route the water
            kinematic_wave_routing(
                i,
                local_delta_t,
                channel_inflow_upstream,
                channel_outflow_sum_upstream_instant,
                channel_length,
                channel_hydraulic_radii,
                channel_flow_velocity,
                channel_slope,
                channel_manning,
                total_drainage_area_single,
                channel_width,
                channel_outflow_downstream,
                channel_cross_section_area,
                channel_lateral_flow_hillslope,
                channel_storage,
                hillslope_wetland_runoff,
                area,
                drainage_fraction,
                channel_delta_storage,
                tiny_value,
                kinematic_wave_parameter,
            )

            # update storage
            channel_storage_previous_timestep[i] = 1.0 * channel_storage[i]
            channel_storage[i] = channel_storage[i] + channel_delta_storage[i] * local_delta_t

            # update channel state
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

            # track the outflow
            outflow = outflow + channel_outflow_downstream[i]

        # update the final outflow
        channel_outflow_downstream[i] = outflow / iterations_main_channel[i]
