import numba as nb

from mosartwmpy.subnetwork.state import update_subnetwork_state
from mosartwmpy.utilities.timing import timing


# @timing
@nb.jit(
    "void("
        "int64, float64, int64, int64,"
        "int64[:], int64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "boolean[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def subnetwork_routing(
    n,
    delta_t,
    routing_iterations,
    max_iterations_subnetwork,
    iterations_subnetwork,
    mosart_mask,
    subnetwork_slope,
    subnetwork_manning,
    subnetwork_length,
    subnetwork_width,
    hillslope_length,
    euler_mask,
    channel_lateral_flow_hillslope,
    subnetwork_flow_velocity,
    subnetwork_discharge,
    subnetwork_lateral_inflow,
    subnetwork_storage,
    subnetwork_storage_previous_timestep,
    subnetwork_delta_storage,
    subnetwork_depth,
    subnetwork_cross_section_area,
    subnetwork_wetness_perimeter,
    subnetwork_hydraulic_radii,
    tiny_value,
):
    """Tracks the storage and flow of water in the subnetwork river channels."""

    for i in nb.prange(n):

        local_delta_t = (delta_t / routing_iterations) / iterations_subnetwork[i]
        channel_lateral_flow_hillslope[i] = 0.0

        if ~euler_mask[i] or ~(mosart_mask[i] > 0):
            continue

        has_tributaries = subnetwork_length[i] > hillslope_length[i]

        # step through max iterations
        for _ in nb.prange(max_iterations_subnetwork):
            if ~(iterations_subnetwork[i] > _):
                continue

            if has_tributaries:
                if subnetwork_hydraulic_radii[i] > 0.0:
                    subnetwork_flow_velocity[i] = (subnetwork_hydraulic_radii[i] ** (2.0/3.0)) * (subnetwork_slope[i] ** (1.0/2.0)) / subnetwork_manning[i]
                else:
                    subnetwork_flow_velocity[i] = 0.0

            if has_tributaries:
                subnetwork_discharge[i] = -subnetwork_flow_velocity[i] * subnetwork_cross_section_area[i]
            else:
                subnetwork_discharge[i] = -1.0 * subnetwork_lateral_inflow[i]

            discharge_condition = has_tributaries and ((subnetwork_storage[i] + (subnetwork_lateral_inflow[i] + subnetwork_discharge[i]) * local_delta_t) < tiny_value)

            if discharge_condition:
                subnetwork_discharge[i] = -(subnetwork_lateral_inflow[i] + subnetwork_storage[i] / local_delta_t)

            if discharge_condition and (subnetwork_cross_section_area[i] > 0.0):
                subnetwork_flow_velocity[i] = -subnetwork_discharge[i] / subnetwork_cross_section_area[i]

            subnetwork_delta_storage[i] = subnetwork_lateral_inflow[i] + subnetwork_discharge[i]

            # update storage
            subnetwork_storage_previous_timestep[i] = subnetwork_storage[i]
            subnetwork_storage[i] = subnetwork_storage[i] + subnetwork_delta_storage[i] * local_delta_t

            # update subnetwork state
            update_subnetwork_state(
                i,
                subnetwork_length,
                subnetwork_width,
                subnetwork_storage,
                subnetwork_cross_section_area,
                subnetwork_depth,
                subnetwork_wetness_perimeter,
                subnetwork_hydraulic_radii,
                tiny_value,
            )

            channel_lateral_flow_hillslope[i] = channel_lateral_flow_hillslope[i] - subnetwork_discharge[i]

        # average lateral flow over substeps
        channel_lateral_flow_hillslope[i] = channel_lateral_flow_hillslope[i] / iterations_subnetwork[i]
