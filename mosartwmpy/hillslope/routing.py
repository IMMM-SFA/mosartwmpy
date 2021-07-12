import numba as nb


@nb.jit(
    "void("
        "int64, float64,"
        "int64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "boolean[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64"
    ")",
    nopython=True,
    nogil=True,
    cache=True,
)
def hillslope_routing(
    i,
    delta_t,
    mosart_mask,
    hillslope,
    hillslope_manning,
    drainage_density,
    drainage_fraction,
    area,
    euler_mask,
    hillslope_depth,
    hillslope_overland_flow,
    hillslope_storage,
    hillslope_surface_runoff,
    hillslope_delta_storage,
    hillslope_subsurface_runoff,
    subnetwork_lateral_inflow,
    tiny_value,
):
    """Tracks the storage of runoff water in the hillslope and the flow of runoff water from the hillslope into the channels."""

    if ~euler_mask[i] or ~(mosart_mask[i] > 0):
        return

    if hillslope_depth[i] > 0.0:
        velocity_hillslope = (hillslope_depth[i] ** (2.0/3.0)) * (hillslope[i] ** (1.0/2.0)) / hillslope_manning[i]
    else:
        velocity_hillslope = 0.0

    hillslope_overland_flow[i] = -hillslope_depth[i] * velocity_hillslope * drainage_density[i]

    if (hillslope_overland_flow[i] < 0.0) and ((hillslope_storage[i] + delta_t * (hillslope_surface_runoff[i] + hillslope_overland_flow[i])) < tiny_value):
        hillslope_overland_flow[i] = -(hillslope_surface_runoff[i] + hillslope_storage[i] / delta_t)

    hillslope_delta_storage[i] = hillslope_surface_runoff[i] + hillslope_overland_flow[i]

    hillslope_storage[i] = hillslope_storage[i] + delta_t * hillslope_delta_storage[i]

    hillslope_depth[i] = hillslope_storage[i]

    subnetwork_lateral_inflow[i] = (hillslope_subsurface_runoff[i] - hillslope_overland_flow[i]) * drainage_fraction[i] * area[i]
