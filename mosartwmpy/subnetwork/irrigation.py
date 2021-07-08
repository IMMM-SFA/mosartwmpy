import numba as nb

from mosartwmpy.subnetwork.state import update_subnetwork_state


@nb.jit(
    "void("
        "int64, int64[:], float64[:], float64[:], boolean[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64, float64"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def subnetwork_irrigation(
    n,
    mosart_mask,
    subnetwork_length,
    subnetwork_width,
    euler_mask,
    subnetwork_depth,
    subnetwork_storage,
    grid_cell_unmet_demand,
    grid_cell_supply,
    subnetwork_cross_section_area,
    subnetwork_wetness_perimeter,
    subnetwork_hydraulic_radii,
    irrigation_extraction_parameter,
    tiny_value,
):
    """Tracks the supply of water from the subnetwork river channels extracted into the grid cells."""

    for i in nb.prange(n):

        # is the channel deep enough to extract from
        depth_condition = (mosart_mask[i] > 0) and euler_mask[i] and (subnetwork_depth[i] >= irrigation_extraction_parameter)

        flow_volume = 1.0 * subnetwork_storage[i]

        # is there enough water to allow full extraction
        volume_condition = flow_volume >= grid_cell_unmet_demand[i]

        if depth_condition:
            if volume_condition:
                grid_cell_supply[i] = grid_cell_supply[i] + grid_cell_unmet_demand[i]
                flow_volume = flow_volume - grid_cell_unmet_demand[i]
                grid_cell_unmet_demand[i] = 0.0
            else:
                grid_cell_supply[i] = grid_cell_supply[i] + flow_volume
                grid_cell_unmet_demand[i] = grid_cell_unmet_demand[i] - flow_volume
                flow_volume = 0.0

            subnetwork_storage[i] = 1.0 * flow_volume
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
