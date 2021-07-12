import numba as nb


@nb.jit(
    "void("
        "int64, float64, float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64, float64"
    ")",
    nopython=True,
    nogil=True,
    cache=True,
)
def kinematic_wave_routing(
    i,
    delta_t,
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
):
    """Tracks the storage and flow of water in the main channel using the kinematic wave routing method."""
    
    # estimation of inflow
    channel_inflow_upstream[i] = -channel_outflow_sum_upstream_instant[i]
    
    # estimation of outflow
    if (channel_length[i] > 0) and (channel_hydraulic_radii[i] > 0):
        channel_flow_velocity[i] = (channel_hydraulic_radii[i] ** (2.0/3.0)) * (channel_slope[i] ** (1.0/2.0)) / channel_manning[i]
    else:
        channel_flow_velocity[i] = 0.0

    kinematic_wave_condition = (channel_length[i] > 0) and ((total_drainage_area_single[i] / channel_width[i] / channel_length[i]) <= kinematic_wave_parameter)

    if kinematic_wave_condition:
        channel_outflow_downstream[i] = -channel_flow_velocity[i] * channel_cross_section_area[i]
    else:
        channel_outflow_downstream[i] = -channel_inflow_upstream[i] - channel_lateral_flow_hillslope[i]

    flow_condition = kinematic_wave_condition and (-channel_outflow_downstream[i] > tiny_value) and ((channel_storage[i] + (channel_lateral_flow_hillslope[i] + channel_inflow_upstream[i] + channel_outflow_downstream[i]) * delta_t) < tiny_value)

    if flow_condition:
        channel_outflow_downstream[i] = -(channel_lateral_flow_hillslope[i] + channel_inflow_upstream[i] + channel_storage[i] / delta_t)

    if flow_condition and (channel_cross_section_area[i] > 0):
        channel_flow_velocity[i] = -channel_outflow_downstream[i] / channel_cross_section_area[i]

    # calculate change in storage, but first round small runoff to zero
    delta_runoff = hillslope_wetland_runoff[i] * area[i] * drainage_fraction[i]
    if abs(delta_runoff) <= tiny_value:
        delta_runoff = 0
    channel_delta_storage[i] = channel_lateral_flow_hillslope[i] + channel_inflow_upstream[i] + channel_outflow_downstream[i] + delta_runoff
