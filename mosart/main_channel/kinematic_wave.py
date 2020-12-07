def kinematic_wave_routing(state, grid, parameters, config, delta_t, base_condition):
    # classic kinematic wave routing method
    
    # estimation of inflow
    state.channel_inflow_upstream = state.channel_inflow_upstream.mask(
        base_condition,
        -state.channel_outflow_sum_upstream_instant
    )
    
    # estimation of outflow
    state.channel_flow_velocity = state.channel_flow_velocity.mask(
        base_condition,
        state.zeros.mask(
            grid.channel_length.gt(0) & state.channel_hydraulic_radii.gt(0),
            (state.channel_hydraulic_radii ** (2/3)) * (grid.channel_slope ** (1/2)) / grid.channel_manning
        )
    )
    condition = grid.channel_length.gt(0) & (grid.total_drainage_area_single / grid.channel_width / grid.channel_length).le(parameters.kinematic_wave_condition)
    state.channel_outflow_downstream = state.channel_outflow_downstream.mask(
        base_condition,
        (-state.channel_inflow_upstream - state.channel_lateral_flow_hillslope).mask(
            condition,
            -state.channel_flow_velocity * state.channel_cross_section_area
        )
    )
    condition = (
        base_condition &
        condition &
        (-state.channel_outflow_downstream).gt(parameters.tiny_value) &
        (state.channel_storage + (state.channel_lateral_flow_hillslope + state.channel_inflow_upstream + state.channel_outflow_downstream) * delta_t).lt(parameters.tiny_value)
    )
    state.channel_outflow_downstream = state.channel_outflow_downstream.mask(
        condition,
        -(state.channel_lateral_flow_hillslope + state.channel_inflow_upstream + state.channel_storage / delta_t)
    )
    state.channel_flow_velocity = state.channel_flow_velocity.mask(
        condition & state.channel_cross_section_area.gt(0),
        -state.channel_outflow_downstream / state.channel_cross_section_area
    )
    
    # calculate change in storage, but first round small runoff to zero
    tmp_delta_runoff = state.zeros.mask(
        base_condition,
        state.hillslope_wetland_runoff * grid.area * grid.drainage_fraction
    )
    tmp_delta_runoff = tmp_delta_runoff.mask(
        base_condition,
        tmp_delta_runoff.mask(
            tmp_delta_runoff.abs().le(parameters.tiny_value),
            0
        )
    )
    state.channel_delta_storage = state.zeros.mask(
        base_condition,
        state.channel_lateral_flow_hillslope + state.channel_inflow_upstream + state.channel_outflow_downstream + tmp_delta_runoff
    )
    
    return state