import numpy as np

def kinematic_wave_routing(state, grid, parameters, delta_t, base_condition):
    # classic kinematic wave routing method
    
    # estimation of inflow
    state.channel_inflow_upstream = np.where(
        base_condition,
        -state.channel_outflow_sum_upstream_instant,
        state.channel_inflow_upstream
    )
    
    # estimation of outflow
    state.channel_flow_velocity = np.where(
        base_condition,
        np.where(
            (grid.channel_length > 0) & (state.channel_hydraulic_radii > 0),
            (state.channel_hydraulic_radii ** (2/3)) * np.sqrt(grid.channel_slope) / grid.channel_manning,
            0
        ),
        state.channel_flow_velocity
    )
    condition = (grid.channel_length > 0) & ((grid.total_drainage_area_single / grid.channel_width / grid.channel_length) <= parameters.kinematic_wave_condition)
    state.channel_outflow_downstream = np.where(
        base_condition,
        np.where(
            condition,
            -state.channel_flow_velocity * state.channel_cross_section_area,
            -state.channel_inflow_upstream - state.channel_lateral_flow_hillslope
        ),
        state.channel_outflow_downstream
    )
    condition = (
        base_condition &
        condition &
        (-state.channel_outflow_downstream > parameters.tiny_value) &
        ((state.channel_storage + (state.channel_lateral_flow_hillslope + state.channel_inflow_upstream + state.channel_outflow_downstream) * delta_t) < parameters.tiny_value)
    )
    state.channel_outflow_downstream = np.where(
        condition,
        -(state.channel_lateral_flow_hillslope + state.channel_inflow_upstream + state.channel_storage / delta_t),
        state.channel_outflow_downstream
    )
    state.channel_flow_velocity = np.where(
        condition & (state.channel_cross_section_area > 0),
        -state.channel_outflow_downstream / state.channel_cross_section_area,
        state.channel_flow_velocity
    )
    
    # calculate change in storage, but first round small runoff to zero
    tmp_delta_runoff = np.where(
        base_condition,
        state.hillslope_wetland_runoff * grid.area * grid.drainage_fraction,
        0
    )
    tmp_delta_runoff = np.where(
        base_condition,
        np.where(
            np.abs(tmp_delta_runoff) <= parameters.tiny_value,
            0,
            tmp_delta_runoff
        ),
        tmp_delta_runoff
    )
    state.channel_delta_storage = np.where(
        base_condition,
        state.channel_lateral_flow_hillslope + state.channel_inflow_upstream + state.channel_outflow_downstream + tmp_delta_runoff,
        0
    )