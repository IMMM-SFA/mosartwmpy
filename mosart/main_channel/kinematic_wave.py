import numpy as np
import pandas as pd

def kinematic_wave_routing(state, grid, parameters, config, delta_t, base_condition):
    # classic kinematic wave routing method
    
    # estimation of inflow
    state.channel_inflow_upstream = pd.DataFrame(np.where(
        base_condition,
        -state.channel_outflow_sum_upstream_instant.values,
        state.channel_inflow_upstream.values
    ))
    
    # estimation of outflow
    state.channel_flow_velocity = pd.DataFrame(np.where(
        base_condition,
        np.where(
            (grid.channel_length.values > 0) & (state.channel_hydraulic_radii.values > 0),
            (state.channel_hydraulic_radii.values ** (2/3)) * (grid.channel_slope.values ** (1/2)) / grid.channel_manning.values,
            0
        ),
        state.channel_flow_velocity.values
    ))
    condition = (grid.channel_length.values > 0) & ((grid.total_drainage_area_single.values / grid.channel_width.values / grid.channel_length.values) <= parameters.kinematic_wave_condition)
    state.channel_outflow_downstream = pd.DataFrame(np.where(
        base_condition,
        np.where(
            condition,
            -state.channel_flow_velocity.values * state.channel_cross_section_area.values,
            -state.channel_inflow_upstream.values - state.channel_lateral_flow_hillslope.values
        ),
        state.channel_outflow_downstream.values
    ))
    condition = (
        base_condition &
        condition &
        (-state.channel_outflow_downstream.values > parameters.tiny_value) &
        ((state.channel_storage.values + (state.channel_lateral_flow_hillslope.values + state.channel_inflow_upstream.values + state.channel_outflow_downstream.values) * delta_t) < parameters.tiny_value)
    )
    state.channel_outflow_downstream = pd.DataFrame(np.where(
        condition,
        -(state.channel_lateral_flow_hillslope.values + state.channel_inflow_upstream.values + state.channel_storage.values / delta_t),
        state.channel_outflow_downstream.values
    ))
    state.channel_flow_velocity = pd.DataFrame(np.where(
        condition & (state.channel_cross_section_area.values > 0),
        -state.channel_outflow_downstream.values / state.channel_cross_section_area.values,
        state.channel_flow_velocity.values
    ))
    
    # calculate change in storage, but first round small runoff to zero
    tmp_delta_runoff = np.where(
        base_condition,
        state.hillslope_wetland_runoff.values * grid.area.values * grid.drainage_fraction.values,
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
    state.channel_delta_storage = pd.DataFrame(np.where(
        base_condition,
        state.channel_lateral_flow_hillslope.values + state.channel_inflow_upstream.values + state.channel_outflow_downstream.values + tmp_delta_runoff,
        0
    ))
    
    return state