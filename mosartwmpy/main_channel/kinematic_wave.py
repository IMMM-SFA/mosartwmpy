import numpy as np
import numexpr as ne

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State

def kinematic_wave_routing(state: State, grid: Grid, parameters: Parameters, delta_t: float, base_condition: np.ndarray) -> None:
    """Tracks the storage and flow of water in the main channel using the kinematic wave routing method.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        parameters (Parameters): the model parameters
        delta_t (float): the timestep for this subcycle (overall timestep / subcycles)
        base_condition (np.ndarray): a boolean array representing where the update should occur in the state
    """
    
    # estimation of inflow
    state.channel_inflow_upstream = calculate_channel_inflow_upstream(base_condition, state.channel_outflow_sum_upstream_instant, state.channel_inflow_upstream)
    
    # estimation of outflow
    state.channel_flow_velocity = calculate_channel_flow_velocity(base_condition, grid.channel_length, state.channel_hydraulic_radii, grid.channel_slope, grid.channel_manning, state.channel_flow_velocity)
    condition = calculate_kinematic_wave_condition(grid.channel_length, grid.total_drainage_area_single, grid.channel_width, grid.channel_length, parameters.kinematic_wave_condition)
    state.channel_outflow_downstream = calculate_base_channel_outflow_downstream(base_condition, condition, state.channel_flow_velocity, state.channel_cross_section_area, state.channel_inflow_upstream, state.channel_lateral_flow_hillslope, state.channel_outflow_downstream)
    condition = calculate_flow_condition(base_condition, condition, state.channel_outflow_downstream, parameters.tiny_value, state.channel_storage, state.channel_lateral_flow_hillslope, state.channel_inflow_upstream, delta_t)
    state.channel_outflow_downstream = calculate_channel_outflow_downstream(condition, state.channel_lateral_flow_hillslope, state.channel_inflow_upstream, state.channel_storage, delta_t, state.channel_outflow_downstream)
    state.channel_flow_velocity = update_channel_flow_velocity(condition, state.channel_cross_section_area, state.channel_outflow_downstream, state.channel_flow_velocity)
    
    # calculate change in storage, but first round small runoff to zero
    tmp_delta_runoff = calculate_tmp_delta_runoff(base_condition, state.hillslope_wetland_runoff, grid.area, grid.drainage_fraction)
    tmp_delta_runoff = update_tmp_delta_runoff(base_condition, tmp_delta_runoff, parameters.tiny_value)
    state.channel_delta_storage = calculate_channel_delta_storage(base_condition, state.channel_lateral_flow_hillslope, state.channel_inflow_upstream, state.channel_outflow_downstream, tmp_delta_runoff)


calculate_channel_inflow_upstream = ne.NumExpr(
    'where('
        'base_condition,'
        '-channel_outflow_sum_upstream_instant,'
        'channel_inflow_upstream'
    ')',
    (('base_condition', np.bool), ('channel_outflow_sum_upstream_instant', np.float64), ('channel_inflow_upstream', np.float64))
)

calculate_channel_flow_velocity = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            '(channel_length > 0) & (channel_hydraulic_radii > 0),'
            '(channel_hydraulic_radii ** (2/3)) * sqrt(channel_slope) / channel_manning,'
            '0'
        '),'
        'channel_flow_velocity'
    ')',
    (('base_condition', np.bool), ('channel_length', np.float64), ('channel_hydraulic_radii', np.float64), ('channel_slope', np.float64), ('channel_manning', np.float64), ('channel_flow_velocity', np.float64))
)

calculate_kinematic_wave_condition = ne.NumExpr(
    '(channel_length > 0) & ((total_drainage_area_single / channel_width / channel_length) <= kinematic_wave_condition)',
    (('channel_length', np.float64), ('total_drainage_area_single', np.float64), ('channel_width', np.float64), ('channel_length', np.float64), ('kinematic_wave_condition', np.float64))
)

calculate_base_channel_outflow_downstream = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            'kinematic_wave_condition,'
            '-channel_flow_velocity * channel_cross_section_area,'
            '-channel_inflow_upstream - channel_lateral_flow_hillslope'
        '),'
        'channel_outflow_downstream'
    ')',
    (('base_condition', np.bool), ('kinematic_wave_condition', np.bool), ('channel_flow_velocity', np.float64), ('channel_cross_section_area', np.float64), ('channel_inflow_upstream', np.float64), ('channel_lateral_flow_hillslope', np.float64), ('channel_outflow_downstream', np.float64))
)

calculate_flow_condition = ne.NumExpr(
    'base_condition &'
    'kinematic_wave_condition &'
    '(-channel_outflow_downstream > tiny_value) &'
    '((channel_storage + (channel_lateral_flow_hillslope + channel_inflow_upstream + channel_outflow_downstream) * delta_t) < tiny_value)',
    (('base_condition', np.bool), ('kinematic_wave_condition', np.bool), ('channel_outflow_downstream', np.float64), ('tiny_value', np.float64), ('channel_storage', np.float64), ('channel_lateral_flow_hillslope', np.float64), ('channel_inflow_upstream', np.float64), ('delta_t', np.float64))
)

calculate_channel_outflow_downstream = ne.NumExpr(
    'where('
        'flow_condition,'
        '-(channel_lateral_flow_hillslope + channel_inflow_upstream + channel_storage / delta_t),'
        'channel_outflow_downstream'
    ')',
    (('flow_condition', np.bool), ('channel_lateral_flow_hillslope', np.float64), ('channel_inflow_upstream', np.float64), ('channel_storage', np.float64), ('delta_t', np.float64), ('channel_outflow_downstream', np.float64))
)

update_channel_flow_velocity = ne.NumExpr(
    'where('
        'flow_condition & (channel_cross_section_area > 0),'
        '-channel_outflow_downstream / channel_cross_section_area,'
        'channel_flow_velocity'
    ')',
    (('flow_condition', np.bool), ('channel_cross_section_area', np.float64), ('channel_outflow_downstream', np.float64), ('channel_flow_velocity', np.float64))
)

calculate_tmp_delta_runoff = ne.NumExpr(
    'where('
        'base_condition,'
        'hillslope_wetland_runoff * area * drainage_fraction,'
        '0'
    ')',
    (('base_condition', np.bool), ('hillslope_wetland_runoff', np.float64), ('area', np.float64), ('drainage_fraction', np.float64))
)

update_tmp_delta_runoff = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            'abs(tmp_delta_runoff) <= tiny_value,'
            '0,'
            'tmp_delta_runoff'
        '),'
        'tmp_delta_runoff'
    ')',
    (('base_condition', np.bool), ('tmp_delta_runoff', np.float64), ('tiny_value', np.float64))
)

calculate_channel_delta_storage = ne.NumExpr(
    'where('
        'base_condition,'
        'channel_lateral_flow_hillslope + channel_inflow_upstream + channel_outflow_downstream + tmp_delta_runoff,'
        '0'
    ')',
    (('base_condition', np.bool), ('channel_lateral_flow_hillslope', np.float64), ('channel_inflow_upstream', np.float64), ('channel_outflow_downstream', np.float64), ('tmp_delta_runoff', np.float64))
)