import numpy as np
import numexpr as ne

from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State
from mosartwmpy.subnetwork.state import update_subnetwork_state
from mosartwmpy.utilities.timing import timing

# @timing
def subnetwork_routing(state: State, grid: Grid, parameters: Parameters, config: Benedict, delta_t: float) -> None:
    """Tracks the storage and flow of water in the subnetwork river channels.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        parameters (Parameters): the model parameters
        config (Benedict): the model configuration
        delta_t (float): the timestep for this subcycle (overall timestep / subcycles)
    """
    
    state.channel_lateral_flow_hillslope[:] = 0
    local_delta_t = (delta_t / config.get('simulation.routing_iterations') / grid.iterations_subnetwork)
    
    # step through max iterations, masking out the unnecessary cells each time
    base_condition = (grid.mosart_mask > 0) & state.euler_mask
    sub_condition = grid.subnetwork_length > grid.hillslope_length # has tributaries
    
    for _ in np.arange(np.nanmax(grid.iterations_subnetwork)):
        iteration_condition = base_condition & (grid.iterations_subnetwork > _)

        state.subnetwork_flow_velocity = calculate_subnetwork_flow_velocity(iteration_condition, sub_condition, state.subnetwork_hydraulic_radii, grid.subnetwork_slope, grid.subnetwork_manning, state.subnetwork_flow_velocity)
        
        state.subnetwork_discharge = calculate_subnetwork_discharge(iteration_condition, sub_condition, state.subnetwork_flow_velocity, state.subnetwork_cross_section_area, state.subnetwork_lateral_inflow, state.subnetwork_discharge)
        
        discharge_condition = calculate_discharge_condition(iteration_condition, sub_condition, state.subnetwork_storage, state.subnetwork_lateral_inflow, state.subnetwork_discharge, local_delta_t, parameters.tiny_value)
        
        state.subnetwork_discharge = update_subnetwork_discharge(discharge_condition, state.subnetwork_lateral_inflow, state.subnetwork_storage, local_delta_t, state.subnetwork_discharge)
        
        state.subnetwork_flow_velocity = update_flow_velocity(discharge_condition, state.subnetwork_cross_section_area, state.subnetwork_discharge, state.subnetwork_flow_velocity)
        
        state.subnetwork_delta_storage = calculate_subnetwork_delta_storage(iteration_condition, state.subnetwork_lateral_inflow, state.subnetwork_discharge, state.subnetwork_delta_storage)
        
        # update storage
        state.subnetwork_storage_previous_timestep = calculate_subnetwork_storage_previous_timestep(iteration_condition, state.subnetwork_storage, state.subnetwork_storage_previous_timestep)
        state.subnetwork_storage = calculate_subnetwork_storage(iteration_condition, state.subnetwork_storage, state.subnetwork_delta_storage, local_delta_t)
        
        update_subnetwork_state(state, grid, parameters, iteration_condition)
        
        state.channel_lateral_flow_hillslope = calculate_channel_lateral_flow_hillslope(iteration_condition, state.channel_lateral_flow_hillslope, state.subnetwork_discharge)
    
    # average lateral flow over substeps
    state.channel_lateral_flow_hillslope = average_channel_lateral_flow_hillslope(base_condition, state.channel_lateral_flow_hillslope, grid.iterations_subnetwork)


calculate_subnetwork_flow_velocity = ne.NumExpr(
    'where('
        'base_condition & length_condition,'
        'where('
            'subnetwork_hydraulic_radii > 0,'
            '(subnetwork_hydraulic_radii ** (2/3)) * sqrt(subnetwork_slope) / subnetwork_manning,'
            '0'
        '),'
        'subnetwork_flow_velocity'
    ')',
    (('base_condition', np.bool), ('length_condition', np.bool), ('subnetwork_hydraulic_radii', np.float64), ('subnetwork_slope', np.float64), ('subnetwork_manning', np.float64), ('subnetwork_flow_velocity', np.float64))
)

calculate_subnetwork_discharge = ne.NumExpr(
    'where('
        'base_condition,'
        'where('
            'length_condition,'
            '-subnetwork_flow_velocity * subnetwork_cross_section_area,'
            '-subnetwork_lateral_inflow'
        '),'
        'subnetwork_discharge'
    ')',
    (('base_condition', np.bool), ('length_condition', np.bool), ('subnetwork_flow_velocity', np.float64), ('subnetwork_cross_section_area', np.float64), ('subnetwork_lateral_inflow', np.float64), ('subnetwork_discharge', np.float64))
)

calculate_discharge_condition = ne.NumExpr(
    'base_condition &'
    'length_condition &'
    '((subnetwork_storage + (subnetwork_lateral_inflow + subnetwork_discharge) * local_delta_t) < tiny_value)',
    (('base_condition', np.bool), ('length_condition', np.bool), ('subnetwork_storage', np.float64), ('subnetwork_lateral_inflow', np.float64), ('subnetwork_discharge', np.float64), ('local_delta_t', np.float64), ('tiny_value', np.float64))
)

update_subnetwork_discharge = ne.NumExpr(
    'where('
        'discharge_condition,'
        '-(subnetwork_lateral_inflow + subnetwork_storage / local_delta_t),'
        'subnetwork_discharge'
    ')',
    (('discharge_condition', np.bool), ('subnetwork_lateral_inflow', np.float64), ('subnetwork_storage', np.float64), ('local_delta_t', np.float64), ('subnetwork_discharge', np.float64))
)

update_flow_velocity = ne.NumExpr(
    'where('
        'discharge_condition & (subnetwork_cross_section_area > 0),'
        '-subnetwork_discharge / subnetwork_cross_section_area,'
        'subnetwork_flow_velocity'
    ')',
    (('discharge_condition', np.bool), ('subnetwork_cross_section_area', np.float64), ('subnetwork_discharge', np.float64), ('subnetwork_flow_velocity', np.float64))
)

calculate_subnetwork_delta_storage = ne.NumExpr(
    'where('
        'base_condition,'
        'subnetwork_lateral_inflow + subnetwork_discharge,'
        'subnetwork_delta_storage'
    ')',
    (('base_condition', np.bool), ('subnetwork_lateral_inflow', np.float64), ('subnetwork_discharge', np.float64), ('subnetwork_delta_storage', np.float64))
)

calculate_subnetwork_storage_previous_timestep = ne.NumExpr(
    'where('
        'base_condition,'
        'subnetwork_storage,'
        'subnetwork_storage_previous_timestep'
    ')',
    (('base_condition', np.bool), ('subnetwork_storage', np.float64), ('subnetwork_storage_previous_timestep', np.float64))
)

calculate_subnetwork_storage = ne.NumExpr(
    'where('
        'base_condition,'
        'subnetwork_storage + subnetwork_delta_storage * local_delta_t,'
        'subnetwork_storage'
    ')',
    (('base_condition', np.bool), ('subnetwork_storage', np.float64), ('subnetwork_delta_storage', np.float64), ('local_delta_t', np.float64))
)

calculate_channel_lateral_flow_hillslope = ne.NumExpr(
    'where('
        'base_condition,'
        'channel_lateral_flow_hillslope - subnetwork_discharge,'
        'channel_lateral_flow_hillslope'
    ')',
    (('base_condition', np.bool), ('channel_lateral_flow_hillslope', np.float64), ('subnetwork_discharge', np.float64))
)

average_channel_lateral_flow_hillslope = ne.NumExpr(
    'where('
        'base_condition,'
        'channel_lateral_flow_hillslope / iterations_subnetwork,'
        'channel_lateral_flow_hillslope'
    ')',
    (('base_condition', np.bool), ('channel_lateral_flow_hillslope', np.float64), ('iterations_subnetwork', np.float64))
)