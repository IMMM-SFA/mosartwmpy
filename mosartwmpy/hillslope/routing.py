import numpy as np
import numexpr as ne

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State
from mosartwmpy.hillslope.state import update_hillslope_state

def hillslope_routing(state: State, grid: Grid, parameters: Parameters, delta_t: float) -> None:
    """Tracks the storage of runoff water in the hillslope and the flow of runoff water from the hillslope into the channels.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        parameters (Parameters): the model parameters
        delta_t (float): the timestep for this subcycle (overall timestep / subcycles)
    """
    
    base_condition = (grid.mosart_mask > 0) & state.euler_mask
    
    velocity_hillslope = calculate_velocity_hillslope(base_condition, state.hillslope_depth, grid.hillslope, grid.hillslope_manning)
    state.hillslope_overland_flow = calculate_base_hillslope_overland_flow(base_condition, state.hillslope_depth, velocity_hillslope, grid.drainage_density, state.hillslope_overland_flow)
    state.hillslope_overland_flow = calculate_hillslope_overland_flow(base_condition, state.hillslope_overland_flow, state.hillslope_storage, delta_t, state.hillslope_surface_runoff, parameters.tiny_value)
    state.hillslope_delta_storage = calculate_hillslope_delta_storage(base_condition, state.hillslope_surface_runoff, state.hillslope_overland_flow, state.hillslope_delta_storage)
    state.hillslope_storage = calculate_hillslope_storage(base_condition, state.hillslope_storage, delta_t, state.hillslope_delta_storage)
    
    update_hillslope_state(state, base_condition)
    
    state.subnetwork_lateral_inflow = calculate_subnetwork_lateral_inflow(base_condition, state.hillslope_subsurface_runoff, state.hillslope_overland_flow, grid.drainage_fraction, grid.area, state.subnetwork_lateral_inflow)

calculate_velocity_hillslope = ne.NumExpr(
    'where('
        'base_condition & (hillslope_depth > 0),'
        '(hillslope_depth ** (2/3)) * sqrt(hillslope) / hillslope_manning,'
        '0'
    ')',
    (('base_condition', np.bool), ('hillslope_depth', np.float64), ('hillslope', np.float64), ('hillslope_manning', np.float64))
)

calculate_base_hillslope_overland_flow = ne.NumExpr(
    'where('
        'base_condition,'
        '-hillslope_depth * velocity_hillslope * drainage_density,'
        'hillslope_overland_flow'
    ')',
    (('base_condition', np.bool), ('hillslope_depth', np.float64), ('velocity_hillslope', np.float64), ('drainage_density', np.float64), ('hillslope_overland_flow', np.float64))
)

calculate_hillslope_overland_flow = ne.NumExpr(
    'where('
        'base_condition & (hillslope_overland_flow < 0) & ((hillslope_storage + delta_t * (hillslope_surface_runoff + hillslope_overland_flow)) < tiny_value),'
        '-(hillslope_surface_runoff + hillslope_storage / delta_t),'
        'hillslope_overland_flow'
    ')',
    (('base_condition', np.bool), ('hillslope_overland_flow', np.float64), ('hillslope_storage', np.float64), ('delta_t', np.float64), ('hillslope_surface_runoff', np.float64), ('tiny_value', np.float64))
)

calculate_hillslope_delta_storage = ne.NumExpr(
    'where('
        'base_condition,'
        'hillslope_surface_runoff + hillslope_overland_flow,'
        'hillslope_delta_storage'
    ')',
    (('base_condition', np.bool), ('hillslope_surface_runoff', np.float64), ('hillslope_overland_flow', np.float64), ('hillslope_delta_storage', np.float64))
)

calculate_hillslope_storage = ne.NumExpr(
    'where('
        'base_condition,'
        'hillslope_storage + delta_t * hillslope_delta_storage,'
        'hillslope_storage'
    ')',
    (('base_condition', np.bool), ('hillslope_storage', np.float64), ('delta_t', np.float64), ('hillslope_delta_storage', np.float64))
)

calculate_subnetwork_lateral_inflow = ne.NumExpr(
    'where('
        'base_condition,'
        '(hillslope_subsurface_runoff - hillslope_overland_flow) * drainage_fraction * area,'
        'subnetwork_lateral_inflow'
    ')',
    (('base_condition', np.bool), ('hillslope_subsurface_runoff', np.float64), ('hillslope_overland_flow', np.float64), ('drainage_fraction', np.float64), ('area', np.float64), ('subnetwork_lateral_inflow', np.float64))
)