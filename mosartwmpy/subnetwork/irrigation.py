import numpy as np
import numexpr as ne

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State
from mosartwmpy.subnetwork.state import update_subnetwork_state
from mosartwmpy.utilities.timing import timing

# @timing
def subnetwork_irrigation(state: State, grid: Grid, parameters: Parameters) -> None:
    """Tracks the supply of water from the subnetwork river channels extracted into the grid cells.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        parameters (Parameters): the model parameters
    """
    
    depth_condition = calculate_depth_condition(grid.mosart_mask, state.euler_mask, state.tracer, parameters.LIQUID_TRACER, state.subnetwork_depth, parameters.irrigation_extraction_condition)
    
    flow_volume = np.empty_like(state.subnetwork_storage)
    np.copyto(flow_volume, state.subnetwork_storage)
    
    volume_condition = calculate_volume_condition(flow_volume, state.grid_cell_unmet_demand)
    
    state.grid_cell_supply = calculate_reservoir_supply(depth_condition, volume_condition, state.grid_cell_supply, state.grid_cell_unmet_demand, flow_volume)
    
    flow_volume = calculate_flow_volume(depth_condition, volume_condition, flow_volume, state.grid_cell_unmet_demand)
    
    state.grid_cell_unmet_demand = calculate_reservoir_demand(depth_condition, volume_condition, state.grid_cell_unmet_demand, flow_volume)
    
    flow_volume = update_flow_volume(depth_condition, volume_condition, flow_volume)
    
    state.subnetwork_storage = calculate_subnetwork_storage(depth_condition, flow_volume, state.subnetwork_storage)
    
    update_subnetwork_state(state, grid, parameters, depth_condition)

calculate_depth_condition = ne.NumExpr(
    '(mosart_mask > 0) &'
    'euler_mask &'
    '(tracer == LIQUID_TRACER) &'
    '(subnetwork_depth >= irrigation_extraction_condition)',
    (('mosart_mask', np.int64), ('euler_mask', np.bool), ('tracer', np.int64), ('LIQUID_TRACER', np.int64), ('subnetwork_depth', np.float64), ('irrigation_extraction_condition', np.float64))
)

calculate_volume_condition = ne.NumExpr(
    'flow_volume >= reservoir_demand',
    (('flow_volume', np.float64), ('reservoir_demand', np.float64))
)

calculate_reservoir_supply = ne.NumExpr(
    'where('
        'depth_condition,'
        'where('
            'volume_condition,'
            'reservoir_supply + reservoir_demand,'
            'reservoir_supply + flow_volume'
        '),'
        'reservoir_supply'
    ')',
    (('depth_condition', np.bool), ('volume_condition', np.bool), ('reservoir_supply', np.float64), ('reservoir_demand', np.float64), ('flow_volume', np.float64))
)

calculate_flow_volume = ne.NumExpr(
    'where('
        'depth_condition & volume_condition,'
        'flow_volume - reservoir_demand,'
        'flow_volume'
    ')',
    (('depth_condition', np.bool), ('volume_condition', np.bool), ('flow_volume', np.float64), ('reservoir_demand', np.float64))
)

calculate_reservoir_demand = ne.NumExpr(
    'where('
        'depth_condition,'
        'where('
            'volume_condition,'
            '0,'
            'reservoir_demand - flow_volume'
        '),'
        'reservoir_demand'
    ')',
    (('depth_condition', np.bool), ('volume_condition', np.bool), ('reservoir_demand', np.float64), ('flow_volume', np.float64))
)

update_flow_volume = ne.NumExpr(
    'where('
        'depth_condition & (~volume_condition),'
        '0,'
        'flow_volume'
    ')',
    (('depth_condition', np.bool), ('volume_condition', np.bool), ('flow_volume', np.float64))
)

calculate_subnetwork_storage = ne.NumExpr(
    'where('
        'depth_condition,'
        'flow_volume,'
        'subnetwork_storage'
    ')',
    (('depth_condition', np.bool), ('flow_volume', np.float64), ('subnetwork_storage', np.float64))
)