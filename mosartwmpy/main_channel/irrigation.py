import numpy as np
import numexpr as ne

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State
from mosartwmpy.main_channel.state import update_main_channel_state
from mosartwmpy.utilities.timing import timing

# @timing
def main_channel_irrigation(state: State, grid: Grid, parameters: Parameters) -> None:
    """Tracks the supply of water from the main river channel extracted into the grid cells.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        parameters (Parameters): the model parameters
    """
    
    depth_condition = calculate_depth_condition(grid.mosart_mask, state.euler_mask, state.tracer, parameters.LIQUID_TRACER, state.channel_depth, parameters.irrigation_extraction_condition)
    
    tiny_condition = calculate_tiny_condition(state.channel_storage, parameters.tinier_value, state.grid_cell_unmet_demand, grid.channel_length)
    
    flow_volume = np.empty_like(state.channel_storage)
    np.copyto(flow_volume, state.channel_storage)
    
    volume_condition = calculate_volume_condition(parameters.irrigation_extraction_maximum_fraction, flow_volume, state.grid_cell_unmet_demand)
    
    state.grid_cell_supply = calculate_reservoir_supply(depth_condition, tiny_condition, volume_condition, state.grid_cell_supply, state.grid_cell_unmet_demand, parameters.irrigation_extraction_maximum_fraction, flow_volume)
    
    flow_volume = calculate_flow_volume(depth_condition, tiny_condition, volume_condition, flow_volume, state.grid_cell_unmet_demand)
    
    state.grid_cell_unmet_demand = calculate_reservoir_demand(depth_condition, tiny_condition, volume_condition, state.grid_cell_unmet_demand, parameters.irrigation_extraction_maximum_fraction, flow_volume)
    
    flow_volume = update_flow_volume(depth_condition, tiny_condition, volume_condition, parameters.irrigation_extraction_maximum_fraction, flow_volume)
    
    state.channel_storage = calculate_channel_storage(depth_condition, tiny_condition, flow_volume, state.channel_storage)
    
    # TODO ? fortran mosart appears to do some more math with temp_erout and TRunoff%erout that looks to me to always be zero
    
    update_main_channel_state(state, grid, parameters, depth_condition)

calculate_depth_condition = ne.NumExpr(
    '(mosart_mask > 0) &'
    'euler_mask &'
    '(tracer == LIQUID_TRACER) &'
    '(channel_depth >= irrigation_extraction_condition)',
    (('mosart_mask', np.int64), ('euler_mask', np.bool), ('tracer', np.int64), ('LIQUID_TRACER', np.int64), ('channel_depth', np.float64), ('irrigation_extraction_condition', np.float64))
)

calculate_tiny_condition = ne.NumExpr(
    '(channel_storage > tinier_value) &'
    '(reservoir_demand > tinier_value) &'
    '(channel_length > tinier_value)',
    (('channel_storage', np.float64), ('tinier_value', np.float64), ('reservoir_demand', np.float64), ('channel_length', np.float64))
)

calculate_volume_condition = ne.NumExpr(
    'irrigation_extraction_maximum_fraction * flow_volume >= reservoir_demand',
    (('irrigation_extraction_maximum_fraction', np.float64), ('flow_volume', np.float64), ('reservoir_demand', np.float64))
)

calculate_reservoir_supply = ne.NumExpr(
    'where('
        'depth_condition & tiny_condition,'
        'where('
            'volume_condition,'
            'reservoir_supply + reservoir_demand,'
            'reservoir_supply + irrigation_extraction_maximum_fraction * flow_volume'
        '),'
        'reservoir_supply'
    ')',
    (('depth_condition', np.bool), ('tiny_condition', np.bool), ('volume_condition', np.bool), ('reservoir_supply', np.float64), ('reservoir_demand', np.float64), ('irrigation_extraction_maximum_fraction', np.float64), ('flow_volume', np.float64))
)

calculate_flow_volume = ne.NumExpr(
    'where('
        'depth_condition & tiny_condition & volume_condition,'
        'flow_volume - reservoir_demand,'
        'flow_volume'
    ')',
    (('depth_condition', np.bool), ('tiny_condition', np.bool), ('volume_condition', np.bool), ('flow_volume', np.float64), ('reservoir_demand', np.float64))
)

calculate_reservoir_demand = ne.NumExpr(
    'where('
        'depth_condition & tiny_condition,'
        'where('
            'volume_condition,'
            '0,'
            'reservoir_demand - irrigation_extraction_maximum_fraction * flow_volume'
        '),'
        'reservoir_demand'
    ')',
    (('depth_condition', np.bool), ('tiny_condition', np.bool), ('volume_condition', np.bool), ('reservoir_demand', np.float64), ('irrigation_extraction_maximum_fraction', np.float64), ('flow_volume', np.float64))
)

update_flow_volume = ne.NumExpr(
    'where('
        'depth_condition & tiny_condition & (~volume_condition),'
        '(1.0 - irrigation_extraction_maximum_fraction) * flow_volume,'
        'flow_volume'
    ')',
    (('depth_condition', np.bool), ('tiny_condition', np.bool), ('volume_condition', np.bool), ('irrigation_extraction_maximum_fraction', np.float64), ('flow_volume', np.float64))
)

calculate_channel_storage = ne.NumExpr(
    'where('
        'depth_condition & tiny_condition,'
        'flow_volume,'
        'channel_storage'
    ')',
    (('depth_condition', np.bool), ('tiny_condition', np.bool), ('flow_volume', np.float64), ('channel_storage', np.float64))
)