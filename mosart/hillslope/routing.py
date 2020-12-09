import numpy as np
import pandas as pd

from mosart.hillslope.state import update_hillslope_state

def hillslope_routing(state, grid, parameters, config, delta_t):
    # perform the hillslope routing for the whole grid
    # TODO describe what is happening heres
    
    base_condition = (grid.mosart_mask.values > 0) & state.euler_mask.values
    
    velocity_hillslope = np.where(
        base_condition & (state.hillslope_depth.values > 0),
        (state.hillslope_depth.values ** (2/3)) * (grid.hillslope.values ** (1/2)) / grid.hillslope_manning.values,
        0
    )
    
    state.hillslope_overland_flow = pd.DataFrame(np.where(
        base_condition,
        -state.hillslope_depth.values * velocity_hillslope * grid.drainage_density.values,
        state.hillslope_overland_flow.values
    ))
    state.hillslope_overland_flow = pd.DataFrame(np.where(
        base_condition &
        (state.hillslope_overland_flow.values < 0) &
        ((state.hillslope_storage.values + delta_t * (state.hillslope_surface_runoff.values + state.hillslope_overland_flow.values)) < parameters.tiny_value),
        -(state.hillslope_surface_runoff.values + state.hillslope_storage.values / delta_t),
        state.hillslope_overland_flow.values
    ))
    
    state.hillslope_delta_storage = pd.DataFrame(np.where(
        base_condition,
        state.hillslope_surface_runoff.values + state.hillslope_overland_flow.values,
        state.hillslope_delta_storage.values
    ))
    
    state.hillslope_storage = pd.DataFrame(np.where(
        base_condition,
        state.hillslope_storage.values + delta_t * state.hillslope_delta_storage.values,
        state.hillslope_storage.values
    ))
    
    state = update_hillslope_state(state, grid, parameters, config, base_condition)
    
    state.subnetwork_lateral_inflow = pd.DataFrame(np.where(
        base_condition,
        (state.hillslope_subsurface_runoff.values - state.hillslope_overland_flow.values) * grid.drainage_fraction.values * grid.area.values,
        state.subnetwork_lateral_inflow.values
    ))
    
    return state