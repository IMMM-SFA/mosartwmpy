import numpy as np

from mosartwmpy.hillslope.state import update_hillslope_state

def hillslope_routing(state, grid, parameters, delta_t):
    # perform the hillslope routing for the whole grid
    # TODO describe what is happening heres
    
    base_condition = (grid.mosart_mask > 0) & state.euler_mask
    
    velocity_hillslope = np.where(
        base_condition & (state.hillslope_depth > 0),
        (state.hillslope_depth ** (2/3)) * np.sqrt(grid.hillslope) / grid.hillslope_manning,
        0
    )
    
    state.hillslope_overland_flow = np.where(
        base_condition,
        -state.hillslope_depth * velocity_hillslope * grid.drainage_density,
        state.hillslope_overland_flow
    )
    state.hillslope_overland_flow = np.where(
        base_condition &
        (state.hillslope_overland_flow < 0) &
        ((state.hillslope_storage + delta_t * (state.hillslope_surface_runoff + state.hillslope_overland_flow)) < parameters.tiny_value),
        -(state.hillslope_surface_runoff + state.hillslope_storage / delta_t),
        state.hillslope_overland_flow
    )
    
    state.hillslope_delta_storage = np.where(
        base_condition,
        state.hillslope_surface_runoff + state.hillslope_overland_flow,
        state.hillslope_delta_storage
    )
    
    state.hillslope_storage = np.where(
        base_condition,
        state.hillslope_storage + delta_t * state.hillslope_delta_storage,
        state.hillslope_storage
    )
    
    update_hillslope_state(state, base_condition)
    
    state.subnetwork_lateral_inflow = np.where(
        base_condition,
        (state.hillslope_subsurface_runoff - state.hillslope_overland_flow) * grid.drainage_fraction * grid.area,
        state.subnetwork_lateral_inflow
    )