import numpy as np

def update_subnetwork_state(state, grid, parameters, base_condition):
    # update the physical properties of the subnetwork
        
    # update state variables
    condition = (grid.subnetwork_length > 0) & (state.subnetwork_storage > 0)
    state.subnetwork_cross_section_area = np.where(
        base_condition,
        np.where(
            condition,
            state.subnetwork_storage / grid.subnetwork_length,
            0
        ),
        state.subnetwork_cross_section_area
    )
    state.subnetwork_depth = np.where(
        base_condition,
        np.where(
            condition & (state.subnetwork_cross_section_area > parameters.tiny_value),
            state.subnetwork_cross_section_area / grid.subnetwork_width,
            0
        ),
        state.subnetwork_depth
    )
    state.subnetwork_wetness_perimeter = np.where(
        base_condition,
        np.where(
            condition & (state.subnetwork_depth > parameters.tiny_value),
            grid.subnetwork_width + 2 * state.subnetwork_depth,
            0
        ),
        state.subnetwork_wetness_perimeter
    )
    state.subnetwork_hydraulic_radii = np.where(
        base_condition,
        np.where(
            condition & (state.subnetwork_wetness_perimeter > parameters.tiny_value),
            state.subnetwork_cross_section_area / state.subnetwork_wetness_perimeter,
            0
        ),
        state.subnetwork_hydraulic_radii
    )