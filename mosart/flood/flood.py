import numpy as np

def flood(state, grid, parameters, config):
    
    ###
    ### Compute Flood
    ### Remove excess liquid water from land
    ###
    ### TODO tcraig leaves a comment here concerning surface_runoff in fortran mosart:
    ### "This seems like an odd approach, you
    ### might create negative forcing.  why not take it out of
    ### the volr directly?  it's also odd to compute this
    ### at the initial time of the time loop.  why not do
    ### it at the end or even during the run loop as the
    ### new volume is computed.  fluxout depends on volr, so
    ### how this is implemented does impact the solution."
    ###

    # flux sent back to land
    state.flood[:] = np.where(
        (grid.land_mask.values == 1) & (state.storage.values > parameters.flood_threshold) & (state.tracer.values == parameters.LIQUID_TRACER),
        (state.storage.values - parameters.flood_threshold) / config.get('simulation.timestep'),
        0
    )
    # remove this flux from the input runoff from land
    state.hillslope_surface_runoff[:] = np.where(
        state.tracer.values == parameters.LIQUID_TRACER,
        state.hillslope_surface_runoff.values - state.flood.values,
        state.hillslope_surface_runoff.values
    )
    
    return state