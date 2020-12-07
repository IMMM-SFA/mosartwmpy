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
    state.flood = state.zeros.mask(
        grid.land_mask.eq(1) & state.storage.gt(parameters.flood_threshold) & state.tracer.eq(parameters.LIQUID_TRACER),
        (state.storage - parameters.flood_threshold) / config.get('simulation.timestep')
    )
    # remove this flux from the input runoff from land
    state.hillslope_surface_runoff = state.hillslope_surface_runoff.mask(
        state.tracer.eq(parameters.LIQUID_TRACER),
        state.hillslope_surface_runoff - state.flood
    )
    
    return state