def direct_to_ocean(state, grid, parameters, config):
    ###
    ### Direct transfer to outlet point
    ###

    # direct to ocean
    # note - in fortran mosart this direct_to_ocean forcing could be provided from LND component, but we don't seem to be using it
    source_direct = state.direct_to_ocean
    
    # wetland runoff
    wetland_runoff_volume = state.hillslope_wetland_runoff * config.get('simulation.timestep') / config.get('simulation.subcycles')
    river_volume_minimum = parameters.river_depth_minimum * grid.area

    # if wetland runoff is negative and it would bring main channel storage below the minimum, send it directly to ocean
    condition = (state.channel_storage + wetland_runoff_volume).lt(river_volume_minimum) & state.hillslope_wetland_runoff.lt(0)
    source_direct = source_direct.mask(condition, source_direct + state.hillslope_wetland_runoff)
    state.hillslope_wetland_runoff = state.hillslope_wetland_runoff.mask(condition, 0)
    # remove remaining wetland runoff (negative and positive)
    source_direct = source_direct + state.hillslope_wetland_runoff
    state.hillslope_wetland_runoff = state.zeros
    
    # runoff from hillslope
    # remove negative subsurface water
    condition = state.hillslope_subsurface_runoff.lt(0)
    source_direct = source_direct.mask(condition, source_direct + state.hillslope_subsurface_runoff)
    state.hillslope_subsurface_runoff = state.hillslope_subsurface_runoff.mask(condition, 0)
    # remove negative surface water
    condition = state.hillslope_surface_runoff.lt(0)
    source_direct = source_direct.mask(condition, source_direct + state.hillslope_surface_runoff)
    state.hillslope_surface_runoff = state.hillslope_surface_runoff.mask(condition, 0)

    # if ocean cell or ice tracer, remove the rest of the sub and surface water
    # other cells will be handled by mosart euler
    condition = grid.mosart_mask.eq(0) | state.tracer.eq(parameters.ICE_TRACER)
    source_direct = source_direct.mask(condition, source_direct + state.hillslope_subsurface_runoff + state.hillslope_surface_runoff)
    state.hillslope_subsurface_runoff = state.hillslope_subsurface_runoff.mask(condition, 0)
    state.hillslope_surface_runoff = state.hillslope_surface_runoff.mask(condition, 0)

    # send the direct water to outlet for each tracer
    # TODO join left on id instead of index, to prepare for splitting grid
    state.direct = grid[['outlet_id']].join(state[['direct']].join(grid.outlet_id).groupby('outlet_id').sum(), how='left').direct.fillna(0.0)
    
    return state