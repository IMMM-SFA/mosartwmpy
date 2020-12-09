import numpy as np
import pandas as pd

def direct_to_ocean(state, grid, parameters, config):
    ###
    ### Direct transfer to outlet point
    ###

    # direct to ocean
    # note - in fortran mosart this direct_to_ocean forcing could be provided from LND component, but we don't seem to be using it
    source_direct = state.direct_to_ocean.values
    
    # wetland runoff
    wetland_runoff_volume = state.hillslope_wetland_runoff.values * config.get('simulation.timestep') / config.get('simulation.subcycles')
    river_volume_minimum = parameters.river_depth_minimum * grid.area.values

    # if wetland runoff is negative and it would bring main channel storage below the minimum, send it directly to ocean
    condition = ((state.channel_storage.values + wetland_runoff_volume) < river_volume_minimum) & (state.hillslope_wetland_runoff.values < 0)
    source_direct = np.where(
        condition,
        source_direct + state.hillslope_wetland_runoff.values,
        source_direct
    )
    state.hillslope_wetland_runoff = pd.DataFrame(np.where(
        condition,
        0,
        state.hillslope_wetland_runoff.values
    ))
    # remove remaining wetland runoff (negative and positive)
    source_direct = source_direct + state.hillslope_wetland_runoff.values
    state.hillslope_wetland_runoff = state.zeros
    
    # runoff from hillslope
    # remove negative subsurface water
    condition = state.hillslope_subsurface_runoff.values < 0
    source_direct = np.where(
        condition,
        source_direct + state.hillslope_subsurface_runoff.values,
        source_direct
    )
    state.hillslope_subsurface_runoff = pd.DataFrame(np.where(
        condition,
        0,
        state.hillslope_subsurface_runoff.values
    ))
    # remove negative surface water
    condition = state.hillslope_surface_runoff.values < 0
    source_direct = np.where(
        condition,
        source_direct + state.hillslope_surface_runoff.values,
        source_direct
    )
    state.hillslope_surface_runoff = pd.DataFrame(np.where(
        condition,
        0,
        state.hillslope_surface_runoff.values
    ))

    # if ocean cell or ice tracer, remove the rest of the sub and surface water
    # other cells will be handled by mosart euler
    condition = (grid.mosart_mask.values == 0) | (state.tracer.values == parameters.ICE_TRACER)
    source_direct = np.where(
        condition,
        source_direct + state.hillslope_subsurface_runoff.values + state.hillslope_surface_runoff.values,
        source_direct
    )
    state.hillslope_subsurface_runoff = pd.DataFrame(np.where(
        condition,
        0,
        state.hillslope_subsurface_runoff.values
    ))
    state.hillslope_surface_runoff = pd.DataFrame(np.where(
        condition,
        0,
        state.hillslope_surface_runoff.values
    ))
    
    state.direct = pd.DataFrame(source_direct)

    # send the direct water to outlet for each tracer
    state.direct = grid[['outlet_id']].join(state[['direct']].join(grid.outlet_id).groupby('outlet_id').sum(), how='left').direct.fillna(0.0)
    
    return state