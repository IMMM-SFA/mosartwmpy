import numpy as np

from xarray import open_dataset

# TODO this currently can only act on the entire domain... need to make more robust so can be handled in parallel
def load_runoff(state, grid, parameters, config, current_time):
    # note that the forcing is provided in mm/s
    # the flood section needs m3/s, but the routing needs m/s, so be aware of the conversions
    # method="pad" means the closest time in the past is selected from the file
    
    runoff = open_dataset(config.get('runoff.path'))
    
    if config.get('runoff.variables.surface_runoff', None) is not None:
        state.hillslope_surface_runoff = 0.001 * grid.area * np.array(
            runoff[config.get('runoff.variables.surface_runoff')].sel({config.get('runoff.time'): current_time}, method='pad')
        ).flatten()
    
    if config.get('runoff.variables.subsurface_runoff', None) is not None:
        state.hillslope_subsurface_runoff = 0.001 * grid.area * np.array(
            runoff[config.get('runoff.variables.subsurface_runoff')].sel({config.get('runoff.time'): current_time}, method='pad')
        ).flatten()
    
    if config.get('runoff.variables.wetland_runoff', None) is not None:
        state.hillslope_wetland_runoff = 0.001 * grid.area * np.array(
            runoff[config.get('runoff.variables.wetland_runoff')].sel({config.get('runoff.time'): current_time}, method='pad')
        ).flatten()
    
    runoff.close()
    
    return state