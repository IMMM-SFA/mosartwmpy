import numpy as np
import pandas as pd
import pdb
from datetime import datetime, time, timedelta
from xarray import open_dataset

def load_runoff(state, grid, parameters, config, current_time):
    # note that the forcing is provided in mm/s
    # the flood section needs m3/s, but the routing needs m/s, so be aware of the conversions
    # method="pad" means the closest time in the past is selected from the file
    
    # TODO hacking in the land_frac modification for now - need to rework appropriately
    land = open_dataset('input/land_grid.nc')
    land_frac = land.frac.values.flatten()
    
    runoff = open_dataset(config.get('runoff.path'))
    
    # TODO hacking in the three hour offset seen in fortran mosart
    sel = {
        config.get('runoff.time'): current_time
    }
    
    if config.get('runoff.variables.surface_runoff', None) is not None:
        state.hillslope_surface_runoff = pd.DataFrame(0.001 * land_frac * grid.area.values * np.array(
            runoff[config.get('runoff.variables.surface_runoff')].sel(sel, method='pad')
        ).flatten())
    
    if config.get('runoff.variables.subsurface_runoff', None) is not None:
        state.hillslope_subsurface_runoff = pd.DataFrame(0.001 * land_frac * grid.area.values * np.array(
            runoff[config.get('runoff.variables.subsurface_runoff')].sel(sel, method='pad')
        ).flatten())
    
    if config.get('runoff.variables.wetland_runoff', None) is not None:
        state.hillslope_wetland_runoff = pd.DataFrame(0.001 * land_frac * grid.area.values * np.array(
            runoff[config.get('runoff.variables.wetland_runoff')].sel(sel, method='pad')
        ).flatten())
    
    runoff.close()
    land.close()
    
    return state