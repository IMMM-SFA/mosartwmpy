import numpy as np
from datetime import datetime, time, timedelta
from xarray import open_dataset

from benedict.dicts import benedict as Benedict
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State
from mosartwmpy.utilities.timing import timing


# @timing
def load_runoff(state: State, grid: Grid, config: Benedict, current_time: datetime) -> None:
    """Loads runoff from file into the state for each grid cell.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        config (Benedict): the model configuration
        current_time (datetime): the current time of the simulation
    """
    
    # note that the forcing is provided in mm/s
    # the flood section needs m3/s, but the routing needs m/s, so be aware of the conversions
    # method="pad" means the closest time in the past is selected from the file
    
    runoff = open_dataset(config.get('runoff.path'))
    
    sel = {
        config.get('runoff.time'): current_time
    }
    
    if config.get('runoff.variables.surface_runoff', None) is not None:
        state.hillslope_surface_runoff = 0.001 * grid.land_fraction * grid.area * np.array(
            runoff[config.get('runoff.variables.surface_runoff')].sel(sel, method='pad')
        ).flatten()
    
    if config.get('runoff.variables.subsurface_runoff', None) is not None:
        state.hillslope_subsurface_runoff = 0.001 * grid.land_fraction * grid.area * np.array(
            runoff[config.get('runoff.variables.subsurface_runoff')].sel(sel, method='pad')
        ).flatten()
    
    if config.get('runoff.variables.wetland_runoff', None) is not None:
        state.hillslope_wetland_runoff = 0.001 * grid.land_fraction * grid.area * np.array(
            runoff[config.get('runoff.variables.wetland_runoff')].sel(sel, method='pad')
        ).flatten()
    
    runoff.close()
