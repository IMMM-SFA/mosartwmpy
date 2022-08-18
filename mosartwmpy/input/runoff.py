import numpy as np
from datetime import datetime
import pandas as pd
import regex as re
from xarray import open_dataset

from benedict.dicts import benedict as Benedict
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State
from mosartwmpy.utilities.timing import timing


# @timing
def load_runoff(state: State, grid: Grid, config: Benedict, current_time: datetime, mask: np.ndarray) -> None:
    """Loads runoff from file into the state for each grid cell.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        config (Benedict): the model configuration
        current_time (datetime): the current time of the simulation
        mask (ndarray): mask of active grid cells
    """
    
    # note that the forcing is provided in mm/s
    # the flood section needs m3/s, but the routing needs m/s, so be aware of the conversions
    # method="pad" means the closest time in the past is selected from the file

    path = config.get('runoff.path')
    path = re.sub('\{(?:Y|y)[^}]*}', current_time.strftime('%Y'), path)
    path = re.sub('\{(?:M|m)[^}]*}', current_time.strftime('%m'), path)
    path = re.sub('\{(?:D|d)[^}]*}', current_time.strftime('%d'), path)

    runoff = open_dataset(path)

    # check for non-standard calendar and convert if needed
    if not isinstance(runoff.indexes[config.get('runoff.time')], pd.DatetimeIndex):
        runoff[config.get('runoff.time')] = runoff.indexes[config.get('runoff.time')].to_datetimeindex()

    # check if time index includes current time (with some slack)
    if not (
        runoff[config.get('runoff.time')].values.min() <= np.datetime64(current_time) <= (runoff[config.get('runoff.time')].values.max() + np.timedelta64(2, 'D'))
    ):
        raise ValueError(
            f"Current simulation date {current_time.strftime('%Y-%m-%d')} not within time bounds of runoff input file {path}. Aborting..."
        )
    
    sel = {
        config.get('runoff.time'): current_time
    }
    
    if config.get('runoff.variables.surface_runoff', None) is not None:
        state.hillslope_surface_runoff = 0.001 * grid.land_fraction * grid.area * np.array(
            runoff[config.get('runoff.variables.surface_runoff')].sel(sel, method='pad')
        ).flatten()[mask]
    
    if config.get('runoff.variables.subsurface_runoff', None) is not None:
        state.hillslope_subsurface_runoff = 0.001 * grid.land_fraction * grid.area * np.array(
            runoff[config.get('runoff.variables.subsurface_runoff')].sel(sel, method='pad')
        ).flatten()[mask]
    
    if config.get('runoff.variables.wetland_runoff', None) is not None:
        state.hillslope_wetland_runoff = 0.001 * grid.land_fraction * grid.area * np.array(
            runoff[config.get('runoff.variables.wetland_runoff')].sel(sel, method='pad')
        ).flatten()[mask]
    
    runoff.close()
