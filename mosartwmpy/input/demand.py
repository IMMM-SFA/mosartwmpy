from benedict.dicts import benedict as Benedict
from datetime import datetime
import numpy as np
import pandas as pd
import regex as re
from xarray import open_dataset

from mosartwmpy.state.state import State
from mosartwmpy.utilities.timing import timing


# @timing
def load_demand(state: State, config: Benedict, current_time: datetime, mask: np.ndarray) -> None:
    """Loads water demand from file into the state for each grid cell.

    Args:
        state (State): the current model state; will be mutated
        config (Benedict): the model configuration
        current_time (datetime): the current time of the simulation
        mask (ndarray): mask of active grid cells
    """

    # demand path can have placeholders for year and month and day, so check for those and replace if needed
    path = config.get('water_management.demand.path')
    path = re.sub('\{(?:Y|y)[^}]*}', current_time.strftime('%Y'), path)
    path = re.sub('\{(?:M|m)[^}]*}', current_time.strftime('%m'), path)
    path = re.sub('\{(?:D|d)[^}]*}', current_time.strftime('%d'), path)

    demand = open_dataset(path)

    # if the demand file has a time axis, use it; otherwise assume data is just 2d
    if config.get('water_management.demand.time', None) in demand:
        # check for non-standard calendar and convert if needed
        if not isinstance(demand.indexes[config.get('water_management.demand.time')], pd.DatetimeIndex):
            demand[config.get('water_management.demand.time')] = demand.indexes[config.get('water_management.demand.time')].to_datetimeindex()
        # check if time index includes current time (with some slack on the end)
        if not (
            demand[config.get('water_management.demand.time')].values.min() <= np.datetime64(current_time) <= (demand[config.get('water_management.demand.time')].values.max() + np.timedelta64(31, 'D'))
        ):
            raise ValueError(
                f"Current simulation date {current_time.strftime('%Y-%m-%d')} not within time bounds of demand input file {path}. Aborting..."
            )
        state.grid_cell_demand_rate = np.array(demand[config.get('water_management.demand.demand')].sel({config.get('water_management.demand.time'): current_time}, method='pad')).flatten()[mask]
    else:
        state.grid_cell_demand_rate = np.array(demand[config.get('water_management.demand.demand')]).flatten()[mask]

    # fill missing values with 0
    state.grid_cell_demand_rate = np.where(
        np.logical_not(np.isfinite(state.grid_cell_demand_rate)),
        0,
        state.grid_cell_demand_rate
    )

    demand.close()
