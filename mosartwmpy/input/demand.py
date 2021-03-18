import numpy as np
import regex as re

from datetime import datetime
from xarray import open_dataset

from benedict.dicts import benedict as Benedict
from mosartwmpy.state.state import State
from mosartwmpy.utilities.timing import timing


# @timing
def load_demand(state: State, config: Benedict, current_time: datetime) -> None:
    """Loads water demand from file into the state for each grid cell.

    Args:
        state (State): the current model state; will be mutated
        config (Benedict): the model configuration
        current_time (datetime): the current time of the simulation
    """

    # demand path can have placeholders for year and month and day, so check for those and replace if needed
    path = config.get('water_management.demand.path')
    path = re.sub('\{y[^}]*}', current_time.strftime('%Y'), path)
    path = re.sub('\{m[^}]*}', current_time.strftime('%m'), path)
    path = re.sub('\{d[^}]*}', current_time.strftime('%d'), path)

    demand = open_dataset(path)

    # if the demand file has a time axis, use it; otherwise assume data is just 2d
    if config.get('water_management.demand.time', None) in demand:
        state.grid_cell_demand_rate = np.array(demand[config.get('water_management.demand.demand')].sel({config.get('water_management.demand.time'): current_time}, method='pad')).flatten()
    else:
        state.grid_cell_demand_rate = np.array(demand[config.get('water_management.demand.demand')]).flatten()

    # fill missing values with 0
    state.grid_cell_demand_rate = np.where(
        np.logical_not(np.isfinite(state.grid_cell_demand_rate)),
        0,
        state.grid_cell_demand_rate
    )

    demand.close()
