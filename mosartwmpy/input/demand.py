import logging
import numpy as np
import regex as re
import sys

from datetime import datetime
from xarray import open_dataset

from benedict.dicts import benedict as Benedict
from mosartwmpy.farmer_abm.farmer_abm import FarmerABM
from mosartwmpy.state.state import State
from mosartwmpy.utilities.timing import timing


# @timing
def load_demand(name: str, state: State, config: Benedict, current_time: datetime, farmerABM: FarmerABM, mask: np.ndarray) -> None:
    """Loads water demand from file into the state for each grid cell.

    Args:
        name (str): name of the simulation
        state (State): the current model state; will be mutated
        config (Benedict): the model configuration
        current_time (datetime): the current time of the simulation
        mask (ndarray): mask of active grid cells
    """

    path = config.get('water_management.demand.path')

    # Demand path can have placeholders for year and month and day, so check for those and replace if needed
    path = re.sub('\{y[^}]*}', current_time.strftime('%Y'), path)
    path = re.sub('\{m[^}]*}', current_time.strftime('%m'), path)
    path = re.sub('\{d[^}]*}', current_time.strftime('%d'), path)

    # Calculate water demand for farmers using an ABM.
    if config.get_bool('water_management.demand.farmer_abm.enabled'):
        try:
            farmerABM.calc_demand()
            path = f"{config.get('simulation.output_path')}/demand/{name}_farmer_abm_demand_{current_time.strftime('%Y')}.nc"
        except:
            logging.info(f"Water demand calculation for farmer ABM failed. Defaulting to precalculated values. ")

    try:
        demand = open_dataset(path)
    except:
        sys.exit(f"Unable to open demand file: {path} ")

    # if the demand file has a time axis, use it; otherwise assume data is just 2D
    if config.get('water_management.demand.time', None) in demand:
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
