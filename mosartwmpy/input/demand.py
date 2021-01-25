import numpy as np
import pandas as pd

from xarray import open_mfdataset

# TODO this currently can only act on the entire domain... need to make more robust so can be handled in parallel
def load_demand(state, config, current_time):

    demand = open_mfdataset(config.get('water_management.demand.path'))

    state.reservoir_monthly_demand = np.array(demand[config.get('water_management.demand.demand')].sel({config.get('water_management.demand.time'): current_time}, method='pad')).flatten()

    state.reservoir_monthly_demand = np.where(
        np.logical_not(np.isfinite(state.reservoir_monthly_demand)),
        0,
        state.reservoir_monthly_demand
    )

    demand.close()
