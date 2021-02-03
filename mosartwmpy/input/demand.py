import numpy as np
import pandas as pd
import regex as re

from xarray import open_dataset

# TODO this currently can only act on the entire domain... need to make more robust so can be handled in parallel
def load_demand(state, config, current_time):

    # demand path can have placeholders for year and month and day, so check for those and replace if needed
    path = config.get('water_management.demand.path')
    path = re.sub('\{y[^}]*}', current_time.strftime('%Y'), path)
    path = re.sub('\{m[^}]*}', current_time.strftime('%m'), path)
    path = re.sub('\{d[^}]*}', current_time.strftime('%d'), path)

    demand = open_dataset(path)

    state.reservoir_monthly_demand = np.array(demand[config.get('water_management.demand.demand')].sel({config.get('water_management.demand.time'): current_time}, method='pad')).flatten()

    state.reservoir_monthly_demand = np.where(
        np.logical_not(np.isfinite(state.reservoir_monthly_demand)),
        0,
        state.reservoir_monthly_demand
    )

    demand.close()
