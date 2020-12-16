import numpy as np
import pandas as pd

from xarray import open_dataset

# TODO this currently can only act on the entire domain... need to make more robust so can be handled in parallel
def load_demand(state, grid, parameters, config, current_time):
    
    demand = open_dataset(config.get('water_management.demand.path'))
    
    state.reservoir_monthly_demand[:] = np.array(demand[config.get('water_management.demand.demand')].sel({config.get('water_management.demand.time'): current_time}, method='pad')).flatten()
    
    demand.close()
    
    return state