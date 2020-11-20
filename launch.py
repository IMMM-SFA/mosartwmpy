from src.mosart import Mosart
from xarray import open_dataset
import dask.array as da
import dask.dataframe as dd
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# print whole dataframes
pd.options.display.max_columns = None

# ignore numpy NaN and invalid warnings
# (i.e. divide by zero and NaN logicals -- in Pandas/Dask, these simply remain NaN instead of becoming infinite)
warnings.filterwarnings('ignore')

# load the grid file and sample fortran mosart first timestep data for comparisons while testing
mos = open_dataset('./input/mosart_sample_data.nc')
g = open_dataset('./input/domains/MOSART_NLDAS_8th_20160426.nc')

def plot(series):
    plt.imshow(series.to_dask_array().compute().reshape(self.get_grid_shape()), origin='lower')
    plt.colorbar()
    plt.show()

# launch simulation
self = Mosart()
self.initialize()
#self.update()
#self.update_until(self.get_end_time())