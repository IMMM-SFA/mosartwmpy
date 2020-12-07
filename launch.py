import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from xarray import open_mfdataset

from mosart.mosart import Mosart

# ignore numpy NaN and invalid warnings
# (i.e. divide by zero and NaN logicals -- in Pandas/Dask, these simply remain NaN instead of becoming infinite)
warnings.filterwarnings('ignore')

# print whole dataframes
pd.options.display.max_columns = None

def plot(series):
    mesh = np.array(series.mask(self.grid.mosart_mask == 0, np.nan)).reshape(self.get_grid_shape())
    plt.imshow(mesh, origin='lower')
    plt.colorbar()
    plt.show()

def plot_sample(v, t):
    series = pd.DataFrame(np.array(v[t,:,:]).flatten())
    plot(series)

# launch simulation
self = Mosart()
self.initialize()
#self.update()
self.update_until(self.get_end_time())
#self.grid.to_parquet(f'./output/{self.name}/grid.parquet')
#self.state.to_parquet(f'./output/{self.name}/state.parquet')
#sample = open_mfdataset('./input/mosart_sample_data.nc')
