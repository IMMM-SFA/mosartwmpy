import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import warnings

from src.mosart import Mosart

# ignore numpy NaN and invalid warnings
# (i.e. divide by zero and NaN logicals -- in Pandas/Dask, these simply remain NaN instead of becoming infinite)
warnings.filterwarnings('ignore')

# print whole dataframes
pd.options.display.max_columns = None

def plot(series):
    plt.imshow(np.array(series).reshape(self.get_grid_shape()), origin='lower')
    plt.colorbar()
    plt.show()

# launch simulation
self = Mosart()
self.initialize()
#self.update()
self.update_until(self.get_end_time())
self.grid.to_parquet(f'./output/{self.name}/grid.parquet')
self.state.to_parquet(f'./output/{self.name}/state.parquet')