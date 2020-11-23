import multiprocessing.popen_spawn_posix # due to bug ?

from dask.distributed import Client, LocalCluster
from xarray import open_dataset

import dask.array as da
import dask.dataframe as dd
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from src.mosart import Mosart

# ignore numpy NaN and invalid warnings
# (i.e. divide by zero and NaN logicals -- in Pandas/Dask, these simply remain NaN instead of becoming infinite)
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    dask_cluster = LocalCluster(
        n_workers=4,
        processes=1,
        #silence_logs=logging.ERROR,
        dashboard_address=None,
        memory_limit='16GB'
    )
    dask_client = Client(dask_cluster)

    # print whole dataframes
    pd.options.display.max_columns = None

    def plot(series):
        plt.imshow(series.to_dask_array().compute().reshape(self.get_grid_shape()), origin='lower')
        plt.colorbar()
        plt.show()

    # launch simulation
    self = Mosart()
    self.initialize()
    #self.update()
    self.update_until(self.get_end_time())
    self.grid.to_parquet(
        f'./output/{self.name}/grid',
        overwrite=True
    )
    self.state.to_parquet(
        f'./output/{self.name}/state',
        overwrite=True
    )