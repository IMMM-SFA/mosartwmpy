from src.mosart import Mosart
from xarray import apply_ufunc, open_dataset
import dask.array as da
import dask.dataframe as dd
import logging
import matplotlib.pyplot as plt
import numpy as np
import warnings

# ignore numpy NaN and invalid warnings
# (i.e. divide by zero and NaN logicals)
warnings.filterwarnings('ignore')

# launch simulation
self = Mosart()
self.initialize()
self.update()