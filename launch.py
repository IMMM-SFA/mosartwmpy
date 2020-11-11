from src.mosart import Mosart
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import warnings

# ignore numpy NaN and invalid warnings
# (i.e. divide by zero and NaN logicals)
warnings.filterwarnings('ignore')

# launch simulation
m = Mosart()
m.initialize()
m.update()