from numba import config as numba_config
numba_config.THREADING_LAYER = 'workqueue'

from .model import Model
from ._version import __version__
