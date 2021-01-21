import numpy as np
import matplotlib.pyplot as plt
from xarray import open_dataset
vars = ['RIVER_DISCHARGE_OVER_LAND_LIQ', 'STORAGE_LIQ', 'QSUR_LIQ', 'QSUB_LIQ', 'WRM_DEMAND', 'WRM_DEMAND0', 'WRM_STORAGE', 'WRM_SUPPLY']
b = open_dataset('validation/wm-validation-1981-1982.nc')
p = open_dataset('output/Fiddling/Fiddling_1981_01.nc')
for var in vars:
  nmae = np.zeros(31)
  for t in np.arange(31):
    nmae[t] = 100 * np.fabs(b[var][t,:,:]  - p[var][t,:,:]).sum() / b[var][t,:,:].sum()
  f = plt.figure()
  plt.plot(nmae)
  plt.title(var)
  f.show()
