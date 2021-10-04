import pdb

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from xarray import open_dataset, open_mfdataset

# TODO accept command line path input as alternative
# TODO accept command line year input
# TODO allow easily toggling between scenarios for variables of interest (no-wm, wm, heat, etc)

years = [1981, 1982]
baseline_data_path = str(Path('validation/mosartwmpy_validation_wm_1981_1982.nc'))
variables_of_interest = ['STORAGE_LIQ', 'RIVER_DISCHARGE_OVER_LAND_LIQ', 'WRM_STORAGE', 'WRM_SUPPLY']
physical_dimensions = ['lat', 'lon']
temporal_dimension = 'time'

print()
print("🎶  MOSARTWMPY VALIDATION 🎶")
print()
print("Thanks for validating your code changes!")
print(f"This tool expects that you have generated data within the years {years}.")
print("Please open an issue on GitHub if you have any trouble or suggestions.")
print("https://github.com/IMMM-SFA/mosartwmpy")
print()

data_path = input("Where are your output files located? Enter the absolute path or path relative to the mosartwmpy root directory: ")
print()

assert os.path.exists(data_path), f"Unable to find this path, {str(data_path)} - please double check and try again."
data = open_mfdataset(f"{data_path}/*.nc" if data_path[-3:] != '.nc' else data_path)

try:
    data = data.sel({temporal_dimension: slice(f"{years[0]}", f"{years[-1]}")})
    timeslice = slice(data[temporal_dimension].values[0], data[temporal_dimension].values[len(data[temporal_dimension].values) - 1])
    baseline_data = open_mfdataset(baseline_data_path)
    baseline_data = baseline_data.sel({temporal_dimension: timeslice})
except:
    print(f"Either your data or the baseline data does not appear to include years {years}.")
    print("Please double check and try again or update this code to look for a different year.")
    quit()

try:
    data = data[variables_of_interest]
    baseline_data = baseline_data[variables_of_interest]
except:
    print(f"Either your data or the baseline data does not contain the expected variables: {variables_of_interest}.")
    print("Please double check and try again or update this code to look for your variables of interest.")
    quit()

# normalize the time indexes to prevent alignment errors
data[temporal_dimension] = data.indexes[temporal_dimension].normalize()
baseline_data[temporal_dimension] = baseline_data.indexes[temporal_dimension].normalize()

nmae = 100.0 * np.abs(baseline_data - data).sum(dim=physical_dimensions) / baseline_data.sum(dim=physical_dimensions)
data_sums = data.sum(dim=physical_dimensions)
baseline_sums = baseline_data.sum(dim=physical_dimensions)

figure, axes = plt.subplots(len(variables_of_interest), 2)

for i in range(len(variables_of_interest)):
    v = variables_of_interest[i]
    axis = axes[i, 0]
    nmae[v].plot(ax=axis)
    axis.set_ylabel(None)
    axis.set_xlabel(None)
    axis.set_ylim([0, 100])
    axis.set_title(v)
    axis = axes[i, 1]
    baseline_sums[v].plot(ax=axis)
    data_sums[v].plot(ax=axis)
    axis.set_ylabel(None)
    axis.set_xlabel(None)
    axis.set_title(f'Sum of {v}')
    axis.legend(['baseline', 'validation'])

figure.text(0.5, 0.04, 'time', ha='center')
figure.text(0.04, 0.5, 'NMAE (%)', va='center', rotation='vertical')
figure.tight_layout()
plt.show()
