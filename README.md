![build](https://github.com/IMMM-SFA/mosartwmpy/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/IMMM-SFA/mosartwmpy/branch/main/graph/badge.svg?token=IPOY8984MB)](https://codecov.io/gh/IMMM-SFA/mosartwmpy)


### getting started

`mosartwmpy` is a python translation of Mosart-WM, a model for water routing and reservoir management written in Fortran. The original code can be found at [IWMM](https://github.com/IMMM-SFA/iwmm) and [E3SM](https://github.com/E3SM-Project/E3SM), in which Mosart is the hdyrological component of a larger suite of earth-science models. The motivation for rewriting is largely for developer convenience -- running, debugging, and adding new capabilities were becoming increasingly difficult due to the complexity of the codebase and lack of familiarity with Fortran. This version aims to be intuitive, lightweight, and well documented, while still being highly interoperable.

Install requirements with `pip install -r requirements.txt`.

`mosartwmpy` implements the [Basic Model Interface](https://csdms.colorado.edu/wiki/BMI) defined by the CSDMS, so driving it should be familiar to those accustomed to the BMI:

```python
from datetime import datetime, time
from mosartwmpy.mosartwmpy import Model

# initialize the model
mosart_wm = Model()
mosart_wm.initialize()

# advance the model one timestep
mosart_wm.update()

# advance until a specificed timestamp
mosart_wm.update_until(datetime.combine(datetime(2030, 12, 31), time.max).timestamp())
```

Settings are defined by the merger of the `config_defaults.yaml` and an optional user specified file which can override any of the default settings:

```python
mosart_wm = Model('path/to/config/file.yaml')
```

Alternatively, one can update the settings via code in the driving script:

```python
 mosart_wm = Model()
 mosart_wm.initialize()
 
 mosart_wm.config['simulation.name'] = 'Water Management'
 mosart_wm.config['simulation.start_date'] = datetime(1981, 1, 1)
 mosart_wm.config['simulation.end_date'] = datetime(1985, 12, 31)
```

By default, key model variables are output on a monthly basis at a daily averaged resolution to `./output/<simulation name>/<simulation name>_<year>_<month>.nc`. See the configuration file for examples of how to modify the outputs, and the `./mosartwmpy/state/state.py` file for state variable names.

Alternatively, certain model outputs deemed most important can be accessed using the BMI interface methods. For example:
```python
# get a list of model output variables
mosart_wm.get_output_var_names()

# get the flattened numpy.ndarray of values for an output variable
mosart_wm.get_value_ptr('supply_water_amount')
```


### input

Several input files in NetCDF format are required to successfully run a simulation, which are not shipped with this repository due to their large size. The grid files, reservoir files, and a small range of runoff and demand input files are available for public download as a zip archive [here](https://zenodo.org/record/4537907/files/mosartwmpy_sample_input_data_1980_1985.zip?download=1). This data can also be obtained using the download utility by running `python download.py` in the repository root and choosing option 1 for "sample_input". Currently, all input files are assumed to be at the same resolution (for the sample files this is 1/8 degree over the CONUS). Below is a summary of the various input files:

Name | Description | Configuration Path | Notes
--- | --- | --- | ---
Grid | Spatial constants dimensioned by latitude and longitude relating to the physical properties of the river channels | `grid.path` |
Land Fraction | Fraction of grid cell that is land (as opposed to i.e. ocean water) dimensioned by latitude and longitude | `grid.land.path` | as a TODO item, this variable should be merged into the grid file (historically it was separate for the coupled land model)
Reservoirs | Locations of reservoirs (possibly aggregated) and their physical and political properties | `water_management.reservoirs.path` |
Runoff | Surface runoff, subsurface runoff, and wetland runoff per grid cell averaged per unit of time; used to drive the river routing | `runoff.path` |
Demand | Water demand of grid cells averaged per unit of time; currently assumed to be monthly | `water_management.reservoirs.demand` | there are plans to support other time scales, such as epiweeks

Alternatively, certain model inputs can be set using the BMI interface. This can be useful for coupling `mosartwmpy` with other models. If setting an input that would typically be read from a file, be sure to disable the `read_from_file` configuration value for that input. For example:
```python
    # get a list of model input variables
    mosart_wm.get_input_var_names()
    
    # disable the runoff read_from_file
    mosart_wm.config['runoff.read_from_file'] = False

    # set the runoff values manually (i.e. from another model's output)
    surface_runoff = np.empty(mosart_wm.get_grid_size())
    surface_runoff[:] = <values from coupled model>
    mosart_wm.set_value('surface_runoff_flux', surface_runoff)
```

### testing and validation

Before running the tests or validation, make sure to download the "sample_input" and "validation" datasets using the download utility `python download.py`.

To execute the tests, run `./test.sh` or `python -m unittest discover mosartwmpy/tests` from the repository root.

To execute the validation, run a model simulation that includes the years 1981 - 1982, note your output directory, and then run `./validation.sh` or `python validation/validate.py` from the repository root. This will ask you for the simulation output directory, think for a moment, and then open a figure with several plots representing the NMAE (Normalized Mean Absolute Error) as a percentage and the spatial sums of several key variables compared between your simulation and the validation scenario. Use these plots to assist you in determining if the changes you have made to the code have caused unintended deviation from the validation scenario. The NMAE should be 0% across time if you have caused no deviations. A non-zero NMAE indicates numerical difference between your simulation and the validation scenario. This might be caused by changes you have made to the code, or alternatively by running a simulation with different configuration or parameters (i.e. larger timestep, fewer iterations, etc). The plots of the spatial sums can assist you in determining what changed and the overall magnitude of the changes.

If you wish to merge code changes that intentionally cause significant deviation from the validation scenario, please work with the maintainers to create a new validation dataset.