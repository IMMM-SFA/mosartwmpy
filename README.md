![build](https://github.com/IMMM-SFA/mosartwmpy/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/IMMM-SFA/mosartwmpy/branch/main/graph/badge.svg?token=IPOY8984MB)](https://codecov.io/gh/IMMM-SFA/mosartwmpy)


### mosartwmpy

`mosartwmpy` is a python translation of Mosart-WM, a model for water routing and reservoir management written in Fortran. The original code can be found at [IWMM](https://github.com/IMMM-SFA/iwmm) and [E3SM](https://github.com/E3SM-Project/E3SM), in which Mosart is the hdyrological component of a larger suite of earth-science models. The motivation for rewriting is largely for developer convenience -- running, debugging, and adding new capabilities were becoming increasingly difficult due to the complexity of the codebase and lack of familiarity with Fortran. This version aims to be intuitive, lightweight, and well documented, while still being highly interoperable.

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

By default, key model variables are output on a monthly basis at a daily averaged resolution to `./output/<simulation name>/<simulation name>_<year>_<month>.nc`. Support for the [CSDMS standard names](https://github.com/csdms/standard_names) will be added shortly, but for now see configuration file and the `./src/_initialize_state.py` file for examples of how to modify the outputs.

## inputs

More details will be added to this section soon.

Several input files are required to successfuly run a simulation, which are not shipped with this repository due to their large size. For now, please reach out to the developers for sample input files. A script will soon be added to download a small subset of input files and validated output files from a public data repository.