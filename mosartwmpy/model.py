import logging
import sys
import regex as re

from benedict import benedict
from bmipy import Bmi
from click import progressbar
from datetime import datetime, time, timedelta
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from numba import get_num_threads, threading_layer
import numpy as np
from pathlib import Path
from pathvalidate import sanitize_filename
import psutil
from timeit import default_timer as timer
from typing import Tuple
import xarray as xr

from mosartwmpy.config.config import get_config
from mosartwmpy.config.parameters import Parameters
from mosartwmpy.farmer_abm.farmer_abm import FarmerABM
from mosartwmpy.grid.grid import Grid
from mosartwmpy.input.runoff import load_runoff
from mosartwmpy.input.demand import load_demand
from mosartwmpy.input_output_variables import IO
from mosartwmpy.output.output import initialize_output, update_output
from mosartwmpy.reservoirs.release import reservoir_release
from mosartwmpy.state.state import State
from mosartwmpy.update.update import update
from mosartwmpy.utilities.pretty_timer import pretty_timer
from mosartwmpy.utilities.inherit_docs import inherit_docs


@inherit_docs
class Model(Bmi):
    """The mosartwmpy basic model interface.

    Args:
        Bmi (Bmi): The Basic Model Interface class

    Returns:
        Model: A BMI instance of the MOSART-WM model.
    """
    
    def __init__(self):
        self.name: str = None
        self.config = benedict()
        self.farmerABM = None
        self.grid = None
        self.restart = None
        self.current_time: datetime = None
        self.parameters = None
        self.state = None
        self.output_buffer = None
        self.output_n = 0
        self.cores = 1
        self.client = None
        self.mask = None

    def __getitem__(self, item):
        return getattr(self, item)

    def initialize(self, config_file_path: str = './config.yaml', grid: Grid = None, state: State = None) -> None:
        
        t = timer()

        try:
            # load config
            self.config = get_config(config_file_path)
            # initialize parameters
            self.parameters = Parameters()
            # sanitize the run name
            self.name = sanitize_filename(self.config.get('simulation.name')).replace(" ", "_")
            # setup logging and output directories
            Path(f'{self.config.get("simulation.output_path")}/{self.name}/restart_files').mkdir(parents=True, exist_ok=True)
            handlers = []
            if self.config.get('simulation.log_to_file'):
                handlers.append(logging.FileHandler(Path(f'{self.config.get("simulation.output_path")}/{self.name}/mosartwmpy.log')))
            if self.config.get('simulation.log_to_std_out'):
                h = logging.StreamHandler(sys.stdout)
                h.setFormatter(logging.Formatter(""))
                handlers.append(h)
            logging.basicConfig(
                level=self.config.get('simulation.log_level', 'INFO'),
                format='%(asctime)s - mosartwmpy: %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p',
                handlers=handlers
            )
            logging.info('Initalizing model...')
            # write config to output directory for posterity
            self.config.to_yaml(filepath=f'{self.config.get("simulation.output_path")}/{self.name}/config.yaml')
            if config_file_path is None or config_file_path == '':
                logging.info("No configuration file provided; initializing with all default values.")
            # ensure that end date is after start date
            if self.config.get('simulation.end_date') < self.config.get('simulation.start_date'):
                raise ValueError(f"Configured `end_date` {self.config.get('simulation.end_date')} is prior to configured `start_date` {self.config.get('simulation.start_date')}; please update and try again.")
            # detect available physical cores
            self.cores = psutil.cpu_count(logical=False)
            logging.debug(f'Cores: {self.cores}.')
            logging.debug(f'Numba threads: {get_num_threads()}.')
            logging.debug(f'Numba threading layer: {threading_layer()}')
        except Exception as e:
            logging.exception('Failed to configure model; see below for stacktrace.')
            raise e

        # load grid
        if grid is not None:
            self.grid = grid
        else:
            try:
                self.grid = Grid(config=self.config, parameters=self.parameters)
            except Exception as e:
                logging.exception('Failed to load grid file; see below for stacktrace.')
                raise e

        # load restart file or initialize state
        if state is not None:
            self.state = state
            self.current_time = datetime.combine(self.config.get('simulation.start_date'), time.min)
        else:
            try:
                # restart file
                if self.config.get('simulation.restart_file') is not None and self.config.get('simulation.restart_file') != '':
                    path = self.config.get('simulation.restart_file')
                    logging.info(f'Loading restart file from: `{path}`.')
                    # set simulation start time based on file name
                    date = re.search(r'\d{4}_\d{2}_\d{2}', path)
                    if date:
                        date = date[len(date) - 1].split('_')
                        self.current_time = datetime(int(date[0]), int(date[1]), int(date[2]))
                    else:
                        logging.warning('Unable to parse date from restart file name, falling back to configured start date.')
                        self.current_time = datetime.combine(self.config.get('simulation.start_date'), time.min)
                    x = xr.open_dataset(path)
                    self.state = State.from_dataframe(x.to_dataframe())
                    x.close()
                else:
                    # simulation start time
                    self.current_time = datetime.combine(self.config.get('simulation.start_date'), time.min)
                    # initialize state
                    self.state = State(grid=self.grid, config=self.config, parameters=self.parameters, grid_size=self.get_grid_size())
            except Exception as e:
                logging.exception('Failed to initialize model; see below for stacktrace.')
                raise e

        if self.config.get('water_management.demand.farmer_abm.enabled', False):
            self.farmerABM = FarmerABM(self)

        # trim all grid and state arrays to match mosart mask
        self.mask = np.where(
            self.grid.mosart_mask > 0,
            True,
            False
        )
        n = self.mask.size
        for key in [key for key in dir(self.grid) if isinstance(getattr(self.grid, key), np.ndarray)]:
            if getattr(self.grid, key).size == n:
                setattr(self.grid, key, getattr(self.grid, key)[self.mask])
        for key in [key for key in dir(self.state) if isinstance(getattr(self.state, key), np.ndarray)]:
            if getattr(self.state, key).size == n:
                setattr(self.state, key, getattr(self.state, key)[self.mask])

        # setup output file averaging
        try:
            initialize_output(self)
        except Exception as e:
            logging.exception('Failed to initialize output; see below for stacktrace.')
            raise e
        
        logging.debug(f'Initialization completed in {pretty_timer(timer() - t)}.')
        logging.info(f'Done.')
        
    def update(self) -> None:
        t = timer()
        step = datetime.fromtimestamp(self.get_current_time()).isoformat(" ")
        # perform one timestep
        logging.debug(f'Beginning timestep {step}...')
        try:
            if self.config.get('runoff.read_from_file', False):
                # read runoff from file
                logging.debug(f'Reading runoff input from file.')
                load_runoff(self.state, self.grid, self.config, self.current_time, self.mask)
            else:
                # convert provided runoff from mm/s to m3/s
                self.state.hillslope_surface_runoff = 0.001 * self.grid.land_fraction * self.grid.area * self.state.hillslope_surface_runoff
                self.state.hillslope_subsurface_runoff = 0.001 * self.grid.land_fraction * self.grid.area * self.state.hillslope_subsurface_runoff
                self.state.hillslope_wetland_runoff = 0.001 * self.grid.land_fraction * self.grid.area * self.state.hillslope_wetland_runoff
            # read demand
            if self.config.get('water_management.enabled', False):
                # only read new demand if it's the very start of simulation or new month
                if self.current_time == datetime.combine(
                    self.config.get('simulation.start_date'), time.min
                ) or self.current_time == datetime(self.current_time.year, self.current_time.month, 1):
                    if self.config.get('water_management.demand.read_from_file', False):
                        logging.debug(f'Reading demand rate input from file.')
                        # load the demand from file
                        load_demand(self.name, self.state, self.config, self.current_time, self.farmerABM, self.mask)
                # update reservoir release targets (logic for when to do this is inside the method)
                reservoir_release(self.state, self.grid, self.config, self.parameters, self.current_time, self.mask)
                # zero supply and demand
                self.state.grid_cell_supply[:] = 0
                self.state.grid_cell_unmet_demand[:] = 0
            # perform simulation for one timestep
            logging.debug('Solving...')
            update(self.state, self.grid, self.parameters, self.config, self.current_time)
            # advance timestep
            self.current_time += timedelta(seconds=self.config.get('simulation.timestep'))
        except Exception as e:
            logging.exception('Failed to complete timestep; see below for stacktrace.')
            raise e
        logging.debug(f'Timestep {step} completed in {pretty_timer(timer() - t)}.')
        try:
            # update the output buffer and write restart file if needed
            update_output(self)
        except Exception as e:
            logging.exception('Failed to write output or restart file; see below for stacktrace.')
            raise e
        if self.config.get('runoff.read_from_file', False):
            # clear runoff input arrays
            self.state.hillslope_surface_runoff[:] = 0
            self.state.hillslope_subsurface_runoff[:] = 0
            self.state.hillslope_wetland_runoff[:] = 0
        else:
            # convert back to mm/s
            self.state.hillslope_surface_runoff = self.state.hillslope_surface_runoff * 1000.0 / self.grid.land_fraction / self.grid.area
            self.state.hillslope_subsurface_runoff = self.state.hillslope_subsurface_runoff * 1000.0 / self.grid.land_fraction / self.grid.area
            self.state.hillslope_wetland_runoff = self.state.hillslope_wetland_runoff * 1000.0 / self.grid.land_fraction / self.grid.area

    def update_until(self, time: float = None) -> None:
        # if time is None, set time to end time
        if time is None:
            time = self.get_end_time()
        # make sure that requested end time is after now
        if time < self.current_time.timestamp():
            logging.error('`time` is prior to current model time. Please choose a new `time` and try again.')
            return
        # perform timesteps until time
        logging.info(f'Beginning simulation for {datetime.fromtimestamp(self.get_current_time()).date().isoformat()} through {datetime.fromtimestamp(time).date().isoformat()}...')
        t = timer()
        with progressbar(
            label='Running mosartwmpy',
            length=int((time - self.current_time.timestamp()) // self.config.get('simulation.timestep')),
            item_show_func=lambda t: t,
        ) as progress:
            while self.get_current_time() < time:
                # update progress bar
                current_datetime = datetime.fromtimestamp(self.get_current_time())
                progress.update(1, current_datetime.isoformat(" "))
                # advance one timestep
                self.update()
        logging.info(f'Simulation completed in {pretty_timer(timer() - t)}.')

    def finalize(self) -> None:
        # simulation is over so free memory, write data, etc
        for handler in logging.getLogger().handlers:
            handler.close()
        logging.getLogger().handlers.clear()
        logging.shutdown()
        return

    def plot_variable(
            self,
            variable: str,
            log_scale: bool = False,
            show: bool = True,
    ):
        """Display a colormap of a spatial variable at the current timestep."""
        data = self.get_value_ptr(variable).reshape(self.get_grid_shape())
        if log_scale:
            data = np.where(data > 0, data, np.nan)
        xr.DataArray(
            data,
            dims=['latitude', 'longitude'],
            coords={'latitude': self.get_grid_x(), 'longitude': self.get_grid_y()},
            name=variable.replace('_', ' ').title(),
            attrs={'units': f'{"Log " if log_scale else ""}{self.get_var_units(variable)}'}
        ).plot(
            robust=True,
            levels=None if log_scale else 16,
            cmap='winter_r',
            norm=colors.LogNorm() if log_scale else None,
        )
        if show:
            plt.show()
        return plt

    def unmask(self, vector: np.ndarray) -> np.ndarray:
        unmasked = np.empty_like(self.mask, dtype=vector.dtype)
        if vector.dtype == float:
            unmasked[:] = np.nan
        elif vector.dtype == int:
            unmasked[:] = -9999
        elif vector.dtype == bool:
            unmasked[:] = False
        unmasked[self.mask] = vector
        return unmasked

    def get_component_name(self) -> str:
        return f'mosartwmpy ({self.git_hash})'

    def get_input_item_count(self) -> int:
        return len(IO.inputs)

    def get_output_item_count(self) -> int:
        return len(IO.outputs)

    def get_input_var_names(self) -> Tuple[str]:
        return tuple(str(var.standard_name) for var in IO.inputs)

    def get_output_var_names(self) -> Tuple[str]:
        return tuple(str(var.standard_name) for var in IO.outputs)

    def get_var_grid(self, name: str) -> int:
        # only one grid used in mosart, so it is the 0th grid
        return 0

    def get_var_type(self, name: str) -> str:
        return next((var.variable_type for var in IO.inputs + IO.outputs if var.standard_name == name), None)

    def get_var_units(self, name: str) -> str:
        return next((var.units for var in IO.inputs + IO.outputs if var.standard_name == name), None)

    def get_var_itemsize(self, name: str) -> int:
        return next((var.variable_item_size for var in IO.inputs + IO.outputs if var.standard_name == name), None)

    def get_var_nbytes(self, name: str) -> int:
        item_size = self.get_var_itemsize(name)
        return item_size * self.get_grid_size()

    def get_var_location(self, name: str) -> str:
        # node, edge, face
        return 'node'

    def get_current_time(self) -> float:
        return self.current_time.timestamp()

    def get_start_time(self) -> float:
        return datetime.combine(self.config.get('simulation.start_date'), time.min).timestamp()

    def get_end_time(self) -> float:
        return datetime.combine(self.config.get('simulation.end_date'), time.max).timestamp()

    def get_time_units(self) -> str:
        return 's'

    def get_time_step(self) -> float:
        return float(self.config.get('simulation.timestep'))

    def get_value(self, name: str, dest: np.ndarray) -> int:
        var = next((var for var in IO.inputs + IO.outputs if var.standard_name == name), None)
        if var is None:
            return 1
        dest[:] = self.unmask(self[var.variable_class][var.variable])
        return 0

    def get_value_ptr(self, name: str) -> np.ndarray:
        var = next((var for var in IO.inputs + IO.outputs if var.standard_name == name), None)
        if var is None:
            raise IOError(f'Variable {name} not found in model input/output definition.')
        return self.unmask(self[var.variable_class][var.variable])

    def get_value_at_indices(self, name: str, dest: np.ndarray, inds: np.ndarray) -> int:
        var = next((var for var in IO.inputs + IO.outputs if var.standard_name == name), None)
        if var is None:
            return 1
        dest[:] = self.unmask(self[var.variable_class][var.variable])[inds]
        return 0

    def set_value(self, name: str, src: np.ndarray) -> int:
        var = next((var for var in IO.inputs + IO.outputs if var.standard_name == name), None)
        if var is None:
            return 1
        self[var.variable_class][var.variable][:] = src[self.mask]
        return 0

    def set_value_at_indices(self, name: str, inds: np.ndarray, src: np.ndarray) -> int:
        var = next((var for var in IO.inputs + IO.outputs if var.standard_name == name), None)
        if var is None:
            return 1
        unmasked = self.unmask(self[var.variable_class][var.variable])
        unmasked[inds] = src
        self[var.variable_class][var.variable][:] = unmasked[self.mask]
        return 0

    def get_grid_type(self, grid: int = 0) -> str:
        return 'uniform_rectilinear'

    def get_grid_rank(self, grid: int = 0) -> int:
        return 2
    
    def get_grid_size(self, grid: int = 0) -> int:
        return self.grid.cell_count

    def get_grid_shape(self, grid: int = 0, shape: np.ndarray = np.empty(2, dtype=np.int64)) -> np.ndarray:
        shape[0] = self.grid.unique_latitudes.size
        shape[1] = self.grid.unique_longitudes.size
        return shape

    def get_grid_spacing(self, grid: int = 0, spacing: np.ndarray = np.empty(2)) -> np.ndarray:
        # assumes uniform grid
        spacing[0] = self.grid.latitude_spacing
        spacing[1] = self.grid.longitude_spacing
        return spacing
    
    def get_grid_origin(self, grid: int = 0, origin: np.ndarray = np.empty(2)) -> np.ndarray:
        origin[0] = self.grid.unique_latitudes[0]
        origin[1] = self.grid.unique_longitudes[0]
        return origin

    def get_grid_x(self, grid: int = 0, x: np.ndarray = None) -> np.ndarray:
        if not x:
            x = np.empty(self.get_grid_shape()[0])
        x[:] = self.grid.unique_latitudes
        return x

    def get_grid_y(self, grid: int = 0, y: np.ndarray = None) -> np.ndarray:
        if not y:
            y = np.empty(self.get_grid_shape()[1])
        y[:] = self.grid.unique_longitudes
        return y

    def get_grid_z(self, grid: int = 0, z: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError

    def get_grid_node_count(self, grid: int = 0) -> int:
        raise NotImplementedError

    def get_grid_edge_count(self, grid: int = 0) -> int:
        raise NotImplementedError

    def get_grid_face_count(self, grid: int = 0) -> int:
        raise NotImplementedError

    def get_grid_edge_nodes(self, grid: int = 0, edge_nodes: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError

    def get_grid_face_edges(self, grid: int = 0, face_edges: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError

    def get_grid_face_nodes(self, grid: int = 0, face_nodes: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError

    def get_grid_nodes_per_face(self, grid: int = 0, nodes_per_face: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError
