import logging
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import pandas as pd
import psutil
import regex as re
import subprocess

from benedict import benedict
from bmipy import Bmi
from datetime import datetime, time, timedelta
from epiweeks import Week
from pathlib import Path
from pathvalidate import sanitize_filename
from timeit import default_timer as timer
from typing import Tuple
from xarray import open_dataset

from mosartwmpy.config.config import get_config
from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.input.runoff import load_runoff
from mosartwmpy.input.demand import load_demand
from mosartwmpy.output.output import initialize_output, update_output, write_restart
from mosartwmpy.reservoirs.reservoirs import reservoir_release
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
        self.name = None
        self.config = benedict()
        self.grid = None
        self.restart = None
        self.current_time = None
        self.parameters = None
        self.state = None
        self.output_buffer = None
        self.output_n = 0
        self.cores = 1
        self.client = None
        self.reservoir_streamflow_schedule = None
        self.reservoir_demand_schedule = None
        self.reservoir_prerelease_schedule = None
        self.git_hash = None
        self.git_untracked = None

    def initialize(self, config_file_path: str) -> None:
        
        t = timer()

        try:
            # load config
            self.config = get_config(config_file_path)
            # initialize parameters
            self.parameters = Parameters()
            # sanitize the run name
            self.name = sanitize_filename(self.config.get('simulation.name')).replace(" ", "_")
            # setup logging and output directory
            Path(f'./output/{self.name}').mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                filename=f'./output/{self.name}/mosartwmpy.log',
                level=self.config.get('simulation.log_level', 'INFO'),
                format='%(asctime)s - mosartwmpy: %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p')
            logging.info('Initalizing model.')
            logging.info(self.config.dump())
            try:
                self.git_hash = subprocess.check_output(['git', 'describe', '--always']).strip().decode('utf-8')
                self.git_untracked = subprocess.check_output(['git', 'diff', '--name-only']).strip().decode('utf-8').split('\n')
                logging.info(f'Version: {self.git_hash}')
                if len(self.git_untracked) > 0:
                    logging.info(f'Uncommitted changes:')
                    for u in self.git_untracked:
                        logging.info(f'  * {u}')
            except:
                pass
            # detect available physical cores
            self.cores = psutil.cpu_count(logical=False)
            logging.info(f'Cores: {self.cores}.')
            ne.set_num_threads(self.cores)
        except Exception as e:
            logging.exception('Failed to configure model; see below for stacktrace.')
            raise e

        # load grid
        try:
            self.grid = Grid(config=self.config, parameters=self.parameters)
        except Exception as e:
            logging.exception('Failed to load grid file; see below for stacktrace.')
            raise e

        # load restart file or initialize state
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
                    logging.warn('Unable to parse date from restart file name, falling back to configured start date.')
                    self.current_time = datetime.combine(self.config.get('simulation.start_date'), time.min)
                x = open_dataset(path)
                self.state = State.from_dataframe(x.to_dataframe())
                x.close()
                # TODO ensure output file writes still workr
            else:
                # simulation start time
                self.current_time = datetime.combine(self.config.get('simulation.start_date'), time.min)
                # initialize state
                self.state = State(grid=self.grid, config=self.config, parameters=self.parameters, grid_size=self.get_grid_size())
        except Exception as e:
            logging.exception('Failed to initialize model; see below for stacktrace.')
            raise e
        
        # setup output file averaging
        try:
            initialize_output(self)
        except Exception as e:
            logging.exception('Failed to initialize output; see below for stacktrace.')
            raise e
        
        logging.info(f'Initialization completed in {pretty_timer(timer() - t)}.')
        
    def update(self) -> None:
        t = timer()
        step = datetime.fromtimestamp(self.get_current_time()).isoformat(" ")
        # perform one timestep
        logging.info(f'Begin timestep {step}.')
        try:
            # read runoff
            if self.config.get('runoff.enabled', False):
                logging.debug(f'Reading runoff input.')
                load_runoff(self.state, self.grid, self.config, self.current_time)
            # read demand
            if self.config.get('water_management.enabled', False):
                # only read new demand and compute new release if it's the very start of simulation or new time period
                # TODO this currently assumes monthly demand input
                if self.current_time == datetime.combine(self.config.get('simulation.start_date'), time.min) or self.current_time == datetime(self.current_time.year, self.current_time.month, 1):
                    logging.debug(f'Reading demand input.')
                    # load the demand from file
                    load_demand(self.state, self.config, self.current_time)
                    # release water from reservoirs
                    reservoir_release(self.state, self.grid, self.config, self.parameters, self.current_time)
                # zero supply and demand
                self.state.reservoir_supply[:] = 0
                self.state.reservoir_demand[:] = 0
                self.state.reservoir_deficit[:] = 0
                # get streamflow for this time period
                # TODO this is still written assuming monthly, but here's the epiweek for when that is relevant
                epiweek = Week.fromdate(self.current_time).week
                month = self.current_time.month
                streamflow_time_name = self.config.get('water_management.reservoirs.streamflow_time_resolution')
                self.state.reservoir_streamflow[:] = self.grid.reservoir_streamflow_schedule.sel({streamflow_time_name: month}).values
            # perform simulation for one timestep
            update(self.state, self.grid, self.parameters, self.config)
            # advance timestep
            self.current_time += timedelta(seconds=self.config.get('simulation.timestep'))
        except Exception as e:
            logging.exception('Failed to complete timestep; see below for stacktrace.')
            raise e
        logging.info(f'Timestep {step} completed in {pretty_timer(timer() - t)}.')
        try:
            # update the output buffer and write restart file if needed
            update_output(self)
        except Exception as e:
            logging.exception('Failed to write output or restart file; see below for stacktrace.')
            raise e

    def update_until(self, time: float) -> None:
        # perform timesteps until time
        t = timer()
        while self.get_current_time() < time:
            self.update()
        logging.info(f'Simulation completed in {pretty_timer(timer() - t)}.')

    def finalize(self) -> None:
        # simulation is over so free memory, write data, etc
        return

    def get_component_name(self) -> str:
        # TODO include version/hash info?
        return f'mosartwmpy ({self.git_hash})'

    def get_input_item_count(self) -> int:
        # TODO
        return 0

    def get_output_item_count(self) -> int:
        # TODO
        return 0

    def get_input_var_names(self) -> Tuple[str]:
        # TODO
        return []

    def get_output_var_names(self) -> Tuple[str]:
        # TODO
        return []

    def get_var_grid(self, name: str) -> int:
        # only one grid used in mosart, so it is the 0th grid
        return 0

    def get_var_type(self, name: str) -> str:
        # TODO
        return 'TODO'

    def get_var_units(self, name: str) -> str:
        # TODO
        return 'TODO'

    def get_var_itemsize(self, name: str) -> int:
        # TODO
        return 0

    def get_var_nbytes(self, name: str) -> int:
        # TODO
        return 0

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

    def get_value(self, name: str, dest: np.ndarray) -> np.ndarray:
        # TODO copy values into array
        return

    def get_value_ptr(self, name: str) -> np.ndarray:
        # TODO set array to current array pointer
        return

    def get_value_at_indices(self, name: str, dest: np.ndarray, inds: np.ndarray) -> np.ndarray:
        # TODO copy values from indices into array
        return

    def set_value(self, name: str, src: np.ndarray) -> None:
        # TODO set values of name from array
        return

    def set_value_at_indices(self, name: str, inds: np.ndarray, src: np.ndarray) -> None:
        # TODO set values of name at indices from array
        return

    def get_grid_type(self, grid: int = 0) -> str:
        return 'uniform_rectilinear'

    def get_grid_rank(self, grid: int = 0) -> int:
        return 2
    
    def get_grid_size(self, grid: int = 0) -> int:
        return self.grid.cell_count

    def get_grid_shape(self, grid: int = 0, shape: np.ndarray = np.empty(2, dtype=int)) -> np.ndarray:
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