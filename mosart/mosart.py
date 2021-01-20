import logging
import matplotlib.pyplot as plt
import numpy as np

from benedict import benedict
from bmipy import Bmi
from datetime import datetime, time
from timeit import default_timer as timer

from mosart.config.setup import setup
from mosart.grid.load_grid import load_grid
from mosart.output.output import initialize_output, update_output, write_restart
from mosart.state.initialize_state import initialize_state
from mosart.update.update import update

class Mosart(Bmi):
    # Mosart Basic Model Interface
    def __init__(self):
        # initialize properties
        self.name = None
        self.config = benedict()
        self.grid = None
        self.longitude = None
        self.latitude = None
        self.cell_count = None
        self.longitude_spacing = None
        self.latitude_spacing = None
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

    def initialize(self, config_file_path: str = None):
        t = timer()

        # load config and setup logging
        try:
            setup(self, config_file_path)
        except Exception as e:
            logging.exception('Failed to configure model; see below for stacktrace.')
            raise e

        # load grid
        try:
            load_grid(self)
        except Exception as e:
            logging.exception('Failed to load grid file; see below for stacktrace.')
            raise e

        # load restart file or initialize state
        try:
            initialize_state(self)
        except Exception as e:
            logging.exception('Failed to initialize model; see below for stacktrace.')
            raise e
        
        # setup output file averaging
        try:
            initialize_output(self)
        except Exception as e:
            logging.exception('Failed to initialize output; see below for stacktrace.')
            raise e
        
        logging.info(f'Initialization completed in {self.pretty_timer(timer() - t)}.')
        
    def update(self):
        t = timer()
        step = datetime.fromtimestamp(self.get_current_time()).isoformat(" ")
        # perform one timestep
        logging.info(f'Begin timestep {step}.')
        try:
            update(self)
        except Exception as e:
            logging.exception('Failed to complete timestep; see below for stacktrace.')
            raise e
        logging.info(f'Timestep {step} completed in {self.pretty_timer(timer() - t)}.')
        try:
            # update the output buffer and write restart file if needed
            update_output(self)
        except Exception as e:
            logging.exception('Failed to write output or restart file; see below for stacktrace.')
            raise e

    def update_until(self, time: float):
        # perform timesteps until time
        while self.get_current_time() < time:
            self.update()

    def finalize(self):
        # simulation is over so free memory, write data, etc
        return

    def pretty_timer(self, seconds):
        # format elapsed times in a human friendly way
        # TODO move to a utitities file
        if seconds < 1:
            return f'{round(seconds * 1.0e3, 0)} milliseconds'
        elif seconds < 60:
            return f'{round(seconds, 3)} seconds'
        elif seconds < 3600:
            return f'{int(round(seconds) // 60)} minutes and {int(round(seconds) % 60)} seconds'
        elif seconds < 86400:
            return f'{int(round(seconds) // 3600)} hours, {int((round(seconds) % 3600) // 60)} minutes, and {int(round(seconds) % 60)} seconds'
        else:
            return f'{int(round(seconds) // 86400)} days, {int((round(seconds) % 86400) // 3600)} hours, and {int((round(seconds) % 3600) // 60)} minutes'

    def get_component_name(self):
        # TODO include version/hash info?
        return 'Mosart'

    def get_input_item_count(self):
        # TODO
        return 0

    def get_output_item_count(self):
        # TODO
        return 0

    def get_input_var_names(self):
        # TODO
        return []

    def get_output_var_names(self):
        # TODO
        return []

    def get_var_grid(self, name: str):
        # only one grid used in mosart, so it is the 0th grid
        return 0

    def get_var_type(self, name: str):
        # TODO
        return 'TODO'

    def get_var_units(self, name: str):
        # TODO
        return 'TODO'

    def get_var_itemsize(self, name: str):
        # TODO
        return 0

    def get_var_nbytes(self, name: str):
        # TODO
        return 0

    def get_var_location(self, name: str):
        # node, edge, face
        return 'node'

    def get_current_time(self):
        return self.current_time.timestamp()

    def get_start_time(self):
        return datetime.combine(self.config.get('simulation.start_date'), time.min).timestamp()

    def get_end_time(self):
        return datetime.combine(self.config.get('simulation.end_date'), time.max).timestamp()

    def get_time_units(self):
        return 's'

    def get_time_step(self):
        return float(self.config.get('simulation.timestep'))

    def get_value(self, name: str, array):
        # TODO copy values into array
        return

    def get_value_ptr(self, name, array):
        # TODO set array to current array pointer
        return

    def get_value_at_indices(self, array, indices):
        # TODO copy values from indices into array
        return

    def set_value(self, name: str, array):
        # TODO set values of name from array
        return

    def set_value_at_indices(self, name: str, indices, array):
        # TODO set values of name at indices from array
        return

    def get_grid_type(self, grid: int = 0):
        return 'uniform_rectilinear'

    def get_grid_rank(self, grid: int = 0):
        return 2
    
    def get_grid_size(self, grid: int = 0):
        return self.cell_count

    def get_grid_shape(self, grid: int = 0, shape = np.empty(2, dtype=int)):
        shape[0] = self.latitude.size
        shape[1] = self.longitude.size
        return shape

    def get_grid_spacing(self, grid: int = 0, spacing = np.empty(2)):
        # assumes uniform grid
        spacing[0] = self.latitude_spacing
        spacing[1] = self.longitude_spacing
        return spacing
    
    def get_grid_origin(self, grid: int = 0, origin = np.empty(2)):
        origin[0] = self.latitude[0]
        origin[1] = self.longitude[0]
        return origin

    def get_grid_x(self, grid: int = 0, x = None):
        if not x:
            x = np.empty(self.get_grid_shape()[0])
        x[:] = self.latitude
        return x

    def get_grid_y(self, grid: int = 0, y = None):
        if not y:
            y = np.empty(self.get_grid_shape()[1])
        y[:] = self.longitude
        return y

    def get_grid_z(self, grid: int = 0, z = None):
        raise NotImplementedError

    def get_grid_node_count(self, grid: int = 0):
        raise NotImplementedError

    def get_grid_edge_count(self, grid: int = 0):
        raise NotImplementedError

    def get_grid_face_count(self, grid: int = 0):
        raise NotImplementedError

    def get_grid_edge_nodes(self, grid: int = 0, edge_nodes = None):
        raise NotImplementedError

    def get_grid_face_edges(self, grid: int = 0, face_edges = None):
        raise NotImplementedError

    def get_grid_face_nodes(self, grid: int = 0, face_nodes = None):
        raise NotImplementedError

    def get_grid_nodes_per_face(self, grid: int = 0, nodes_per_face = None):
        raise NotImplementedError