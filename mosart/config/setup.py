import logging
import numpy as np
import psutil
from benedict import benedict
from pathlib import Path
from pathvalidate import sanitize_filename

def setup(self, config_file_path):
    # load default config and user config and merge
    self.config = benedict('./config_defaults.yaml', format='yaml')
    if config_file_path and config_file_path != '':
        self.config.merge(benedict(config_file_path, format='yaml'), overwrite=True)
    
    self.name = sanitize_filename(self.config.get('simulation.name')).replace(" ", "_")
    
    # setup logging and output directory
    Path(f'./output/{self.name}').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=f'./output/{self.name}/mosart.log',
        level=self.config.get('simulation.log_level', 'INFO'),
        format='%(asctime)s - Mosart: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info('Initalizing model.')
    logging.debug(self.config.dump())
    
    # multiprocessing
    if self.config.get('multiprocessing.enabled', False) or self.config.get('batch.enabled', False):
        max_cores = psutil.cpu_count(logical=False)
        requested = self.config.get('multiprocessing.cores', None)
        if requested is None or requested > max_cores:
            requested = max_cores
        self.cores = requested
        logging.info(f'Cores: {self.cores}.')
    
    # parameters
    self.parameters = Parameters()

class Parameters:
    def __init__(self):
        # some constants used throughout the code
        # TODO better document what these are used for and what they should be and maybe they should be part of config?
        # TINYVALUE
        self.tiny_value = 1.0e-14
        # radius of the earth [m]
        self.radius_earth = 6.37122e6
        # a small value in order to avoid abrupt change of hydraulic radius
        self.slope_1_def = 0.1
        self.inverse_sin_atan_slope_1_def = 1.0 / (np.sin(np.arctan(self.slope_1_def)))
        # flood threshold - excess water will be sent back to ocean
        self.flood_threshold = 1.0e36 # [m3]?
        # liquid/ice effective velocity # TODO is this used anywhere
        self.effective_tracer_velocity = 10.0 # [m/s]?
        # minimum river depth
        self.river_depth_minimum = 1.0e-4 # [m]?
        # coefficient to adjust the width of the subnetwork channel
        self.subnetwork_width_parameter = 1.0
        # minimum hillslope (replaces 0s from grid file)
        self.hillslope_minimum = 0.005
        # minimum subnetwork slope (replaces 0s from grid file)
        self.subnetwork_slope_minimum = 0.0001
        # minimum main channel slope (replaces 0s from grid file)
        self.channel_slope_minimum = 0.0001
        # kinematic wave condition # TODO what is it?
        self.kinematic_wave_condition =  1.0e6
        # just a string... probably can dispose of these if we never do ICE separately
        self.LIQUID_TRACER = 'LIQUID'
        self.ICE_TRACER = 'ICE'