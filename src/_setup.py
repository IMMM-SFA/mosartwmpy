import logging
import numpy as np
from benedict import benedict
from pathlib import Path
from pathvalidate import sanitize_filename

def _setup(self, config_file_path):
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
    
    # some constants used throughout the code
    # TODO better document what these are used for and what they should be and maybe they should be part of config?
    # TINYVALUE
    self.parameters['tiny_value'] = 1.0e-14
    # a small value in order to avoid abrupt change of hydraulic radius
    self.parameters['slope_1_def'] = 0.1
    self.parameters['1_over_sin_atan_slope_1_def'] = 1.0 / (np.sin(np.arctan(self.parameters['slope_1_def'])))
    # flood threshold - excess water will be sent back to ocean
    self.parameters['flood_threshold'] = 1.0e36 # [m3]?
    # liquid/ice effective velocity
    self.parameters['effective_tracer_velocity'] = 10.0 # [m/s]?
    # minimum river depth
    self.parameters['river_depth_minimum'] = 1.0e-4 # [m]?
    # coefficient to adjust the width of the subnetwork channel
    self.parameters['subnetwork_width_parameter'] = 1.0