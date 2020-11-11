import logging
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