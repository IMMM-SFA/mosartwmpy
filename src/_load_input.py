import dask.dataframe as dd
import logging

from xarray import open_dataset

def _load_input(self):
    # load input files into lazy dataframe
    logging.info('Loading input files.')
    try:
        
        # runoff forcing
        if self.config.get('runoff.enabled', False):
            self.runoff = open_dataset(self.config.get('runoff.path'))
        else:
            self.runoff = None
        
        # water management files
        if self.config.get('water_management.enabled', False):
            # TODO
            pass
        else:
            self.demand = None
            self.reservoirs = None
        
    except Exception as e:
        logging.exception('Failed to load input files; see below for stacktrace.')
        raise e