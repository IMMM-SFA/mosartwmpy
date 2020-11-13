import dask.dataframe as dd
import logging

from xarray import open_dataset

def _load_input(self):
    # load input files into dataframe
    # TODO convert to dataframe
    # TODO WM, heat, etc
    logging.info('Loading input files.')
    try:
        # water management files
        if self.config.get('water_management.enabled', False):
            pass
            # self.demand = open_dataset(self.config.get('water_management.demand.path'), chunks={})
            # self.reservoirs = open_dataset(self.config.get('water_management.reservoirs.path'), chunks={})
            # self.runoff = open_dataset(self.config.get('water_management.runoff.path'), chunks={})
        else:
            self.demand = None
            self.reservoirs = None
            self.runoff = None
        
    except Exception as e:
        logging.exception('Failed to load input files; see below for stacktrace.')
        raise e