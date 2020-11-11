import logging

from xarray import open_dataset

def _load_input(self):
    logging.info('Loading input files.')
    try:
        # water management files
        if self.config.get('water_management.enabled', False):
            self.demand = open_dataset(self.config.get('water_management.demand.path'), chunks={
                self.config.get('water_management.demand.longitude'): self.chunk_size_longitude,
                self.config.get('water_management.demand.latitude'): self.chunk_size_latitude,
                self.config.get('water_management.demand.time'): self.chunk_size_time
            })
            self.reservoirs = open_dataset(self.config.get('water_management.reservoirs.path'), chunks={
                self.config.get('water_management.reservoirs.longitude'): self.chunk_size_longitude,
                self.config.get('water_management.reservoirs.latitude'): self.chunk_size_latitude,
            })
            self.runoff = open_dataset(self.config.get('water_management.runoff.path'), chunks={
                self.config.get('water_management.runoff.longitude'): self.chunk_size_longitude,
                self.config.get('water_management.runoff.latitude'): self.chunk_size_latitude,
                self.config.get('water_management.runoff.time'): self.chunk_size_time
            })
        else:
            self.demand = None
            self.reservoirs = None
            self.runoff = None
        
    except Exception as e:
        logging.exception('Failed to load input files; see below for stacktrace.')
        raise e