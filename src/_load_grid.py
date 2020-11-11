import logging

from xarray import open_dataset

def _load_grid(self):
    # load grid
        logging.info('Loading grid file.')
        try:
            self.grid = open_dataset(self.config.get('grid.path'), chunks={
                self.config.get('grid.longitude'): self.chunk_size_longitude,
                self.config.get('grid.latitude'): self.chunk_size_latitude
            })
        except Exception as e:
            logging.exception('Failed to load grid file; see below for stacktrace.')
            raise e