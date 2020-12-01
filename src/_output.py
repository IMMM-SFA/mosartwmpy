import logging
import numpy as np
import pandas as pd

from datetime import datetime, timezone, timedelta
from xarray import concat, open_dataset

def _initialize_output(self):
    # setup output buffer and averaging
    logging.info('Initalizing output buffer.')
    if self.config.get('simulation.output_resolution') % self.config.get('simulation.timestep') != 0 or self.config.get('simulation.output_resolution')  <  self.config.get('simulation.timestep'):
        raise Exception('The `simulation.output_resolution` must be greater than or equal to and evenly divisible by the `simulation.timestep`.')
    for output in self.config.get('simulation.output'):
        if self.output_buffer is None:
            self.output_buffer = pd.DataFrame(self.state.zeros.rename(output.get('name')))
        else:
            self.output_buffer = self.output_buffer.join(self.state.zeros.rename(output.get('name')))

def _update_output(self):
    # handle updating output buffer and writing to file when appropriate
    
    # update buffer
    for output in self.config.get('simulation.output'):
        self.output_buffer[output.get('name')] += self.state[output.get('variable')]
        
    # if a new period has begun: average output buffer, write to file, and zero output buffer
    if self.current_time.replace(tzinfo=timezone.utc).timestamp() % self.config.get('simulation.output_resolution') == 0:
        logging.info('Writing to output file.')
        self.output_buffer = self.output_buffer / (self.config.get('simulation.output_resolution') / self.config.get('simulation.timestep'))
        _write_output(self)
        for output in self.config.get('simulation.output'):
            self.output_buffer[output.get('name')] = self.state.zeros

def _write_output(self):
    # handle writing output to file
    # TODO only daily resolution is currently supported - need to support arbitrary resolutions
    
    # check the write frequency to see if writing to new file or appending to existing file
    # also construct the file name
    period = self.config.get('simulation.output_file_frequency')
    is_new_period = False
    true_date = self.current_time - timedelta(days=1)
    filename = f'./output/{self.name}/{self.name}_{true_date.year}'
    if period == 'daily':
        filename += f'_{true_date.strftime("%m")}_{true_date.strftime("%d")}'
        if self.current_time.hour == 0 and self.current_time.second == 0:
            is_new_period = True
    if period == 'monthly':
        filename += f'_{true_date.strftime("%m")}'
        if self.current_time.day == 2 and self.current_time.hour == 0 and self.current_time.second == 0:
            is_new_period = True
    if period == 'yearly':
        if self.current_time.month == 1 and self.current_time.day == 2 and self.current_time.hour == 0 and self.current_time.second == 0:
            is_new_period = True
    filename += '.nc'
    
    # create the data frame
    frame = self.grid[['latitude', 'longitude']].join(
        pd.DataFrame(np.full(self.get_grid_size(), pd.to_datetime(true_date)), columns=['time'])
    ).join(
        self.output_buffer
    ).rename(columns={
        'latitude': 'lat',
        'longitude': 'lon'
    }).set_index(
        ['time', 'lat', 'lon']
    ).to_xarray().astype(
        np.float32
    )
    
    # add other useful grid fields - id, outlet_id, downstream_id
    # TODO
    
    # restrict lat/lon to 32 bit precision
    frame = frame.assign_coords(
        lat=frame.lat.astype(np.float32),
        lon=frame.lon.astype(np.float32)
    )
    
    # assign metadata
    frame.lat.attrs['units'] = 'degrees_north'
    frame.lon.attrs['units'] = 'degrees_east'
    for output in self.config.get('simulation.output'):
        if output.get('long_name'):
            frame[output.get('name')].attrs['long_name'] = output.get('long_name')
        if output.get('units'):
            frame[output.get('name')].attrs['units'] = output.get('units')
    
    # if new period, write to new file otherwise update file
    if not is_new_period:
        nc = open_dataset(filename)
        frame = concat([nc, frame], dim='time')
        nc.close()
    frame.to_netcdf(filename, unlimited_dims=['time'])