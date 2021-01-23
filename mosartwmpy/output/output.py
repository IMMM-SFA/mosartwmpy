import logging
import numpy as np
import pandas as pd

from datetime import datetime, timezone, timedelta
from xarray import concat, open_dataset

def initialize_output(self):
    # setup output buffer and averaging
    logging.info('Initalizing output buffer.')
    if self.config.get('simulation.output_resolution') % self.config.get('simulation.timestep') != 0 or self.config.get('simulation.output_resolution') < self.config.get('simulation.timestep'):
        raise Exception('The `simulation.output_resolution` must be greater than or equal to and evenly divisible by the `simulation.timestep`.')
    for output in self.config.get('simulation.output'):
        if getattr(self.state, output.get('variable'), None) is not None:
            if self.output_buffer is None:
                self.output_buffer = pd.DataFrame(self.state.zeros, columns=[output.get('name')])
            else:
                self.output_buffer = self.output_buffer.join(pd.DataFrame(self.state.zeros, columns=[output.get('name')]))

def update_output(self):
    # handle updating output buffer and writing to file when appropriate
    
    # update buffer
    self.output_n += 1
    for output in self.config.get('simulation.output'):
        if getattr(self.state, output.get('variable'), None) is not None:
            self.output_buffer.loc[:, output.get('name')] += getattr(self.state, output.get('variable'))
        
    # if a new period has begun: average output buffer, write to file, and zero output buffer
    if self.current_time.replace(tzinfo=timezone.utc).timestamp() % self.config.get('simulation.output_resolution') == 75600: # 0: TODO
        logging.info('Writing to output file.')
        self.output_buffer = self.output_buffer / self.output_n
        write_output(self)
        self.output_n = 0
        for output in self.config.get('simulation.output'):
            if getattr(self.state, output.get('variable'), None) is not None:
                self.output_buffer.loc[:, output.get('name')] = 0.0 * self.state.zeros
        # check if restart file if need
        check_restart(self)

def write_output(self):
    # handle writing output to file
    # TODO only daily resolution is currently supported - need to support arbitrary resolutions
    
    # logging.info(f'WRM_DEMAND0 sum: {self.output_buffer.WRM_DEMAND0.sum()}')
    # logging.info(f'WRM_DEMAND sum: {self.output_buffer.WRM_DEMAND.sum()}')
    # logging.info(f'WRM_SUPPLY sum: {self.output_buffer.WRM_SUPPLY.sum()}')
    # logging.info(f'QSUR sum: {self.output_buffer.QSUR_LIQ.sum()}')
    
    # check the write frequency to see if writing to new file or appending to existing file
    # also construct the file name
    period = self.config.get('simulation.output_file_frequency')
    is_new_period = False
    true_date = self.current_time #- timedelta(days=1) TODO
    filename = f'./output/{self.name}/{self.name}_{true_date.year}'
    if period == 'daily':
        filename += f'_{true_date.strftime("%m")}_{true_date.strftime("%d")}'
        if self.current_time.hour == 0 and self.current_time.second == 0:
            is_new_period = True
    if period == 'monthly':
        filename += f'_{true_date.strftime("%m")}'
        if self.current_time.day == 1 and self.current_time.hour == 21: #2 and self.current_time.hour == 0 and self.current_time.second == 0: TODO
            is_new_period = True
    if period == 'yearly':
        if self.current_time.month == 1 and self.current_time.day == 2 and self.current_time.hour == 0 and self.current_time.second == 0:
            is_new_period = True
    filename += '.nc'

    # create the data frame
    frame = pd.DataFrame(self.grid.latitude, columns=['latitude']).join(pd.DataFrame(self.grid.longitude, columns=['longitude'])).join(
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

    # restrict lat/lon to 32 bit precision
    frame = frame.assign_coords(
        lat=frame.lat.astype(np.float32),
        lon=frame.lon.astype(np.float32)
    )

    # assign metadata
    frame.lat.attrs['units'] = 'degrees_north'
    frame.lon.attrs['units'] = 'degrees_east'
    for output in self.config.get('simulation.output'):
        if getattr(self.state, output.get('variable'), None) is not None:
            if output.get('long_name'):
                frame[output.get('name')].attrs['long_name'] = output.get('long_name')
            if output.get('units'):
                frame[output.get('name')].attrs['units'] = output.get('units')

    # if new period, write to new file and include grid variables, otherwise update file
    if not is_new_period:
        nc = open_dataset(filename).load()
        frame = concat([nc, frame], dim='time', data_vars='minimal')
        nc.close()
    else:
        if len(self.config.get('simulation.grid_output', [])) > 0:
            grid_frame = pd.DataFrame(self.grid.latitude, columns=['latitude']).join(pd.DataFrame(self.grid.longitude, columns=['longitude']))
            for grid_output in self.config.get('simulation.grid_output'):
                grid_frame = grid_frame.join(pd.DataFrame(getattr(self.grid, grid_output.get('variable')), columns=[grid_output.get('variable')]))
            grid_frame = grid_frame.rename(columns={
                'latitude': 'lat',
                'longitude': 'lon'
            }).set_index(['lat', 'lon']).to_xarray()
            grid_frame = grid_frame.assign_coords(
                lat=grid_frame.lat.astype(np.float32),
                lon=grid_frame.lon.astype(np.float32)
            )
            for grid_output in self.config.get('simulation.grid_output', []):
                frame = frame.assign({
                    grid_output.get('name'): grid_frame[grid_output.get('variable')]
                })
                if grid_output.get('long_name'):
                    frame[grid_output.get('name')].attrs['long_name'] = grid_output.get('long_name')
                if grid_output.get('units'):
                    frame[grid_output.get('name')].attrs['units'] = grid_output.get('units')
    frame.to_netcdf(filename, unlimited_dims=['time'])

def check_restart(self):
    # check if new restart file is desired
    frequency = self.config.get('simulation.restart_file_frequency')
    is_needed = False
    true_date = self.current_time - timedelta(days=1)
    if frequency == 'daily':
        if self.current_time.hour == 0 and self.current_time.second == 0:
            is_needed = True
    if frequency == 'monthly':
        if self.current_time.day == 2 and self.current_time.hour == 0 and self.current_time.second == 0:
            is_needed = True
    if frequency == 'yearly':
        if self.current_time.month == 1 and self.current_time.day == 2 and self.current_time.hour == 0 and self.current_time.second == 0:
            is_needed = True
    if is_needed:
        write_restart(self)

def write_restart(self):
    # TODO
    # need to save state and possibly some wm stuff? ideally just state
    pass