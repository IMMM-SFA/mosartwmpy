import logging
import numpy as np
import pandas as pd

from datetime import timezone, timedelta
from pathlib import Path
from xarray import concat, open_dataset
import rioxarray

from mosartwmpy.utilities.timing import timing


def initialize_output(self):
    """Initializes the output buffer."""
    
    logging.debug('Initalizing output buffer.')
    if self.config.get('simulation.output_resolution') % self.config.get('simulation.timestep') != 0 or self.config.get('simulation.output_resolution') < self.config.get('simulation.timestep'):
        raise Exception('The `simulation.output_resolution` must be greater than or equal to and evenly divisible by the `simulation.timestep`.')
    for output in self.config.get('simulation.output'):
        if getattr(self.state, output.get('variable'), None) is not None and len(getattr(self.state, output.get('variable'))) > 0:
            if self.output_buffer is None:
                self.output_buffer = pd.DataFrame(self.state.zeros, columns=[output.get('name')])
            else:
                self.output_buffer = self.output_buffer.join(pd.DataFrame(self.state.zeros, columns=[output.get('name')]))


# @timing
def update_output(self):
    """Updates the output buffer based on the current state, and averages and writes to file when appropriate."""

    # update buffer
    self.output_n += 1
    for output in self.config.get('simulation.output'):
        if getattr(self.state, output.get('variable'), None) is not None and len(getattr(self.state, output.get('variable'))) > 0:
            self.output_buffer.loc[:, output.get('name')] += getattr(self.state, output.get('variable'))

    # if a new period has begun: average output buffer, write to file, and zero output buffer
    if self.current_time.replace(tzinfo=timezone.utc).timestamp() % self.config.get('simulation.output_resolution') == 0:
        self.output_buffer = self.output_buffer / self.output_n
        write_output(self)
        self.output_n = 0
        for output in self.config.get('simulation.output'):
            if getattr(self.state, output.get('variable'), None) is not None and len(getattr(self.state, output.get('variable'))) > 0:
                self.output_buffer.loc[:, output.get('name')] = 0.0 * self.state.zeros
    
    # check if restart file if need
    check_restart(self)


def write_output(self):
    """Writes the output buffer and requested grid variables to a netcdf file."""
    # TODO only daily resolution is currently supported - need to support arbitrary resolutions

    # check the write frequency to see if writing to new file or appending to existing file
    # also construct the file name
    period = self.config.get('simulation.output_file_frequency')
    is_new_period = False
    # use yesterday's date as the file name, to match with what is actually being averaged
    true_date = self.current_time if not (self.current_time.hour == 0 and self.current_time.minute == 0 and self.current_time.second == 0) else (self.current_time - timedelta(days=1))
    filename = f'{self.config.get("simulation.output_path")}/{self.name}/{self.name}_{true_date.year}'
    if period == 'daily':
        filename += f'_{true_date.strftime("%m")}_{true_date.strftime("%d")}'
        if self.current_time.hour == 0 and self.current_time.second == 0:
            is_new_period = True
    elif period == 'monthly':
        filename += f'_{true_date.strftime("%m")}'
        if self.current_time.day == 2 and self.current_time.hour == 0 and self.current_time.second == 0:
            is_new_period = True
    elif period == 'yearly':
        if self.current_time.month == 1 and self.current_time.day == 2 and self.current_time.hour == 0 and self.current_time.second == 0:
            is_new_period = True
    else:
        logging.warning(f'Configuration value for `simulation.output_file_frequency: {period}` is not recognized.')
        return
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
        if getattr(self.state, output.get('variable'), None) is not None and len(getattr(self.state, output.get('variable'))) > 0:
            if output.get('long_name'):
                frame[output.get('name')].attrs['long_name'] = output.get('long_name')
            if output.get('units'):
                frame[output.get('name')].attrs['units'] = output.get('units')

    # if file exists and it's not a new period, update existing file else write to new file and include grid variables
    logging.debug(f'Writing to output file: {Path(filename)}.')
    if not is_new_period and Path(filename).is_file():
        nc = open_dataset(Path(filename)).load()
        # slice the existing data to account for restarts
        # TODO this assumes daily averaged output
        nc = nc.sel(time=slice(None, pd.to_datetime(self.current_time) - pd.Timedelta('1d 1s')))
        frame = concat([nc, frame], dim='time', data_vars='minimal')
        nc.close()
    else:
        if len(self.config.get('simulation.grid_output', [])) > 0:
            grid_frame = pd.DataFrame(self.grid.latitude, columns=['latitude']).join(pd.DataFrame(self.grid.longitude, columns=['longitude']))
            for grid_output in self.config.get('simulation.grid_output'):
                if getattr(self.grid, grid_output.get('variable'), None) is not None:
                    grid_frame = grid_frame.join(pd.DataFrame(getattr(self.grid, grid_output.get('variable')), columns=[grid_output.get('variable')]))
            grid_frame = grid_frame.rename(columns={
                'latitude': 'lat',
                'longitude': 'lon'
            }).set_index(['lat', 'lon']).to_xarray()
            grid_frame = grid_frame.assign_coords(
                lat=grid_frame.lat.astype(np.float32),
                lon=grid_frame.lon.astype(np.float32)
            )
            for grid_output in self.config.get('simulation.grid_output'):
                if getattr(self.grid, grid_output.get('variable'), None) is not None:
                    frame = frame.assign({
                        grid_output.get('name'): grid_frame[grid_output.get('variable')]
                    })
                    if grid_output.get('long_name'):
                        frame[grid_output.get('name')].attrs['long_name'] = grid_output.get('long_name')
                    if grid_output.get('units'):
                        frame[grid_output.get('name')].attrs['units'] = grid_output.get('units')
    frame = frame.rio.write_crs(4326)
    frame.to_netcdf(filename, unlimited_dims=['time'])


def check_restart(self):
    """Checks if a restart file is needed based on the current simulation time."""
    
    frequency = self.config.get('simulation.restart_file_frequency')
    is_needed = False
    if frequency == 'daily':
        if self.current_time.hour == 0 and self.current_time.second == 0:
            is_needed = True
    elif frequency == 'monthly':
        if self.current_time.day == 1 and self.current_time.hour == 0 and self.current_time.second == 0:
            is_needed = True
    elif frequency == 'yearly':
        if self.current_time.month == 1 and self.current_time.day == 1 and self.current_time.hour == 0 and self.current_time.second == 0:
            is_needed = True
    if self.current_time.timestamp() >= self.get_end_time():
        # always write a restart file at the end of simulation
        is_needed = True
    if is_needed:
        write_restart(self)


def write_restart(self):
    """Writes the state to a netcdf file, with the current simulation time in the file name."""

    x = self.state.to_dataframe().to_xarray()
    filename = Path(f'{self.config.get("simulation.output_path")}/{self.name}/restart_files/{self.name}_restart_{self.current_time.year}_{self.current_time.strftime("%m")}_{self.current_time.strftime("%d")}.nc')
    x = x.rio.write_crs(4326)
    logging.debug(f'Writing restart file: {filename}.')
    x.to_netcdf(filename)
