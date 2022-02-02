import numpy as np
import pandas as pd
import xarray as xr

from numba.core import types
from numba.typed import Dict
from xarray import concat, open_dataset, DataArray
from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters


def load_reservoirs(self, config: Benedict, parameters: Parameters) -> None:
    """Loads the reservoir information from file onto the grid.

    Args:
        config (Benedict): the model configuration
        parameters (Parameters): the model parameters
    """

    # reservoir parameter file
    reservoirs_file = open_dataset(config.get('water_management.reservoirs.parameters.path'))
    reservoirs = pd.DataFrame(index=self.id).merge(
        reservoirs_file.to_dataframe(),
        how='left',
        left_index=True,
        right_on=config.get('water_management.reservoirs.parameters.grid_cell_index'),
    )
    reservoirs_file.close()

    # load reservoir variables
    for key, value in config.get('water_management.reservoirs.parameters.variables').items():
        setattr(self, key, reservoirs[value].values)

    # correct the fields with different units
    # surface area from km^2 to m^2
    self.reservoir_surface_area = self.reservoir_surface_area * 1.0e6
    # capacity from millions m^3 to m^3
    self.reservoir_storage_capacity = self.reservoir_storage_capacity * 1.0e6

    # reservoir dependency database file
    self.reservoir_dependency_database = pd.read_parquet(
        config.get('water_management.reservoirs.dependencies.path')
    ).rename(columns={
        config.get('water_management.reservoirs.dependencies.variables.dependent_reservoir_id'): 'reservoir_id',
        config.get('water_management.reservoirs.dependencies.variables.dependent_cell_index'): 'grid_cell_id'
    })
    # drop nan grid ids
    self.reservoir_dependency_database = self.reservoir_dependency_database[self.reservoir_dependency_database.grid_cell_id.notna()]
    # set to integer
    self.reservoir_dependency_database = self.reservoir_dependency_database.astype(np.int64)

    # create a numba typed dict with key = <grid cell id> and value = <list of reservoir_ids that feed the cell>
    self.grid_index_to_reservoirs_map = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:],
    )
    for grid_cell_id, group in self.reservoir_dependency_database.groupby('grid_cell_id'):
        self.grid_index_to_reservoirs_map[grid_cell_id] = group.reservoir_id.values

    # index by grid cell
    self.reservoir_dependency_database = self.reservoir_dependency_database.set_index('grid_cell_id').sort_index()

    # prepare the month based reservoir schedules mapped to the domain
    prepare_reservoir_schedule(self, config)

    # calculate reservoir Biemans & Hanasaki Runoff/Capacity
    self.reservoir_runoff_capacity = self.reservoir_streamflow_schedule.mean(
        dim='month') * 365 * 24 * 60 * 60 / self.reservoir_storage_capacity

    # calculate the long term mean flow across the reservoirs
    self.reservoir_computed_meanflow_cumecs = self.reservoir_streamflow_schedule.weighted(xr.DataArray(
        data=np.array([31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0]),
        dims=['month'],
        coords=dict(month=np.arange(1, 13, 1))
    )).mean(dim='month').values


def prepare_reservoir_schedule(self, config: Benedict) -> None:
    """Establishes the reservoir schedule and flow.

    Args:
        config (Benedict): the model configuration
    """

    # streamflow file
    streamflow = pd.read_parquet(config.get('water_management.reservoirs.streamflow.path')).astype(np.float64)
    # demand file
    demand = pd.read_parquet(config.get('water_management.reservoirs.demand.path')).astype(np.float64)

    flow_schedule = []
    demand_schedule = []

    # for each month, map mean flow and demand for reservoir onto the grid
    for m in np.arange(12):
        flow_schedule.append(pd.DataFrame(self.reservoir_id, index=self.id, columns=['reservoir_id']).merge(
            streamflow[
                streamflow[config.get('water_management.reservoirs.streamflow.variables.streamflow_month_index')] == m
            ][[
                config.get('water_management.reservoirs.streamflow.variables.streamflow_reservoir_id'),
                config.get('water_management.reservoirs.streamflow.variables.streamflow')
            ]],
            how='left',
            left_on='reservoir_id',
            right_on=config.get('water_management.reservoirs.streamflow.variables.streamflow_reservoir_id')
        )[config.get('water_management.reservoirs.streamflow.variables.streamflow')].values)
        demand_schedule.append(pd.DataFrame(self.reservoir_id, index=self.id, columns=['reservoir_id']).merge(
            demand[
                demand[config.get('water_management.reservoirs.demand.variables.demand_month_index')] == m
            ][[
                config.get('water_management.reservoirs.demand.variables.demand_reservoir_id'),
                config.get('water_management.reservoirs.demand.variables.demand')
            ]],
            how='left',
            left_on='reservoir_id',
            right_on=config.get('water_management.reservoirs.demand.variables.demand_reservoir_id')
        )[config.get('water_management.reservoirs.demand.variables.demand')].values)
    self.reservoir_streamflow_schedule = DataArray(
        data=np.array(flow_schedule),
        dims=['month', 'index'],
        coords=dict(
            index=self.id,
            month=(np.arange(12) + 1),  # convert month to 1 based indexing
        )
    )
    self.reservoir_demand_schedule = DataArray(
        data=np.array(demand_schedule),
        dims=['month', 'index'],
        coords=dict(
            index=self.id,
            month=(np.arange(12) + 1),  # convert month to 1 based indexing
        )
    )

    # initialize prerelease based on long term mean flow and demand (Biemans 2011)
    # TODO weighted average by days in month?
    flow_avg = self.reservoir_streamflow_schedule.mean(dim='month')
    demand_avg = self.reservoir_demand_schedule.mean(dim='month')
    prerelease = (1.0 * self.reservoir_streamflow_schedule)
    prerelease[:, :] = flow_avg
    # note that xarray `where` modifies the false values
    condition = (demand_avg >= (0.5 * flow_avg)) & (flow_avg > 0)
    prerelease = prerelease.where(
        ~condition,
        demand_avg / 10 + 9 / 10 * flow_avg * self.reservoir_demand_schedule / demand_avg
    )
    prerelease = prerelease.where(
        condition,
        prerelease.where(
            ~((flow_avg + self.reservoir_demand_schedule - demand_avg) > 0),
            flow_avg + self.reservoir_demand_schedule - demand_avg
        )
    )
    self.reservoir_prerelease_schedule = prerelease
