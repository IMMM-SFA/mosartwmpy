import numpy as np
import pandas as pd

from numba.core import types
from numba.typed import Dict
from xarray import concat, open_dataset, Dataset
from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters


def load_reservoirs(self, config: Benedict, parameters: Parameters) -> None:
    """Loads the reservoir information from file onto the grid.

    Args:
        config (Benedict): the model configuration
        parameters (Parameters): the model parameters
    """

    # reservoir parameter file
    reservoirs = open_dataset(config.get('water_management.reservoirs.path'))

    # load reservoir variables
    for key, value in config.get('water_management.reservoirs.variables').items():
        setattr(self, key, np.array(reservoirs[value]).flatten())

    # correct the fields with different units
    # surface area from km^2 to m^2
    self.reservoir_surface_area = self.reservoir_surface_area * 1.0e6
    # capacity from millions m^3 to m^3
    self.reservoir_storage_capacity = self.reservoir_storage_capacity * 1.0e6

    # map dams to all their dependent grid cells
    # this will be a table of many to many relationship of grid cell ids to reservoir ids
    self.reservoir_to_grid_mapping = reservoirs[
        config.get('water_management.reservoirs.grid_to_reservoir')
    ].to_dataframe().reset_index()[[
        config.get('water_management.reservoirs.grid_to_reservoir_reservoir_dimension'),
        config.get('water_management.reservoirs.grid_to_reservoir')
    ]].rename(columns={
        config.get('water_management.reservoirs.grid_to_reservoir_reservoir_dimension'): 'reservoir_id',
        config.get('water_management.reservoirs.grid_to_reservoir'): 'grid_cell_id'
    })
    # drop nan grid ids
    self.reservoir_to_grid_mapping = self.reservoir_to_grid_mapping[self.reservoir_to_grid_mapping.grid_cell_id.notna()]
    # correct to zero-based grid indexing for grid cell
    self.reservoir_to_grid_mapping.loc[:, self.reservoir_to_grid_mapping.grid_cell_id.name] = self.reservoir_to_grid_mapping.grid_cell_id.values - 1
    # set to integer
    self.reservoir_to_grid_mapping = self.reservoir_to_grid_mapping.astype(int)

    # create a numba typed dict with key = <grid cell id> and value = <list of reservoir_ids that feed the cell>
    self.reservoir_to_grid_map = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:],
    )
    for grid_cell_id, group in self.reservoir_to_grid_mapping.groupby('grid_cell_id'):
        self.reservoir_to_grid_map[grid_cell_id] = group.reservoir_id.values

    # index by grid cell
    self.reservoir_to_grid_mapping = self.reservoir_to_grid_mapping.set_index('grid_cell_id').sort_index()

    # prepare the month based reservoir schedules mapped to the domain
    prepare_reservoir_schedule(self, config, parameters, reservoirs)

    reservoirs.close()


def prepare_reservoir_schedule(self, config: Benedict, parameters: Parameters, reservoirs: Dataset) -> None:
    """Establishes the reservoir schedule and flow.

    Args:
        config (Benedict): the model configuration
        parameters (Parameters): the model parameters
        reservoirs (Dataset): the reservoir dataset loaded from file
    """

    # the reservoir streamflow and demand are specified by the time resolution and reservoir id
    # so let's remap those to the actual mosart domain for ease of use

    # streamflow flux
    streamflow_time_name = config.get('water_management.reservoirs.streamflow_time_resolution')
    streamflow = reservoirs[config.get('water_management.reservoirs.streamflow')]
    schedule = None
    for t in np.arange(streamflow.shape[0]):
        flow = streamflow[t, :].to_pandas().to_frame('streamflow')
        sched = pd.DataFrame(self.reservoir_id, columns=['reservoir_id']).merge(flow, how='left', left_on='reservoir_id', right_index=True)[['streamflow']].to_xarray().expand_dims(
            {streamflow_time_name: 1},
            axis=0
        )
        if schedule is None:
            schedule = sched
        else:
            schedule = concat([schedule, sched], dim=streamflow_time_name)
    self.reservoir_streamflow_schedule = schedule.assign_coords(
        # convert to 1 based month index (instead of starting from 0)
        {streamflow_time_name: (streamflow_time_name, schedule[streamflow_time_name].values + (1 if streamflow_time_name == 'month' else 0))}
    ).streamflow

    # demand volume
    demand_time_name = config.get('water_management.reservoirs.demand_time_resolution')
    demand = reservoirs[config.get('water_management.reservoirs.demand')]
    schedule = None
    for t in np.arange(demand.shape[0]):
        dem = demand[t, :].to_pandas().to_frame('demand')
        sched = pd.DataFrame(self.reservoir_id, columns=['reservoir_id']).merge(dem, how='left', left_on='reservoir_id', right_index=True)[['demand']].to_xarray().expand_dims(
            {demand_time_name: 1}, axis=0
        )
        if schedule is None:
            schedule = sched
        else:
            schedule = concat([schedule, sched], dim=demand_time_name)
    self.reservoir_demand_schedule = schedule.assign_coords(
        # convert to 1 based month index (instead of starting from 0)
        {demand_time_name: (demand_time_name, schedule[demand_time_name].values + (1 if demand_time_name == 'month' else 0))}
    ).demand

    # initialize prerelease based on long term mean flow and demand (Biemans 2011)
    flow_avg = self.reservoir_streamflow_schedule.mean(dim=streamflow_time_name)
    demand_avg = self.reservoir_demand_schedule.mean(dim=demand_time_name)
    prerelease = (1.0 * self.reservoir_streamflow_schedule)
    prerelease[:,:] = flow_avg
    # note that xarray `where` modifies the false values
    condition = (demand_avg >= (0.5 * flow_avg)) & (flow_avg > 0)
    prerelease = prerelease.where(
        ~condition,
        demand_avg/ 10 + 9 / 10 * flow_avg * self.reservoir_demand_schedule / demand_avg
    )
    prerelease = prerelease.where(
        condition,
        prerelease.where(
            ~((flow_avg + self.reservoir_demand_schedule - demand_avg) > 0),
            flow_avg + self.reservoir_demand_schedule - demand_avg
        )
    )
    self.reservoir_prerelease_schedule = prerelease
