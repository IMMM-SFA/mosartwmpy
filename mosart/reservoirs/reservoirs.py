from epiweeks import Week
import numpy as np
import pandas as pd

from xarray import concat, open_dataset
from xarray.ufuncs import logical_not

def load_reservoirs(self):
    
    # reservoir parameter file
    reservoirs = open_dataset(self.config.get('water_management.reservoirs.path'))
    
    # load reservoir variables
    for frame in [
        pd.DataFrame(reservoirs[value].values.flatten(), columns=[key]) for key, value in self.config.get('water_management.reservoirs.variables').items()
    ]:
        self.grid = self.grid.join(frame)
    
    # correct the fields with different units
    # surface area from km^2 to m^2
    self.grid.reservoir_surface_area[:] = self.grid.reservoir_surface_area.values * 1.0e6
    # capacity from millions m^3 to m^3
    self.grid.reservoir_storage_capacity[:] = self.grid.reservoir_storage_capacity * 1.0e6
    
    # prepare the epiweek based reservoir schedules mapped to the domain
    prepare_reservoir_schedule(self, reservoirs)
    
    reservoirs.close()


def prepare_reservoir_schedule(self, reservoirs):
    # the reservoir streamflow and demand are specified by the time resolution and reservoir id
    # so let's remap those to the actual mosart domain for ease of use
    
    # TODO i had wanted to convert these all to epiweeks no matter what format provided, but we don't know what year all the data came from
    
    # streamflow flux
    streamflow_time_name = self.config.get('water_management.reservoirs.streamflow_time_resolution')
    streamflow = reservoirs[self.config.get('water_management.reservoirs.streamflow')]
    schedule = None
    for t in np.arange(streamflow.shape[0]):
        flow = streamflow[t, :].to_pandas().to_frame('streamflow')
        sched = self.grid[['reservoir_id']].merge(flow, how='left', left_on='reservoir_id', right_index=True)[['streamflow']].to_xarray().expand_dims(
            {streamflow_time_name: 1},
            axis=0
        )
        if schedule is None:
            schedule = sched
        else:
            schedule = concat([schedule, sched], dim=streamflow_time_name)
    self.reservoir_streamflow_schedule = schedule.assign_coords(
        # if monthly, convert to 1 based month index (instead of starting from 0)
        {streamflow_time_name: (streamflow_time_name, schedule[streamflow_time_name].values + (1 if streamflow_time_name == 'month' else 0))}
    ).streamflow
    
    # demand volume
    demand_time_name = self.config.get('water_management.reservoirs.demand_time_resolution')
    demand = reservoirs[self.config.get('water_management.reservoirs.demand')]
    schedule = None
    for t in np.arange(demand.shape[0]):
        dem = demand[t, :].to_pandas().to_frame('demand')
        sched = self.grid[['reservoir_id']].merge(dem, how='left', left_on='reservoir_id', right_index=True)[['demand']].to_xarray().expand_dims(
            {demand_time_name: 1}, axis=0
        )
        if schedule is None:
            schedule = sched
        else:
            schedule = concat([schedule, sched], dim=demand_time_name)
    self.reservoir_demand_schedule = schedule.assign_coords(
        # if monthly, convert to 1 based month index (instead of starting from 0)
        {demand_time_name: (demand_time_name, schedule[demand_time_name].values + (1 if demand_time_name == 'month' else 0))}
    ).demand
    
    # initialize prerelease based on long term mean flow and demand (Biemans 2011)
    # TODO this assumes demand and flow use the same timescale :(
    flow_avg = self.reservoir_streamflow_schedule.mean(dim=streamflow_time_name)
    demand_avg = self.reservoir_demand_schedule.mean(dim=demand_time_name)
    prerelease = (1.0 * self.reservoir_streamflow_schedule)
    prerelease[:,:] = flow_avg
    # note that xarray `where` modifies the false values
    condition = (demand_avg >= (0.5 * flow_avg)) & (flow_avg > 0)
    prerelease = prerelease.where(
        logical_not(condition),
        demand_avg/ 10 + 9 / 10 * flow_avg * self.reservoir_demand_schedule / demand_avg
    )
    prerelease = prerelease.where(
        condition,
        prerelease.where(
            logical_not((flow_avg + self.reservoir_demand_schedule - demand_avg) > 0),
            flow_avg + self.reservoir_demand_schedule - demand_avg
        )
    )
    self.reservoir_prerelease_schedule = prerelease


def initialize_reservoir_state(self):
    
    # reservoir storage at the start of the operation year
    self.state = self.state.join(pd.DataFrame(
        0.85 * self.grid.reservoir_storage_capacity.values,
        columns=['reservoir_storage_operation_year_start']
    ))
    
    # initial storage in each reservoir
    self.state = self.state.join(pd.DataFrame(
        0.9 * self.grid.reservoir_storage_capacity.values,
        columns=['reservoir_storage']
    ))
    
    initialize_reservoir_start_of_operation_year(self)
    
    reservoir_release(self)


def initialize_reservoir_start_of_operation_year(self):
    # define the start of the operation - define irrigation releases pattern
    # multiple hydrograph - 1 peak, 2 peaks, multiple small peaks
    # TODO this all depends on the schedules being monthly :(
    
    streamflow_time_name = self.config.get('water_management.reservoirs.streamflow_time_resolution')
    
    # find the peak flow and peak flow month for each reservoir
    peak = self.reservoir_streamflow_schedule.max(dim=streamflow_time_name).values
    month_start_operations = self.reservoir_streamflow_schedule.idxmax(dim=streamflow_time_name).values
    
    # correct the month start for reservoirs where average flow is greater than a small value and magnitude of peak flow difference from average is greater than smaller value
    # TODO a little hard to follow the logic here but it seem to be related to number of peaks/troughs
    flow_avg = self.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values
    condition = flow_avg > self.parameters.reservoir_minimum_flow_condition
    number_of_sign_changes = 0 * flow_avg
    count = 1 + number_of_sign_changes
    count_max = 1 + number_of_sign_changes
    month = 1 * month_start_operations
    sign = np.where(
        np.abs(peak - flow_avg) > self.parameters.reservoir_small_magnitude_difference,
        np.where(
            peak - flow_avg > 0,
            1,
            -1
        ),
        1
    )
    current_sign = 1 * sign
    for t in self.reservoir_streamflow_schedule[streamflow_time_name].values:
        # if not an xarray object with coords, the sel doesn't work, so that why the strange definition here
        i = self.reservoir_streamflow_schedule.idxmax(dim=streamflow_time_name)
        i = i.where(
            i + t > 12,
            i + t
        ).where(
            i + t <= 12,
            i + t - 12
        )
        flow = self.reservoir_streamflow_schedule.sel({
            streamflow_time_name: i.fillna(1)
        })
        flow = np.where(np.isfinite(i), flow, np.nan)
        current_sign = np.where(
            np.abs(flow - flow_avg) > self.parameters.reservoir_small_magnitude_difference,
            np.where(
                flow - flow_avg > 0,
                1,
                -1
            ),
            sign
        )
        number_of_sign_changes = np.where(
            current_sign != sign,
            number_of_sign_changes + 1,
            number_of_sign_changes
        )
        change_condition = (current_sign != sign) & (current_sign > 0) & (number_of_sign_changes > 0) & (count > count_max)
        count_max = np.where(
            change_condition,
            count,
            count_max
        )
        month_start_operations = np.where(
            condition & change_condition,
            month,
            month_start_operations
        )
        month = np.where(
            current_sign != sign,
            i,
            month
        )
        count = np.where(
            current_sign != sign,
            1,
            count + 1
        )
        sign = 1 * current_sign
    
    # setup flood control for reservoirs with average flow greater than a larger value
    # TODO this is also hard to follow, but seems related to months in a row with high or low flow
    month_flood_control_start = 0 * month_start_operations
    month_flood_control_end = 0 * month_start_operations
    condition = flow_avg > self.parameters.reservoir_flood_control_condition
    match = 0 * month
    keep_going = np.where(
        np.isfinite(month_start_operations),
        True,
        False
    )
    # TODO why 8?
    for j in np.arange(8):
        t = j+1
        # if not an xarray object with coords, the sel doesn't work, so that why the strange definitions here
        month = self.reservoir_streamflow_schedule.idxmax(dim=streamflow_time_name)
        month = month.where(
            month_start_operations - t < 1,
            month_start_operations - t
        ).where(
            month_start_operations - t >= 1,
            month_start_operations - t + 12
        )
        month_1 = month.where(
            month_start_operations - t + 1 < 1,
            month_start_operations - t + 1
        ).where(
            month_start_operations - t + 1 >= 1,
            month_start_operations - t + 1 + 12
        )
        month_2 = month.where(
            month_start_operations - t - 1 < 1,
            month_start_operations - t - 1
        ).where(
            month_start_operations - t - 1 >= 1,
            month_start_operations - t - 1 + 12
        )
        flow = self.reservoir_streamflow_schedule.sel({
            streamflow_time_name: month.fillna(1)
        })
        flow = np.where(np.isfinite(month), flow, np.nan)
        flow_1 = self.reservoir_streamflow_schedule.sel({
            streamflow_time_name: month_1.fillna(1)
        })
        flow_1 = np.where(np.isfinite(month_1), flow_1, np.nan)
        flow_2 = self.reservoir_streamflow_schedule.sel({
            streamflow_time_name: month_2.fillna(1)
        })
        flow_2 = np.where(np.isfinite(month_2), flow_2, np.nan)
        end_condition = (flow >= flow_avg) & (flow_2 <= flow_avg) & (match == 0)
        month_flood_control_end = np.where(
            condition & end_condition & keep_going,
            month,
            month_flood_control_end
        )
        match = np.where(
            condition & end_condition & keep_going,
            1,
            match
        )
        start_condition = (flow <= flow_1) & (flow <= flow_2) & (flow <= flow_avg)
        month_flood_control_start = np.where(
            condition & start_condition & keep_going,
            month,
            month_flood_control_start
        )
        keep_going = np.where(
            condition & start_condition & keep_going,
            False,
            keep_going
        )
        # note: in fortran mosart there's a further condition concerning hydropower, but it doesn't seem to be used
    
    # if flood control is active, enforce the flood control targets
    flood_control_condition = (self.grid.reservoir_use_flood_control > 0) & (month_flood_control_start == 0)
    month = np.where(
        month_flood_control_end - 2 < 0,
        month_flood_control_end - 2 + 12,
        month_flood_control_end - 2
    )
    month_flood_control_start = np.where(
        condition & flood_control_condition,
        month,
        month_flood_control_start
    )
    
    # TODO should probably initialize these along with all the others in the main state file, so it's easier to see all the state variables
    self.state = self.state.join(pd.DataFrame(
        month_start_operations,
        columns=['reservoir_month_start_operations']
    )).join(pd.DataFrame(
        month_flood_control_start,
        columns=['reservoir_month_flood_control_start']
    )).join(pd.DataFrame(
        month_flood_control_end,
        columns=['reservoir_month_flood_control_end']
    )).join(pd.DataFrame(
        self.state.zeros.values,
        columns=['reservoir_release']
    ))


def reservoir_release(self):
    # compute release from reservoirs
    
    # TODO so much logic was dependent on monthly, so still assuming monthly for now, but here's the epiweek for when that is relevant
    epiweek = Week.fromdate(self.current_time).week
    month = self.current_time.month
    
    # if it's the start of the operational year for the reservoir, set it's start of op year storage to the current storage
    self.state.reservoir_storage_operation_year_start[:] = np.where(
        self.state.reservoir_month_start_operations.values == month,
        self.state.reservoir_storage.values,
        self.state.reservoir_storage_operation_year_start.values
    )
    
    regulation_release(self)
    
    storage_targets(self)


def regulation_release(self):
    # compute the expected monthly release based on Biemans (2011)
    
    # TODO this is still written assuming monthly, but here's the epiweek for when that is relevant
    epiweek = Week.fromdate(self.current_time).week
    month = self.current_time.month
    
    streamflow_time_name = self.config.get('water_management.reservoirs.streamflow_time_resolution')
    
    # initialize to the average flow
    self.state.reservoir_release[:] = self.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values
    
    # TODO what is k
    k = self.state.reservoir_storage_operation_year_start.values / ( self.parameters.reservoir_regulation_release_parameter * self.grid.reservoir_storage_capacity.values)
    
    # TODO what is factor
    factor = np.where(
        self.grid.reservoir_runoff_capacity.values > self.parameters.reservoir_runoff_capacity_condition,
        (2.0 / self.grid.reservoir_runoff_capacity.values) ** 2.0,
        0
    )
    
    # release is some combination of prerelease, average flow in the time period, and total average flow
    self.state.reservoir_release[:] = np.where(
        (self.grid.reservoir_use_electricity.values > 0) | (self.grid.reservoir_use_irrigation.values > 0),
        np.where(
            self.grid.reservoir_runoff_capacity.values <= 2.0,
            k * self.reservoir_prerelease_schedule.sel({streamflow_time_name: month}).values,
            k * factor * self.reservoir_prerelease_schedule.sel({streamflow_time_name: month}).values + (1 - factor) * self.reservoir_streamflow_schedule.sel({streamflow_time_name: month}).values
        ),
        np.where(
            self.grid.reservoir_runoff_capacity.values <= 2.0,
            k * self.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values,
            k * factor * self.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values + (1 - factor) * self.reservoir_streamflow_schedule.sel({streamflow_time_name: month}).values
        )
    )

def storage_targets(self):
    # define the necessary drop in storage based on storage targets at the start of the month
    # should not be run during the euler solve # TODO is that because it's expensive?
    pass