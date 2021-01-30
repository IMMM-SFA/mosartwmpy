import logging
import numpy as np
import pandas as pd

from epiweeks import Week
from xarray import concat, open_dataset
from xarray.ufuncs import logical_not

# TODO in fortran mosart there is a StorCalibFlag that affects how storage targets are calculated -- code so far is written assuming that it == 0

def load_reservoirs(grid, config, parameters):
    """[summary]

    Args:
        grid (Grid): Grid instance for this simulation
        config (Benedict): Model configuration benedict instance
        parameters (Parameters): Model parameters instance
    """
    
    logging.info('Loading reservoir file.')
    
    # reservoir parameter file
    reservoirs = open_dataset(config.get('water_management.reservoirs.path'))
    
    # load reservoir variables
    for key, value in config.get('water_management.reservoirs.variables').items():
        setattr(grid, key, np.array(reservoirs[value]).flatten())
    
    # correct the fields with different units
    # surface area from km^2 to m^2
    grid.reservoir_surface_area = grid.reservoir_surface_area * 1.0e6
    # capacity from millions m^3 to m^3
    grid.reservoir_storage_capacity = grid.reservoir_storage_capacity * 1.0e6
    
    # map dams to all their dependent grid cells
    # this will be a table of many to many relationship of grid cell ids to reservoir ids
    grid.reservoir_to_grid_mapping = reservoirs[
        config.get('water_management.reservoirs.grid_to_reservoir')
    ].to_dataframe().reset_index()[[
        config.get('water_management.reservoirs.grid_to_reservoir_reservoir_dimension'),
        config.get('water_management.reservoirs.grid_to_reservoir')
    ]].rename(columns={
        config.get('water_management.reservoirs.grid_to_reservoir_reservoir_dimension'): 'reservoir_id',
        config.get('water_management.reservoirs.grid_to_reservoir'): 'grid_cell_id'
    })
    # drop nan grid ids
    grid.reservoir_to_grid_mapping = grid.reservoir_to_grid_mapping[grid.reservoir_to_grid_mapping.grid_cell_id.notna()]
    # correct to zero-based grid indexing
    grid.reservoir_to_grid_mapping.loc[:, grid.reservoir_to_grid_mapping.grid_cell_id.name] = grid.reservoir_to_grid_mapping.grid_cell_id.values - 1
    grid.reservoir_to_grid_mapping.loc[:, grid.reservoir_to_grid_mapping.reservoir_id.name] = grid.reservoir_to_grid_mapping.reservoir_id.values - 1
    # set to integer
    grid.reservoir_to_grid_mapping = grid.reservoir_to_grid_mapping.astype(int)
    
    # count of the number of reservoirs that can supply each grid cell
    grid.reservoir_count = np.array(pd.DataFrame(grid.id).join(
        grid.reservoir_to_grid_mapping.groupby('grid_cell_id').count().rename(columns={'reservoir_id': 'reservoir_count'}),
        how='left'
    ).reservoir_count)
    
    # index by grid cell
    grid.reservoir_to_grid_mapping = grid.reservoir_to_grid_mapping.set_index('grid_cell_id')
    
    # prepare the month or epiweek based reservoir schedules mapped to the domain
    prepare_reservoir_schedule(grid, config, parameters, reservoirs)
    
    reservoirs.close()


def prepare_reservoir_schedule(grid, config, parameters, reservoirs):
    # the reservoir streamflow and demand are specified by the time resolution and reservoir id
    # so let's remap those to the actual mosart domain for ease of use
    
    # TODO i had wanted to convert these all to epiweeks no matter what format provided, but we don't know what year all the data came from
    
    # streamflow flux
    streamflow_time_name = config.get('water_management.reservoirs.streamflow_time_resolution')
    streamflow = reservoirs[config.get('water_management.reservoirs.streamflow')]
    schedule = None
    for t in np.arange(streamflow.shape[0]):
        flow = streamflow[t, :].to_pandas().to_frame('streamflow')
        sched = pd.DataFrame(grid.reservoir_id, columns=['reservoir_id']).merge(flow, how='left', left_on='reservoir_id', right_index=True)[['streamflow']].to_xarray().expand_dims(
            {streamflow_time_name: 1},
            axis=0
        )
        if schedule is None:
            schedule = sched
        else:
            schedule = concat([schedule, sched], dim=streamflow_time_name)
    grid.reservoir_streamflow_schedule = schedule.assign_coords(
        # if monthly, convert to 1 based month index (instead of starting from 0)
        {streamflow_time_name: (streamflow_time_name, schedule[streamflow_time_name].values + (1 if streamflow_time_name == 'month' else 0))}
    ).streamflow
    
    # demand volume
    demand_time_name = config.get('water_management.reservoirs.demand_time_resolution')
    demand = reservoirs[config.get('water_management.reservoirs.demand')]
    schedule = None
    for t in np.arange(demand.shape[0]):
        dem = demand[t, :].to_pandas().to_frame('demand')
        sched = pd.DataFrame(grid.reservoir_id, columns=['reservoir_id']).merge(dem, how='left', left_on='reservoir_id', right_index=True)[['demand']].to_xarray().expand_dims(
            {demand_time_name: 1}, axis=0
        )
        if schedule is None:
            schedule = sched
        else:
            schedule = concat([schedule, sched], dim=demand_time_name)
    grid.reservoir_demand_schedule = schedule.assign_coords(
        # if monthly, convert to 1 based month index (instead of starting from 0)
        {demand_time_name: (demand_time_name, schedule[demand_time_name].values + (1 if demand_time_name == 'month' else 0))}
    ).demand
    
    # initialize prerelease based on long term mean flow and demand (Biemans 2011)
    # TODO this assumes demand and flow use the same timescale :(
    flow_avg = grid.reservoir_streamflow_schedule.mean(dim=streamflow_time_name)
    demand_avg = grid.reservoir_demand_schedule.mean(dim=demand_time_name)
    prerelease = (1.0 * grid.reservoir_streamflow_schedule)
    prerelease[:,:] = flow_avg
    # note that xarray `where` modifies the false values
    condition = (demand_avg >= (0.5 * flow_avg)) & (flow_avg > 0)
    prerelease = prerelease.where(
        logical_not(condition),
        demand_avg/ 10 + 9 / 10 * flow_avg * grid.reservoir_demand_schedule / demand_avg
    )
    prerelease = prerelease.where(
        condition,
        prerelease.where(
            logical_not((flow_avg + grid.reservoir_demand_schedule - demand_avg) > 0),
            flow_avg + grid.reservoir_demand_schedule - demand_avg
        )
    )
    grid.reservoir_prerelease_schedule = prerelease


def initialize_reservoir_state(state, grid, config, parameters):
    """[summary]

    Args:
        state (State): Model state instance
        grid (Grid): Model grid instance
        config (Config): Model config instance
        parameters (Parameters): Model parameters instance
    """
    
    # reservoir storage at the start of the operation year
    state.reservoir_storage_operation_year_start = 0.85 * grid.reservoir_storage_capacity
    
    # initial storage in each reservoir
    state.reservoir_storage = 0.9 * grid.reservoir_storage_capacity
    
    initialize_reservoir_start_of_operation_year(state, grid, config, parameters)


def initialize_reservoir_start_of_operation_year(state, grid, config, parameters):
    # define the start of the operation - define irrigation releases pattern
    # multiple hydrograph - 1 peak, 2 peaks, multiple small peaks
    # TODO this all depends on the schedules being monthly :(
    
    streamflow_time_name = config.get('water_management.reservoirs.streamflow_time_resolution')
    
    # find the peak flow and peak flow month for each reservoir
    peak = np.max(grid.reservoir_streamflow_schedule.values, axis=0)
    month_start_operations = grid.reservoir_streamflow_schedule.idxmax(dim=streamflow_time_name).values
    
    # correct the month start for reservoirs where average flow is greater than a small value and magnitude of peak flow difference from average is greater than smaller value
    # TODO a little hard to follow the logic here but it seem to be related to number of peaks/troughs
    flow_avg = grid.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values
    condition = flow_avg > parameters.reservoir_minimum_flow_condition
    number_of_sign_changes = 0 * flow_avg
    count = 1 + number_of_sign_changes
    count_max = 1 + number_of_sign_changes
    month = 1 * month_start_operations
    sign = np.where(
        np.abs(peak - flow_avg) > parameters.reservoir_small_magnitude_difference,
        np.where(
            peak - flow_avg > 0,
            1,
            -1
        ),
        1
    )
    current_sign = 1 * sign
    for t in grid.reservoir_streamflow_schedule[streamflow_time_name].values:
        # if not an xarray object with coords, the sel doesn't work, so that why the strange definition here
        i = grid.reservoir_streamflow_schedule.idxmax(dim=streamflow_time_name)
        i = i.where(
            i + t > 12,
            i + t
        ).where(
            i + t <= 12,
            i + t - 12
        )
        flow = grid.reservoir_streamflow_schedule.sel({
            streamflow_time_name: i.fillna(1)
        })
        flow = np.where(np.isfinite(i), flow, np.nan)
        current_sign = np.where(
            np.abs(flow - flow_avg) > parameters.reservoir_small_magnitude_difference,
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
    condition = flow_avg > parameters.reservoir_flood_control_condition
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
        month = grid.reservoir_streamflow_schedule.idxmax(dim=streamflow_time_name)
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
        flow = grid.reservoir_streamflow_schedule.sel({
            streamflow_time_name: month.fillna(1)
        })
        flow = np.where(np.isfinite(month), flow, np.nan)
        flow_1 = grid.reservoir_streamflow_schedule.sel({
            streamflow_time_name: month_1.fillna(1)
        })
        flow_1 = np.where(np.isfinite(month_1), flow_1, np.nan)
        flow_2 = grid.reservoir_streamflow_schedule.sel({
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
    flood_control_condition = (grid.reservoir_use_flood_control > 0) & (month_flood_control_start == 0)
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
    
    state.reservoir_month_start_operations = month_start_operations
    state.reservoir_month_flood_control_start = month_flood_control_start
    state.reservoir_month_flood_control_end = month_flood_control_end


def reservoir_release(state, grid, config, parameters, current_time):
    # compute release from reservoirs
    
    # TODO so much logic was dependent on monthly, so still assuming monthly for now, but here's the epiweek for when that is relevant
    epiweek = Week.fromdate(current_time).week
    month = current_time.month
    
    # if it's the start of the operational year for the reservoir, set it's start of op year storage to the current storage
    state.reservoir_storage_operation_year_start = np.where(
        state.reservoir_month_start_operations == month,
        state.reservoir_storage,
        state.reservoir_storage_operation_year_start
    )
    
    regulation_release(state, grid, config, parameters, current_time)
    
    storage_targets(state, grid, config, parameters, current_time)


def regulation_release(state, grid, config, parameters, current_time):
    # compute the expected monthly release based on Biemans (2011)
    
    # TODO this is still written assuming monthly, but here's the epiweek for when that is relevant
    epiweek = Week.fromdate(current_time).week
    month = current_time.month
    streamflow_time_name = config.get('water_management.reservoirs.streamflow_time_resolution')
    
    # initialize to the average flow
    state.reservoir_release = grid.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values
    
    # TODO what is k
    k = state.reservoir_storage_operation_year_start / ( parameters.reservoir_regulation_release_parameter * grid.reservoir_storage_capacity)
    
    # TODO what is factor
    factor = np.where(
        grid.reservoir_runoff_capacity > parameters.reservoir_runoff_capacity_condition,
        (2.0 / grid.reservoir_runoff_capacity) ** 2.0,
        0
    )
    
    # release is some combination of prerelease, average flow in the time period, and total average flow
    state.reservoir_release = np.where(
        (grid.reservoir_use_electricity > 0) | (grid.reservoir_use_irrigation > 0),
        np.where(
            grid.reservoir_runoff_capacity <= 2.0,
            k * grid.reservoir_prerelease_schedule.sel({streamflow_time_name: month}).values,
            k * factor * grid.reservoir_prerelease_schedule.sel({streamflow_time_name: month}).values + (1 - factor) * grid.reservoir_streamflow_schedule.sel({streamflow_time_name: month}).values
        ),
        np.where(
            grid.reservoir_runoff_capacity <= 2.0,
            k * grid.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values,
            k * factor * grid.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values + (1 - factor) * grid.reservoir_streamflow_schedule.sel({streamflow_time_name: month}).values
        )
    )

def storage_targets(state, grid, config, parameters, current_time):
    # define the necessary drop in storage based on storage targets at the start of the month
    # should not be run during the euler solve # TODO is that because it's expensive?
    
    # TODO the logic here is really hard to follow... can it be simplified or made more readable?
    
    # TODO this is still written assuming monthly, but here's the epiweek for when that is relevant
    epiweek = Week.fromdate(current_time).week
    month = current_time.month
    streamflow_time_name = config.get('water_management.reservoirs.streamflow_time_resolution')
    
    # if flood control active and has a flood control start
    flood_control_condition = (grid.reservoir_use_flood_control > 0) & (state.reservoir_month_flood_control_start > 0)
    # modify release in order to maintain a certain storage level
    month_condition = state.reservoir_month_flood_control_start <= state.reservoir_month_flood_control_end
    total_condition = flood_control_condition & (
        (month_condition &
        (month >= state.reservoir_month_flood_control_start) &
        (month < state.reservoir_month_flood_control_end)) |
        (np.logical_not(month_condition) &
        (month >= state.reservoir_month_flood_control_start) |
        (month < state.reservoir_month_flood_control_end))
    )
    drop = 0 * state.reservoir_month_flood_control_start
    n_month = 0 * drop
    for m in np.arange(1,13): # TODO assumes monthly
        m_and_condition = (m >= state.reservoir_month_flood_control_start) & (m < state.reservoir_month_flood_control_end)
        m_or_condition = (m >= state.reservoir_month_flood_control_start) | (m < state.reservoir_month_flood_control_end)
        drop = np.where(
            (month_condition & m_and_condition) | (np.logical_not(month_condition) & m_or_condition),
            np.where(
                grid.reservoir_streamflow_schedule.sel({streamflow_time_name: m}).values >= grid.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values,
                drop + 0,
                drop + np.abs(grid.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values - grid.reservoir_streamflow_schedule.sel({streamflow_time_name: m}).values)
            ),
            drop
        )
        n_month = np.where(
            (month_condition & m_and_condition) | (np.logical_not(month_condition) & m_or_condition),
            n_month + 1,
            n_month
        )
    state.reservoir_release = np.where(
        total_condition & (n_month > 0),
        state.reservoir_release + drop / n_month,
        state.reservoir_release
    )
    # now need to make sure it will fill up but issue with spilling in certain hydro-climate conditions
    month_condition = state.reservoir_month_flood_control_end <= state.reservoir_month_start_operations
    first_condition = flood_control_condition & month_condition & (
        (month >= state.reservoir_month_flood_control_end) &
        (month < state.reservoir_month_start_operations)
    )
    second_condition = flood_control_condition & np.logical_not(month_condition) & (
        (month >= state.reservoir_month_flood_control_end) |
        (month < state.reservoir_month_start_operations)
    )
    # TODO this logic exists in fortran mosart but isn't used...
    # fill = 0 * drop
    # n_month = 0 * drop
    # for m in np.arange(1,13): # TODO assumes monthly
    #     m_condition = (m >= self.state.reservoir_month_flood_control_end.values) &
    #         (self.reservoir_streamflow_schedule.sel({streamflow_time_name: m}).values > self.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values) & (
    #             (first_condition & (m <= self.state.reservoir_month_start_operations)) |
    #             (second_condition & (m <= 12))
    #         )
    #     fill = np.where(
    #         m_condition,
    #         fill + np.abs(self.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values - self.reservoir_streamflow_schedule.sel({streamflow_time_name: m}).values),
    #         fill
    #     )
    #     n_month = np.where(
    #         m_condition,
    #         n_month + 1,
    #         n_month
    #     )
    state.reservoir_release = np.where(
        (state.reservoir_release> grid.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values) & (first_condition | second_condition),
        grid.reservoir_streamflow_schedule.mean(dim=streamflow_time_name).values,
        state.reservoir_release
    )