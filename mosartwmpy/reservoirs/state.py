import numpy as np

from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid


def initialize_reservoir_state(self, grid: Grid, config: Benedict, parameters: Parameters) -> None:
    """Initializes the reservoir state.

    Args:
        grid (Grid): the model grid
        config (Config): the model configuration
        parameters (Parameters): the model parameters
    """

    # reservoir storage at the start of the operation year
    self.reservoir_storage_operation_year_start = 0.85 * grid.reservoir_storage_capacity

    # initial storage in each reservoir
    self.reservoir_storage = 0.9 * grid.reservoir_storage_capacity

    initialize_reservoir_start_of_operation_year(self, grid, config, parameters)


def initialize_reservoir_start_of_operation_year(self, grid: Grid, config: Benedict, parameters: Parameters) -> None:
    """Determines the start of operation for each reservoir, which influences the irrigation release patterns

    Args:
        grid (Grid): the model grid
        config (Config): the model configuration
        parameters (Parameters): the model parameters
    """

    # Note from fortran mosart:
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

    self.reservoir_month_start_operations = month_start_operations
    self.reservoir_month_flood_control_start = month_flood_control_start
    self.reservoir_month_flood_control_end = month_flood_control_end
