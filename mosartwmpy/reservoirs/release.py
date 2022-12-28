import numpy as np

from datetime import datetime, time
from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.state.state import State
from mosartwmpy.grid.grid import Grid
from mosartwmpy.reservoirs.istarf import istarf_release


def reservoir_release(state: State, grid: Grid, config: Benedict, parameters: Parameters, current_time: datetime, mask: np.ndarray):
    # compute release from reservoirs

    # to support enabling some reservoirs to use generic rules and others to use special rules,
    # let's always compute release with the generic rules at the beginning of the month, and then let the
    # special rules update the release afterward, where required

    # at the beginning of simulation and start of each month,
    # apply the generic reservoir operating rules to update the release
    if (
        (current_time == datetime.combine(config.get('simulation.start_date'), time.min)) or
        (current_time == datetime(current_time.year, current_time.month, 1))
    ):
        month = current_time.month
        # if it's the start of the operational year for the reservoir,
        # set it's start of op year storage to the current storage
        state.reservoir_storage_operation_year_start = np.where(
            state.reservoir_month_start_operations == month,
            state.reservoir_storage,
            state.reservoir_storage_operation_year_start
        )
        regulation_release(state, grid, parameters, current_time, mask)
        storage_targets(state, grid, current_time, mask)

    # if ISTARF is enabled, and it is the start of a day, update the release targets for non-generic reservoirs
    if (
        config.get('water_management.reservoirs.enable_istarf') and
        (current_time == datetime(current_time.year, current_time.month, current_time.day, 0, 0, 0))
    ):
        istarf_release(state, grid, current_time)


def regulation_release(state, grid, parameters, current_time, mask):
    # compute the expected monthly release based on Biemans (2011)
    
    # initialize to the average flow
    state.reservoir_release = grid.reservoir_streamflow_schedule.mean(dim='month').values[mask]
    
    # TODO what is k
    k = state.reservoir_storage_operation_year_start / (
            parameters.reservoir_regulation_release_parameter * grid.reservoir_storage_capacity)
    
    # TODO what is factor
    factor = np.where(
        grid.reservoir_runoff_capacity.values[mask] > parameters.reservoir_runoff_capacity_parameter,
        (2.0 / grid.reservoir_runoff_capacity.values[mask]) ** 2.0,
        0
    )
    
    # release is some combination of prerelease, average flow in the time period, and total average flow
    state.reservoir_release = np.where(
        np.logical_or(grid.reservoir_use_electricity == True, grid.reservoir_use_irrigation == True),
        np.where(
            grid.reservoir_runoff_capacity.values[mask] <= 2.0,
            k * grid.reservoir_prerelease_schedule.sel({'month': current_time.month}).values[mask],
            k * factor * grid.reservoir_prerelease_schedule.sel({
                'month': current_time.month}).values[mask] + (1 - factor) * grid.reservoir_streamflow_schedule.sel({
                    'month': current_time.month}).values[mask]
        ),
        np.where(
            grid.reservoir_runoff_capacity.values[mask] <= 2.0,
            k * grid.reservoir_streamflow_schedule.mean(dim='month').values[mask],
            k * factor * grid.reservoir_streamflow_schedule.mean(
                dim='month').values[mask] + (1 - factor) * grid.reservoir_streamflow_schedule.sel({
                    'month': current_time.month}).values[mask]
        )
    )


def storage_targets(state: State, grid: Grid, current_time: datetime, mask: np.ndarray) -> None:
    """Define the necessary drop in storage based on the reservoir storage targets at the start of the month.

    Args:
        state (State): the model state
        grid (Grid): the model grid
        config (Config): the model configuration
        parameters (Parameters): the model parameters
        current_time (datetime): the current simulation time
    """

    # TODO the logic here is really hard to follow... can it be simplified or made more readable?

    # if flood control active and has a flood control start
    flood_control_condition = (grid.reservoir_use_flood_control == True) & (state.reservoir_month_flood_control_start > 0)
    # modify release in order to maintain a certain storage level
    month_condition = state.reservoir_month_flood_control_start <= state.reservoir_month_flood_control_end
    total_condition = flood_control_condition & (
        (month_condition &
        (current_time.month >= state.reservoir_month_flood_control_start) &
        (current_time.month < state.reservoir_month_flood_control_end)) |
        (np.logical_not(month_condition) &
        (current_time.month >= state.reservoir_month_flood_control_start) |
        (current_time.month < state.reservoir_month_flood_control_end))
    )
    drop = 0 * state.reservoir_month_flood_control_start
    n_month = 0 * drop
    for m in np.arange(1, 13):
        m_and_condition = (m >= state.reservoir_month_flood_control_start) & (m < state.reservoir_month_flood_control_end)
        m_or_condition = (m >= state.reservoir_month_flood_control_start) | (m < state.reservoir_month_flood_control_end)
        drop = np.where(
            (month_condition & m_and_condition) | (np.logical_not(month_condition) & m_or_condition),
            np.where(
                grid.reservoir_streamflow_schedule.sel({'month': m}).values[mask] >= grid.reservoir_streamflow_schedule.mean(dim='month').values[mask],
                drop + 0,
                drop + np.abs(grid.reservoir_streamflow_schedule.mean(dim='month').values[mask] - grid.reservoir_streamflow_schedule.sel({'month': m}).values[mask])
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
        (current_time.month >= state.reservoir_month_flood_control_end) &
        (current_time.month < state.reservoir_month_start_operations)
    )
    second_condition = flood_control_condition & np.logical_not(month_condition) & (
        (current_time.month >= state.reservoir_month_flood_control_end) |
        (current_time.month < state.reservoir_month_start_operations)
    )
    state.reservoir_release = np.where(
        (state.reservoir_release > grid.reservoir_streamflow_schedule.mean(dim='month').values[mask]) & (first_condition | second_condition),
        grid.reservoir_streamflow_schedule.mean(dim='month').values[mask],
        state.reservoir_release
    )
