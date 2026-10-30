from benedict.dicts import benedict as Benedict
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import regex as re
from xarray import open_dataset

from mosartwmpy.farmer_abm.farmer_abm import FarmerABM
from mosartwmpy.state.state import State
from mosartwmpy.utilities.timing import timing


def _open_error(error: Exception, path: str) -> Exception:
    """Builds the error to raise when the demand file cannot be opened.

    The original type is kept where it can be rebuilt, since that is what a caller branches on:
    an OSError subclass (FileNotFoundError, PermissionError, ...) keeps its errno and gets the
    path as its filename; a ValueError, which xarray raises for a file it cannot read as netCDF,
    keeps its type unless the subclass has its own constructor signature (UnicodeDecodeError),
    in which case a plain ValueError is returned. The original is chained by the caller.

    Args:
        error (Exception): the exception raised by open_dataset
        path (str): the demand file path

    Returns:
        Exception: the exception to raise, naming the path
    """

    message = 'Unable to open demand file'
    if isinstance(error, OSError) and error.errno is not None:
        detail = f'{message}: {error.strerror}' if error.strerror else message
        return type(error)(error.errno, detail, path)
    try:
        return type(error)(f'{message}: {path}')
    except TypeError:
        return ValueError(f'{message}: {path}')


# consumption above withdrawal by less than this fraction of the withdrawal (of 1 m3/s where the
# withdrawal is smaller) is round-off, not a violation: a float32 withdrawal beside a float64
# consumption that mean the same value differ by up to about 6e-8 relative, which an exact
# comparison flagged in a quarter of the cells of the packaged test file
CONSUMPTION_TOLERANCE = 1e-6
# the fraction of cells with demand that may have consumption above withdrawal before the file is
# rejected as misconfigured instead of clipped
MAX_CLIPPED_FRACTION = 0.5


# @timing
def load_demand(name: str, state: State, config: Benedict, current_time: datetime, farmer_abm: FarmerABM, mask: np.ndarray) -> None:
    """Loads water demand from file into the state for each grid cell.

    Args:
        name (str): name of the simulation
        state (State): the current model state; will be mutated
        config (Benedict): the model configuration
        current_time (datetime): the current time of the simulation
        mask (ndarray): mask of active grid cells
    """

    path = config.get('water_management.demand.path')
    
    # Calculate water demand for farmers using an ABM.
    if config.get_bool('water_management.demand.farmer_abm.enabled'):
        try:
            farmer_abm.calc_demand()
            path = f"{config.get('simulation.output_path')}/demand/{name}_farmer_abm_demand_{current_time.strftime('%Y')}.nc"
        except:
            logging.info(f"Water demand calculation for farmer ABM failed. Defaulting to precalculated values. ")

    path = re.sub(r'\{(?:Y|y)[^}]*}', current_time.strftime('%Y'), path)
    path = re.sub(r'\{(?:M|m)[^}]*}', current_time.strftime('%m'), path)
    path = re.sub(r'\{(?:D|d)[^}]*}', current_time.strftime('%d'), path)

    # raise rather than exit: this runs inside a BMI host process, and a typed error with the cause
    # chained lets Model.update log the failure and the caller tell a missing file from a bad name
    try:
        opened = open_dataset(path)
    except (OSError, ValueError) as e:
        raise _open_error(e, path) from e
    # the context closes the file however the body exits, including the validation errors below
    with opened:
        _read_demand(opened, state, config, current_time, mask, path)


def _read_demand(demand, state: State, config: Benedict, current_time: datetime, mask: np.ndarray, path: str) -> None:
    # outside the open's try so a wrong coordinate name surfaces as its own KeyError naming the coordinate
    demand = demand.sortby([
        config.get('water_management.demand.latitude'), config.get('water_management.demand.longitude')
    ])

    # If returnflow is enabled, demand will have four facets:
    #  - irrigation withdrawal
    #  - nonirrigation withdrawal
    #  - irrigation consumption
    #  - nonirrigation consumption
    # In this case, total demand will represent total withdrawal demand
    # But the consumptive deficit will be disaggregated into irrigation/nonirrigation
    # For now let's assume this type of demand will always have a time axis

    if config.get('water_management.demand.time', None) in demand:
        # check for non-standard calendar and convert if needed
        # unsafe=True silences the calendar-mismatch warning: mapping noleap/360_day dates onto the
        # standard calendar is intended, and the time selection below pads to the nearest earlier date
        if not isinstance(demand.indexes[config.get('water_management.demand.time')], pd.DatetimeIndex):
            demand[config.get('water_management.demand.time')] = demand.indexes[config.get('water_management.demand.time')].to_datetimeindex(time_unit='ns', unsafe=True)
        # check if time index includes current time (with some slack on the end)
        if not (
            demand[config.get('water_management.demand.time')].values.min() <= np.datetime64(current_time) <= (demand[config.get('water_management.demand.time')].values.max() + np.timedelta64(31, 'D'))
        ):
            raise ValueError(
                f"Current simulation date {current_time.strftime('%Y-%m-%d')} not within time bounds of demand input file {path}. Aborting..."
            )

    if config.get('water_management.demand.return_flow_enabled', False):

        time_key = config.get('water_management.demand.time')
        # the four disaggregated variables, keyed by the config entry that names each of them
        variables = {
            key: config.get(f'water_management.demand.{key}')
            for key in ('irrigation_withdrawal', 'irrigation_consumption', 'nonirrigation_withdrawal', 'nonirrigation_consumption')
        }

        # validate before reading, so a misnamed variable or a 2-D file is reported with the config
        # key that controls it rather than as a bare KeyError from xarray
        missing = [
            f'`{variable}` (config key `water_management.demand.{key}`)'
            for key, variable in variables.items() if variable not in demand.data_vars
        ]
        if missing:
            raise ValueError(
                f'Return flow demand file {path} has no variable(s) {", ".join(missing)}; return flow requires demand '
                f'disaggregated into irrigation and nonirrigation withdrawal and consumption.'
            )
        if time_key not in demand:
            raise ValueError(
                f'Return flow demand file {path} has no `{time_key}` coordinate (config key `water_management.demand.time`); '
                f'return flow demand must have a time axis.'
            )

        def read(key):
            return np.array(
                demand[variables[key]].sel({time_key: current_time}, method='pad').fillna(0), dtype=np.float64
            ).flatten()[mask]

        state.irrigation_withdrawal_rate = read('irrigation_withdrawal')
        state.irrigation_consumption_rate = read('irrigation_consumption')
        state.nonirrigation_withdrawal_rate = read('nonirrigation_withdrawal')
        state.nonirrigation_consumption_rate = read('nonirrigation_consumption')

        # consumption cannot exceed withdrawal: their difference is the return flow, which would
        # otherwise be negative and be removed from the soil column and the channel without a check.
        # the clip to withdrawal is unconditional, so the return flow is non-negative whatever the
        # file holds, including where the excess is round-off. cells over by more than round-off are
        # also warned about, so a few bad cells do not reject an otherwise usable file; a file where
        # most cells with demand are over is rejected, since that is a misconfiguration (most likely
        # the withdrawal and consumption variable names swapped) and clipping it would silently zero
        # the return flow everywhere
        for use in ('irrigation', 'nonirrigation'):
            withdrawal = getattr(state, f'{use}_withdrawal_rate')
            consumption = getattr(state, f'{use}_consumption_rate')
            excess = consumption - withdrawal
            over = excess > CONSUMPTION_TOLERANCE * np.maximum(withdrawal, 1.0)
            if over.any():
                count = int(over.sum())
                demanding = int(((withdrawal > 0) | (consumption > 0)).sum())
                largest = excess[over].max()
                if count > MAX_CLIPPED_FRACTION * demanding:
                    withdrawal_variable = variables[f'{use}_withdrawal']
                    consumption_variable = variables[f'{use}_consumption']
                    raise ValueError(
                        f'{count} of {demanding} cell(s) with {use} demand in demand file {path} have consumption above '
                        f'withdrawal (largest excess {largest:.4g} m3/s), more than the {MAX_CLIPPED_FRACTION:.0%} that would be '
                        f'clipped; check that `{withdrawal_variable}` (config key `water_management.demand.{use}_withdrawal`) '
                        f'and `{consumption_variable}` (config key `water_management.demand.{use}_consumption`) are not swapped.'
                    )
                logging.warning(
                    f'{count} cell(s) in demand file {path} have {use} consumption above withdrawal '
                    f'(largest excess {largest:.4g} m3/s); consumption clipped to withdrawal in those cells.'
                )
            setattr(state, f'{use}_consumption_rate', np.minimum(consumption, withdrawal))

        state.grid_cell_demand_rate = state.irrigation_withdrawal_rate + state.nonirrigation_withdrawal_rate

    else:

        # if the demand file has a time axis, use it; otherwise assume data is just 2D
        if config.get('water_management.demand.time', None) in demand:
            state.grid_cell_demand_rate = np.array(demand[config.get('water_management.demand.demand')].sel({config.get('water_management.demand.time'): current_time}, method='pad'), dtype=np.float64).flatten()[mask]
        else:
            state.grid_cell_demand_rate = np.array(demand[config.get('water_management.demand.demand')], dtype=np.float64).flatten()[mask]

        # fill missing values with 0
        state.grid_cell_demand_rate = np.where(
            np.logical_not(np.isfinite(state.grid_cell_demand_rate)),
            0,
            state.grid_cell_demand_rate
        )
