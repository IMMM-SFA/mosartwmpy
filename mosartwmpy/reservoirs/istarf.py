import numba as nb
import numpy as np
import pandas as pd

from datetime import datetime
from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.state.state import State
from mosartwmpy.grid.grid import Grid
from mosartwmpy.utilities.epiweek import get_epiweek_from_datetime


def istarf_release(state: State, grid: Grid, current_time: datetime):
    # estimate reservoir release using ISTARF which is based on harmonic functions

    # restrict epiweek to [1, 52]
    epiweek = np.minimum(float(get_epiweek_from_datetime(current_time)), 52.0)

    # boolean array indicating which cells use the istarf rules;
    # if behavior is "generic", then just keep the monthly generic release value instead
    uses_istarf = np.array([(x.lower() != 'generic') if pd.notna(x) else False for x in grid.reservoir_behavior])

    # initialize the release array
    daily_release = np.zeros(len(grid.reservoir_id))

    compute_istarf_release(
        epiweek,
        uses_istarf,
        grid.reservoir_id,
        grid.reservoir_upper_min,
        grid.reservoir_upper_max,
        grid.reservoir_upper_alpha,
        grid.reservoir_upper_beta,
        grid.reservoir_upper_mu,
        grid.reservoir_lower_min,
        grid.reservoir_lower_max,
        grid.reservoir_lower_alpha,
        grid.reservoir_lower_beta,
        grid.reservoir_lower_mu,
        grid.reservoir_release_min,
        grid.reservoir_release_max,
        grid.reservoir_release_alpha_one,
        grid.reservoir_release_alpha_two,
        grid.reservoir_release_beta_one,
        grid.reservoir_release_beta_two,
        grid.reservoir_release_p_one,
        grid.reservoir_release_p_two,
        grid.reservoir_release_c,
        grid.reservoir_storage_capacity,
        grid.reservoir_computed_meanflow_cumecs,
        state.reservoir_storage,
        state.channel_inflow_upstream,
        daily_release,
    )

    # join release back into the grid as m3/s
    # except where generic behavior is requested
    state.reservoir_release = np.where(
        uses_istarf,
        daily_release / (24.0 * 60.0 * 60.0),
        state.reservoir_release
    )


@nb.jit(
    "void("
        "float64, boolean[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], "
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], "
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def compute_istarf_release(
    epiweek,
    uses_istarf,
    reservoir_id,
    upper_min,
    upper_max,
    upper_alpha,
    upper_beta,
    upper_mu,
    lower_min,
    lower_max,
    lower_alpha,
    lower_beta,
    lower_mu,
    release_min_parameter,
    release_max_parameter,
    release_alpha_one,
    release_alpha_two,
    release_beta_one,
    release_beta_two,
    release_p_one,
    release_p_two,
    release_c,
    capacity,
    inflow_mean,
    storage,
    inflow,
    release,
):

    omega = 1.0 / 52.0

    for i in nb.prange(len(release)):

        if np.isfinite(reservoir_id[i]) and uses_istarf[i]:

            max_normal = np.minimum(
                upper_max[i],
                np.maximum(
                    upper_min[i],
                    upper_mu[i] +
                    upper_alpha[i] * np.sin(2.0 * np.pi * omega * epiweek) +
                    upper_beta[i] * np.cos(2.0 * np.pi * omega * epiweek)
                )
            )

            min_normal = np.minimum(
                lower_max[i],
                np.maximum(
                    lower_min[i],
                    lower_mu[i] +
                    lower_alpha[i] * np.sin(2.0 * np.pi * omega * epiweek) +
                    lower_beta[i] * np.cos(2.0 * np.pi * omega * epiweek)
                )
            )

            # TODO could make a better forecast?
            forecasted_weekly_volume = 7.0 * inflow[i] * 24.0 * 60.0 * 60.0

            mean_weekly_volume = 7.0 * inflow_mean[i] * 24.0 * 60.0 * 60.0

            standardized_inflow = (forecasted_weekly_volume / mean_weekly_volume) - 1.0

            standardized_weekly_release = (
                release_alpha_one[i] * np.sin(2.0 * np.pi * omega * epiweek) +
                release_alpha_two[i] * np.sin(4.0 * np.pi * omega * epiweek) +
                release_beta_one[i] * np.cos(2.0 * np.pi * omega * epiweek) +
                release_beta_two[i] * np.cos(4.0 * np.pi * omega * epiweek)
            )

            release_min = mean_weekly_volume * (1 + release_min_parameter[i]) / 7.0
            release_max = mean_weekly_volume * (1 + release_max_parameter[i]) / 7.0

            availability_status = (100.0 * storage[i] / capacity[i] - min_normal) / (max_normal - min_normal)

            # above normal
            if availability_status > 1:
                release[i] = (storage[i] - (capacity[i] * max_normal / 100.0) + forecasted_weekly_volume) / 7.0

            # below normal
            elif availability_status < 0:
                release[i] = (storage[i] - (capacity[i] * min_normal / 100.0) + forecasted_weekly_volume) / 7.0

            # within normal
            else:
                release[i] = (mean_weekly_volume * (1 + (
                    standardized_weekly_release + release_c[i] +
                    release_p_one[i] * availability_status +
                    release_p_two[i] * standardized_inflow
                ))) / 7.0

            # enforce boundaries on release
            if release[i] < release_min:
                # shouldn't release less than min
                release[i] = release_min
            elif release[i] > release_max:
                # shouldn't release more than max
                release[i] = release_max

            # storage update and boundaries are enforced during the regulation step
