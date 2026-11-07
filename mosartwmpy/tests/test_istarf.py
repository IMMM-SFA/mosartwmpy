import numpy as np
import unittest

from datetime import date, datetime

from mosartwmpy.grid.grid import Grid
from mosartwmpy.reservoirs.istarf import compute_istarf_release, istarf_release
from mosartwmpy.state.state import State
from mosartwmpy.utilities.epiweek import get_epiweek_from_datetime


class IstarfTest(unittest.TestCase):

    def _default_inputs(self):
        capacity = np.array([6670.7 * 1.0e6])
        return dict(
            epiweek=np.minimum(float(get_epiweek_from_datetime(date(1994, 10, 1))), 52.0),
            uses_istarf=np.array([True]),
            reservoir_id=np.array([0.0]),
            upper_min=np.array([-np.inf]),
            upper_max=np.array([np.inf]),
            upper_alpha=np.array([-1.74]),
            upper_beta=np.array([-13.62]),
            upper_mu=np.array([78.17]),
            lower_min=np.array([48.02]),
            lower_max=np.array([np.inf]),
            lower_alpha=np.array([0.64]),
            lower_beta=np.array([-3.34]),
            lower_mu=np.array([50.12]),
            release_min_parameter=np.array([-0.992]),
            release_max_parameter=np.array([3.312]),
            release_alpha_one=np.array([0.0836]),
            release_alpha_two=np.array([0.2017]),
            release_beta_one=np.array([-0.1011]),
            release_beta_two=np.array([-0.0162]),
            release_p_one=np.array([1.502]),
            release_p_two=np.array([0.006]),
            release_c=np.array([-0.617]),
            capacity=capacity,
            minimum_storage=0.1 * capacity,
            inflow_mean=np.array([176.297]),
            storage=np.array([3493.943 * 1.0e6]),
            inflow=np.array([10.512893 * 1.0e6 / (24.0 * 60.0 * 60.0)]),
            release=np.array([0.0]),
        )

    def _run_kernel(self, **overrides):
        inputs = self._default_inputs()
        inputs.update(overrides)
        compute_istarf_release(
            inputs['epiweek'],
            inputs['uses_istarf'],
            inputs['reservoir_id'],
            inputs['upper_min'],
            inputs['upper_max'],
            inputs['upper_alpha'],
            inputs['upper_beta'],
            inputs['upper_mu'],
            inputs['lower_min'],
            inputs['lower_max'],
            inputs['lower_alpha'],
            inputs['lower_beta'],
            inputs['lower_mu'],
            inputs['release_min_parameter'],
            inputs['release_max_parameter'],
            inputs['release_alpha_one'],
            inputs['release_alpha_two'],
            inputs['release_beta_one'],
            inputs['release_beta_two'],
            inputs['release_p_one'],
            inputs['release_p_two'],
            inputs['release_c'],
            inputs['capacity'],
            inputs['minimum_storage'],
            inputs['inflow_mean'],
            inputs['storage'],
            inputs['inflow'],
            inputs['release'],
        )
        return inputs['release']

    def test_is_correct(self):
        release = self._run_kernel()
        self.assertAlmostEqual(release[0], 6957353.29, 2)

    def test_release_is_capped_at_active_storage_above_minimum_storage(self):
        """When the harmonic rule or release_min exceeds active storage above CAP_MIN, release is capped."""
        storage = np.array([3493.943 * 1.0e6])
        daily_inflow_volume = 10.512893 * 1.0e6
        # Leave only 1.0e6 m3 of active storage above CAP_MIN (plus the 10.512893e6 daily inflow -> 11.512893e6 total),
        # or set CAP_MIN so storage - CAP_MIN + daily_inflow == 2.0e6 m3 (< 6.957e6 unconstrained release)
        minimum_storage = storage + daily_inflow_volume - 2.0e6
        release = self._run_kernel(minimum_storage=minimum_storage)
        self.assertAlmostEqual(release[0], 2.0e6, 2)

    def test_release_is_zero_when_below_minimum_storage(self):
        """A reservoir at or below CAP_MIN after accounting for daily inflow cannot release."""
        storage = np.array([500.0 * 1.0e6])
        minimum_storage = np.array([600.0 * 1.0e6])
        release = self._run_kernel(storage=storage, minimum_storage=minimum_storage)
        self.assertEqual(release[0], 0.0)

    def test_istarf_release_passes_grid_minimum_storage(self):
        """istarf_release converts the CAP_MIN-capped daily volume to m3/s and leaves generic reservoirs untouched."""
        inputs = self._default_inputs()
        grid = Grid(empty=True)
        grid.reservoir_id = np.array([1.0, 2.0])
        grid.reservoir_behavior = np.array(['full', 'generic'], dtype=object)
        for grid_attr, key in (
            ('reservoir_upper_min', 'upper_min'),
            ('reservoir_upper_max', 'upper_max'),
            ('reservoir_upper_alpha', 'upper_alpha'),
            ('reservoir_upper_beta', 'upper_beta'),
            ('reservoir_upper_mu', 'upper_mu'),
            ('reservoir_lower_min', 'lower_min'),
            ('reservoir_lower_max', 'lower_max'),
            ('reservoir_lower_alpha', 'lower_alpha'),
            ('reservoir_lower_beta', 'lower_beta'),
            ('reservoir_lower_mu', 'lower_mu'),
            ('reservoir_release_min', 'release_min_parameter'),
            ('reservoir_release_max', 'release_max_parameter'),
            ('reservoir_release_alpha_one', 'release_alpha_one'),
            ('reservoir_release_alpha_two', 'release_alpha_two'),
            ('reservoir_release_beta_one', 'release_beta_one'),
            ('reservoir_release_beta_two', 'release_beta_two'),
            ('reservoir_release_p_one', 'release_p_one'),
            ('reservoir_release_p_two', 'release_p_two'),
            ('reservoir_release_c', 'release_c'),
            ('reservoir_storage_capacity', 'capacity'),
            ('reservoir_computed_meanflow_cumecs', 'inflow_mean'),
        ):
            setattr(grid, grid_attr, np.repeat(inputs[key], 2))

        daily_inflow_volume = 10.512893 * 1.0e6
        allowed_daily_volume = 1.5e6
        grid.reservoir_minimum_storage = np.repeat(
            inputs['storage'] + daily_inflow_volume - allowed_daily_volume, 2
        )

        state = State(empty=True)
        state.reservoir_storage = np.repeat(inputs['storage'], 2)
        state.channel_inflow_upstream = np.repeat(inputs['inflow'], 2)
        state.reservoir_release = np.array([999.0, 42.0])

        istarf_release(state, grid, datetime(1994, 10, 1))
        self.assertAlmostEqual(state.reservoir_release[0], allowed_daily_volume / 86400.0, places=5)
        self.assertEqual(state.reservoir_release[1], 42.0)


if __name__ == '__main__':
    unittest.main()
