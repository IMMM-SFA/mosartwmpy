import inspect
import numpy as np
import tempfile
import unittest

import importlib.resources
from pathlib import Path

import xarray as xr

import mosartwmpy.update.update as update_module
from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid

# position of `hillslope_subsurface_runoff` in the jitted `_subcycle` signature, resolved
# rather than hardcoded so these tests fail loudly if that signature is reordered
SUBCYCLE_RUNOFF_ARG = list(
    inspect.signature(update_module._subcycle.py_func).parameters
).index('hillslope_subsurface_runoff')


class ReturnFlowTest(unittest.TestCase):
    """Smoke test for the opt-in return-flow feature.

    Exercises the disaggregated-demand loader (input/demand.py) and the
    per-subcycle return-flow accounting (update.py) with return flow enabled
    and no irrigation-first mask file supplied. The packaged test grid is
    loaded via ``Grid.from_files`` (which bypasses ``Grid.__init__``), so the
    serialized ``irrigation_first_mask`` is empty; we set it to a full-grid
    zeros array here, which is exactly what the ``Grid.__init__`` fallback
    produces for a freshly built grid when no mask path is configured
    (nonirrigation-first everywhere).
    """

    GRID_FILE = str(importlib.resources.files('mosartwmpy').joinpath('tests', 'grid.zip'))
    CONFIG_FILE = str(importlib.resources.files('mosartwmpy').joinpath('tests', 'test_config.yaml'))
    RUNOFF_FILE = str(importlib.resources.files('mosartwmpy').joinpath('tests', 'runoff_1981_01_01.nc'))
    DEMAND_FILE = str(importlib.resources.files('mosartwmpy').joinpath('tests', 'demand_1981_01_01.nc'))
    RESERVOIRS_FILE = str(importlib.resources.files('mosartwmpy').joinpath('tests', 'reservoirs.nc'))

    @classmethod
    def setUpClass(cls):
        # build a disaggregated demand file from the packaged total-demand file
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.DISAGG_DEMAND_FILE = str(Path(cls._tmpdir.name) / 'demand_returnflow_1981_01_01.nc')
        with xr.open_dataset(cls.DEMAND_FILE) as demand:
            total = demand['totalDemand']
            # split the total demand into irrigation vs nonirrigation withdrawal,
            # and a consumed fraction of each; withdrawal >= consumption by construction
            irrigation_withdrawal = total * 0.6
            nonirrigation_withdrawal = total * 0.4
            irrigation_consumption = irrigation_withdrawal * 0.4
            nonirrigation_consumption = nonirrigation_withdrawal * 0.7
            disagg = xr.Dataset(
                {
                    'irrigation_withdrawal': irrigation_withdrawal,
                    'irrigation_consumption': irrigation_consumption,
                    'nonirrigation_withdrawal': nonirrigation_withdrawal,
                    'nonirrigation_consumption': nonirrigation_consumption,
                }
            )
            disagg.to_netcdf(cls.DISAGG_DEMAND_FILE)

        cls.model = Model()
        cls.grid = Grid.from_files(cls.GRID_FILE)
        cls.model.initialize(cls.CONFIG_FILE, grid=cls.grid)

        # point inputs at the packaged test data
        cls.model.config['runoff.path'] = cls.RUNOFF_FILE
        cls.model.config['water_management.reservoirs.path'] = cls.RESERVOIRS_FILE

        # enable return flow and point at the disaggregated demand file
        cls.model.config['water_management.demand.path'] = cls.DISAGG_DEMAND_FILE
        cls.model.config['water_management.demand.return_flow_enabled'] = True

        # the packaged grid was serialized with return flow off, so the mask is
        # empty; set it to zeros (nonirrigation-first everywhere), matching the
        # Grid.__init__ fallback. initialize() has already trimmed grid/state
        # arrays to the mosart mask, so size this to the trimmed grid, not the
        # full grid.
        cls.model.grid.irrigation_first_mask = np.zeros(int(cls.model.mask.sum()))

    @classmethod
    def tearDownClass(cls):
        cls.model.finalize()
        cls._tmpdir.cleanup()

    def test_can_run_with_return_flow(self):
        # two steps so the returnflow accumulated on step 1 is applied on step 2
        self.model.update()
        self.model.update()
        self.assertTrue(True, "model runs with return flow enabled and no mask file")

    def test_return_flow_state_is_finite(self):
        self.model.update()
        for name in (
            'irrigation_returnflow',
            'nonirrigation_returnflow',
            'irrigation_consumption_deficit',
            'nonirrigation_consumption_deficit',
        ):
            values = getattr(self.model.state, name)
            self.assertTrue(np.all(np.isfinite(values)), f"{name} is finite")

        # deficits accumulate unmet consumptive demand and must be non-negative
        self.assertTrue(
            np.all(self.model.state.irrigation_consumption_deficit >= 0.0),
            "irrigation consumption deficit is non-negative",
        )
        self.assertTrue(
            np.all(self.model.state.nonirrigation_consumption_deficit >= 0.0),
            "nonirrigation consumption deficit is non-negative",
        )

    # -- conversion regression tests ------------------------------------------------
    #
    # The two return flow conversions are inline in `update`, so they are pinned here by
    # intercepting the jitted calls that bracket them. Between the return flow block and
    # each of these call sites, nothing else writes the array under test, so the
    # difference across the boundary is exactly what the return flow contributed.
    #
    # Water management is switched off for these tests so the return flow rates are not
    # recomputed mid-timestep; the tests supply them directly instead.

    def _disable_water_management(self):
        self.model.config['water_management.enabled'] = False
        self.addCleanup(self.model.config.__setitem__, 'water_management.enabled', True)

    def _expected_irrigation_flux(self, rate):
        return rate / (self.model.grid.area * self.model.grid.land_fraction)

    def test_irrigation_return_flow_converts_without_the_subcycle_duration(self):
        """A m3/s return flow becomes a m/s flux by area alone, not by area and duration."""
        self._disable_water_management()
        rate = 10.0
        self.model.state.irrigation_returnflow[:] = rate
        self.model.state.nonirrigation_returnflow[:] = 0.0

        snapshots = []
        original = update_module._subcycle
        self.addCleanup(setattr, update_module, '_subcycle', original)

        def probe(*args):
            snapshots.append(args[SUBCYCLE_RUNOFF_ARG].copy())
            return original(*args)

        update_module._subcycle = probe
        self.model.update()

        self.assertEqual(len(snapshots), self.model.config.get('simulation.subcycles'))

        # the rate is consumed by the first subcycle, so the second subcycle must give the
        # flux back rather than carry it forward
        np.testing.assert_allclose(
            snapshots[1] - snapshots[0],
            -self._expected_irrigation_flux(rate),
            rtol=1e-9,
            err_msg='irrigation return flow flux is not the m3/s rate divided by area and land fraction',
        )
        # and with nothing left to apply, later subcycles must not move
        for i in range(2, len(snapshots)):
            np.testing.assert_allclose(
                snapshots[i], snapshots[i - 1], rtol=1e-12,
                err_msg='subsurface runoff changed with no return flow to apply',
            )

    def test_irrigation_return_flow_does_not_compound_across_subcycles(self):
        """With a return flow rate present every subcycle, the flux must not accumulate."""
        self._disable_water_management()
        rate = 10.0
        self.model.state.irrigation_returnflow[:] = rate
        self.model.state.nonirrigation_returnflow[:] = 0.0

        snapshots = []
        original = update_module._subcycle
        self.addCleanup(setattr, update_module, '_subcycle', original)

        def probe(*args):
            snapshots.append(args[SUBCYCLE_RUNOFF_ARG].copy())
            # stand in for the per-subcycle recompute that water management would do
            self.model.state.irrigation_returnflow[:] = rate
            return original(*args)

        update_module._subcycle = probe
        self.model.update()

        self.assertGreater(len(snapshots), 1)
        for i in range(1, len(snapshots)):
            np.testing.assert_allclose(
                snapshots[i], snapshots[0], rtol=1e-12,
                err_msg='irrigation return flow flux compounded across subcycles',
            )

    def test_nonirrigation_return_flow_adds_a_volume_to_channel_storage(self):
        """A m3/s return flow becomes a volume by multiplying the subcycle duration."""
        self._disable_water_management()
        rate = 10.0
        self.model.state.irrigation_returnflow[:] = 0.0
        self.model.state.nonirrigation_returnflow[:] = rate

        subcycle_delta_t = (
            self.model.config.get('simulation.timestep') / self.model.config.get('simulation.subcycles')
        )

        events = []
        original_routing = update_module.main_channel_routing
        original_average = update_module._average_over_routing_iterations
        self.addCleanup(setattr, update_module, 'main_channel_routing', original_routing)
        self.addCleanup(setattr, update_module, '_average_over_routing_iterations', original_average)

        def routing_probe(*args):
            result = original_routing(*args)
            events.append(('routed', self.model.state.channel_storage.copy()))
            return result

        def average_probe(*args):
            # runs just after the return flow block, which rebinds channel_storage
            events.append(('returned', self.model.state.channel_storage.copy()))
            # stand in for the per-subcycle recompute that water management would do
            self.model.state.nonirrigation_returnflow[:] = rate
            return original_average(*args)

        update_module.main_channel_routing = routing_probe
        update_module._average_over_routing_iterations = average_probe
        self.model.update()

        boundaries = [
            (events[i - 1][1], storage)
            for i, (kind, storage) in enumerate(events)
            if kind == 'returned' and i > 0 and events[i - 1][0] == 'routed'
        ]
        self.assertEqual(len(boundaries), self.model.config.get('simulation.subcycles'))
        for routed, returned in boundaries:
            # 21 cells in the packaged grid carry a NaN channel_slope, which makes their
            # routed storage NaN before the return flow block is reached; this is present
            # on main as well and is a property of the test grid, not of return flow
            finite = np.isfinite(routed)
            self.assertGreater(finite.sum(), 0)
            np.testing.assert_allclose(
                returned[finite] - routed[finite],
                rate * subcycle_delta_t,
                rtol=1e-6,
                err_msg='channel return flow is not the m3/s rate times the subcycle duration',
            )
            # and return flow must not spread non-finite values to any further cells
            np.testing.assert_array_equal(
                np.isfinite(returned), finite,
                err_msg='channel return flow introduced non-finite storage',
            )

    def test_channel_geometry_stays_consistent_with_storage(self):
        """Injecting return flow must resync the derived geometry, as channel irrigation does.

        `channel_cross_section_area == channel_storage / channel_length` is an invariant
        that `update_main_channel_state` maintains, and it holds exactly when return flow
        is off. Adding to channel storage without recomputing the geometry breaks it, which
        leaves the next kinematic wave evaluation routing on a stale cross section.
        """
        self.model.update()

        storage = self.model.state.channel_storage
        length = self.model.grid.channel_length
        active = np.isfinite(storage) & (storage > 0.0) & (length > 0.0)
        self.assertGreater(active.sum(), 0)

        np.testing.assert_allclose(
            self.model.state.channel_cross_section_area[active],
            storage[active] / length[active],
            rtol=1e-12,
            err_msg='channel cross section area is stale with respect to channel storage',
        )

    def test_runs_with_irrigation_demand_met_first(self):
        """Cover the irrigation-first arm of the priority logic, which the mask zeros skip."""
        original = self.model.grid.irrigation_first_mask
        self.addCleanup(setattr, self.model.grid, 'irrigation_first_mask', original)
        self.model.grid.irrigation_first_mask = np.ones_like(original)

        self.model.update()

        for name in ('irrigation_returnflow', 'nonirrigation_returnflow',
                     'irrigation_consumption_deficit', 'nonirrigation_consumption_deficit'):
            values = getattr(self.model.state, name)
            self.assertTrue(np.all(np.isfinite(values)), f'{name} is finite with irrigation first')


if __name__ == '__main__':
    unittest.main()
