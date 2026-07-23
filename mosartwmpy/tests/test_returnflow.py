import numpy as np
import tempfile
import unittest

import importlib.resources
from pathlib import Path

import xarray as xr

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid


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


if __name__ == '__main__':
    unittest.main()
