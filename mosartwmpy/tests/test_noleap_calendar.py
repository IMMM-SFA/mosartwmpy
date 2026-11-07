import importlib.resources
import tempfile
import unittest
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid
from mosartwmpy.input.demand import load_demand
from mosartwmpy.input.runoff import load_runoff


class NoLeapCalendarTest(unittest.TestCase):
    """Runoff and demand files on a non-standard calendar are converted to a datetime index.

    The packaged fixtures use a standard calendar, so without these tests the conversion
    branch in each loader never runs. Both loaders must convert the same way, with the
    `time_unit` keyword that fixes the xarray floor declared in setup.py.
    """

    TESTS = importlib.resources.files('mosartwmpy').joinpath('tests')
    GRID_FILE = str(TESTS.joinpath('grid.zip'))
    CONFIG_FILE = str(TESTS.joinpath('test_config.yaml'))
    RUNOFF_FILE = str(TESTS.joinpath('runoff_1981_01_01.nc'))
    DEMAND_FILE = str(TESTS.joinpath('demand_1981_01_01.nc'))

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        workdir = Path(cls._tmpdir.name)
        cls.NOLEAP_RUNOFF_FILE = cls.write_noleap_copy(cls.RUNOFF_FILE, workdir / 'runoff_noleap.nc')
        cls.NOLEAP_DEMAND_FILE = cls.write_noleap_copy(cls.DEMAND_FILE, workdir / 'demand_noleap.nc')

        cls.model = Model()
        cls.model.initialize(cls.CONFIG_FILE, grid=Grid.from_files(cls.GRID_FILE))
        cls.model.config['runoff.path'] = cls.RUNOFF_FILE
        cls.model.config['water_management.demand.path'] = cls.DEMAND_FILE

    @classmethod
    def tearDownClass(cls):
        cls.model.finalize()
        cls._tmpdir.cleanup()

    @staticmethod
    def write_noleap_copy(source, destination):
        """Copy a fixture with its single time step re-dated to 1981-01-01 on the noleap calendar."""
        with xr.open_dataset(source) as dataset:
            noleap = dataset.load().assign_coords(time=[cftime.DatetimeNoLeap(1981, 1, 1)])
            noleap.to_netcdf(destination)
        with xr.open_dataset(destination) as check:
            # precondition: the file decodes to a CFTimeIndex, so the loaders take the conversion branch
            assert not isinstance(check.indexes['time'], pd.DatetimeIndex)
        return str(destination)

    def load_runoff_from(self, path):
        self.model.config['runoff.path'] = path
        self.addCleanup(self.model.config.__setitem__, 'runoff.path', self.RUNOFF_FILE)
        load_runoff(self.model.state, self.model.grid, self.model.config, self.model.current_time, self.model.mask)
        return self.model.state.hillslope_surface_runoff.copy(), self.model.state.hillslope_subsurface_runoff.copy()

    def load_demand_from(self, path):
        self.model.config['water_management.demand.path'] = path
        self.addCleanup(self.model.config.__setitem__, 'water_management.demand.path', self.DEMAND_FILE)
        load_demand(self.model.name, self.model.state, self.model.config, self.model.current_time, None, self.model.mask)
        return self.model.state.grid_cell_demand_rate.copy()

    def test_noleap_runoff_matches_standard_calendar_runoff(self):
        standard_surface, standard_subsurface = self.load_runoff_from(self.RUNOFF_FILE)
        noleap_surface, noleap_subsurface = self.load_runoff_from(self.NOLEAP_RUNOFF_FILE)
        self.assertTrue(np.any(standard_surface != 0.0), 'fixture carries runoff')
        np.testing.assert_array_equal(noleap_surface, standard_surface)
        np.testing.assert_array_equal(noleap_subsurface, standard_subsurface)

    def test_noleap_demand_matches_standard_calendar_demand(self):
        standard = self.load_demand_from(self.DEMAND_FILE)
        noleap = self.load_demand_from(self.NOLEAP_DEMAND_FILE)
        self.assertTrue(np.any(standard != 0.0), 'fixture carries demand')
        np.testing.assert_array_equal(noleap, standard)


if __name__ == '__main__':
    unittest.main()
