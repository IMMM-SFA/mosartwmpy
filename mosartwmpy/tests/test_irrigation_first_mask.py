import importlib.resources
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from mosartwmpy.config.config import get_config
from mosartwmpy.grid.grid import Grid


class IrrigationFirstMaskTest(unittest.TestCase):
    """The irrigation-first mask used by return flow must be on the model grid.

    A mask on a different grid flattens to a different size. Unchecked, it passes the
    size-gated trim in `Model.initialize` and only fails on the first timestep, deep inside
    `update`, with a broadcast error that does not name the file.
    """

    GRID_FILE = str(importlib.resources.files('mosartwmpy').joinpath('tests', 'grid.zip'))

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.workdir = Path(cls._tmpdir.name)
        cls.grid = Grid.from_files(cls.GRID_FILE)
        cls.config = get_config()
        cls.lat = cls.config.get('water_management.demand.latitude')
        cls.lon = cls.config.get('water_management.demand.longitude')

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def write_mask(self, name, latitudes, longitudes):
        """Write a mask of ones and zeros on the given coordinates, latitudes descending to exercise the sort."""
        rng = np.random.default_rng(0)
        mask = xr.DataArray(
            rng.integers(0, 2, size=(latitudes.size, longitudes.size)).astype(float),
            dims=(self.lat, self.lon),
            coords={self.lat: latitudes[::-1], self.lon: longitudes},
            name='irrigation_first',
        )
        path = self.workdir / f'{name}.nc'
        mask.to_netcdf(path)
        return str(path), mask

    def config_with_mask(self, path):
        config = get_config()
        config['water_management.demand.return_flow_enabled'] = True
        config['water_management.demand.irrigation_first_mask_path'] = path
        return config

    def test_mask_on_a_coarser_grid_raises_with_both_sizes(self):
        path, mask = self.write_mask('coarse', self.grid.unique_latitudes[::2], self.grid.unique_longitudes[::2])
        with self.assertRaises(ValueError) as context:
            self.grid.load_irrigation_first_mask(self.config_with_mask(path))
        message = str(context.exception)
        self.assertIn(path, message)
        self.assertIn(str(mask.size), message)
        self.assertIn(str(self.grid.id.size), message)

    def test_mask_on_the_model_grid_is_loaded_sorted(self):
        path, mask = self.write_mask('matching', self.grid.unique_latitudes, self.grid.unique_longitudes)
        self.grid.load_irrigation_first_mask(self.config_with_mask(path))
        self.assertEqual(self.grid.irrigation_first_mask.size, self.grid.id.size)
        expected = mask.sortby([self.lat, self.lon]).values.flatten()
        np.testing.assert_array_equal(self.grid.irrigation_first_mask, expected)

    def test_no_mask_path_defaults_to_nonirrigation_first(self):
        self.grid.load_irrigation_first_mask(self.config_with_mask(None))
        self.assertEqual(self.grid.irrigation_first_mask.size, self.grid.id.size)
        self.assertFalse(self.grid.irrigation_first_mask.any())


if __name__ == '__main__':
    unittest.main()
