import datetime
import importlib.resources
import io
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid


class GridCacheTest(unittest.TestCase):
    """Grid caches written by an older mosartwmpy lack arrays added since.

    `Grid.from_files` leaves such arrays at their size-0 class default, and the numba kernels
    do no bounds checking, so without a guard they read past the end of the array silently.
    `Grid.validate`, called from `Model.initialize`, fills in `reservoir_minimum_storage` with
    its historical value and rejects anything else.

    The packaged cache `tests/grid.zip` is the full CONUS NLDAS domain built with the default
    configuration (water management on, return flow off) and the reservoir fixtures in this
    directory, written with `Grid.to_files`.
    """

    TESTS = importlib.resources.files('mosartwmpy').joinpath('tests')
    GRID_FILE = str(TESTS.joinpath('grid.zip'))

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.workdir = Path(cls._tmpdir.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def write_config(self, name, return_flow=False):
        """Write a config equivalent to tests/test_config.yaml with output under the temp dir."""
        config = {
            'simulation': {
                'name': name,
                'start_date': datetime.date(1981, 1, 1),
                'end_date': datetime.date(1981, 1, 31),
                'output_path': str(self.workdir / 'output'),
                'log_to_std_out': False,
                'log_to_file': False,
            },
            'runoff': {'path': str(self.TESTS.joinpath('runoff_1981_01_01.nc'))},
            'water_management': {
                'demand': {
                    'path': str(self.TESTS.joinpath('demand_1981_01_01.nc')),
                    'return_flow_enabled': return_flow,
                    'farmer_abm': {'enabled': False},
                },
                'reservoirs': {
                    'parameters': {'path': str(self.TESTS.joinpath('reservoirs.nc'))},
                    'dependencies': {'path': str(self.TESTS.joinpath('dependency_database.parquet'))},
                    'streamflow': {'path': str(self.TESTS.joinpath('mean_flow.parquet'))},
                    'demand': {'path': str(self.TESTS.joinpath('mean_demand.parquet'))},
                },
            },
        }
        path = self.workdir / f'{name}.yaml'
        path.write_text(yaml.safe_dump(config))
        return str(path)

    def write_old_style_cache(self, name, drop_columns, keep_version=False):
        """Copy the packaged cache without the given columns and, unless keep_version, without its version stamp."""
        path = self.workdir / f'{name}.zip'
        with zipfile.ZipFile(self.GRID_FILE) as source, zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as target:
            for entry in source.namelist():
                data = source.read(entry)
                if entry == 'special.pickle' and not keep_version:
                    special = pickle.loads(data)
                    special.pop('cache_version', None)
                    data = pickle.dumps(special)
                elif entry == 'np.feather':
                    frame = pd.read_feather(io.BytesIO(data)).drop(columns=list(drop_columns))
                    buffer = io.BytesIO()
                    frame.to_feather(buffer)
                    data = buffer.getvalue()
                target.writestr(entry, data)
        return str(path)

    def test_packaged_cache_is_current(self):
        with self.assertNoLogs(level='WARNING'):
            grid = Grid.from_files(self.GRID_FILE)
        self.assertEqual(grid.cache_version, Grid.CACHE_VERSION)
        self.assertEqual(grid.reservoir_minimum_storage.size, grid.cell_count)

    def test_old_cache_without_minimum_storage_falls_back(self):
        cache = self.write_old_style_cache('old_minimum_storage', ['reservoir_minimum_storage'])
        with self.assertLogs(level='WARNING') as logs:
            grid = Grid.from_files(cache)
        self.assertEqual(grid.cache_version, 0)
        self.assertEqual(grid.reservoir_minimum_storage.size, 0)
        self.assertTrue(any('reservoir_minimum_storage' in line for line in logs.output), 'from_files names the missing array')

        model = Model()
        with self.assertLogs(level='WARNING') as logs:
            model.initialize(self.write_config('old_minimum_storage'), grid=grid)
        self.assertTrue(any('reservoir_minimum_storage' in line for line in logs.output), 'initialize warns about the fallback')
        n = int(model.mask.sum())
        self.assertEqual(model.grid.reservoir_minimum_storage.size, n)
        np.testing.assert_array_equal(
            model.grid.reservoir_minimum_storage,
            model.parameters.reservoir_default_minimum_storage_fraction * model.grid.reservoir_storage_capacity,
            'fallback is the historical fraction of storage capacity',
        )
        # the regulation kernel reads the array on the first step; the packaged reservoir fixture has
        # no CAP_MIN column, so the current cache holds exactly the fallback values and one step on
        # the old cache must reproduce one step on the current cache
        model.update()
        reference = Model()
        reference.initialize(self.write_config('current_minimum_storage'), grid=Grid.from_files(self.GRID_FILE))
        reference.update()
        np.testing.assert_array_equal(model.state.reservoir_storage, reference.state.reservoir_storage)
        np.testing.assert_array_equal(model.state.channel_outflow_downstream, reference.state.channel_outflow_downstream)
        model.finalize()
        reference.finalize()

    def test_old_cache_missing_required_array_raises(self):
        cache = self.write_old_style_cache('old_channel_width', ['channel_width'])
        with self.assertLogs(level='WARNING'):
            grid = Grid.from_files(cache)
        model = Model()
        with self.assertRaises(ValueError) as context:
            model.initialize(self.write_config('old_channel_width'), grid=grid)
        message = str(context.exception)
        self.assertIn('channel_width', message)
        self.assertIn('predates this array', message)
        self.assertIn('Rebuild the grid cache', message)

    def test_current_cache_missing_required_array_is_not_called_outdated(self):
        """A cache with the current version lacks an array only because it was empty when written."""

        cache = self.write_old_style_cache('current_channel_width', ['channel_width'], keep_version=True)
        with self.assertNoLogs(level='WARNING'):
            grid = Grid.from_files(cache)
        self.assertEqual(grid.cache_version, Grid.CACHE_VERSION)
        model = Model()
        with self.assertRaises(ValueError) as context:
            model.initialize(self.write_config('current_channel_width'), grid=grid)
        message = str(context.exception)
        self.assertIn('channel_width', message)
        self.assertNotIn('predates', message)
        self.assertIn('written with the feature that fills this array disabled', message)
        self.assertIn('Rebuild the grid cache', message)

    def test_return_flow_on_a_cache_written_without_it_defaults_the_mask(self):
        """The packaged cache was written with return flow off, so it has no irrigation-first mask.

        With return flow enabled the mask must default to nonirrigation-first everywhere, as a
        freshly built grid with no mask path does, instead of failing with a message that asks for
        a rebuild that would produce the same cache.
        """

        grid = Grid.from_files(self.GRID_FILE)
        self.assertEqual(grid.irrigation_first_mask.size, 0)
        model = Model()
        with self.assertLogs(level='INFO') as logs:
            model.initialize(self.write_config('return_flow_mask_default', return_flow=True), grid=grid)
        self.assertTrue(any('irrigation_first_mask' in line and 'return flow off' in line for line in logs.output))
        n = int(model.mask.sum())
        self.assertEqual(model.grid.irrigation_first_mask.shape, (n,))
        self.assertFalse(model.grid.irrigation_first_mask.any())
        model.finalize()

    def test_outlet_with_nonexistent_nonnegative_downstream_id_does_not_loop(self):
        """An outlet cell whose raw dnID >= 0 points outside the domain is treated as an outlet (#108).

        Previously, `land_mask` was computed before `downstream_id` was remapped through `id_hashmap`,
        leaving `land_mask == 1` with `downstream_id == -1`. In the downstream traversal loop `j = -1`
        then indexed the last grid cell (`self.land_mask[-1]`), causing an infinite loop when the last
        cell drained into the boundary cell.
        """
        import xarray as xr
        from benedict.dicts import benedict as Benedict
        from mosartwmpy.config.parameters import Parameters

        # 2x2 synthetic domain: cell 3 (last cell) drains to cell 1, and cell 1 has dnID=999
        # (nonnegative, but absent from ID=[10, 20, 30, 40]). Under the old code:
        # cell 1 got land_mask=1 and downstream_id=-1, so tracing cell 1 or cell 3 jumped to
        # j=-1 (cell 3, land_mask=1) -> cell 1 -> j=-1 in an infinite loop.
        lat = np.array([45.0, 46.0])
        lon = np.array([-120.0, -119.0])
        ones = np.ones((2, 2), dtype=np.float64)
        ds = xr.Dataset(
            data_vars={
                'ID': (('lat', 'lon'), np.array([[10, 20], [30, 40]], dtype=np.int64)),
                'dnID': (('lat', 'lon'), np.array([[-9999, 999], [-9999, 20]], dtype=np.int64)),
                'fdir': (('lat', 'lon'), np.array([[0, 1], [-1, 1]], dtype=np.int64)),
                'frac': (('lat', 'lon'), np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)),
                'area': (('lat', 'lon'), 1.0e8 * ones),
                'areaTotal': (('lat', 'lon'), 2.0e8 * ones),
                'areaTotal2': (('lat', 'lon'), 2.0e8 * ones),
                'nh': (('lat', 'lon'), 0.1 * ones),
                'nt': (('lat', 'lon'), 0.05 * ones),
                'nr': (('lat', 'lon'), 0.03 * ones),
                'hslp': (('lat', 'lon'), 0.01 * ones),
                'gxr': (('lat', 'lon'), 1.0e-3 * ones),
                'tslp': (('lat', 'lon'), 0.005 * ones),
                'twid': (('lat', 'lon'), 10.0 * ones),
                'rlen': (('lat', 'lon'), 1.0e4 * ones),
                'rslp': (('lat', 'lon'), 0.001 * ones),
                'rwid': (('lat', 'lon'), 50.0 * ones),
                'rwid0': (('lat', 'lon'), 150.0 * ones),
                'rdep': (('lat', 'lon'), 2.0 * ones),
            },
            coords={'lat': lat, 'lon': lon},
        )
        grid_nc = self.workdir / 'boundary_outlet_grid.nc'
        ds.to_netcdf(grid_nc)

        defaults = Path(str(importlib.resources.files('mosartwmpy').joinpath('config_defaults.yaml')))
        config = Benedict(yaml.safe_load(defaults.read_text()))
        config['grid.path'] = str(grid_nc)
        config['water_management.enabled'] = False

        grid = Grid(config, Parameters())
        # Cell 1 (ID=20, dnID=999 outside domain) must be classified as an outlet (land_mask=3, mosart_mask=2)
        # and both cell 3 (ID=40 -> ID=20) and cell 1 must have outlet_id == 1.
        self.assertEqual(grid.downstream_id[1], -1)
        self.assertEqual(grid.downstream_id[3], 1)
        self.assertEqual(grid.land_mask[1], 3)
        self.assertEqual(grid.land_mask[3], 1)
        self.assertEqual(grid.mosart_mask[1], 2)
        self.assertEqual(grid.outlet_id[1], 1)
        self.assertEqual(grid.outlet_id[3], 1)
        self.assertEqual(grid.upstream_cell_count[1], 2)


if __name__ == '__main__':
    unittest.main()
