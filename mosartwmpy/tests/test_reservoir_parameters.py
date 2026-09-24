import numpy as np
import pandas as pd
import unittest
import yaml

import importlib.resources

from pathlib import Path
from tempfile import TemporaryDirectory
from xarray import open_dataset

from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.reservoirs.grid import load_reservoirs
from mosartwmpy.reservoirs.state import _load_initial_storage


class ReservoirParametersTest(unittest.TestCase):
    """Test the reservoir parameter file formats, minimum storage, and initial storage."""

    # package data
    DEFAULTS_FILE = str(importlib.resources.files('mosartwmpy').joinpath('config_defaults.yaml'))
    RESERVOIRS_FILE = str(importlib.resources.files('mosartwmpy').joinpath('tests', 'reservoirs.nc'))
    DEPENDENCIES_FILE = str(importlib.resources.files('mosartwmpy').joinpath('tests', 'dependency_database.parquet'))
    FLOW_FILE = str(importlib.resources.files('mosartwmpy').joinpath('tests', 'mean_flow.parquet'))
    DEMAND_FILE = str(importlib.resources.files('mosartwmpy').joinpath('tests', 'mean_demand.parquet'))

    # the sample reservoir file covers a grid larger than the reservoir count
    GRID_SIZE = 20000

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = TemporaryDirectory()
        base = Path(cls.tmpdir.name)

        # the bundled sample file predates the minimum storage field, so build the
        # variants used by these tests from it, adding CAP_MIN where needed
        with open_dataset(cls.RESERVOIRS_FILE) as ds:
            frame = ds.to_dataframe()

        cls.no_cap_min_path = str(base / 'reservoirs_no_cap_min.nc')
        frame.to_xarray().to_netcdf(cls.no_cap_min_path)

        with_cap_min = frame.copy()
        with_cap_min['CAP_MIN'] = 0.25 * with_cap_min['CAP_MCM']

        cls.paths_with_cap_min = {
            '.nc': str(base / 'reservoirs.nc'),
            '.csv': str(base / 'reservoirs.csv'),
            '.parquet': str(base / 'reservoirs.parquet'),
        }
        with_cap_min.to_xarray().to_netcdf(cls.paths_with_cap_min['.nc'])
        with_cap_min.to_csv(cls.paths_with_cap_min['.csv'], index=False)
        with_cap_min.to_parquet(cls.paths_with_cap_min['.parquet'], index=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def make_config(self, reservoir_path):
        """Builds a config from the packaged defaults, pointed at the test reservoir files."""
        with open(self.DEFAULTS_FILE) as file:
            config = Benedict(yaml.safe_load(file))
        config['water_management.reservoirs.parameters.path'] = reservoir_path
        config['water_management.reservoirs.dependencies.path'] = self.DEPENDENCIES_FILE
        config['water_management.reservoirs.streamflow.path'] = self.FLOW_FILE
        config['water_management.reservoirs.demand.path'] = self.DEMAND_FILE
        return config

    def load(self, reservoir_path):
        """Loads reservoirs onto an otherwise empty grid."""
        grid = Grid(empty=True)
        grid.id = np.arange(self.GRID_SIZE)
        load_reservoirs(grid, self.make_config(reservoir_path), Parameters())
        return grid

    def test_file_formats_are_equivalent(self):
        """netCDF, csv, and parquet reservoir files should produce an identical grid."""

        grids = {suffix: self.load(path) for suffix, path in self.paths_with_cap_min.items()}
        reference = grids['.nc']

        # the reservoir arrays that come straight from the parameter file
        keys = [
            key for key in self.make_config(
                self.paths_with_cap_min['.nc']
            ).get('water_management.reservoirs.parameters.variables').keys()
        ] + ['reservoir_minimum_storage']

        for suffix, grid in grids.items():
            if suffix == '.nc':
                continue
            for key in keys:
                expected = getattr(reference, key)
                actual = getattr(grid, key)
                self.assertEqual(expected.shape, actual.shape, f'{key} shape differs for {suffix}')
                if np.issubdtype(expected.dtype, np.number):
                    np.testing.assert_allclose(
                        actual.astype(np.float64),
                        expected.astype(np.float64),
                        equal_nan=True,
                        err_msg=f'{key} differs for {suffix}',
                    )
                else:
                    # object/string columns, compared with nulls normalized
                    self.assertTrue(
                        pd.Series(actual).astype('object').equals(pd.Series(expected).astype('object')),
                        f'{key} differs for {suffix}',
                    )

    def test_minimum_storage_read_from_file(self):
        """CAP_MIN in the parameter file should set the minimum storage, converted to m3."""

        grid = self.load(self.paths_with_cap_min['.nc'])
        is_reservoir = np.isfinite(grid.reservoir_storage_capacity) & (grid.reservoir_storage_capacity > 0)

        self.assertGreater(is_reservoir.sum(), 0, 'sample data contains reservoirs')
        # CAP_MIN was written as 0.25 * CAP_MCM, and both are scaled from million m3 to m3
        np.testing.assert_allclose(
            grid.reservoir_minimum_storage[is_reservoir],
            0.25 * grid.reservoir_storage_capacity[is_reservoir],
        )

    def test_minimum_storage_falls_back_without_cap_min(self):
        """A parameter file with no CAP_MIN column should fall back to the historical default."""

        grid = self.load(self.no_cap_min_path)
        is_reservoir = np.isfinite(grid.reservoir_storage_capacity) & (grid.reservoir_storage_capacity > 0)

        self.assertGreater(is_reservoir.sum(), 0, 'sample data contains reservoirs')
        np.testing.assert_allclose(
            grid.reservoir_minimum_storage[is_reservoir],
            Parameters().reservoir_runoff_capacity_parameter * grid.reservoir_storage_capacity[is_reservoir],
        )

    def test_minimum_storage_falls_back_per_reservoir(self):
        """Reservoirs with a missing CAP_MIN value should fall back individually."""

        with open_dataset(self.paths_with_cap_min['.nc']) as ds:
            frame = ds.to_dataframe()
        # blank out CAP_MIN for a few reservoirs that land on the test grid
        on_grid = frame[frame['GRID_CELL_INDEX'].between(0, self.GRID_SIZE - 1)]
        blanked_ids = on_grid['GRAND_ID'].values[:5]
        frame.loc[frame['GRAND_ID'].isin(blanked_ids), 'CAP_MIN'] = np.nan
        path = str(Path(self.tmpdir.name) / 'reservoirs_partial_cap_min.csv')
        frame.to_csv(path, index=False)

        grid = self.load(path)
        blanked = np.isin(grid.reservoir_id, blanked_ids)
        capacity = grid.reservoir_storage_capacity

        self.assertGreater(blanked.sum(), 0, 'blanked reservoirs are present on the grid')
        np.testing.assert_allclose(
            grid.reservoir_minimum_storage[blanked],
            Parameters().reservoir_runoff_capacity_parameter * capacity[blanked],
        )

        supplied = np.isfinite(capacity) & (capacity > 0) & ~blanked
        np.testing.assert_allclose(
            grid.reservoir_minimum_storage[supplied],
            0.25 * capacity[supplied],
        )

    def test_minimum_storage_falls_back_on_non_positive(self):
        """A CAP_MIN of zero or below is missing data, not a real zero floor.

        The published sample data carries CAP_MIN = 0 for a handful of reservoirs.
        Honoring that literally would let them draw down to empty, which is a
        regression against the 10%-of-capacity floor that applied before this
        column was read.
        """

        with open_dataset(self.paths_with_cap_min['.nc']) as ds:
            frame = ds.to_dataframe()
        on_grid = frame[frame['GRID_CELL_INDEX'].between(0, self.GRID_SIZE - 1)]
        zeroed_ids = on_grid['GRAND_ID'].values[:3]
        negative_ids = on_grid['GRAND_ID'].values[3:5]
        frame.loc[frame['GRAND_ID'].isin(zeroed_ids), 'CAP_MIN'] = 0.0
        frame.loc[frame['GRAND_ID'].isin(negative_ids), 'CAP_MIN'] = -1.0
        path = str(Path(self.tmpdir.name) / 'reservoirs_non_positive_cap_min.csv')
        frame.to_csv(path, index=False)

        grid = self.load(path)
        capacity = grid.reservoir_storage_capacity
        expected = Parameters().reservoir_runoff_capacity_parameter

        for label, ids in (('zeroed', zeroed_ids), ('negative', negative_ids)):
            selected = np.isin(grid.reservoir_id, ids)
            self.assertGreater(selected.sum(), 0, f'{label} reservoirs are present on the grid')
            np.testing.assert_allclose(
                grid.reservoir_minimum_storage[selected],
                expected * capacity[selected],
                err_msg=f'{label} CAP_MIN should fall back to the historical default',
            )

        # reservoirs with a real value are untouched
        overridden = np.isin(grid.reservoir_id, np.concatenate([zeroed_ids, negative_ids]))
        supplied = np.isfinite(capacity) & (capacity > 0) & ~overridden
        np.testing.assert_allclose(
            grid.reservoir_minimum_storage[supplied],
            0.25 * capacity[supplied],
        )

    def test_minimum_storage_is_declared_on_grid(self):
        """The field must exist on a grid loaded from cache, which skips load_reservoirs."""

        self.assertTrue(hasattr(Grid(empty=True), 'reservoir_minimum_storage'))


class InitialStorageTest(unittest.TestCase):
    """Test the optional initial reservoir storage file."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.grid = Grid(empty=True)
        # a realistic grid: mostly non-reservoir cells, NaN padded
        self.grid.reservoir_id = np.array([np.nan, 7.0, np.nan, 9.0])
        self.grid.reservoir_grid_index = np.array([np.nan, 1.0, np.nan, 3.0])
        self.default = np.array([1.0e6, 2.0e6, 3.0e6, 4.0e6])

    def tearDown(self):
        self.tmpdir.cleanup()

    def make_config(self, path):
        return Benedict({'water_management': {'reservoirs': {
            'initial_storage': {'path': path},
            'parameters': {
                'grid_cell_index': 'GRID_CELL_INDEX',
                'variables': {'reservoir_id': 'GRAND_ID'},
            },
        }}})

    def write(self, frame, name='initial_storage.csv'):
        path = str(Path(self.tmpdir.name) / name)
        if name.endswith('.parquet'):
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        return path

    def load(self, path):
        return _load_initial_storage(self.grid, self.make_config(path), self.default.copy())

    def test_no_path_returns_default(self):
        self.assertIs(_load_initial_storage(self.grid, self.make_config(None), self.default), self.default)

    def test_missing_file_returns_default(self):
        path = str(Path(self.tmpdir.name) / 'absent.csv')
        np.testing.assert_array_equal(self.load(path), self.default)

    def test_matches_on_reservoir_id(self):
        """CAP_INIT is in million m3 and applies only to matched reservoirs."""

        path = self.write(pd.DataFrame({'GRAND_ID': [7], 'CAP_INIT': [50.0]}))
        np.testing.assert_allclose(self.load(path), [1.0e6, 50.0e6, 3.0e6, 4.0e6])

    def test_matches_from_parquet(self):
        path = self.write(pd.DataFrame({'GRAND_ID': [7], 'CAP_INIT': [50.0]}), 'initial_storage.parquet')
        np.testing.assert_allclose(self.load(path), [1.0e6, 50.0e6, 3.0e6, 4.0e6])

    def test_falls_through_to_grid_cell_index(self):
        """A present-but-unmatched identifier should not stop the search."""

        path = self.write(pd.DataFrame({
            'GRAND_ID': [999], 'GRID_CELL_INDEX': [3], 'CAP_INIT': [77.0],
        }))
        np.testing.assert_allclose(self.load(path), [1.0e6, 2.0e6, 3.0e6, 77.0e6])

    def test_duplicate_identifiers_keep_last(self):
        """Duplicate ids should be tolerated rather than raising."""

        path = self.write(pd.DataFrame({'GRAND_ID': [7, 7], 'CAP_INIT': [50.0, 60.0]}))
        np.testing.assert_allclose(self.load(path), [1.0e6, 60.0e6, 3.0e6, 4.0e6])

    def test_no_overlap_returns_default(self):
        path = self.write(pd.DataFrame({'GRAND_ID': [999], 'CAP_INIT': [77.0]}))
        np.testing.assert_array_equal(self.load(path), self.default)

    def test_missing_cap_init_column_returns_default(self):
        path = self.write(pd.DataFrame({'GRAND_ID': [7], 'STORAGE': [50.0]}))
        np.testing.assert_array_equal(self.load(path), self.default)

    def test_unsupported_format_returns_default(self):
        path = str(Path(self.tmpdir.name) / 'initial_storage.txt')
        Path(path).write_text('GRAND_ID,CAP_INIT\n7,50.0\n')
        np.testing.assert_array_equal(self.load(path), self.default)

    def test_does_not_mutate_default(self):
        """The caller's default array must not be modified in place."""

        path = self.write(pd.DataFrame({'GRAND_ID': [7], 'CAP_INIT': [50.0]}))
        default = self.default.copy()
        _load_initial_storage(self.grid, self.make_config(path), default)
        np.testing.assert_array_equal(default, self.default)


if __name__ == '__main__':
    unittest.main()
