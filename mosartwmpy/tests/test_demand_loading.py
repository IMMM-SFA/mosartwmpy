import importlib.resources
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid
from mosartwmpy.input.demand import CONSUMPTION_TOLERANCE, load_demand


class DemandLoadingTest(unittest.TestCase):
    """Failures to open the demand file are typed errors, not a process exit.

    The loader used to wrap the open and the coordinate sort in a bare except that called
    sys.exit, so a wrong coordinate name was reported as an unopenable file, Ctrl-C was
    swallowed, and the SystemExit bypassed the `except Exception` in Model.update that logs
    a failed timestep before terminating a BMI host process.
    """

    TESTS = importlib.resources.files('mosartwmpy').joinpath('tests')
    GRID_FILE = str(TESTS.joinpath('grid.zip'))
    CONFIG_FILE = str(TESTS.joinpath('test_config.yaml'))
    RUNOFF_FILE = str(TESTS.joinpath('runoff_1981_01_01.nc'))
    DEMAND_FILE = str(TESTS.joinpath('demand_1981_01_01.nc'))

    @classmethod
    def setUpClass(cls):
        cls.model = Model()
        cls.model.initialize(cls.CONFIG_FILE, grid=Grid.from_files(cls.GRID_FILE))
        cls.model.config['runoff.path'] = cls.RUNOFF_FILE
        cls.model.config['water_management.demand.path'] = cls.DEMAND_FILE

    @classmethod
    def tearDownClass(cls):
        cls.model.finalize()

    def set_config(self, key, value):
        original = self.model.config.get(key)
        self.model.config[key] = value
        self.addCleanup(self.model.config.__setitem__, key, original)

    def load(self):
        load_demand(self.model.name, self.model.state, self.model.config, self.model.current_time, None, self.model.mask)

    def test_missing_file_raises_file_not_found(self):
        missing = str(self.TESTS.joinpath('does_not_exist.nc'))
        self.set_config('water_management.demand.path', missing)
        with self.assertRaises(FileNotFoundError) as context:
            self.load()
        # the path is the filename attribute, so it is in the message as well; errno survives
        self.assertEqual(context.exception.filename, missing)
        self.assertIn('Unable to open demand file', str(context.exception))
        self.assertEqual(context.exception.errno, context.exception.__cause__.errno)
        self.assertIsNotNone(context.exception.errno)

    def test_unreadable_file_keeps_permission_error(self):
        """An OSError subclass other than a missing file must not be reported as a missing file."""

        with patch('mosartwmpy.input.demand.open_dataset', side_effect=PermissionError(13, 'Permission denied')):
            with self.assertRaises(PermissionError) as context:
                self.load()
        self.assertNotIsInstance(context.exception, FileNotFoundError)
        self.assertEqual(context.exception.errno, 13)
        self.assertEqual(context.exception.filename, self.DEMAND_FILE)
        self.assertIn('Permission denied', str(context.exception))
        self.assertIsInstance(context.exception.__cause__, PermissionError)

    def test_value_error_subclass_with_its_own_constructor_is_not_masked(self):
        """A ValueError subclass that cannot be rebuilt from a message must not turn into a TypeError."""

        original = UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')
        with patch('mosartwmpy.input.demand.open_dataset', side_effect=original):
            with self.assertRaises(ValueError) as context:
                self.load()
        self.assertIn(self.DEMAND_FILE, str(context.exception))
        self.assertIs(context.exception.__cause__, original)

    def test_corrupt_file_raises_value_error_naming_the_path(self):
        """A file xarray cannot read as netCDF raises ValueError, which must carry the path."""

        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt = str(Path(tmpdir) / 'corrupt.nc')
            Path(corrupt).write_bytes(b'this is not a netcdf file')
            self.set_config('water_management.demand.path', corrupt)
            with self.assertRaises(ValueError) as context:
                self.load()
        self.assertIn(corrupt, str(context.exception))
        self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_wrong_coordinate_name_raises_key_error_naming_it(self):
        self.set_config('water_management.demand.latitude', 'latitude')
        with self.assertRaises(KeyError) as context:
            self.load()
        self.assertIn('latitude', str(context.exception))

    def test_failed_timestep_is_logged_through_model_update(self):
        self.set_config('water_management.demand.latitude', 'latitude')
        with self.assertLogs(level='ERROR') as logs:
            with self.assertRaises(KeyError):
                self.model.update()
        self.assertTrue(any('Failed to complete timestep' in line for line in logs.output))


class ReturnFlowDemandValidationTest(unittest.TestCase):
    """Return flow demand files are checked before their four variables are read.

    A missing variable or a file without a time axis is reported with the config key that
    controls it, and the file is closed on the way out. Consumption above withdrawal would make
    the return flow negative and be removed from the soil column and channel unchecked, so
    consumption is clipped to withdrawal everywhere. Cells over by more than round-off between
    float32 and float64 variables also warn, and a file where most cells with demand are over is
    rejected as misconfigured rather than clipped to zero return flow everywhere.
    """

    TESTS = importlib.resources.files('mosartwmpy').joinpath('tests')
    GRID_FILE = str(TESTS.joinpath('grid.zip'))
    CONFIG_FILE = str(TESTS.joinpath('test_config.yaml'))
    DEMAND_FILE = str(TESTS.joinpath('demand_1981_01_01.nc'))

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.workdir = Path(cls._tmpdir.name)
        cls.model = Model()
        cls.model.initialize(cls.CONFIG_FILE, grid=Grid.from_files(cls.GRID_FILE))
        # the loader alone is exercised here, so return flow is switched on after initialize
        cls.model.config['water_management.demand.return_flow_enabled'] = True

    @classmethod
    def tearDownClass(cls):
        cls.model.finalize()
        cls._tmpdir.cleanup()

    def disaggregated(self, name, consumption_factor=0.4, drop=None, no_time=False, over_cells=0, withdrawal_dtype=None):
        """Split the packaged total demand into the four return flow variables.

        `over_cells` sets irrigation consumption to 1.5 times withdrawal in that many of the
        cells with demand; `withdrawal_dtype` stores the two withdrawal variables in that type.
        """
        path = self.workdir / f'{name}.nc'
        with xr.open_dataset(self.DEMAND_FILE) as demand:
            total = demand['totalDemand']
            irrigation, nonirrigation = total * 0.6, total * 0.4
            irrigation_consumption = irrigation * consumption_factor
            if over_cells:
                flat = irrigation.isel(time=0).values.flatten()
                over = np.zeros(flat.shape, dtype=bool)
                over[np.flatnonzero(flat > 0)[:over_cells]] = True
                over = xr.DataArray(over.reshape(irrigation.isel(time=0).shape), dims=irrigation.isel(time=0).dims)
                irrigation_consumption = irrigation_consumption.where(~over, irrigation * 1.5)
            if withdrawal_dtype is not None:
                irrigation, nonirrigation = irrigation.astype(withdrawal_dtype), nonirrigation.astype(withdrawal_dtype)
            dataset = xr.Dataset({
                'irrigation_withdrawal': irrigation,
                'irrigation_consumption': irrigation_consumption,
                'nonirrigation_withdrawal': nonirrigation,
                'nonirrigation_consumption': nonirrigation * 0.7,
            })
            if drop:
                dataset = dataset.drop_vars(drop)
            if no_time:
                dataset = dataset.isel(time=0, drop=True)
            dataset.to_netcdf(path)
        return str(path)

    def load(self, path):
        self.model.config['water_management.demand.path'] = path
        load_demand(self.model.name, self.model.state, self.model.config, self.model.current_time, None, self.model.mask)

    def test_well_formed_file_loads_without_warnings(self):
        with self.assertNoLogs(level='WARNING'):
            self.load(self.disaggregated('demand_ok'))
        state = self.model.state
        self.assertTrue(np.any(state.irrigation_withdrawal_rate > 0))
        self.assertTrue(np.all(state.irrigation_consumption_rate <= state.irrigation_withdrawal_rate))

    def test_missing_variable_names_it_and_its_config_key(self):
        path = self.disaggregated('demand_missing_variable', drop=['irrigation_consumption'])
        with self.assertRaises(ValueError) as context:
            self.load(path)
        message = str(context.exception)
        self.assertIn('irrigation_consumption', message)
        self.assertIn('water_management.demand.irrigation_consumption', message)
        self.assertIn(path, message)

    def test_file_without_a_time_axis_is_rejected(self):
        path = self.disaggregated('demand_no_time', no_time=True)
        with self.assertRaises(ValueError) as context:
            self.load(path)
        message = str(context.exception)
        self.assertIn('water_management.demand.time', message)
        self.assertIn(path, message)

    def test_file_is_closed_when_validation_fails(self):
        path = self.disaggregated('demand_closed_on_error', drop=['irrigation_consumption'])
        with patch.object(xr.Dataset, 'close', autospec=True, side_effect=xr.Dataset.close) as close:
            with self.assertRaises(ValueError):
                self.load(path)
        self.assertTrue(close.called, 'the demand dataset was not closed after the validation error')

    def test_a_few_cells_with_consumption_above_withdrawal_are_clipped_with_a_warning(self):
        path = self.disaggregated('demand_consumption_above_withdrawal', over_cells=20)
        with self.assertLogs(level='WARNING') as logs:
            self.load(path)
        self.assertTrue(any(
            line.startswith('WARNING') and '20 cell(s)' in line and 'irrigation consumption above withdrawal' in line and 'clipped' in line
            for line in logs.output
        ))
        state = self.model.state
        clipped = np.isclose(state.irrigation_consumption_rate, state.irrigation_withdrawal_rate) & (state.irrigation_withdrawal_rate > 0)
        self.assertEqual(int(clipped.sum()), 20)
        # the other cells keep the consumption the file gave them
        np.testing.assert_allclose(state.irrigation_consumption_rate[~clipped], 0.4 * state.irrigation_withdrawal_rate[~clipped])
        # the return flow is withdrawal minus consumption and can no longer be negative anywhere
        self.assertTrue(np.all(state.irrigation_withdrawal_rate - state.irrigation_consumption_rate >= 0.0))
        self.assertTrue(np.all(state.nonirrigation_withdrawal_rate - state.nonirrigation_consumption_rate >= 0.0))
        # the untouched nonirrigation variables are unchanged
        np.testing.assert_allclose(state.nonirrigation_consumption_rate, 0.7 * state.nonirrigation_withdrawal_rate)

    def test_round_off_between_float32_and_float64_variables_is_not_a_violation(self):
        """Withdrawal stored as float32 beside a float64 consumption that means "fully consumed"."""
        path = self.disaggregated('demand_mixed_precision', consumption_factor=1.0, withdrawal_dtype=np.float32)

        # the round-off is really in the file, so an exact comparison would have fired, and all of
        # it is inside the tolerance the loader applies
        with xr.open_dataset(path) as demand:
            withdrawal = np.nan_to_num(demand['irrigation_withdrawal'].values.astype(np.float64))
            consumption = np.nan_to_num(demand['irrigation_consumption'].values.astype(np.float64))
        excess = consumption - withdrawal
        self.assertTrue(np.any(excess > 0.0))
        self.assertTrue(np.all(excess <= CONSUMPTION_TOLERANCE * np.maximum(withdrawal, 1.0)))

        with self.assertNoLogs(level='WARNING'):
            self.load(path)
        state = self.model.state
        # nothing was counted as a violation, and the clip still leaves no negative return flow
        self.assertTrue(np.all(state.irrigation_withdrawal_rate - state.irrigation_consumption_rate >= 0.0))

    def test_file_where_most_cells_violate_is_rejected(self):
        path = self.disaggregated('demand_mostly_over', consumption_factor=1.5)
        with self.assertRaises(ValueError) as context:
            self.load(path)
        message = str(context.exception)
        self.assertIn(path, message)
        self.assertIn('irrigation demand', message)
        self.assertIn('water_management.demand.irrigation_withdrawal', message)
        self.assertIn('water_management.demand.irrigation_consumption', message)

    def test_swapped_withdrawal_and_consumption_names_are_rejected(self):
        path = self.disaggregated('demand_swapped_names')
        for key, value in (('irrigation_withdrawal', 'irrigation_consumption'), ('irrigation_consumption', 'irrigation_withdrawal')):
            original = self.model.config.get(f'water_management.demand.{key}')
            self.model.config[f'water_management.demand.{key}'] = value
            self.addCleanup(self.model.config.__setitem__, f'water_management.demand.{key}', original)
        with self.assertRaises(ValueError) as context:
            self.load(path)
        self.assertIn('not swapped', str(context.exception))


if __name__ == '__main__':
    unittest.main()
