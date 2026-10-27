import importlib.resources
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid
from mosartwmpy.input.demand import load_demand


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


if __name__ == '__main__':
    unittest.main()
