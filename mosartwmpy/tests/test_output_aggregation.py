import datetime
import importlib.resources
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid


class OutputAggregationTest(unittest.TestCase):
    """The `aggregation` key on `simulation.output` entries is optional and defaults to `mean`.

    User configs that list their own outputs replace the default list wholesale, so their
    entries normally carry no `aggregation` key. Demand is loaded once per month and is
    constant across a day, so the written daily value equals the state value under `mean`
    and `steps_per_day` times the state value under `sum`.
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

    def write_config(self, name, output):
        """Write a config equivalent to tests/test_config.yaml with the given output list."""
        config = {
            'simulation': {
                'name': name,
                'start_date': datetime.date(1981, 1, 1),
                'end_date': datetime.date(1981, 1, 31),
                'output_path': str(self.workdir / 'output'),
                'log_to_std_out': False,
                'log_to_file': False,
                'output': output,
            },
            'runoff': {'path': str(self.TESTS.joinpath('runoff_1981_01_01.nc'))},
            'water_management': {
                'demand': {
                    'path': str(self.TESTS.joinpath('demand_1981_01_01.nc')),
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

    def test_missing_aggregation_key_writes_daily_mean(self):
        output = [
            {
                # user-style entry: no `aggregation` key
                'variable': 'grid_cell_demand_rate',
                'name': 'WRM_DEMAND',
                'units': 'm3/s',
            },
            {
                # same variable, explicitly summed, to pin the other branch
                'variable': 'grid_cell_demand_rate',
                'name': 'WRM_DEMAND_SUM',
                'units': 'm3/s',
                'aggregation': 'sum',
            },
        ]
        model = Model()
        # a fresh grid per model: initialize trims grid arrays in place
        model.initialize(self.write_config('aggregation_default', output), grid=Grid.from_files(self.GRID_FILE))
        steps_per_day = 86400 // int(model.config.get('simulation.timestep'))
        for _ in range(steps_per_day):
            model.update()
        model.finalize()

        files = sorted((Path(model.config.get('simulation.output_path')) / model.name).glob('*.nc'))
        self.assertEqual(len(files), 1, 'one daily output file is written')
        with xr.open_dataset(files[0]) as ds:
            written_mean = ds['WRM_DEMAND'].isel(time=0).values.flatten()
            written_sum = ds['WRM_DEMAND_SUM'].isel(time=0).values.flatten()
        state = model.unmask(model.state.grid_cell_demand_rate)
        active = np.isfinite(state) & (state > 0)
        self.assertGreater(active.sum(), 0, 'some cells have demand')

        # output files are float32
        np.testing.assert_allclose(written_mean[active], state[active], rtol=1e-5,
                                   err_msg='entry without `aggregation` is written as a daily mean')
        np.testing.assert_allclose(written_sum[active], steps_per_day * state[active], rtol=1e-5,
                                   err_msg='entry with `aggregation: sum` is written as a daily sum')

    def test_invalid_aggregation_raises_at_initialize(self):
        output = [{
            'variable': 'grid_cell_demand_rate',
            'name': 'WRM_DEMAND',
            'units': 'm3/s',
            'aggregation': 'median',
        }]
        model = Model()
        with self.assertRaises(ValueError) as context:
            model.initialize(self.write_config('aggregation_invalid', output), grid=Grid.from_files(self.GRID_FILE))
        message = str(context.exception)
        self.assertIn('median', message)
        self.assertIn('WRM_DEMAND', message)


if __name__ == '__main__':
    unittest.main()
