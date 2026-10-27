import datetime
import importlib.resources
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid


class RegulatedFlowReferenceTest(unittest.TestCase):
    """The water-management state after one simulated day on the packaged fixture is pinned.

    `tests/regulated_flow_reference.npz` holds supply, deficit, channel outflow and reservoir
    storage after 8 steps of 3 h with the default configuration, written by
    `extraction_regulated_flow` as it stood when the dense index landed (#144), which was
    bit-identical to the typed-dict kernel before it. A change to the regulated flow kernels
    that alters results fails here instead of passing a suite that only checks the model runs.
    Regenerate the file only for an intended, understood result change
    (`.nino/regenerate_regulated_flow_reference.py` holds the recipe: this test's configuration,
    eight steps, `np.savez_compressed`).
    """

    TESTS = importlib.resources.files('mosartwmpy').joinpath('tests')
    REFERENCE_FILE = str(TESTS.joinpath('regulated_flow_reference.npz'))

    def test_one_day_matches_the_stored_reference(self):
        reference = np.load(self.REFERENCE_FILE)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'simulation': {
                    'name': 'regulated_flow_reference',
                    'start_date': datetime.date(1981, 1, 1),
                    'end_date': datetime.date(1981, 1, 31),
                    'output_path': tmpdir,
                    'log_to_std_out': False,
                    'log_to_file': False,
                },
                'runoff': {'path': str(self.TESTS.joinpath('runoff_1981_01_01.nc'))},
                'water_management': {
                    'demand': {'path': str(self.TESTS.joinpath('demand_1981_01_01.nc')), 'farmer_abm': {'enabled': False}},
                    'reservoirs': {
                        'parameters': {'path': str(self.TESTS.joinpath('reservoirs.nc'))},
                        'dependencies': {'path': str(self.TESTS.joinpath('dependency_database.parquet'))},
                        'streamflow': {'path': str(self.TESTS.joinpath('mean_flow.parquet'))},
                        'demand': {'path': str(self.TESTS.joinpath('mean_demand.parquet'))},
                    },
                },
            }
            path = Path(tmpdir) / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            model = Model()
            model.initialize(str(path), grid=Grid.from_files(str(self.TESTS.joinpath('grid.zip'))))
            for _ in range(int(reference['steps'])):
                model.update()
            model.finalize()

        for name in ('grid_cell_supply', 'grid_cell_deficit', 'channel_outflow_downstream', 'reservoir_storage'):
            # a tight relative tolerance rather than bit equality, to allow for fused multiply-add
            # differences between CPUs while still catching any real change in the arithmetic
            np.testing.assert_allclose(
                getattr(model.state, name), reference[name], rtol=1e-9, atol=0.0, equal_nan=True,
                err_msg=f'{name} after {int(reference["steps"])} steps differs from the stored reference',
            )


class RegulatedFlowTest(unittest.TestCase):
    """Dependencies on reservoirs absent from the grid are skipped by the regulated flow kernel.

    The kernel resolves dependency reservoir ids through a dense id-to-index array. An id that
    is not on the grid, whether inside or beyond the range of ids that are, must contribute
    nothing rather than index out of range. The packaged dependency database already carries
    one such orphan; this test adds more, including ids larger than any reservoir on the grid.
    """

    TESTS = importlib.resources.files('mosartwmpy').joinpath('tests')
    GRID_FILE = str(TESTS.joinpath('grid.zip'))
    CONFIG_FILE = str(TESTS.joinpath('test_config.yaml'))
    RUNOFF_FILE = str(TESTS.joinpath('runoff_1981_01_01.nc'))
    DEMAND_FILE = str(TESTS.joinpath('demand_1981_01_01.nc'))

    def run_one_step(self, inject_orphans):
        model = Model()
        model.initialize(self.CONFIG_FILE, grid=Grid.from_files(self.GRID_FILE))
        model.config['runoff.path'] = self.RUNOFF_FILE
        model.config['water_management.demand.path'] = self.DEMAND_FILE
        self.addCleanup(model.finalize)

        if inject_orphans:
            on_grid = set(int(r) for r in model.grid.reservoir_id[np.isfinite(model.grid.reservoir_id)])
            max_id = max(on_grid)
            inside = next(r for r in range(max_id, 0, -1) if r not in on_grid)
            orphans = np.array([inside, max_id + 1, max_id + 1000], dtype=np.int64)
            dependencies = model.grid.grid_index_to_reservoirs_map
            for key in list(dependencies.keys())[:200]:
                dependencies[key] = np.concatenate([dependencies[key], orphans])

        model.update()
        return model

    def test_orphaned_dependencies_contribute_nothing(self):
        reference = self.run_one_step(inject_orphans=False)
        with_orphans = self.run_one_step(inject_orphans=True)
        self.assertTrue(np.any(reference.state.grid_cell_supply > 0.0), 'reservoirs supply some demand')
        for name in ('grid_cell_supply', 'grid_cell_deficit', 'channel_outflow_downstream', 'reservoir_storage'):
            np.testing.assert_array_equal(
                getattr(with_orphans.state, name), getattr(reference.state, name),
                err_msg=f'{name} changed when orphaned dependencies were added',
            )


if __name__ == '__main__':
    unittest.main()
