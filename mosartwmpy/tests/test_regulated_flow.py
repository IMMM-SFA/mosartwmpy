import importlib.resources
import unittest

import numpy as np

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid


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
