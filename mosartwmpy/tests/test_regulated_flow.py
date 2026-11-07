import datetime
import importlib.resources
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from numba import types
from numba.typed import Dict

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid
from mosartwmpy.reservoirs.regulation import extraction_regulated_flow


class RegulatedFlowReferenceTest(unittest.TestCase):
    """The water-management state after one simulated day on the packaged fixture is pinned.

    `tests/regulated_flow_reference.npz` holds supply, deficit, channel outflow and reservoir
    storage after 8 steps of 3 h with the default configuration, written by
    `extraction_regulated_flow` as it stood when the dense index landed (#144), which was
    bit-identical to the typed-dict kernel before it. A change to the regulated flow kernels
    that alters results fails here instead of passing a suite that only checks the model runs.
    Only case 1 of the supply algorithm fires on this fixture; cases 2 and 3 are pinned by
    `ExtractionCasesTest` below. Regenerate the file only for an intended, understood result change
    (`mosartwmpy/tests/regenerate_regulated_flow_reference.py` holds the recipe: this test's configuration,
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


class ExtractionCasesTest(unittest.TestCase):
    """Each supply case of `extraction_regulated_flow` is pinned on hand-built inputs.

    The kernel converts reservoir flow volume into grid cell supply in one of three ways per
    iteration, chosen by the demand fraction (flow volume over total demand) of the reservoirs:
    case 1 when some reservoir can meet all demand on it, case 2 when a cell's reservoirs can do
    so together, case 3 otherwise. The packaged fixture only reaches case 1, so these tests call
    the kernel directly on a few cells with demands chosen to land in each case, and compare
    against values worked out by hand, which the kernel as it stood before #145 reproduces
    bit for bit. Reservoir ids are deliberately neither zero-based nor consecutive, and the
    dependency lists carry an id inside the range of ids on the grid that belongs to no reservoir
    (5) and one beyond the range (99); both must contribute nothing.

    Two reservoirs sit in cells 0 and 1 with ids 3 and 7; with a 3600 s step and a flow volume
    ratio of 1 their inflows of 10 and 5 m3/s give flow volumes of 36000 and 18000 m3. Cells 2-5
    only have demand: cell 2 depends on reservoir 3, cell 4 on reservoir 7, cells 3 and 5 on both.
    """

    DELTA_T = 3600
    RATIO = 1.0
    RESERVOIR_ID = [3.0, 7.0, np.nan, np.nan, np.nan, np.nan]
    OUTFLOW = [-10.0, -5.0, -1.0, -1.0, -1.0, -1.0]
    DEPENDENCIES = {2: [3], 3: [3, 7], 4: [7], 5: [5, 3, 7, 99]}

    def run_kernel(self, unmet, iterations, reservoir_id=RESERVOIR_ID, outflow=OUTFLOW, dependencies=DEPENDENCIES):
        n = len(reservoir_id)
        reservoir_id = np.array(reservoir_id, dtype=np.float64)
        dependency_map = Dict.empty(key_type=types.int64, value_type=types.int64[:])
        for cell, ids in dependencies.items():
            dependency_map[cell] = np.array(ids, dtype=np.int64)
        state = {
            'channel_outflow_downstream': np.array(outflow, dtype=np.float64),
            'outflow_before_regulation': np.zeros(n),
            'outflow_after_regulation': np.zeros(n),
            'channel_flow': np.zeros(n),
            'grid_cell_unmet_demand': np.array(unmet, dtype=np.float64),
            'grid_cell_supply': np.zeros(n),
            'grid_cell_deficit': np.zeros(n),
        }
        extraction_regulated_flow(
            n, int(np.isfinite(reservoir_id).sum()), self.DELTA_T, np.arange(n, dtype=np.int64),
            reservoir_id, dependency_map,
            state['channel_outflow_downstream'], state['outflow_before_regulation'], state['outflow_after_regulation'],
            state['channel_flow'], state['grid_cell_unmet_demand'], state['grid_cell_supply'], state['grid_cell_deficit'],
            iterations, self.RATIO,
        )
        return state

    def assert_state(self, state, outflow, supply, deficit, outflow_after):
        """Supply, deficit and residual outflow match, and the bookkeeping arrays follow from them."""
        close = dict(rtol=1e-12, atol=1e-9)
        outflow = np.asarray(outflow, dtype=np.float64)
        outflow_after = np.asarray(outflow_after, dtype=np.float64)
        np.testing.assert_allclose(state['grid_cell_supply'], supply, err_msg='supply', **close)
        np.testing.assert_allclose(state['grid_cell_deficit'], deficit, err_msg='deficit', **close)
        np.testing.assert_allclose(state['grid_cell_unmet_demand'], deficit, err_msg='unmet demand', **close)
        np.testing.assert_allclose(state['channel_outflow_downstream'], outflow_after, err_msg='outflow', **close)
        np.testing.assert_allclose(state['outflow_before_regulation'], -outflow, err_msg='outflow before', **close)
        np.testing.assert_allclose(state['outflow_after_regulation'], -outflow_after, err_msg='outflow after', **close)
        np.testing.assert_allclose(state['channel_flow'], outflow - outflow_after, err_msg='channel flow', **close)

    def test_case_1_meets_demand_in_full_split_equally_across_reservoirs_that_can(self):
        # demand on reservoir 3 is 18000 (fraction 2.0), on reservoir 7 it is 10000 (fraction 1.8):
        # both can meet all demand on them, so every cell is met and the cells depending on both
        # take half from each
        state = self.run_kernel([0, 0, 10000.0, 5000.0, 2000.0, 3000.0], iterations=1)
        self.assert_state(
            state, self.OUTFLOW,
            supply=[0, 0, 10000.0, 5000.0, 2000.0, 3000.0],
            deficit=[0, 0, 0, 0, 0, 0],
            outflow_after=[-(36000 - 10000 - 2500 - 1500) / 3600, -(18000 - 2500 - 2000 - 1500) / 3600, -1, -1, -1, -1],
        )

    def test_case_2_meets_demand_in_full_prorated_where_reservoirs_suffice_together(self):
        # demand on reservoir 3 is 60000 (fraction 0.6), on reservoir 7 it is 40000 (fraction 0.45):
        # neither suffices alone, but together (1.05) they meet cells 3 and 5 in full, each
        # reservoir giving its share of the fraction; cells 2 and 4 get nothing this iteration
        state = self.run_kernel([0, 0, 30000.0, 20000.0, 10000.0, 10000.0], iterations=1)
        self.assert_state(
            state, self.OUTFLOW,
            supply=[0, 0, 0, 20000.0, 0, 10000.0],
            deficit=[0, 0, 30000.0, 0, 10000.0, 0],
            outflow_after=[-(36000 - 30000 * 0.6 / 1.05) / 3600, -(18000 - 30000 * 0.45 / 1.05) / 3600, -1, -1, -1, -1],
        )

    def test_case_3_meets_the_fraction_of_demand_the_reservoirs_can_supply(self):
        # demand on reservoir 3 is 120000 (fraction 0.3), on reservoir 7 it is 80000 (fraction
        # 0.225); no cell reaches 1.0 even from both, so each cell takes its demand times each
        # reservoir's fraction, which drains both reservoirs exactly
        state = self.run_kernel([0, 0, 60000.0, 40000.0, 20000.0, 20000.0], iterations=1)
        self.assert_state(
            state, self.OUTFLOW,
            supply=[0, 0, 60000 * 0.3, 40000 * 0.3 + 40000 * 0.225, 20000 * 0.225, 20000 * 0.3 + 20000 * 0.225],
            deficit=[0, 0, 60000 - 18000, 40000 - 21000, 20000 - 4500, 20000 - 10500],
            outflow_after=[0, 0, -1, -1, -1, -1],
        )

    def test_iterations_move_from_case_1_to_case_3_as_demand_is_met(self):
        # three reservoirs in cells 0-2 with ids 3, 7 and 11 (flow volumes 72000, 18000, 18000)
        # and demand in cells 3-7. iteration 1: reservoir 3 has fraction 2.4 (demand 30000) while
        # 7 and 11 have 0.36 (demand 50000 each), so case 1 meets cells 3 and 4 from reservoir 3
        # alone. iteration 2: reservoir 3 has no demand left, 7 has fraction 0.45 (demand 40000)
        # and 11 has 0.36 (demand 50000), no sum reaches 1.0, so case 3 drains both into cells
        # 5-7. iteration 3 finds no water and changes nothing.
        reservoir_id = [3.0, 7.0, 11.0, np.nan, np.nan, np.nan, np.nan, np.nan]
        outflow = [-20.0, -5.0, -5.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        dependencies = {3: [3], 4: [3, 7], 5: [7, 11], 6: [5, 11, 99], 7: [7]}
        unmet = [0, 0, 0, 20000.0, 10000.0, 20000.0, 30000.0, 20000.0]

        after_one = self.run_kernel(unmet, 1, reservoir_id, outflow, dependencies)
        self.assert_state(
            after_one, outflow,
            supply=[0, 0, 0, 20000.0, 10000.0, 0, 0, 0],
            deficit=[0, 0, 0, 0, 0, 20000.0, 30000.0, 20000.0],
            outflow_after=[-(72000 - 30000) / 3600, -5, -5, -1, -1, -1, -1, -1],
        )
        for iterations in (2, 3):
            state = self.run_kernel(unmet, iterations, reservoir_id, outflow, dependencies)
            self.assert_state(
                state, outflow,
                supply=[0, 0, 0, 20000.0, 10000.0, 20000 * 0.45 + 20000 * 0.36, 30000 * 0.36, 20000 * 0.45],
                deficit=[0, 0, 0, 0, 0, 20000 - 16200, 30000 - 10800, 20000 - 9000],
                outflow_after=[-(72000 - 30000) / 3600, 0, 0, -1, -1, -1, -1, -1],
            )


if __name__ == '__main__':
    unittest.main()
