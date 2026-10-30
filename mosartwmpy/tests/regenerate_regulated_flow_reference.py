"""Regenerate mosartwmpy/tests/regulated_flow_reference.npz: eight steps on the packaged fixture.

The file pins the water-management state after one simulated day (8 steps of 3 h) on the
packaged test grid, runoff, demand and reservoir fixtures with the default configuration.
`tests/test_regulated_flow.py` compares a fresh run against it, so an edit to the regulated
flow kernels that changes results fails the suite instead of passing silently.

The script writes whatever the kernel at the current commit produces, so run it only at a
commit whose results are known to be correct. The committed file was written by
`extraction_regulated_flow` as it stood when the dense index landed (#144), which was
bit-identical to the typed-dict kernel before it. Regenerate ONLY when a result change is
intended and understood:
    python -m mosartwmpy.tests.regenerate_regulated_flow_reference
"""

import datetime
import hashlib
import importlib.resources
import tempfile
from pathlib import Path

import numpy as np
import yaml

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid

TESTS = Path(str(importlib.resources.files('mosartwmpy').joinpath('tests')))
REFERENCE = TESTS / 'regulated_flow_reference.npz'
ARRAYS = ('grid_cell_supply', 'grid_cell_deficit', 'channel_outflow_downstream', 'reservoir_storage')
STEPS = 8


def main():
    with tempfile.TemporaryDirectory() as tmp:
        config = {
            'simulation': {'name': 'regulated_flow_reference', 'start_date': datetime.date(1981, 1, 1),
                           'end_date': datetime.date(1981, 1, 31), 'output_path': tmp,
                           'log_to_std_out': False, 'log_to_file': False},
            'runoff': {'path': str(TESTS / 'runoff_1981_01_01.nc')},
            'water_management': {
                'demand': {'path': str(TESTS / 'demand_1981_01_01.nc'), 'farmer_abm': {'enabled': False}},
                'reservoirs': {
                    'parameters': {'path': str(TESTS / 'reservoirs.nc')},
                    'dependencies': {'path': str(TESTS / 'dependency_database.parquet')},
                    'streamflow': {'path': str(TESTS / 'mean_flow.parquet')},
                    'demand': {'path': str(TESTS / 'mean_demand.parquet')},
                },
            },
        }
        path = Path(tmp) / 'config.yaml'
        path.write_text(yaml.safe_dump(config))
        model = Model()
        model.initialize(str(path), grid=Grid.from_files(str(TESTS / 'grid.zip')))
        for _ in range(STEPS):
            model.update()
        arrays = {name: getattr(model.state, name).copy() for name in ARRAYS}
        model.finalize()

    np.savez_compressed(REFERENCE, steps=np.int64(STEPS), **arrays)
    digest = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()[:16]
    print(f'wrote {REFERENCE} ({REFERENCE.stat().st_size / 1024:.0f} KB, sha256 {digest})')
    for name, values in arrays.items():
        print(f'  {name:28s} nonzero {int(np.count_nonzero(values)):6d}  finite {int(np.isfinite(values).sum()):6d}  '
              f'sum {np.nansum(values):.6e}')


if __name__ == '__main__':
    main()
