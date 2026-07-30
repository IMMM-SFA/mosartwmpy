# Changelog

All notable changes to `mosartwmpy` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - unreleased

First release since v0.6.2. Consolidates the numpy 2 migration, two validated reservoir
bug fixes, and the new opt-in return-flow feature. The `0.7.0` version that briefly lived
on `main` was never published; this release supersedes it.

### Added
- **Return flow** (opt-in, off by default). When `water_management.demand.return_flow_enabled`
  is set, the unconsumed portion of met withdrawals is routed back into the water system:
  irrigation return flow to the soil column (hillslope subsurface runoff) and nonirrigation
  return flow to the channel. Requires demand disaggregated into four fields
  (`irrigation_withdrawal`, `irrigation_consumption`, `nonirrigation_withdrawal`,
  `nonirrigation_consumption`). An optional `irrigation_first_mask_path` marks cells that meet
  irrigation demand before nonirrigation (value `1`) versus nonirrigation first (`0`). Adds
  `irrigation_consumption_deficit` and `nonirrigation_consumption_deficit` output variables.
  See the "return flow" section of the README for configuration.
- **Per-reservoir minimum storage.** An optional `CAP_MIN` column in the reservoir parameter file
  (million m<sup>3</sup>, like `CAP_MCM`) sets the minimum storage below which a reservoir will not
  release, replacing the hardcoded 10% of capacity. The column name is configurable via
  `water_management.reservoirs.parameters.minimum_storage_variable`. Reservoirs with no value, and
  files with no such column, fall back to the previous 10% of capacity, so existing input keeps
  working unchanged. Note that ISTARF release rules still derive their normal operating range from
  storage capacity alone and do not respect `CAP_MIN`. (Dan Broman)
- **Reservoir parameter files in Parquet or CSV.** The reservoir parameter file holds static
  per-reservoir values with a single dimension, so netCDF is no longer required; `.parquet` and
  `.csv` are now accepted and selected by file extension, which makes the values easier to inspect
  and edit. `create_grand_parameters` writes whichever of the three formats the output path names.
  (Dan Broman)
- **Optional initial reservoir storage.** `water_management.reservoirs.initial_storage.path` accepts
  a Parquet or CSV file with a `CAP_INIT` column (million m<sup>3</sup>) to set starting storage,
  joined on `GRAND_ID` or `GRID_CELL_INDEX`. Values are optional per reservoir; anything unmatched
  falls back to the default 90% of capacity. (Dan Broman)

### Changed
- **numpy 2 support.** Requires `numpy>=2.0` and `numba>=0.60`; minimum Python raised to 3.10.
  The CI test matrix and README were updated to Python 3.10-3.12 accordingly. This is a
  compatibility break, hence the major version bump.
- Input reading fills NaNs more robustly regardless of how the input arrives, and sorts
  grid/runoff/demand datasets by coordinate on open.
- The model logs its version on startup.

### Fixed
- **Reservoir regulation numba race condition.** Inner accumulation loops in
  `reservoirs/regulation.py` that write to shared arrays were running under `nb.prange`,
  causing data races that could silently corrupt results; those loops are now serial while
  the confirmed-safe outer loops remain parallel. (Youngjun Son)
- **Flood-control window operator precedence.** A missing set of parentheses in
  `reservoirs/release.py` `storage_targets()` caused the flood-control adjustment to fire for
  nearly all flood-control dams whenever the current month preceded the window end. The
  wraparound-window condition now matches the MOSART Fortran reference. (Cameron Bracken)
- **Orphaned reservoir dependency guard.** `extraction_regulated_flow()` now guards lookups of
  reservoir IDs that are absent from the current domain's `reservoir_id_to_index`, avoiding a
  `KeyError` (previously silently swallowed under parallel execution). (Cameron Bracken)
- **numpy 2 / pyarrow string columns.** Under modern pandas with pyarrow, string reservoir
  columns (e.g. the reservoir behavior field) load as an `ArrowStringArray` rather than a numpy
  array, so the grid/state mask-trimming loop in `model.py` skipped them, leaving those arrays
  at full-grid size and breaking downstream broadcasts (e.g. in `istarf_release`). Reservoir
  variables and grid columns loaded from files are now coerced to numpy arrays. This restores
  the core model test suite under the numpy 2 stack.
- **SyntaxWarning on the multi-file path regexes** in `input/demand.py` (invalid escape sequence
  `\{` under Python 3.12+) resolved by using raw strings. (#111)

### Notes
- The return-flow return-flux scaling in `update.py` (division factors applied when adding
  return flow back to the soil column and channel) and the default nonirrigation-first mask
  behavior are pending confirmation from the return-flow author; they only affect runs that
  explicitly enable return flow.

[1.0.0]: https://github.com/IMMM-SFA/mosartwmpy/compare/v0.6.2...v1.0.0
