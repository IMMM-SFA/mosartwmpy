# Changelog

All notable changes to `mosartwmpy` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-09-09

### Added

Two new reservoir release methods were added to the model, GDROM and C+GDROM. These
methods can be used in place of the generic and ISTARF methods. An optional parameter
can be added to the reservoir file to specify which method is desired. The model
now outputs a reservoir method file that documents which reservoir release method
was used for each reservoir - providing method as tracking as the model has a default
fallback behavior if the data required for a specified method is unavailable.

- **GDROM — Generic Data-driven Reservoir Operation Model** (opt-in, off by default).
  When `water_management.reservoirs.enable_gdrom: true`, reservoirs with GDROM rule
  files use a two-stage inference pipeline: a CART module-condition classifier selects
  an operating module, and a Decision Tree regressor computes the daily release target.
  Three rule-file formats are supported: tree, single-line constant, and linear
  regression (`Release = a*Inflow + b*Storage + c`). PDSI (Palmer Drought Severity
  Index) is a required input; the file and time-series or climatology mode are
  configured via `water_management.reservoirs.gdrom.pdsi`. Rule files are parsed
  once at model init and all thresholds are converted to SI units (m³, m³/s).
  GDROM sits in the method-priority chain above ISTARF/generic and below C-GDROM.
  References: Li et al. (2024) *Water Resources Research*; GDROM v2.0.0 dataset
  (doi: 10.57931/*).

- **C-GDROM — Conceptual + Generic Data-driven Reservoir Operation Model**
  (opt-in, off by default).  When `water_management.reservoirs.enable_cgdrom: true`,
  C-GDROM takes final priority for its reservoirs (above GDROM/ISTARF/generic).
  Three per-reservoir operation tiers are supported, resolved automatically from
  available calibration data:
  - *General* — no calibration; empirical S-curve + inflow-percentile thresholds.
  - *Flood control* — calibrated linear modules for up to 127 demonstration reservoirs.
  - *Irrigation* — calibrated seasonal constant releases for up to 64 demonstration
    reservoirs.
  A 365-element typical-storage curve (S_ty) is pre-computed at init from four-piece
  S-curve parameters. Inflow percentile statistics (I10/I30/I50/I80/I99) and daily
  ramping constraints (α upward / β downward) are loaded at init; storage and inflow
  are read each day at runtime. Required inputs: a flow-statistics parquet
  (`cgdrom.flow_stats.path`) and a storage-curve CSV (`cgdrom.storage_curve.path`).
  Optional FC and irrigation module CSVs enable the calibrated tiers.
  References: Zhao et al. (2025) C-GDROM; github.com/fzfz12138/C-GDROM.

- **Per-reservoir method override column.**  An optional column in the reservoir
  parameter file (key `water_management.reservoirs.parameters.release_method_variable`)
  lets individual reservoirs pin a method (`cgdrom`, `gdrom`, `istarf`, or `generic`).
  Reservoirs without a value use auto-detection (priority chain C-GDROM > GDROM >
  ISTARF > generic).  A `reservoir_methods.csv` is written to the simulation output
  directory at init and updated with GDROM runtime fallback statistics at finalization.

- **Reservoir method output file.** A reservoir method csv is now written to the output
directory containing the reservoir method used for each reservoir.  


## [1.0.0] - 2026-08-18

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
  working unchanged. A `CAP_MIN` of zero or below counts as no value rather than a real zero floor,
  so it also takes the 10% default — the published sample data carries `CAP_MIN = 0` for 13 of its
  1860 reservoirs, and honoring that literally would have let them draw down to empty. Note that
  ISTARF release rules still derive their normal operating range from storage capacity alone and do
  not respect `CAP_MIN` (tracked in #124). (Dan Broman)
- **Reservoir parameter files in Parquet or CSV.** The reservoir parameter file holds static
  per-reservoir values with a single dimension, so netCDF is no longer required; `.parquet` and
  `.csv` are now accepted and selected by file extension, which makes the values easier to inspect
  and edit. `create_grand_parameters` writes whichever of the three formats the output path names.
  (Dan Broman)
- **Optional initial reservoir storage.** `water_management.reservoirs.initial_storage.path` accepts
  a Parquet or CSV file with a `CAP_INIT` column (million m<sup>3</sup>) to set starting storage,
  joined on `GRAND_ID` or `GRID_CELL_INDEX`. Values are optional per reservoir; anything unmatched
  falls back to the default 90% of capacity. (Dan Broman)
- **MSD-LIVE downloads without a plain URL.** Files on MSD-LIVE records created from July 2023
  onward live in a project owned S3 bucket rather than InvenioRDM's managed storage, so the
  Invenio file API lists only a placeholder and no fetchable URL exists. `mosartwmpy.download`
  now recognizes a MSD-LIVE record URL and retrieves those files by requesting anonymous,
  read only credentials and signing the request with AWS Signature Version 4. No account is
  required and no new dependency was added. Zenodo hosted datasets are still downloaded
  directly. A manifest entry for such a record gives the record URL plus an optional
  `filename`; without it the largest archive in the record is used.

### Changed
- **numpy 2 support.** Requires `numpy>=2.0` and `numba>=0.60`; minimum Python raised to 3.10.
  The CI test matrix and README were updated to Python 3.10-3.12 accordingly. This is a
  compatibility break, hence the major version bump.
- Input reading fills NaNs more robustly regardless of how the input arrives, and sorts
  grid/runoff/demand datasets by coordinate on open.
- The model logs its version on startup.
- The `sample_input` dataset now points at MSD-LIVE v0.0.8 (doi `10.57931/3398687`), whose
  reservoir parameters carry `CAP_MIN` and are provided as netCDF, Parquet, and CSV. Reservoir
  operating rule assignments are unchanged from v0.0.6.
- **Docker image rebased.** The `Dockerfile` built `FROM python:3.9-slim-bullseye`, which no
  longer satisfies `python_requires>=3.10` as of this release, so the image could not be built
  at all. It now uses `ghcr.io/msd-live/jupyter/python-notebook:latest` (Python 3.11) and
  installs the published `mosartwmpy` from PyPI rather than the source in the build context;
  it no longer downloads the sample dataset or sets a command to run the model. (Emily Rexer)

### Fixed
- **Reservoir regulation numba race condition.** Four loops in
  `reservoirs/regulation.py` accumulate into `reservoir_demand` and `reservoir_flow_volume` at
  indices shared by many grid cells, since several cells can depend on the same reservoir. Running
  them under `nb.prange` made those read-modify-write updates race, which could silently corrupt
  supply and demand. They are now serial; the loops that only touch their own index, and the outer
  loops, remain parallel. (Youngjun Son)
- **Flood-control window operator precedence.** A missing set of parentheses in
  `reservoirs/release.py` `storage_targets()` caused the flood-control adjustment to fire for
  nearly all flood-control dams whenever the current month preceded the window end. The
  wraparound-window condition now matches the MOSART Fortran reference. (Cameron Bracken)
- **Missing `h5py` dependency.** Grid serialization in `grid/grid.py` pins `engine='h5netcdf'`,
  which requires an `h5py` backend, but `h5py` was never declared. Installs that did not happen
  to pull it in transitively failed with `ImportError: No module named 'h5py'` when reading or
  writing a grid file. (Cameron Bracken)
- **Missing `python-benedict[io]` extra.** Configuration and the data manifest are read as YAML
  through `benedict`, which needs its `[io]` extra for the parser. Without it, configuring a model
  raised `ExtrasRequireModuleNotFoundError`. The dependency now requests the extra explicitly.
  (Cameron Bracken)
- **`create_grand_parameters` broken on a clean install.** The console script imports
  `scipy.spatial.KDTree`, but `scipy` was never declared, so the command failed immediately with
  `ModuleNotFoundError`. `scipy` is now a dependency, along with `rasterio` and `shapely`, which
  back the `bil_to_parquet` script and had been arriving only incidentally through `geopandas`.
  (Cameron Bracken)
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
