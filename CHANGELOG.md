# Changelog

All notable changes to `mosartwmpy` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-18

First release since v0.6.2. Consolidates the numpy 2 migration, two validated reservoir
bug fixes, and the new opt-in return-flow feature. The `0.7.0` version that briefly lived
on `main` was never published; this release supersedes it.

**Rebuild any grid cache written before this release and re-run.** Grid caches saved with
`Grid.to_files` before 1.0.0 lack the `reservoir_minimum_storage` array that the reservoir
regulation kernel now reads. Until the fix in this release (#143), the 1.0.0 development code
read past the end of that empty array with no error and no log message, so reservoir
operations in any run that reused such a cache were wrong: on the packaged test fixture 1,498
cells differ after 8 steps, by up to 299 m<sup>3</sup>/s in channel outflow and
2.27e7 m<sup>3</sup> in reservoir storage. Rebuild the cache from the domain file and re-run.
Caches are now versioned and checked when the model initializes.

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
- **Output `aggregation` defaults to `mean`.** Each `simulation.output` entry accepts an
  `aggregation` of `mean` or `sum`; a missing key now means `mean`, which is how every output was
  written before the key existed, and any other value is rejected when the model initializes. The
  default configuration is unaffected, since all of its entries carry the key; only user configs
  that supply their own `simulation.output` list are affected, and for them this restores daily
  averages in place of daily sums. (#129)
- **xarray floor raised to 2025.1.2**, the release that added the `time_unit` keyword the input
  loaders use when converting `noleap` and `360_day` calendars. The demand loader now converts
  calendars the same way as the runoff loader. (#131)
- **Misconfigured initial storage file is an error.** When
  `water_management.reservoirs.initial_storage.path` is set, a missing file, an unsupported
  extension, a file without `CAP_INIT`, or a file with neither `GRAND_ID` nor `GRID_CELL_INDEX`
  raises instead of logging a warning and running on 90% of capacity. A null path still means no
  file, and reservoirs absent from the file still take the default. `RES_NAME` is no longer
  documented as an identifier column, since it was never joined on. (#135, #136)
- **Regulated flow about three times faster.** `extraction_regulated_flow` resolves dependency
  reservoir ids through a dense array instead of a numba typed dictionary, removing roughly 70%
  of the kernel's time (about 60% of a 20-thread timestep) with bit-identical results. (#144)
- **Downloads stream to disk.** `mosartwmpy.download` writes the archive to a temporary file and
  extracts from it instead of holding the whole archive in memory, and reads 1 MiB chunks on both
  transports; peak memory for the 702 MB sample archive drops from roughly the archive size to a
  few megabytes. (#141)

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
- **Grid caches are versioned and validated.** `Grid.to_files` writes a cache schema version and
  `Grid.from_files` warns when it loads an older cache. `Model.initialize` checks that every
  per-cell array the kernels read has one value per active cell: a missing
  `reservoir_minimum_storage` is restored to its historical tenth of storage capacity with a
  warning, and any other missing array is an error asking for the cache to be rebuilt. Previously
  the regulation kernel read past the end of the empty array silently (see the notice at the top
  of this release). The test suite now also runs under `NUMBA_BOUNDSCHECK=1` in CI, which catches
  this class of defect outright. (#143)
- **Return flow on a cached grid written without a mask.** A grid cache saved with return flow off
  carries no `irrigation_first_mask`, since empty arrays are not written. Enabling return flow on
  such a cache used to fail at initialization with a message asking for a rebuild that would have
  produced the same cache; the mask now defaults to nonirrigation-first everywhere, as a freshly
  built grid does with no mask path, and the remaining cache errors state whether the cache is
  outdated or was written with the relevant feature disabled. (#130)
- **Return flow in cells with zero land fraction.** The irrigation return flow flux divided by
  area times land fraction, which is 0 / 0 in an active cell with no land and, with numpy errors
  ignored process-wide, a silent NaN in hillslope, subnetwork and channel storage. The flux is now
  zero where there is no land area. Latent on the published CONUS grid, where every active cell
  has a land fraction of 1. (#132)
- **Irrigation-first mask shape is checked at grid load.** A mask on a different grid used to pass
  grid construction and fail on the first timestep with a broadcast error that did not name the
  file; it now raises at load naming the file and both cell counts. (#133)
- **Demand file open errors are typed and logged.** The demand loader no longer wraps the file open
  in a bare `except` that called `sys.exit`, which reported a wrong coordinate name as an
  unopenable file, swallowed Ctrl-C, and bypassed the failed-timestep log entry in `Model.update`
  before terminating a BMI host process. A missing file raises `FileNotFoundError`, an unreadable
  one `PermissionError`, a file that is not netCDF `ValueError`, all naming the path, and a wrong
  coordinate name raises its own `KeyError`. (#138)
- **Reservoir parameter file type and columns are validated.** A file extension other than
  `.nc`, `.nc4`, `.parquet` or `.csv` raises naming the file and the supported extensions instead
  of falling through to the netCDF reader, and a file missing the grid cell index column or any
  configured variable column raises naming the file and the columns. `create_grand_parameters`
  refuses an unsupported output extension before doing any work. (#134)
- **The download utility leaves the root logger alone.** `InstallSupplement` added a stdout handler
  to the root logger on every instantiation and, when done, removed every root handler and shut
  logging down, so a notebook or coupled driver downloading data next to a running model lost the
  model's log file, and a failed download leaked a handler per attempt. It now logs through its
  own module logger: with logging already configured, messages propagate to the existing handlers;
  with nothing configured, as in `python -m mosartwmpy.download`, a stdout handler is attached to
  the module logger for the duration of the download. The downloaded archive is also spooled into
  the destination directory rather than the system temporary directory, which on compute nodes is
  often small or RAM-backed. (#140)
- **HTTP errors during a download are reported as such.** The plain HTTP transport sent its request
  with no timeout and did not check the status, so a 404 or 500 body, or an error page served with
  a 200, was buffered and only failed later as "File is not a zip file" with no mention of the URL
  or the status. The request now has a 120 s timeout and checks the status, so an HTTP error raises
  `requests.HTTPError` naming the URL, and a response that is not a zip archive raises
  `zipfile.BadZipFile` naming the URL on both transports; both are logged before they propagate.
  (#139)
- **MSD-LIVE responses are validated.** Credentials missing a nested key, listing entries without a
  key or size, malformed listing XML, and a truncated listing without a continuation token each
  raise a typed error naming the problem, where before they surfaced as bare `KeyError`,
  `AttributeError` or `ParseError`, or as a silently partial file list. (#142)

### Notes
- The return-flow return-flux scaling in `update.py` (division factors applied when adding
  return flow back to the soil column and channel) and the default nonirrigation-first mask
  behavior are pending confirmation from the return-flow author; they only affect runs that
  explicitly enable return flow.

[1.0.0]: https://github.com/IMMM-SFA/mosartwmpy/compare/v0.6.2...v1.0.0
