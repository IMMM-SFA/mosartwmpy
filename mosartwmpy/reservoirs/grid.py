import logging

import numpy as np
import pandas as pd
import xarray as xr

from numba.core import types
from numba.typed import Dict
from pathlib import Path
from xarray import concat, open_dataset, DataArray
from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters


def load_reservoirs(self, config: Benedict, parameters: Parameters) -> None:
    """Loads the reservoir information from file onto the grid.

    Args:
        config (Benedict): the model configuration
        parameters (Parameters): the model parameters
    """

    # reservoir parameter file — supports .nc (netCDF), .parquet, or .csv
    reservoir_path = config.get('water_management.reservoirs.parameters.path')
    suffix = Path(reservoir_path).suffix.lower()
    if suffix in ('.parquet',):
        reservoir_df = pd.read_parquet(reservoir_path)
    elif suffix in ('.csv',):
        reservoir_df = pd.read_csv(reservoir_path)
    else:
        reservoirs_file = open_dataset(reservoir_path)
        reservoir_df = reservoirs_file.to_dataframe()
        reservoirs_file.close()
    reservoirs = pd.DataFrame(index=self.id).merge(
        reservoir_df,
        how='left',
        left_index=True,
        right_on=config.get('water_management.reservoirs.parameters.grid_cell_index'),
    )

    # load reservoir variables
    # coerce to numpy arrays: under pandas with pyarrow, string columns (e.g. the
    # reservoir behavior field) come back as an ArrowStringArray rather than an
    # ndarray, which the mask-trimming loop in model.py would then skip -- leaving
    # that array at full-grid size and breaking downstream broadcasts.
    for key, value in config.get('water_management.reservoirs.parameters.variables').items():
        setattr(self, key, np.asarray(reservoirs[value].values))

    # correct the fields with different units
    # surface area from km^2 to m^2
    self.reservoir_surface_area = self.reservoir_surface_area * 1.0e6
    # capacity from millions m^3 to m^3
    self.reservoir_storage_capacity = self.reservoir_storage_capacity * 1.0e6

    # minimum storage: use CAP_MIN [million m3] from the file when it gives a usable
    # value, otherwise fall back to reservoir_runoff_capacity_parameter * storage_capacity.
    # a non-positive CAP_MIN is treated as missing rather than as a real zero floor: the
    # published sample data carries CAP_MIN = 0 for a handful of reservoirs, and honoring
    # that would let them draw down to empty, which is a regression against the historical
    # 10%-of-capacity floor that applied before this column was read at all.
    cap_min_col = config.get('water_management.reservoirs.parameters.minimum_storage_variable', 'CAP_MIN')
    if cap_min_col in reservoirs.columns and reservoirs[cap_min_col].notna().any():
        cap_min_m3 = np.asarray(reservoirs[cap_min_col].values, dtype=np.float64) * 1.0e6
        fallback = parameters.reservoir_runoff_capacity_parameter * self.reservoir_storage_capacity
        usable = np.isfinite(cap_min_m3) & (cap_min_m3 > 0)
        self.reservoir_minimum_storage = np.where(usable, cap_min_m3, fallback)
    else:
        self.reservoir_minimum_storage = parameters.reservoir_runoff_capacity_parameter * self.reservoir_storage_capacity

    # reservoir dependency database file
    self.reservoir_dependency_database = pd.read_parquet(
        config.get('water_management.reservoirs.dependencies.path')
    ).rename(columns={
        config.get('water_management.reservoirs.dependencies.variables.dependent_reservoir_id'): 'reservoir_id',
        config.get('water_management.reservoirs.dependencies.variables.dependent_cell_index'): 'grid_cell_id'
    })
    # drop nan grid ids
    self.reservoir_dependency_database = self.reservoir_dependency_database[self.reservoir_dependency_database.grid_cell_id.notna()]
    # set to integer
    self.reservoir_dependency_database = self.reservoir_dependency_database.astype(np.int64)

    # create a numba typed dict with key = <grid cell id> and value = <list of reservoir_ids that feed the cell>
    self.grid_index_to_reservoirs_map = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:],
    )
    for grid_cell_id, group in self.reservoir_dependency_database.groupby('grid_cell_id'):
        self.grid_index_to_reservoirs_map[grid_cell_id] = group.reservoir_id.values.copy()

    # index by grid cell
    self.reservoir_dependency_database = self.reservoir_dependency_database.set_index('grid_cell_id').sort_index()

    # prepare the month based reservoir schedules mapped to the domain
    prepare_reservoir_schedule(self, config)

    # calculate reservoir Biemans & Hanasaki Runoff/Capacity
    self.reservoir_runoff_capacity = self.reservoir_streamflow_schedule.mean(
        dim='month') * 365 * 24 * 60 * 60 / self.reservoir_storage_capacity

    # calculate the long term mean flow across the reservoirs
    self.reservoir_computed_meanflow_cumecs = self.reservoir_streamflow_schedule.weighted(xr.DataArray(
        data=np.array([31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0]),
        dims=['month'],
        coords=dict(month=np.arange(1, 13, 1))
    )).mean(dim='month').values

    # resolve per-reservoir release method (produces uses_gdrom, uses_istarf, reservoir_resolved_method)
    # and write reservoir_methods.csv to the output directory
    _resolve_and_load_gdrom(self, config, reservoir_df, parameters)


def prepare_reservoir_schedule(self, config: Benedict) -> None:
    """Establishes the reservoir schedule and flow.

    Args:
        config (Benedict): the model configuration
    """

    # streamflow file
    streamflow = pd.read_parquet(config.get('water_management.reservoirs.streamflow.path')).astype(np.float64)
    # demand file
    demand = pd.read_parquet(config.get('water_management.reservoirs.demand.path')).astype(np.float64)

    flow_schedule = []
    demand_schedule = []

    # for each month, map mean flow and demand for reservoir onto the grid
    for m in np.arange(12):
        flow_schedule.append(pd.DataFrame(self.reservoir_id, index=self.id, columns=['reservoir_id']).merge(
            streamflow[
                streamflow[config.get('water_management.reservoirs.streamflow.variables.streamflow_month_index')] == m
            ][[
                config.get('water_management.reservoirs.streamflow.variables.streamflow_reservoir_id'),
                config.get('water_management.reservoirs.streamflow.variables.streamflow')
            ]],
            how='left',
            left_on='reservoir_id',
            right_on=config.get('water_management.reservoirs.streamflow.variables.streamflow_reservoir_id')
        )[config.get('water_management.reservoirs.streamflow.variables.streamflow')].values)
        demand_schedule.append(pd.DataFrame(self.reservoir_id, index=self.id, columns=['reservoir_id']).merge(
            demand[
                demand[config.get('water_management.reservoirs.demand.variables.demand_month_index')] == m
            ][[
                config.get('water_management.reservoirs.demand.variables.demand_reservoir_id'),
                config.get('water_management.reservoirs.demand.variables.demand')
            ]],
            how='left',
            left_on='reservoir_id',
            right_on=config.get('water_management.reservoirs.demand.variables.demand_reservoir_id')
        )[config.get('water_management.reservoirs.demand.variables.demand')].values)
    self.reservoir_streamflow_schedule = DataArray(
        data=np.array(flow_schedule),
        dims=['month', 'index'],
        coords=dict(
            index=self.id,
            month=(np.arange(12) + 1),  # convert month to 1 based indexing
        )
    )
    self.reservoir_demand_schedule = DataArray(
        data=np.array(demand_schedule),
        dims=['month', 'index'],
        coords=dict(
            index=self.id,
            month=(np.arange(12) + 1),  # convert month to 1 based indexing
        )
    )

    # initialize prerelease based on long term mean flow and demand (Biemans 2011)
    # TODO weighted average by days in month?
    flow_avg = self.reservoir_streamflow_schedule.mean(dim='month')
    demand_avg = self.reservoir_demand_schedule.mean(dim='month')
    prerelease = (1.0 * self.reservoir_streamflow_schedule)
    prerelease[:, :] = flow_avg
    # note that xarray `where` modifies the false values
    condition = (demand_avg >= (0.5 * flow_avg)) & (flow_avg > 0)
    prerelease = prerelease.where(
        ~condition,
        demand_avg / 10 + 9 / 10 * flow_avg * self.reservoir_demand_schedule / demand_avg
    )
    prerelease = prerelease.where(
        condition,
        prerelease.where(
            ~((flow_avg + self.reservoir_demand_schedule - demand_avg) > 0),
            flow_avg + self.reservoir_demand_schedule - demand_avg
        )
    )
    self.reservoir_prerelease_schedule = prerelease


# --------------------------------------------------------------------------- #
# GDROM loading and method resolution
# --------------------------------------------------------------------------- #

def _resolve_and_load_gdrom(self, config: Benedict, reservoir_df: pd.DataFrame, parameters: Parameters) -> None:
    """Load GDROM rules and PDSI data, resolve per-reservoir methods, write CSV.

    Sets grid.uses_gdrom, grid.uses_istarf, grid.reservoir_resolved_method,
    grid.gdrom_rules, grid.reservoir_state_name, and grid.pdsi_lookup.
    Writes reservoir_methods.csv to the simulation output directory.

    Called once from load_reservoirs() after all other grid attributes are set.
    """
    n = len(self.reservoir_id)
    enable_gdrom = config.get('water_management.reservoirs.enable_gdrom', False)
    enable_istarf = config.get('water_management.reservoirs.enable_istarf', True)

    # --- optional per-reservoir method column ---
    method_col_key = config.get(
        'water_management.reservoirs.parameters.variables.reservoir_release_method', None
    )
    if method_col_key and method_col_key in reservoir_df.columns:
        specified_methods = np.asarray(reservoir_df[method_col_key].values, dtype=object)
        # NaN → None so downstream logic can use `is None` uniformly
        specified_methods = np.where(pd.isnull(specified_methods), None, specified_methods)
    else:
        specified_methods = np.array([None] * n, dtype=object)

    # --- load GDROM rule files and metadata (if enabled) ---
    has_rule_files = np.zeros(n, dtype=bool)
    state_names = np.empty(n, dtype=object)

    if enable_gdrom:
        rules_path = Path(config.get('water_management.reservoirs.gdrom.rules_path'))
        metadata_path = rules_path / 'reservoir_metadata.csv'

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"GDROM reservoir_metadata.csv not found at {metadata_path}. "
                f"Stage the GDROM rule files into {rules_path} before enabling enable_gdrom."
            )

        gdrom_meta = pd.read_csv(metadata_path)[['GRAND_ID', 'ADMIN_UNIT']].dropna()
        gdrom_meta = gdrom_meta.set_index('GRAND_ID')['ADMIN_UNIT'].to_dict()

        # determine which reservoirs in this domain have rule files
        mod_dir = rules_path / 'modules'
        for i in range(n):
            gid = int(self.reservoir_id[i]) if np.isfinite(self.reservoir_id[i]) else -1
            if gid <= 0:
                continue
            if (mod_dir / f"{gid}_0.txt").exists():
                has_rule_files[i] = True
                state_names[i] = gdrom_meta.get(gid, None)

        # load and validate PDSI
        self.pdsi_lookup = _load_pdsi(config, gdrom_meta, has_rule_files, self.reservoir_id)

        # parse rule files for all GDROM-eligible reservoirs
        eligible_ids = [
            int(self.reservoir_id[i])
            for i in range(n)
            if has_rule_files[i] and np.isfinite(self.reservoir_id[i])
        ]
        from mosartwmpy.reservoirs.gdrom import load_gdrom_rules
        logging.info("GDROM: parsing rule files for %d reservoirs...", len(eligible_ids))
        self.gdrom_rules = load_gdrom_rules(rules_path, eligible_ids)
        logging.info("GDROM: rule files loaded.")

    self.reservoir_state_name = state_names

    # --- resolve per-reservoir method ---
    uses_gdrom = np.zeros(n, dtype=bool)
    uses_istarf = np.zeros(n, dtype=bool)
    resolved_methods = np.empty(n, dtype=object)
    fallback_reasons = np.empty(n, dtype=object)

    # istarf eligibility from the existing behavior column (unchanged logic)
    istarf_eligible = np.array(
        [(str(x).lower() != 'generic') if pd.notna(x) else False
         for x in self.reservoir_behavior],
        dtype=bool
    )

    for i in range(n):
        specified = specified_methods[i]
        # normalise to lower-case string or None
        if specified is not None:
            specified = str(specified).lower().strip()
            if specified == 'nan' or specified == '':
                specified = None

        gdrom_avail = enable_gdrom and has_rule_files[i]
        istarf_avail = enable_istarf and istarf_eligible[i]

        if specified is None:
            # auto-detect: GDROM > ISTARF > generic
            if gdrom_avail:
                resolved_methods[i] = 'gdrom'
                uses_gdrom[i] = True
            elif istarf_avail:
                resolved_methods[i] = 'istarf'
                uses_istarf[i] = True
            else:
                resolved_methods[i] = 'generic'

        elif specified == 'gdrom':
            if gdrom_avail:
                resolved_methods[i] = 'gdrom'
                uses_gdrom[i] = True
            elif istarf_avail:
                resolved_methods[i] = 'istarf'
                uses_istarf[i] = True
                fallback_reasons[i] = 'gdrom_unavailable' if not enable_gdrom else 'no_rule_file'
            else:
                resolved_methods[i] = 'generic'
                if not enable_gdrom:
                    fallback_reasons[i] = 'gdrom_disabled'
                elif not has_rule_files[i]:
                    fallback_reasons[i] = 'no_rule_file'
                else:
                    fallback_reasons[i] = 'istarf_disabled_no_params'

        elif specified == 'istarf':
            if istarf_avail:
                resolved_methods[i] = 'istarf'
                uses_istarf[i] = True
            else:
                resolved_methods[i] = 'generic'
                fallback_reasons[i] = 'istarf_disabled' if not enable_istarf else 'no_istarf_params'

        elif specified == 'generic':
            resolved_methods[i] = 'generic'

        else:
            logging.warning(
                "GDROM: unknown reservoir_release_method value '%s' for index %d; "
                "falling back to auto-detect.",
                specified, i,
            )
            # re-run auto-detect
            if gdrom_avail:
                resolved_methods[i] = 'gdrom'
                uses_gdrom[i] = True
            elif istarf_avail:
                resolved_methods[i] = 'istarf'
                uses_istarf[i] = True
            else:
                resolved_methods[i] = 'generic'

    # emit batched fallback warnings
    _warn_fallbacks(resolved_methods, specified_methods, fallback_reasons, self.reservoir_id)

    self.uses_gdrom = uses_gdrom
    self.uses_istarf = uses_istarf
    self.reservoir_resolved_method = resolved_methods

    # --- write reservoir_methods.csv ---
    output_dir = Path(config.get('simulation.output_path')) / config.get('simulation.name', '')
    _write_methods_csv(
        output_dir, self.reservoir_id, specified_methods, resolved_methods, fallback_reasons
    )


def _load_pdsi(config: Benedict, gdrom_meta: dict, has_rule_files: np.ndarray, reservoir_ids: np.ndarray):
    """Load and validate PDSI data; return a lookup callable.

    Parameters
    ----------
    config : Benedict
    gdrom_meta : dict  {GRAND_ID: ADMIN_UNIT (state full name)}
    has_rule_files : bool array aligned to active-cell index
    reservoir_ids : float array of GRanD IDs

    Returns
    -------
    callable : (state_name: str, year: int, month: int) -> float
    """
    pdsi_config = config.get('water_management.reservoirs.gdrom.pdsi', {})
    pdsi_path = pdsi_config.get('path') if pdsi_config else None
    mode = (pdsi_config.get('mode', 'timeseries') or 'timeseries').lower() if pdsi_config else 'timeseries'

    if not pdsi_path:
        raise ValueError(
            "GDROM is enabled (enable_gdrom: true) but "
            "water_management.reservoirs.gdrom.pdsi.path is not set."
        )

    pdsi_df = pd.read_parquet(pdsi_path)
    _validate_pdsi_schema(pdsi_df, pdsi_path)

    # determine which state names are required by GDROM-eligible reservoirs
    required_states = set()
    for i in range(len(reservoir_ids)):
        if not has_rule_files[i]:
            continue
        gid = int(reservoir_ids[i]) if np.isfinite(reservoir_ids[i]) else -1
        state = gdrom_meta.get(gid)
        if state:
            required_states.add(state)

    available_states = set(pdsi_df['state'].unique())
    missing_states = required_states - available_states
    if missing_states:
        raise ValueError(
            f"GDROM PDSI file is missing data for the following states required "
            f"by GDROM-eligible reservoirs: {sorted(missing_states)}. "
            f"Check that {pdsi_path} covers all necessary states."
        )

    if mode == 'timeseries':
        return _build_timeseries_lookup(pdsi_df, config, required_states, pdsi_path)
    elif mode == 'climatology':
        return _build_climatology_lookup(pdsi_df, config, required_states, pdsi_path)
    else:
        raise ValueError(
            f"Unknown PDSI mode '{mode}'. "
            f"Set water_management.reservoirs.gdrom.pdsi.mode to 'timeseries' or 'climatology'."
        )


def _validate_pdsi_schema(pdsi_df: pd.DataFrame, path) -> None:
    required = {'state', 'year', 'month', 'pdsi'}
    missing = required - set(pdsi_df.columns)
    if missing:
        raise ValueError(
            f"PDSI file {path} is missing required columns: {sorted(missing)}. "
            f"Expected columns: state, year, month, pdsi."
        )


def _build_timeseries_lookup(pdsi_df: pd.DataFrame, config: Benedict, required_states: set, pdsi_path):
    """Validate full simulation-period coverage and return a dict-based lookup."""
    sim_start = config.get('simulation.start_date')
    sim_end = config.get('simulation.end_date')

    if sim_start is None or sim_end is None:
        raise ValueError(
            "GDROM PDSI mode 'timeseries' requires simulation.start_date and "
            "simulation.end_date to be set in the config."
        )

    start_year, start_month = sim_start.year, sim_start.month
    end_year, end_month = sim_end.year, sim_end.month

    # check coverage per required state
    for state in required_states:
        state_df = pdsi_df[pdsi_df['state'] == state]
        available = set(zip(state_df['year'].astype(int), state_df['month'].astype(int)))
        # build expected (year, month) pairs covering simulation period
        missing = []
        y, m = start_year, start_month
        while (y, m) <= (end_year, end_month):
            if (y, m) not in available:
                missing.append((y, m))
            m += 1
            if m > 12:
                m = 1
                y += 1
        if missing:
            raise ValueError(
                f"GDROM PDSI timeseries for state '{state}' is missing {len(missing)} "
                f"year-month entries needed by the simulation period "
                f"({sim_start} to {sim_end}). First missing: {missing[0]}. "
                f"Check {pdsi_path} or switch to pdsi.mode: climatology."
            )

    lookup = {
        (row['state'], int(row['year']), int(row['month'])): float(row['pdsi'])
        for _, row in pdsi_df.iterrows()
    }
    return lambda state_name, year, month: lookup[(state_name, year, month)]


def _build_climatology_lookup(pdsi_df: pd.DataFrame, config: Benedict, required_states: set, pdsi_path):
    """Validate clim-year coverage, compute means, return a lambda-based lookup."""
    pdsi_cfg = config.get('water_management.reservoirs.gdrom.pdsi', {})
    clim_start = pdsi_cfg.get('climatology_start_year')
    clim_end = pdsi_cfg.get('climatology_end_year')

    if clim_start is None or clim_end is None:
        raise ValueError(
            "GDROM PDSI mode 'climatology' requires "
            "water_management.reservoirs.gdrom.pdsi.climatology_start_year and "
            "climatology_end_year to be set."
        )
    clim_start, clim_end = int(clim_start), int(clim_end)

    # check that every required state has all 12 months for every year in the clim range
    for state in required_states:
        state_df = pdsi_df[pdsi_df['state'] == state]
        available = set(zip(state_df['year'].astype(int), state_df['month'].astype(int)))
        missing = [
            (y, m)
            for y in range(clim_start, clim_end + 1)
            for m in range(1, 13)
            if (y, m) not in available
        ]
        if missing:
            raise ValueError(
                f"GDROM PDSI climatology for state '{state}' is missing {len(missing)} "
                f"entries in the year range {clim_start}–{clim_end}. "
                f"First missing: {missing[0]}. "
                f"Check {pdsi_path} or adjust climatology_start_year/climatology_end_year."
            )

    clim_df = pdsi_df[
        (pdsi_df['year'].astype(int) >= clim_start) &
        (pdsi_df['year'].astype(int) <= clim_end)
    ]
    clim_means = (
        clim_df.groupby(['state', 'month'])['pdsi']
        .mean()
        .to_dict()
    )
    # clim_means key: (state_name, month_int)
    return lambda state_name, year, month: clim_means[(state_name, month)]


def _warn_fallbacks(resolved, specified, reasons, reservoir_ids) -> None:
    """Emit one batched warning per fallback reason."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, reason in enumerate(reasons):
        if reason is not None:
            groups[reason].append(int(reservoir_ids[i]) if np.isfinite(reservoir_ids[i]) else i)

    messages = {
        'gdrom_unavailable': (
            "GDROM specified but not available (not enabled or missing rule file); "
            "fell back to ISTARF"
        ),
        'gdrom_disabled':    "GDROM specified but enable_gdrom is false; fell back to generic",
        'no_rule_file':      "GDROM specified but no rule file found; fell back",
        'istarf_disabled_no_params': (
            "GDROM specified, no rule file, ISTARF also unavailable; fell back to generic"
        ),
        'istarf_disabled':   "ISTARF specified but enable_istarf is false; fell back to generic",
        'no_istarf_params':  (
            "ISTARF specified but reservoir_behavior is 'generic'; fell back to generic"
        ),
    }
    for reason, ids in groups.items():
        msg = messages.get(reason, reason)
        logging.warning(
            "Reservoir method fallback (%s) for %d reservoir(s) (GRAND_IDs: %s%s).",
            msg, len(ids),
            ", ".join(str(x) for x in ids[:10]),
            "..." if len(ids) > 10 else "",
        )


def _write_methods_csv(output_dir: Path, reservoir_ids, specified, resolved, reasons) -> None:
    """Write reservoir_methods.csv to output_dir."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(len(reservoir_ids)):
            gid = int(reservoir_ids[i]) if np.isfinite(reservoir_ids[i]) else None
            rows.append({
                'GRAND_ID': gid,
                'SPECIFIED_METHOD': specified[i] if specified[i] is not None else '',
                'RESOLVED_METHOD': resolved[i],
                'FALLBACK_REASON': reasons[i] if reasons[i] is not None else '',
            })
        pd.DataFrame(rows).to_csv(output_dir / 'reservoir_methods.csv', index=False)
        logging.info("GDROM: wrote reservoir_methods.csv to %s", output_dir)
    except Exception as exc:
        logging.warning("GDROM: could not write reservoir_methods.csv: %s", exc)
