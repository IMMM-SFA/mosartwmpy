import numpy as np
import pandas as pd
import xarray as xr

from numba.core import types
from numba.typed import Dict
from pathlib import Path
from xarray import concat, open_dataset, DataArray
from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters

# file formats accepted for the reservoir parameter file, by extension; shared with the writer in
# utilities/create_grand_parameters.py so the two cannot drift apart
RESERVOIR_PARAMETER_FORMATS = {
    '.nc': 'netcdf',
    '.nc4': 'netcdf',
    '.parquet': 'parquet',
    '.csv': 'csv',
}


# Core reservoir parameter variables required whenever reservoirs are enabled
REQUIRED_RESERVOIR_VARIABLES = (
    'reservoir_id',
    'reservoir_grid_index',
    'reservoir_surface_area',
    'reservoir_storage_capacity',
    'reservoir_use_irrigation',
    'reservoir_use_electricity',
    'reservoir_use_flood_control',
)

# Additional reservoir parameter variables required only when ISTARF release rules are enabled (#109)
ISTARF_RESERVOIR_VARIABLES = (
    'reservoir_upper_alpha',
    'reservoir_upper_beta',
    'reservoir_upper_max',
    'reservoir_upper_min',
    'reservoir_upper_mu',
    'reservoir_lower_alpha',
    'reservoir_lower_beta',
    'reservoir_lower_max',
    'reservoir_lower_min',
    'reservoir_lower_mu',
    'reservoir_release_alpha_one',
    'reservoir_release_alpha_two',
    'reservoir_release_beta_one',
    'reservoir_release_beta_two',
    'reservoir_release_c',
    'reservoir_release_max',
    'reservoir_release_min',
    'reservoir_release_p_one',
    'reservoir_release_p_two',
    'reservoir_behavior',
)


def reservoir_parameter_format(path: str) -> str:
    """Determines the reservoir parameter file format from the path's extension.

    Args:
        path (str): path of a reservoir parameter file to read or write

    Returns:
        str: one of 'netcdf', 'parquet' or 'csv'

    Raises:
        ValueError: if the extension is not a supported format
    """

    suffix = Path(path).suffix.lower()
    if suffix not in RESERVOIR_PARAMETER_FORMATS:
        raise ValueError(
            f'Unsupported reservoir parameter file `{path}`: the extension `{suffix}` is not one of '
            f'{", ".join(sorted(RESERVOIR_PARAMETER_FORMATS))}.'
        )
    return RESERVOIR_PARAMETER_FORMATS[suffix]


def load_reservoirs(self, config: Benedict, parameters: Parameters) -> None:
    """Loads the reservoir information from file onto the grid.

    Args:
        config (Benedict): the model configuration
        parameters (Parameters): the model parameters
    """

    # reservoir parameter file — supports .nc (netCDF), .parquet, or .csv
    reservoir_path = config.get('water_management.reservoirs.parameters.path')
    file_format = reservoir_parameter_format(reservoir_path)
    if file_format == 'parquet':
        reservoir_df = pd.read_parquet(reservoir_path)
    elif file_format == 'csv':
        reservoir_df = pd.read_csv(reservoir_path)
    else:
        reservoirs_file = open_dataset(reservoir_path)
        reservoir_df = reservoirs_file.to_dataframe()
        reservoirs_file.close()

    # every required column must be present before the merge and the attribute loop below
    # dereference them, so a renamed column is reported with the file and the config keys;
    # ISTARF columns are only required when `enable_istarf` is True (#109)
    grid_cell_index = config.get('water_management.reservoirs.parameters.grid_cell_index')
    variables = config.get('water_management.reservoirs.parameters.variables')
    enable_istarf = bool(config.get('water_management.reservoirs.enable_istarf', False))
    required_keys = list(REQUIRED_RESERVOIR_VARIABLES)
    if enable_istarf:
        required_keys.extend(ISTARF_RESERVOIR_VARIABLES)
    required_columns = [grid_cell_index] + [
        variables[k] for k in required_keys if k in variables and variables[k] is not None
    ]
    missing = [column for column in dict.fromkeys(required_columns) if column not in reservoir_df.columns]
    if missing:
        raise KeyError(
            f'Reservoir parameter file `{reservoir_path}` has no column(s) {missing}; check the '
            f'`water_management.reservoirs.parameters` field names in the configuration.'
        )

    reservoirs = pd.DataFrame(index=self.id).merge(
        reservoir_df,
        how='left',
        left_index=True,
        right_on=grid_cell_index,
    )

    # load reservoir variables
    # coerce to numpy arrays: under pandas with pyarrow, string columns (e.g. the
    # reservoir behavior field) come back as an ArrowStringArray rather than an
    # ndarray, which the mask-trimming loop in model.py would then skip -- leaving
    # that array at full-grid size and breaking downstream broadcasts.
    # Optional metadata or disabled-ISTARF variables absent from the file default to NaN
    # (or 'generic' for reservoir_behavior) sized to the grid (#109).
    for key, value in variables.items():
        if value is not None and value in reservoirs.columns:
            setattr(self, key, np.asarray(reservoirs[value].values))
        elif key == 'reservoir_behavior':
            setattr(self, key, np.full(self.id.size, 'generic', dtype=object))
        else:
            setattr(self, key, np.full(self.id.size, np.nan, dtype=np.float64))

    # correct the fields with different units
    # surface area from km^2 to m^2
    self.reservoir_surface_area = self.reservoir_surface_area * 1.0e6
    # capacity from millions m^3 to m^3
    self.reservoir_storage_capacity = self.reservoir_storage_capacity * 1.0e6

    # minimum storage: use CAP_MIN [million m3] from the file when it gives a usable
    # value, otherwise fall back to reservoir_default_minimum_storage_fraction * storage_capacity.
    # a non-positive CAP_MIN is treated as missing rather than as a real zero floor: the
    # published sample data carries CAP_MIN = 0 for a handful of reservoirs, and honoring
    # that would let them draw down to empty, which is a regression against the historical
    # 10%-of-capacity floor that applied before this column was read at all.
    cap_min_col = config.get('water_management.reservoirs.parameters.minimum_storage_variable', 'CAP_MIN')
    if cap_min_col in reservoirs.columns and reservoirs[cap_min_col].notna().any():
        cap_min_m3 = np.asarray(reservoirs[cap_min_col].values, dtype=np.float64) * 1.0e6
        fallback = parameters.reservoir_default_minimum_storage_fraction * self.reservoir_storage_capacity
        usable = np.isfinite(cap_min_m3) & (cap_min_m3 > 0)
        self.reservoir_minimum_storage = np.where(usable, cap_min_m3, fallback)
    else:
        self.reservoir_minimum_storage = parameters.reservoir_default_minimum_storage_fraction * self.reservoir_storage_capacity

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
