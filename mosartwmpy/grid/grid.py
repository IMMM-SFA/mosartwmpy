import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import tempfile
from typing import Union
import xarray as xr

from benedict.dicts import benedict as Benedict
from numba.core import types
from numba.typed import Dict
from xarray import open_dataset
from zipfile import ZIP_DEFLATED, ZipFile

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.reservoirs.grid import load_reservoirs

# ignore numpy NaN and invalid warnings
# (i.e. divide by zero and NaN logicals -- in Pandas/Dask, these simply remain NaN instead of becoming infinite)
np.seterr(all='ignore')


class Grid:
    """Class to store grid related values that are constant throughout a simulation."""
    
    # initialize all properties
    id: np.ndarray = np.empty(0)
    nldas_id: np.ndarray = np.empty(0)
    downstream_id: np.ndarray = np.empty(0)
    longitude: np.ndarray = np.empty(0)
    latitude: np.ndarray = np.empty(0)
    unique_longitudes: np.ndarray = np.empty(0)
    unique_latitudes: np.ndarray = np.empty(0)
    cell_count: int = 0
    longitude_spacing: float = 0.0
    latitude_spacing: float = 0.0
    land_mask: np.ndarray = np.empty(0)
    mosart_mask: np.ndarray = np.empty(0)
    area: np.ndarray = np.empty(0)
    outlet_id: np.ndarray = np.empty(0)
    upstream_id: np.ndarray = np.empty(0)
    upstream_cell_count: np.ndarray = np.empty(0)
    land_fraction: np.ndarray = np.empty(0)
    iterations_main_channel: np.ndarray = np.empty(0)
    iterations_subnetwork: np.ndarray = np.empty(0)
    drainage_fraction: np.ndarray = np.empty(0)
    local_drainage_area: np.ndarray = np.empty(0)
    total_drainage_area_multi: np.ndarray = np.empty(0)
    total_drainage_area_single: np.ndarray = np.empty(0)
    flow_direction: np.ndarray = np.empty(0)
    hillslope_manning: np.ndarray = np.empty(0)
    subnetwork_manning: np.ndarray = np.empty(0)
    channel_manning: np.ndarray = np.empty(0)
    hillslope: np.ndarray = np.empty(0)
    hillslope_length: np.ndarray = np.empty(0)
    drainage_density: np.ndarray = np.empty(0)
    subnetwork_slope: np.ndarray = np.empty(0)
    subnetwork_width: np.ndarray = np.empty(0)
    subnetwork_length: np.ndarray = np.empty(0)
    channel_length: np.ndarray = np.empty(0)
    channel_slope: np.ndarray = np.empty(0)
    channel_width: np.ndarray = np.empty(0)
    channel_floodplain_width: np.ndarray = np.empty(0)
    total_channel_length: np.ndarray = np.empty(0)
    grid_channel_depth: np.ndarray = np.empty(0)
    
    # Reservoir related properties
    reservoir_id: np.ndarray = np.empty(0)
    reservoir_grid_index: np.ndarray = np.empty(0)
    reservoir_runoff_capacity: np.ndarray = np.empty(0)
    reservoir_height: np.ndarray = np.empty(0)
    reservoir_length: np.ndarray = np.empty(0)
    reservoir_surface_area: np.ndarray = np.empty(0)
    reservoir_storage_capacity: np.ndarray = np.empty(0)
    reservoir_depth: np.ndarray = np.empty(0)
    reservoir_use_irrigation: np.ndarray = np.empty(0)
    reservoir_use_electricity: np.ndarray = np.empty(0)
    reservoir_use_supply: np.ndarray = np.empty(0)
    reservoir_use_flood_control: np.ndarray = np.empty(0)
    reservoir_use_recreation: np.ndarray = np.empty(0)
    reservoir_use_navigation: np.ndarray = np.empty(0)
    reservoir_use_fish_protection: np.ndarray = np.empty(0)
    reservoir_grand_meanflow_cumecs: np.ndarray = np.empty(0)
    reservoir_observed_meanflow_cumecs: np.ndarray = np.empty(0)
    reservoir_computed_meanflow_cumecs = np.empty(0)
    reservoir_upper_alpha: np.ndarray = np.empty(0)
    reservoir_upper_beta: np.ndarray = np.empty(0)
    reservoir_upper_max: np.ndarray = np.empty(0)
    reservoir_upper_min: np.ndarray = np.empty(0)
    reservoir_upper_mu: np.ndarray = np.empty(0)
    reservoir_lower_alpha: np.ndarray = np.empty(0)
    reservoir_lower_beta: np.ndarray = np.empty(0)
    reservoir_lower_max: np.ndarray = np.empty(0)
    reservoir_lower_min: np.ndarray = np.empty(0)
    reservoir_lower_mu: np.ndarray = np.empty(0)
    reservoir_release_alpha_one: np.ndarray = np.empty(0)
    reservoir_release_alpha_two: np.ndarray = np.empty(0)
    reservoir_release_beta_one: np.ndarray = np.empty(0)
    reservoir_release_beta_two: np.ndarray = np.empty(0)
    reservoir_release_c: np.ndarray = np.empty(0)
    reservoir_release_max: np.ndarray = np.empty(0)
    reservoir_release_min: np.ndarray = np.empty(0)
    reservoir_release_p_one: np.ndarray = np.empty(0)
    reservoir_release_p_two: np.ndarray = np.empty(0)
    reservoir_behavior: np.ndarray = np.empty(0)
    reservoir_dependency_database: pd.DataFrame = pd.DataFrame()
    grid_index_to_reservoirs_map: Dict = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:],
    )
    reservoir_streamflow_schedule: xr.DataArray = xr.DataArray()
    reservoir_demand_schedule: xr.DataArray = xr.DataArray()
    reservoir_prerelease_schedule: xr.DataArray = xr.DataArray()
    
    def __init__(self, config: Benedict = None, parameters: Parameters = None, empty: bool = False):
        """Initialize the Grid class.
        
        Args:
            config (Benedict): the model configuration
            parameters (Parameters): the model parameters
            empty (bool): if true will return an empty instance
        """
        
        # shortcut to get an empty grid instance
        if empty:
            return
        
        # open dataset
        grid_dataset = open_dataset(config.get('grid.path'))
    
        # create grid from longitude and latitude dimensions
        self.unique_longitudes = np.array(grid_dataset[config.get('grid.longitude')])
        self.unique_latitudes = np.array(grid_dataset[config.get('grid.latitude')])
        self.cell_count = self.unique_longitudes.size * self.unique_latitudes.size
        self.longitude_spacing = abs(self.unique_longitudes[1] - self.unique_longitudes[0])
        self.latitude_spacing = abs(self.unique_latitudes[1] - self.unique_latitudes[0])
        self.longitude, self.latitude = np.meshgrid(
            grid_dataset[config.get('grid.longitude')],
            grid_dataset[config.get('grid.latitude')]
        )
        self.longitude = self.longitude.flatten()
        self.latitude = self.latitude.flatten()
        
        for key, value in config.get('grid.variables').items():
            setattr(self, key, np.array(grid_dataset[value]).flatten())
        
        # free memory
        grid_dataset.close()
        
        # use ID and dnID field to calculate masks, upstream, downstream, and outlet indices, as well as count of upstream cells
    
        # ocean/land mask
        # 1 == land
        # 2 == ocean
        # 3 == ocean outlet from land
        # rtmCTL%mask
        self.land_mask = np.where(
            self.downstream_id >= 0,
            1,
            np.where(
                np.isin(self.id, self.downstream_id),
                3,
                2
            )
        ).astype(np.int64)
        
        # TODO this is basically the same as the above... should consolidate code to just use one of these masks
        # mosart ocean/land mask
        # 0 == ocean
        # 1 == land
        # 2 == outlet
        # TUnit%mask
        self.mosart_mask = np.where(
            np.array(self.flow_direction) < 0,
            np.where(
                self.land_fraction > 0,
                2,
                0,
            ),
            np.where(
                np.array(self.flow_direction) == 0,
                2,
                np.where(
                    np.array(self.flow_direction) > 0,
                    1,
                    0
                )
            )
        ).astype(np.int64)
        
        # determine final downstream outlet of each cell
        # this essentially slices up the grid into discrete basins
        # first remap cell ids into cell indices for the (reshaped) 1d grid
        id_hashmap = {}
        for i, _id in enumerate(self.id):
            id_hashmap[int(_id)] = int(i)
        # convert downstream ids into downstream indices
        self.downstream_id = np.array([id_hashmap[int(i)] if int(i) in id_hashmap else -1 for i in self.downstream_id], dtype=np.int64)
        # update the id to be zero-indexed (note this makes them one less than fortran mosart ids)
        self.id = np.arange(self.id.size).astype(np.int64)
        
        # follow each cell downstream to compute outlet id
        size = self.downstream_id.size
        self.outlet_id = np.full(size, -1, dtype=np.int64)
        self.upstream_id = np.full(size, -1, dtype=np.int64)
        self.upstream_cell_count = np.full(size, 0, dtype=np.int64)
        for i in np.arange(size):
            if self.downstream_id[i] >= 0:
                # mark as upstream cell of downstream cell
                self.upstream_id[self.downstream_id[i]] = i
            if self.land_mask[i] == 1:
                # land
                j = i
                while self.land_mask[j] == 1:
                    self.upstream_cell_count[j] += 1
                    j = int(self.downstream_id[j])
                if self.land_mask[j] == 3:
                    # found the ocean outlet
                    self.upstream_cell_count[j] += 1
                    self.outlet_id[i] = j
            else:
                # ocean
                self.upstream_cell_count[i] += 1
                self.outlet_id[i] = i

        # if a subdomain is desired, update mosart_mask to disable out-of-subdomain cells
        subdomain = config.get('grid.subdomain', None)
        if subdomain is not None:
            if not isinstance(subdomain, list):
                subdomain = [subdomain]
            outlet_ids = set()
            for point in subdomain:
                point = [float(x) for x in point.split(',')]
                distance = Grid.haversine(self.latitude, self.longitude, point[0], point[1])
                index = np.argmin(distance)
                outlet_ids.add(self.outlet_id[index])
            self.mosart_mask = np.where(
                np.in1d(self.outlet_id, list(outlet_ids)),
                self.mosart_mask,
                0
            )

        # recalculate area to fill in missing values
        # assumes grid spacing is in degrees and uniform
        deg2rad = np.pi / 180.0
        self.area = np.where(
            self.local_drainage_area <= 0,
            np.absolute(
                parameters.radius_earth ** 2 * deg2rad * self.longitude_spacing * (
                    np.sin(deg2rad * (self.latitude + 0.5 * self.latitude_spacing)) - np.sin(deg2rad * (self.latitude - 0.5 * self.latitude_spacing))
                )
            ),
            self.local_drainage_area,
        )
        
        # update zero slopes to a small number
        self.hillslope = np.where(
            self.hillslope <= 0,
            parameters.hillslope_minimum,
            self.hillslope
        )
        self.subnetwork_slope = np.where(
            self.subnetwork_slope <= 0,
            parameters.subnetwork_slope_minimum,
            self.subnetwork_slope
        )
        self.channel_slope = np.where(
            self.channel_slope <= 0,
            parameters.channel_slope_minimum,
            self.channel_slope
        )

        # parameter for calculating number of main channel iterations needed
        # phi_r
        phi_main = np.where(
            (self.mosart_mask > 0) & (self.channel_length > 0),
            self.total_drainage_area_single * np.sqrt(self.channel_slope) / (self.channel_length * self.channel_width),
            0
        )
        # sub timesteps needed for main channel
        # numDT_r
        self.iterations_main_channel = np.where(
            phi_main >= 10,
            np.maximum(1, np.floor(1 + config.get('simulation.subcycles') * np.log10(phi_main))),
            np.where(
                (self.mosart_mask > 0) & (self.channel_length > 0),
                1 + config.get('simulation.subcycles'),
                1
            )
        ).astype(np.int64)

        # total main channel length [m]
        # rlenTotal
        self.total_channel_length = self.area * self.drainage_density
        self.total_channel_length = np.where(
            self.channel_length > self.total_channel_length,
            self.channel_length,
            self.total_channel_length
        )

        # hillslope length [m]
        # constrain hillslope length
        # there is a TODO in fortran mosart that says: "allievate the outlier in drainage density estimation."
        # hlen
        channel_length_minimum = np.sqrt(self.area)
        hillslope_max_length = np.maximum(channel_length_minimum, 1000)
        self.hillslope_length = np.where(
            self.channel_length > 0,
            self.area / self.total_channel_length / 2.0,
            0
        )
        self.hillslope_length = np.where(
            self.hillslope_length > hillslope_max_length,
            hillslope_max_length,
            self.hillslope_length
        )

        # subnetwork channel length [m]
        # tlen
        self.subnetwork_length = np.where(
            self.channel_length > 0,
            np.where(
                self.channel_length >= channel_length_minimum,
                self.area / self.channel_length / 2.0 - self.hillslope_length,
                self.area / channel_length_minimum / 2.0 - self.hillslope_length
            ),
            0
        )
        self.subnetwork_length = np.where(
            self.subnetwork_length < 0,
            0,
            self.subnetwork_length
        )

        # subnetwork channel width (adjusted from input file) [m]
        # twidth
        self.subnetwork_width = np.where(
            (self.channel_length > 0) & (self.subnetwork_width >= 0),
            np.where(
                (self.subnetwork_length > 0) & ((self.total_channel_length - self.channel_length) / self.subnetwork_length > 1),
                parameters.subnetwork_width_parameter * self.subnetwork_width * ((self.total_channel_length - self.channel_length) / self.subnetwork_length),
                self.subnetwork_width
            ),
            0
        )
        self.subnetwork_width = np.where(
            (self.subnetwork_length > 0) & (self.subnetwork_width <= 0),
            0,
            self.subnetwork_width
        )

        # parameter for calculating number of subnetwork iterations needed
        # phi_t
        phi_sub = np.where(
            self.subnetwork_length > 0,
            (self.area * np.sqrt(self.subnetwork_slope)) / (self.subnetwork_length * self.subnetwork_width),
            0,
        )

        # sub timesteps needed for subnetwork
        # numDT_t
        self.iterations_subnetwork = np.where(
            self.subnetwork_length > 0,
            np.where(
                phi_sub >= 10,
                np.maximum(np.floor(1 + config.get('simulation.subcycles') * np.log10(phi_sub)), 1),
                np.where(
                    self.subnetwork_length > 0,
                    1 + config.get('simulation.subcycles'),
                    1
                )
            ),
            1
        ).astype(np.int64)

        # if water management is enabled, load the reservoir parameters and build the grid cell mapping
        # note that reservoir grid is assumed to be the same as the domain grid
        if config.get('water_management.enabled', False):
            load_reservoirs(self, config, parameters)

    def __getitem__(self, item):
        return getattr(self, item)

    def to_files(self, path: str, mask: np.ndarray) -> None:
        """Builds a dataframe from all the grid values.
        
        Args:
            path (str): the file path to save the grid zip file to
            mask (ndarray): the mask applied by the model
        """
        
        if not path.endswith('.zip'):
            path += '.zip'
        
        keys = dir(self)
        
        paths = []
        names = []
        
        # handle special cases
        special = ['unique_longitudes', 'unique_latitudes', 'cell_count', 'longitude_spacing', 'latitude_spacing']
        to_pickle = {}
        for key in special:
            keys.remove(key)
            to_pickle[key] = getattr(self, key)
        
        # handle numpy arrrays
        npdf = pd.DataFrame()
        for key in [key for key in keys if isinstance(getattr(self, key), np.ndarray)]:
            if getattr(self, key).size > 0:
                vector = getattr(self, key)
                unmasked = np.empty_like(mask, dtype=vector.dtype)
                if vector.dtype == float:
                    unmasked[:] = np.nan
                elif vector.dtype == int:
                    unmasked[:] = -9999
                elif vector.dtype == bool:
                    unmasked[:] = False
                elif vector.dtype == np.object:
                    unmasked[:] = np.nan
                unmasked[mask] = vector
                npdf[key] = unmasked
        
        # handle dataframes
        dfs = []
        for key in [key for key in keys if isinstance(getattr(self, key), pd.DataFrame)]:
            if getattr(self, key).size > 0:
                dfs.append({
                    'key': key,
                    'frame': getattr(self, key)
                })
        
        # handle xarrays
        xrs = []
        for key in [key for key in keys if isinstance(getattr(self, key), xr.DataArray)]:
            if getattr(self, key).size > 0:
                xrs.append({
                    'key': key,
                    'data_array': getattr(self, key)
                })
        
        # write them all to files and zip
        with tempfile.TemporaryDirectory() as tmpdir:
            names.append('special.pickle')
            paths.append(f'{tmpdir}/{names[-1]}')
            with open(paths[-1], 'wb') as file:
                pickle.dump(to_pickle, file)
            names.append('np.feather')
            paths.append(f'{tmpdir}/{names[-1]}')
            npdf.to_feather(paths[-1])
            for df in dfs:
                names.append(f'{df["key"]}.df.nc')
                paths.append(f'{tmpdir}/{names[-1]}')
                df['frame'].to_xarray().to_netcdf(paths[-1], engine='h5netcdf')
            for ds in xrs:
                names.append(f'{ds["key"]}.xr.nc')
                paths.append(f'{tmpdir}/{names[-1]}')
                ds['data_array'].to_netcdf(paths[-1], engine='h5netcdf')
            with ZipFile(path, 'w', compression=ZIP_DEFLATED, compresslevel=9) as zip:
                for i, filename in enumerate(paths):
                    zip.write(filename, names[i])
    
    @staticmethod
    def from_files(path: Path) -> 'Grid':
        """Creates a Grid instance from columns in a dataframe.

        Args:
            path (str): the file path to the zip file to load the grid from

        Returns:
            Grid: a Grid instance populated with the columns from the dataframe
        """
        if not Path(path).suffix == '.zip':
            path += '.zip'

        grid = Grid(empty=True)
        
        with ZipFile(path, 'r') as zip:
            for filename in zip.namelist():
                with zip.open(filename) as file:
                    if filename.endswith('.pickle'):
                        from_pickle = pickle.load(file)
                        for key in from_pickle.keys():
                            setattr(grid, key, from_pickle[key])
                    if filename.endswith('np.feather'):
                        npdf = pd.read_feather(file)
                        for key in npdf.columns:
                            setattr(grid, key, npdf[key].values)
                    if filename.endswith('df.nc'):
                        key = filename.split('.')[0]
                        ds = xr.open_dataset(file, engine='h5netcdf')
                        df = ds.to_dataframe()
                        setattr(grid, key, df)
                        ds.close()
                    if filename.endswith('xr.nc'):
                        key = filename.split('.')[0]
                        ds = xr.open_dataarray(file, engine='h5netcdf').load()
                        setattr(grid, key, ds)
                        ds.close()

        # recreate the numba grid to reservoir map
        if grid.reservoir_dependency_database.size > 0:
            for grid_cell_id, group in grid.reservoir_dependency_database.reset_index().groupby('grid_cell_id'):
                grid.grid_index_to_reservoirs_map[grid_cell_id] = group.reservoir_id.values
        
        return grid

    @staticmethod
    def haversine(
            lat1: Union[float, np.ndarray],
            lon1: Union[float, np.ndarray],
            lat2: float,
            lon2: float
    ) -> Union[float, np.ndarray]:
        """Calculates the haversine distance between points

                Args:
                    lat1 (float|ArrayLike[float]): latitude of the origin point or array of points
                    lon1 (float|ArrayLike[float]): longitude of the origin point or array of points
                    lat2 (float): latitude of the point in question
                    lon2 (float): longitude of the point in question

                Returns:
                    float or array of floats of the haversine distance between points
                """
        p = 0.017453292519943295
        hav = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
        return 12742 * np.arcsin(np.sqrt(hav))
