import dask.array as da
import dask.dataframe as dd
import logging
import numpy as np

from xarray import open_dataset

def _load_grid(self):
    # load grid into dataframe
    # TODO clean grid file and precalculate and store cell, upstream, downstream, and outlet indices
    logging.info('Loading grid file.')
    
    # open dataset
    grid_dataset = open_dataset(self.config.get('grid.path'))
    
    # create grid from longitude and latitude dimensions
    self.longitude = da.array(grid_dataset[self.config.get('grid.longitude')]).compute()
    self.latitude = da.array(grid_dataset[self.config.get('grid.latitude')]).compute()
    self.cell_count = self.longitude.size * self.latitude.size
    self.longitude_spacing = abs(self.longitude[1] - self.longitude[0])
    self.latitude_spacing = abs(self.latitude[1] - self.latitude[0])
    longitude, latitude = da.meshgrid(
        grid_dataset[self.config.get('grid.longitude')],
        grid_dataset[self.config.get('grid.latitude')]
    )
    
    # create dataframe from dimensions and data
    grid_dataframe = dd.from_array(
        longitude.flatten(), columns=['longitude']
    ).join(
        dd.from_array(latitude.flatten(), columns=['latitude'])
    ).persist()
    for frame in [
        dd.from_array(da.array(grid_dataset[value]).flatten(), columns=[key]) for key, value in self.config.get('grid.variables').items()
    ]:
        grid_dataframe = grid_dataframe.join(frame).persist()
    
    # use ID and dnID field to calculate masks, upstream, downstream, and outlet indices, as well as count of upstream cells
    logging.debug(' - masks, downstream, upstream, and outlet cell indices')
    
    # ocean/land mask
    # 1 == land
    # 2 == ocean
    # 3 == ocean outlet from land
    # rtmCTL%mask
    grid_dataframe = grid_dataframe.join(dd.from_array(da.where(
        da.array(grid_dataframe.downstream_id) >= 0,
        1,
        da.where(
            da.isin(grid_dataframe.id, grid_dataframe.downstream_id),
            3,
            2
        )
    ), columns=['land_mask'])).persist()
    
    # TODO this is basically the same as the above... should consolidate code to just use one of these masks
    # mosart ocean/land mask
    # 0 == ocean
    # 1 == land
    # 2 == outlet
    # TUnit%mask
    grid_dataframe = grid_dataframe.join(dd.from_array(da.array(da.where(
        da.array(grid_dataframe.flow_direction) < 0,
        0,
        da.where(
            da.array(grid_dataframe.flow_direction) == 0,
            2,
            da.where(
                da.array(grid_dataframe.flow_direction) > 0,
                1,
                0
            )
        )
    ), dtype=int), columns=['mosart_mask'])).persist()
    
    # determine final downstream outlet of each cell
    # this essentially slices up the grid into discrete basins
    # first remap cell ids into cell indices for the (reshaped) 1d grid
    id_hashmap = {}
    for i, _id in enumerate(da.array(grid_dataframe.id).compute()):
        id_hashmap[int(_id)] = int(i)
    # convert downstream ids into downstream indices
    downstream_ids = da.array(grid_dataframe.downstream_id.map(
        lambda i: id_hashmap[int(i)] if int(i) in id_hashmap else -1
    )).compute()
    # follow each cell downstream to compute outlet id
    mask = da.array(grid_dataframe.land_mask).compute()
    size = mask.size
    outlet_ids = np.full(size, -1)
    upstream_ids = np.full(size, -1)
    upstream_cell_counts = np.full(size, 0)
    for i in np.arange(size):
        if downstream_ids[i] >= 0:
            # mark as upstream cell of downstream cell
            upstream_ids[downstream_ids[i]] = i
        if mask[i] == 1:
            # land
            j = i
            while mask[j] == 1:
                upstream_cell_counts[j] += 1
                j = int(downstream_ids[j])
            if mask[j] == 3:
                # found the ocean outlet
                upstream_cell_counts[j] += 1
                outlet_ids[i] = j
        else:
            # ocean
            upstream_cell_counts[i] += 1
            outlet_ids[i] = i
    
    # recalculate area to fill in missing values
    # assumes grid spacing is in degrees and uniform
    logging.debug(' - area')
    radius_earth = 6.37122e6
    deg2rad = np.pi / 180.0
    lats = da.array(grid_dataframe.latitude)
    grid_dataframe = grid_dataframe.join(dd.from_array(da.where(
        da.array(grid_dataframe.local_drainage_area) <= 0,
        da.absolute(
            radius_earth ** 2 * deg2rad * self.get_grid_spacing()[1] * da.subtract(
                da.sin(deg2rad * (lats + 0.5 * self.get_grid_spacing()[0])),
                da.sin(deg2rad * (lats - 0.5 * self.get_grid_spacing()[0]))
            )
        ),
        da.array(grid_dataframe.local_drainage_area),
    ), columns=['area'])).persist()
    
    # drop columns that aren't needed
    grid_dataframe = grid_dataframe.drop(
        'downstream_id', 1
    ).drop(
        'id', 1
    )
    
    # add the updated ids
    grid_dataframe = grid_dataframe.join(
        dd.from_array(da.array(outlet_ids, dtype=int), columns=['outlet_id'])
    ).join(
        dd.from_array(da.array(downstream_ids, dtype=int), columns=['downstream_id'])
    ).join(
        dd.from_array(da.array(upstream_ids, dtype=int), columns=['upstream_id'])
    ).join(
        dd.from_array(da.array(upstream_cell_counts, dtype=int),  columns=['upstream_cell_count'])
    ).persist()
    
    logging.debug(' - main channel iterations')
    
    # parameter for calculating number of main channel iterations needed
    # phi_r
    phi_main = da.where(
        da.array(grid_dataframe.mosart_mask.gt(0) & grid_dataframe.channel_length.gt(0)).compute(),
        grid_dataframe.total_drainage_area_single * da.sqrt(grid_dataframe.channel_slope) / (grid_dataframe.channel_length * grid_dataframe.channel_width),
        0
    ).persist()
    # sub timesteps needed for main channel
    # numDT_r
    grid_dataframe = grid_dataframe.join(dd.from_array(da.where(
        phi_main >= 10,
        da.maximum(1, da.ceil(1 + self.config.get('simulation.subcycles') * da.log10(phi_main))),
        da.where(
            da.array(grid_dataframe.mosart_mask.gt(0) & grid_dataframe.channel_length.gt(0)).compute(),
            1 + self.config.get('simulation.subcycles'),
            1
        )
    ), columns=['iterations_main_channel'])).persist()
    
    logging.debug(' - subnetwork substeps')
    
    # total main channel length [m]
    # rlenTotal
    grid_dataframe = grid_dataframe.join(dd.from_array(da.array(grid_dataframe.area * grid_dataframe.drainage_density), columns=['total_channel_length'])).persist()
    grid_dataframe.total_channel_length = grid_dataframe.total_channel_length.mask(
        grid_dataframe.channel_length.gt(grid_dataframe.total_channel_length),
        grid_dataframe.channel_length
    ).persist()
    
    # hillslope length [m]
    # constrain hillslope length
    # there is a TODO in fortran mosart that says: "allievate the outlier in drainage density estimation."
    # hlen
    channel_length_minimum = grid_dataframe.area ** (1/2)
    hillslope_max_length = da.maximum(channel_length_minimum, 1000)
    grid_dataframe = grid_dataframe.join(dd.from_array(da.zeros(self.get_grid_size()), columns=['hillslope_length']).hillslope_length.mask(
        grid_dataframe.channel_length.gt(0),
        grid_dataframe.area / grid_dataframe.total_channel_length / 2.0
    ).to_frame()).persist()
    grid_dataframe.hillslope_length = grid_dataframe.hillslope_length.mask(
        grid_dataframe.hillslope_length.gt(hillslope_max_length),
        hillslope_max_length
    ).persist()
    
    # subnetwork channel length [m]
    # tlen
    grid_dataframe = grid_dataframe.join(dd.from_array(da.zeros(self.get_grid_size()), columns=['subnetwork_length']).subnetwork_length.mask(
        grid_dataframe.channel_length.gt(0),
        (grid_dataframe.area / channel_length_minimum / 2.0 - grid_dataframe.hillslope_length).mask(
            channel_length_minimum.le(grid_dataframe.channel_length),
            grid_dataframe.area / grid_dataframe.channel_length / 2.0 - grid_dataframe.hillslope_length
        )
    ).to_frame()).persist()
    grid_dataframe.subnetwork_length = grid_dataframe.subnetwork_length.mask(
        grid_dataframe.subnetwork_length.lt(0),
        0
    ).persist()
    
    # subnetwork channel width (adjusted from input file) [m]
    # twidth
    subnetwork_width = dd.from_array(da.zeros(self.get_grid_size()), columns=['subnetwork_width']).subnetwork_width.mask(
        grid_dataframe.channel_length.gt(0) & grid_dataframe.subnetwork_width.ge(0),
        grid_dataframe.subnetwork_width.mask(
            grid_dataframe.subnetwork_length.gt(0) & ((grid_dataframe.total_channel_length - grid_dataframe.channel_length) / grid_dataframe.subnetwork_length).gt(1),
            self.parameters['subnetwork_width_parameter'] * grid_dataframe.subnetwork_width * ((grid_dataframe.total_channel_length - grid_dataframe.channel_length) / grid_dataframe.subnetwork_length)
        )
    ).persist()
    grid_dataframe.subnetwork_width = subnetwork_width.mask(
        grid_dataframe.subnetwork_length.gt(0) & subnetwork_width.le(0),
        0
    ).persist()
    
    # parameter for calculating number of subnetwork iterations needed
    # phi_t
    phi_sub = da.where(
        da.array(grid_dataframe.subnetwork_length.gt(0)).compute(),
        (grid_dataframe.area * grid_dataframe.subnetwork_slope ** (1/2)) / (grid_dataframe.subnetwork_length * grid_dataframe.subnetwork_width),
        0,
    ).persist()
    
    # sub timesteps needed for subnetwork
    # numDT_t
    grid_dataframe = grid_dataframe.join(dd.from_array(da.where(
        da.array(grid_dataframe.subnetwork_length.gt(0)).compute(),
        da.where(
            phi_sub > 10,
            da.maximum(da.ceil(1 + self.config.get('simulation.subcycles') * da.log10(phi_sub)), 1),
            da.where(
                da.array(grid_dataframe.subnetwork_length.gt(0)).compute(),
                1 + self.config.get('simulation.subcycles'),
                1
            )
        ),
        1
    ), columns=['iterations_subnetwork'])).persist()
    
    # add the dataframe to self and load it into memory
    logging.debug(' - persisting')
    self.grid = grid_dataframe.persist()
    
    # free memory
    grid_dataset.close()