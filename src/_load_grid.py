import logging
import numpy as np
import pandas as pd

from xarray import open_dataset

def _load_grid(self):
    # load grid into dataframe
    # TODO clean grid file and precalculate and store cell, upstream, downstream, and outlet indices
    logging.info('Loading grid file.')
    
    # open dataset
    grid_dataset = open_dataset(self.config.get('grid.path'))
    
    # create grid from longitude and latitude dimensions
    self.longitude = np.array(grid_dataset[self.config.get('grid.longitude')])
    self.latitude = np.array(grid_dataset[self.config.get('grid.latitude')])
    self.cell_count = self.longitude.size * self.latitude.size
    self.longitude_spacing = abs(self.longitude[1] - self.longitude[0])
    self.latitude_spacing = abs(self.latitude[1] - self.latitude[0])
    longitude, latitude = np.meshgrid(
        grid_dataset[self.config.get('grid.longitude')],
        grid_dataset[self.config.get('grid.latitude')]
    )
    
    # create dataframe from dimensions and data
    grid_dataframe = pd.DataFrame(
        longitude.flatten(), columns=['longitude']
    ).join(
        pd.DataFrame(latitude.flatten(), columns=['latitude'])
    )
    for frame in [
        pd.DataFrame(np.array(grid_dataset[value]).flatten(), columns=[key]) for key, value in self.config.get('grid.variables').items()
    ]:
        grid_dataframe = grid_dataframe.join(frame)
    
    # use ID and dnID field to calculate masks, upstream, downstream, and outlet indices, as well as count of upstream cells
    logging.debug(' - masks, downstream, upstream, and outlet cell indices')
    
    # ocean/land mask
    # 1 == land
    # 2 == ocean
    # 3 == ocean outlet from land
    # rtmCTL%mask
    grid_dataframe = grid_dataframe.join(pd.DataFrame(np.where(
        np.array(grid_dataframe.downstream_id) >= 0,
        1,
        np.where(
            np.isin(grid_dataframe.id, grid_dataframe.downstream_id),
            3,
            2
        )
    ), columns=['land_mask']))
    
    # TODO this is basically the same as the above... should consolidate code to just use one of these masks
    # mosart ocean/land mask
    # 0 == ocean
    # 1 == land
    # 2 == outlet
    # TUnit%mask
    grid_dataframe = grid_dataframe.join(pd.DataFrame(np.array(np.where(
        np.array(grid_dataframe.flow_direction) < 0,
        0,
        np.where(
            np.array(grid_dataframe.flow_direction) == 0,
            2,
            np.where(
                np.array(grid_dataframe.flow_direction) > 0,
                1,
                0
            )
        )
    ), dtype=int), columns=['mosart_mask']))
    
    # determine final downstream outlet of each cell
    # this essentially slices up the grid into discrete basins
    # first remap cell ids into cell indices for the (reshaped) 1d grid
    id_hashmap = {}
    for i, _id in enumerate(np.array(grid_dataframe.id)):
        id_hashmap[int(_id)] = int(i)
    # convert downstream ids into downstream indices
    downstream_ids = np.array(grid_dataframe.downstream_id.map(
        lambda i: id_hashmap[int(i)] if int(i) in id_hashmap else -1
    ))
    # follow each cell downstream to compute outlet id
    mask = np.array(grid_dataframe.land_mask)
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
    lats = np.array(grid_dataframe.latitude)
    grid_dataframe = grid_dataframe.join(pd.DataFrame(np.where(
        np.array(grid_dataframe.local_drainage_area) <= 0,
        np.absolute(
            radius_earth ** 2 * deg2rad * self.get_grid_spacing()[1] * (
                np.sin(deg2rad * (lats + 0.5 * self.get_grid_spacing()[0])) - np.sin(deg2rad * (lats - 0.5 * self.get_grid_spacing()[0]))
            )
        ),
        np.array(grid_dataframe.local_drainage_area),
    ), columns=['area']))
    
    # drop columns that aren't needed
    grid_dataframe = grid_dataframe.drop(
        'downstream_id', 1
    ).drop(
        'id', 1
    )
    
    # add the updated ids
    grid_dataframe = grid_dataframe.join(
        pd.DataFrame(np.array(outlet_ids, dtype=int), columns=['outlet_id'])
    ).join(
        pd.DataFrame(np.array(downstream_ids, dtype=int), columns=['downstream_id'])
    ).join(
        pd.DataFrame(np.array(upstream_ids, dtype=int), columns=['upstream_id'])
    ).join(
        pd.DataFrame(np.array(upstream_cell_counts, dtype=int),  columns=['upstream_cell_count'])
    )
    
    # update zero slopes to a small number
    grid_dataframe.hillslope = grid_dataframe.hillslope.mask(
        grid_dataframe.hillslope.le(0),
        self.parameters.hillslope_minimum
    )
    grid_dataframe.subnetwork_slope = grid_dataframe.subnetwork_slope.mask(
        grid_dataframe.subnetwork_slope.le(0),
        self.parameters.subnetwork_slope_minimum
    )
    grid_dataframe.channel_slope = grid_dataframe.channel_slope.mask(
        grid_dataframe.channel_slope.le(0),
        self.parameters.channel_slope_minimum
    )
    
    logging.debug(' - main channel iterations')
    
    # parameter for calculating number of main channel iterations needed
    # phi_r
    phi_main = np.where(
        np.array(grid_dataframe.mosart_mask.gt(0) & grid_dataframe.channel_length.gt(0)),
        grid_dataframe.total_drainage_area_single * np.sqrt(grid_dataframe.channel_slope) / (grid_dataframe.channel_length * grid_dataframe.channel_width),
        0
    )
    # sub timesteps needed for main channel
    # numDT_r
    grid_dataframe = grid_dataframe.join(pd.DataFrame(np.where(
        phi_main >= 10,
        np.maximum(1, np.floor(1 + self.config.get('simulation.subcycles') * np.log10(phi_main))),
        np.where(
            np.array(grid_dataframe.mosart_mask.gt(0) & grid_dataframe.channel_length.gt(0)),
            1 + self.config.get('simulation.subcycles'),
            1
        )
    ), columns=['iterations_main_channel']))
    
    logging.debug(' - subnetwork substeps')
    
    # total main channel length [m]
    # rlenTotal
    grid_dataframe = grid_dataframe.join(pd.DataFrame(np.array(grid_dataframe.area * grid_dataframe.drainage_density), columns=['total_channel_length']))
    grid_dataframe.total_channel_length = grid_dataframe.total_channel_length.mask(
        grid_dataframe.channel_length.gt(grid_dataframe.total_channel_length),
        grid_dataframe.channel_length
    )
    
    # hillslope length [m]
    # constrain hillslope length
    # there is a TODO in fortran mosart that says: "allievate the outlier in drainage density estimation."
    # hlen
    channel_length_minimum = grid_dataframe.area ** (1/2)
    hillslope_max_length = np.maximum(channel_length_minimum, 1000)
    grid_dataframe = grid_dataframe.join(pd.DataFrame(np.zeros(self.get_grid_size()), columns=['hillslope_length']).hillslope_length.mask(
        grid_dataframe.channel_length.gt(0),
        grid_dataframe.area / grid_dataframe.total_channel_length / 2.0
    ).to_frame())
    grid_dataframe.hillslope_length = grid_dataframe.hillslope_length.mask(
        grid_dataframe.hillslope_length.gt(hillslope_max_length),
        hillslope_max_length
    )
    
    # subnetwork channel length [m]
    # tlen
    grid_dataframe = grid_dataframe.join(pd.DataFrame(np.zeros(self.get_grid_size()), columns=['subnetwork_length']).subnetwork_length.mask(
        grid_dataframe.channel_length.gt(0),
        (grid_dataframe.area / channel_length_minimum / 2.0 - grid_dataframe.hillslope_length).mask(
            channel_length_minimum.le(grid_dataframe.channel_length),
            grid_dataframe.area / grid_dataframe.channel_length / 2.0 - grid_dataframe.hillslope_length
        )
    ).to_frame())
    grid_dataframe.subnetwork_length = grid_dataframe.subnetwork_length.mask(
        grid_dataframe.subnetwork_length.lt(0),
        0
    )
    
    # subnetwork channel width (adjusted from input file) [m]
    # twidth
    subnetwork_width = pd.DataFrame(np.zeros(self.get_grid_size()), columns=['subnetwork_width']).subnetwork_width.mask(
        grid_dataframe.channel_length.gt(0) & grid_dataframe.subnetwork_width.ge(0),
        grid_dataframe.subnetwork_width.mask(
            grid_dataframe.subnetwork_length.gt(0) & ((grid_dataframe.total_channel_length - grid_dataframe.channel_length) / grid_dataframe.subnetwork_length).gt(1),
            self.parameters.subnetwork_width_parameter * grid_dataframe.subnetwork_width * ((grid_dataframe.total_channel_length - grid_dataframe.channel_length) / grid_dataframe.subnetwork_length)
        )
    )
    grid_dataframe.subnetwork_width = subnetwork_width.mask(
        grid_dataframe.subnetwork_length.gt(0) & subnetwork_width.le(0),
        0
    )
    
    # parameter for calculating number of subnetwork iterations needed
    # phi_t
    phi_sub = np.where(
        np.array(grid_dataframe.subnetwork_length.gt(0)),
        (grid_dataframe.area * grid_dataframe.subnetwork_slope ** (1/2)) / (grid_dataframe.subnetwork_length * grid_dataframe.subnetwork_width),
        0,
    )
    
    # sub timesteps needed for subnetwork
    # numDT_t
    grid_dataframe = grid_dataframe.join(pd.DataFrame(np.where(
        np.array(grid_dataframe.subnetwork_length.gt(0)),
        np.where(
            phi_sub >= 10,
            np.maximum(np.floor(1 + self.config.get('simulation.subcycles') * np.log10(phi_sub)), 1),
            np.where(
                np.array(grid_dataframe.subnetwork_length.gt(0)),
                1 + self.config.get('simulation.subcycles'),
                1
            )
        ),
        1
    ), columns=['iterations_subnetwork']))
    
    # add the dataframe to self
    self.grid = grid_dataframe
    
    # free memory
    grid_dataset.close()