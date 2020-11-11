import dask.array as da
import logging
import numpy as np
import sparse
from datetime import datetime, time
from xarray import open_dataset, apply_ufunc

def _initialize_state(self):
    
    # some random constants used throughout the code
    # TODO better document what these are used for and what they should be
    # TINYVALUE
    self.state['tiny_value'] = 1.0e-14
    # a small value in order to avoid abrupt change of hydraulic radius
    self.state['slope_1_def'] = 0.1
    self.state['sin_atan_slope_1_def'] = 1.0 / (da.sin(da.arctan(self.state['slope_1_def'])))
    # what are these parameters? seem like sub-sub-timesteps
    # DLevelH2R
    self.state['dlevelh2r'] = 5
    # DLevelR
    self.state['dlevelr'] = 3

    # restart file
    if self.config.get('simulation.restart_file') is not None and self.config.get('simulation.restart_file') != '':
        self.restart = open_dataset(self.config.get('simulation.restart_file'), chunks={})
        # TODO set current timestep based on restart
        # TODO initialize state from restart file
        logging.error('Restart file not yet implemented. Aborting.')
        raise NotImplementedError

    # initialize all the state variables
    logging.debug('Initializing state variables.')
    
    # current timestep
    self.current_time = datetime.combine(self.config.get('simulation.start_date'), time.min)

    # tracers
    logging.debug(' - tracers')
    self.state['tracers'] = (self.LIQUID_TRACER, self.ICE_TRACER) if self.config.get('water_management.ice_runoff_enabled') else ('LIQUID',)
    self.state[['tracer', self.LIQUID_TRACER]] = da.concatenate([
        da.atleast_3d(da.full(self.get_grid_shape(), True)),
        da.atleast_3d(da.full(self.get_grid_shape(), False))
    ], axis=2).compute() if self.ICE_TRACER in self.state['tracers'] else da.atleast_3d(da.full(self.get_grid_shape(), True)).compute()
    self.state[['tracer', self.ICE_TRACER]] = da.logical_not(self.state[['tracer', self.LIQUID_TRACER]])

    # based on tracers, this is the shape of most matrices
    self.DATA_SHAPE = np.append(self.get_grid_shape(), len(self.state['tracers']))

    logging.debug(' - masks')

    # ocean/land mask
    # 1 == land
    # 2 == ocean
    # 3 == ocean outlet from land
    # rtmCTL%mask
    self.state['mask'] = da.atleast_3d(da.where(
        self.grid[self.config.get('grid.downstream_id')] > 0,
        1,
        da.where(
            da.isin(self.grid[self.config.get('grid.id')], self.grid[self.config.get('grid.downstream_id')]),
            3,
            2
        )
    )).compute()

    # mosart ocean/land mask
    # 0 == ocean
    # 1 == land
    # 2 == outlet
    # TUnit%mask
    self.state['mosart_mask'] = da.atleast_3d(da.where(
        self.grid[self.config.get('grid.flow_direction')] < 0,
        0,
        da.where(
            self.grid[self.config.get('grid.flow_direction')] == 0,
            2,
            da.where(
                da.logical_not(self.grid[self.config.get('grid.flow_direction')] > 0),
                da.empty(self.get_grid_shape()),
                1
            )
        )
    )).compute()

    logging.debug(' - downstream, upstream, and outlet cell indices')

    # determine final downstream outlet of each cell
    # this essentially slices up the grid into discrete basins
    # first remap cell ids into cell indices for the (reshaped) 1d grid
    id_hashmap = {}
    for i, _id in enumerate(self.grid[self.config.get('grid.id')].data.flatten().compute()):
        id_hashmap[int(_id)] = int(i)
    # convert downstream ids into downstream indices
    downstream_ids = apply_ufunc(
        lambda i: id_hashmap[int(i)] if int(i) in id_hashmap else int(i),
        self.grid[self.config.get('grid.downstream_id')],
        vectorize=True, dask='parallelized'
    ).data.flatten().compute()
    # follow each cell downstream to compute outlet id
    mask = self.state['mask'][:,:,0].flatten()
    outlet_ids = np.empty(self.get_grid_size())
    upstream_ids = np.full(self.get_grid_size(), -9999)
    upstream_cell_counts = np.full(self.get_grid_size(), 0)
    for i in np.arange(self.get_grid_size()):
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
    
    self.state['outlet_id'] = outlet_ids.reshape(self.get_grid_shape())
    self.state['downstream_id'] = downstream_ids.reshape(self.get_grid_shape())
    self.state['upstream_id'] = upstream_ids.reshape(self.get_grid_shape())

    # mask on whether or not to perform euler calculations
    logging.debug(' - euler mask')
    self.state['euler_mask'] = da.where(
        self.state[['tracer', self.LIQUID_TRACER]],
        True,
        False
    ).compute()

    # local drainage area from domain file, adjusted to fill in missing values
    # assumes grid spacing is in degrees and uniform
    # area
    logging.debug(' - area')
    radius_earth = 6.37122e6
    deg2rad = 0.0174533
    lats = da.tile(self.grid[self.config.get('grid.latitude')], (self.get_grid_shape()[1], 1)).transpose().flatten()
    self.state['area'] = da.where(
        self.grid[self.config.get('grid.local_drainage_area')] <= 0,
        da.absolute(
            radius_earth * radius_earth * (
                deg2rad * self.get_grid_spacing()[1]
            ) * da.subtract(
                da.sin(deg2rad * (lats + 0.5 * self.get_grid_spacing()[0])),
                da.sin(deg2rad * (lats - 0.5 * self.get_grid_spacing()[0]))
            )
        ).reshape(self.get_grid_shape()),
        self.grid[self.config.get('grid.local_drainage_area')]
    )
    
    logging.debug(' - state variables')

    # effective tracer velocity [m/s]
    # evel
    self.state['effective_tracer_velocity'] = da.full(self.DATA_SHAPE, 10)

    # water flood threshold [m3]
    # fthresh
    self.state['flood_threshold'] = da.full(self.DATA_SHAPE, 1e36)

    # flow [m3/s]
    # flow
    self.state['flow'] = da.zeros(self.DATA_SHAPE)

    # outflow into downstream links from previous timestep [m3/s]
    # eroup_lagi
    self.state['outlow_downstream_previous_timestep'] = da.zeros(self.DATA_SHAPE)

    # outflow into downstream links from current timestep [m3/s]
    # eroup_lagf
    self.state['outflow_downstream_current_timestep'] = da.zeros(self.DATA_SHAPE)

    # initial outflow before dam regulation at current timestep [m3/s]
    # erowm_regi
    self.state['outflow_before_regulation'] = da.zeros(self.DATA_SHAPE)

    # final outflow after dam regulation at current timestep [m3/s]
    # erowm_regf
    self.state['outflow_after_regulation'] = da.zeros(self.DATA_SHAPE)

    # outflow sum of upstream gridcells, average [m3/s]
    # eroutUp_avg
    self.state['outflow_sum_upstream_average'] = da.zeros(self.DATA_SHAPE)

    # lateral flow from hillslope, including surface and subsurface runoff generation components, average [m3/s]
    # erlat_avg
    self.state['lateral_flow_hillslope_average'] = da.zeros(self.DATA_SHAPE)

    # routing storage [m3]
    # volr
    self.state['storage'] = da.zeros(self.DATA_SHAPE)

    # routing change in storage [m3/s]
    # dvolrdt
    self.state['delta_storage'] = da.zeros(self.DATA_SHAPE)

    # routing change in storage masked for land [m3/s]
    self.state['delta_storage_land'] = da.zeros(self.DATA_SHAPE)

    # routing change in storage masked for ocean [m3/s]
    self.state['delta_storage_ocean'] = da.zeros(self.DATA_SHAPE)

    # basin derived flow [m3/s]
    # runoff
    self.state['runoff'] = da.zeros(self.DATA_SHAPE)
    
    # return direct flow [m3/s]
    # runofftot
    self.state['runoff_total'] = da.zeros(self.DATA_SHAPE)

    # runoff masked for land [m3/s]
    # runofflnd
    self.state['runoff_land'] = da.empty(self.DATA_SHAPE)

    # runoff masked for ocean [m3/s]
    # runoffocn
    self.state['runoff_ocean'] = da.empty(self.DATA_SHAPE)

    # total runoff masked for ocean [m3/s]
    self.state['total_runoff_ocean'] = da.empty(self.DATA_SHAPE)

    # direct flow [m3/s]
    # direct
    self.state['direct'] = da.zeros(self.DATA_SHAPE)

    # direct-to-ocean forcing [m3/s]
    # qdto
    self.state['direct_to_ocean'] = da.zeros(self.DATA_SHAPE)

    # flood water [m3/s]
    # flood
    self.state['flood'] = da.zeros(self.DATA_SHAPE)

    # hillslope surface water storage [m]
    # wh
    self.state['hillslope.storage'] = da.zeros(self.DATA_SHAPE)

    # change of hillslope water storage [m/s]
    # dwh
    self.state['hillslope.delta_storage'] = da.zeros(self.DATA_SHAPE)

    # depth of hillslope surface water [m]
    # yh
    self.state['hillslope.depth'] = da.zeros(self.DATA_SHAPE)

    # surface runoff from hillslope [m/s]
    # qsur
    self.state['hillslope.surface_runoff'] = da.zeros(self.DATA_SHAPE)

    # subsurface runoff from hillslope [m/s]
    # qsub
    self.state['hillslope.subsurface_runoff'] = da.zeros(self.DATA_SHAPE)

    # runoff from glacier, wetlands, and lakes [m/s]
    # qgwl
    self.state['hillslope.wetland_runoff'] = da.zeros(self.DATA_SHAPE)

    # overland flor from hillslope into subchannel (outflow is negative) [m/s]
    # ehout
    self.state['hillslope.overland_flow'] = da.zeros(self.DATA_SHAPE)

    # subnetwork area of water surface [m2]
    # tarea
    self.state['subnetwork.area'] = da.zeros(self.DATA_SHAPE)

    # subnetwork water storage [m3]
    # wt
    self.state['subnetwork.storage'] = da.zeros(self.DATA_SHAPE)
    
    # subnetwork water storage at previous timestep [m3]
    # wt_last
    self.state['subnetwork.storage_previous_timestep'] = da.zeros(self.DATA_SHAPE)

    # change of subnetwork water storage [m3]
    # dwt
    self.state['subnetwork.delta_storage'] = da.zeros(self.DATA_SHAPE)

    # depth of subnetwork water [m]
    # yt
    self.state['subnetwork.depth'] = da.zeros(self.DATA_SHAPE)

    # cross section area of subnetwork [m2]
    # mt
    self.state['subnetwork.cross_section_area'] = da.zeros(self.DATA_SHAPE)

    # hydraulic radii of subnetwork [m]
    # rt
    self.state['subnetwork.hydraulic_radii'] = da.zeros(self.DATA_SHAPE)

    # wetness perimeter of subnetwork [m]
    # pt
    self.state['subnetwork.wetness_perimeter'] = da.zeros(self.DATA_SHAPE)

    # subnetwork flow velocity [m/s]
    # vt
    self.state['subnetwork.flow_velocity'] = da.zeros(self.DATA_SHAPE)

    # subnetwork mean travel time of water within travel [s]
    # tt
    self.state['subnetwork.mean_travel_time'] = da.zeros(self.DATA_SHAPE)

    # subnetwork evaporation [m/s]
    # tevap
    self.state['subnetwork.evaporation'] = da.zeros(self.DATA_SHAPE)

    # subnetwork lateral inflow from hillslope [m3/s]
    # etin
    self.state['subnetwork.lateral_inflow'] = da.zeros(self.DATA_SHAPE)

    # subnetwork discharge into main channel (outflow is negative) [m3/s]
    # etout
    self.state['subnetwork.discharge'] = da.zeros(self.DATA_SHAPE)

    # irrigation demand [m/s]
    # qdem
    self.state['subnetwork.irrigation_demand'] = da.zeros(self.DATA_SHAPE)

    # main channel area [m/2]
    # rarea
    self.state['channel.area'] = da.zeros(self.DATA_SHAPE)

    # main channel storage [m3]
    # wr
    self.state['channel.storage'] = da.zeros(self.DATA_SHAPE)

    # change in main channel storage [m3]
    # dwr
    self.state['channel.delta_storage'] = da.zeros(self.DATA_SHAPE)

    # main channel storage at last timestep [m3]
    # wr_last
    self.state['channel.storage_previous_timestep'] = da.zeros(self.DATA_SHAPE)

    # main channel water depth [m]
    # yr
    self.state['channel.depth'] = da.zeros(self.DATA_SHAPE)

    # cross section area of main channel [m2]
    # mr
    self.state['channel.cross_section_area'] = da.zeros(self.DATA_SHAPE)

    # hydraulic radii of main channel [m]
    # rr
    self.state['channel.hydraulic_radii'] = da.zeros(self.DATA_SHAPE)

    # wetness perimeter of main channel[m]
    # pr
    self.state['channel.wetness_perimeter'] = da.zeros(self.DATA_SHAPE)

    # main channel flow velocity [m/s]
    # vr
    self.state['channel.flow_velocity'] = da.zeros(self.DATA_SHAPE)

    # main channel mean travel time of water within travel [s]
    # tr
    self.state['channel.mean_travel_time'] = da.zeros(self.DATA_SHAPE)

    # main channel evaporation [m/s]
    # erlg
    self.state['channel.evaporation'] = da.zeros(self.DATA_SHAPE)

    # lateral flow from hillslope [m3/s]
    # erlateral
    self.state['channel.lateral_flow_hillslope'] = da.zeros(self.DATA_SHAPE)

    # inflow from upstream links [m3/s]
    # erin
    self.state['channel.inflow_upstream'] = da.zeros(self.DATA_SHAPE)

    # outflow into downstream links [m3/s]
    # erout
    self.state['channel.outflow_downstream'] = da.zeros(self.DATA_SHAPE)

    # outflow into downstream links from previous timestep [m3/s]
    # TRunoff%eroup_lagi
    self.state['channel.outlow_downstream_previous_timestep'] = da.zeros(self.DATA_SHAPE)

    # outflow into downstream links from current timestep [m3/s]
    # TRunoff%eroup_lagf
    self.state['channel.outflow_downstream_current_timestep'] = da.zeros(self.DATA_SHAPE)

    # initial outflow before dam regulation at current timestep [m3/s]
    # TRunoff%erowm_regi
    self.state['channel.outflow_before_regulation'] = da.zeros(self.DATA_SHAPE)

    # final outflow after dam regulation at current timestep [m3/s]
    # TRunoff%erowm_regf
    self.state['channel.outflow_after_regulation'] = da.zeros(self.DATA_SHAPE)

    # outflow sum of upstream gridcells, instantaneous [m3/s]
    # eroutUp
    self.state['channel.outflow_sum_upstream_instant'] = da.zeros(self.DATA_SHAPE)

    # outflow sum of upstream gridcells, average [m3/s]
    # TRunoff%eroutUp_avg
    self.state['channel.outflow_sum_upstream_average'] = da.zeros(self.DATA_SHAPE)

    # lateral flow from hillslope, including surface and subsurface runoff generation components, average [m3/s]
    # TRunoff%erlat_avg
    self.state['channel.lateral_flow_hillslope_average'] = da.zeros(self.DATA_SHAPE)
    
    # flux for adjustment of water balance residual in glacier, wetlands, and lakes [m3/s]
    # ergwl
    self.state['channel.wetland_flux'] = da.zeros(self.DATA_SHAPE)

    # streamflow from outlet, positive is out [m3/s]
    # flow
    self.state['channel.flow'] = da.zeros(self.DATA_SHAPE)

    logging.debug(' - main channel substeps')

    # parameter for calculating number of main channel subtimesteps
    # phi_r
    phi_main = da.where(
        da.logical_and(
            self.state['mosart_mask'][:,:,0] > 0,
            self.grid[self.config.get('grid.channel_length')] > 0
        ),
        da.divide(
            da.multiply(
                self.grid[self.config.get('grid.total_drainage_area_single')],
                da.sqrt(self.grid[self.config.get('grid.channel_slope')])
            ),
            da.multiply(
                self.grid[self.config.get('grid.channel_length')],
                self.grid[self.config.get('grid.channel_width')]
            )
        ),
        da.zeros(self.get_grid_shape())
    )

    # sub timesteps needed for main channel
    # numDT_r
    self.state['subtimesteps.main'] = da.where(
        phi_main >= 10,
        1 + self.state['dlevelr'] * da.log10(phi_main),
        da.where(
            da.logical_and(
                self.state['mosart_mask'][:,:,0] > 0,
                self.grid[self.config.get('grid.channel_length')] > 0
            ),
            da.full(self.get_grid_shape(), self.state['dlevelr'] + 1),
            da.full(self.get_grid_shape(), 1)
        )
    )
    self.state['subtimesteps.main'] = da.ceil(da.where(
        self.state['subtimesteps.main'] < 1,
        1,
        self.state['subtimesteps.main']
    ))
    
    logging.debug(' - subnetwork substeps')

    # total main channel length [m]
    # rlenTotal
    total_channel_length = da.where(
        self.grid[self.config.get('grid.drainage_density')] > 0,
        da.multiply(
            self.state['area'],
            self.grid[self.config.get('grid.drainage_density')]
        ),
        0
    )
    total_channel_length = da.where(
        self.grid[self.config.get('grid.channel_length')] > total_channel_length,
        self.grid[self.config.get('grid.channel_length')],
        total_channel_length
    )

    # hillslope length [m]
    # hlen
    self.state['hillslope.length'] = da.where(
        self.grid[self.config.get('grid.channel_length')] > 0,
        da.true_divide(
            da.true_divide(
                self.state['area'],
                total_channel_length
            ),
            2
        ),
        0
    )
    # constrain hillslope length
    # there is a TODO in fortran mosart that says: "allievate the outlier in drainage density estimation."
    channel_length_minimum = da.sqrt(self.state['area'])
    hillslope_max_length = da.where(
        channel_length_minimum > 1000,
        1000,
        channel_length_minimum
    )
    self.state['hillslope.length'] = da.where(
        self.state['hillslope.length'] > hillslope_max_length,
        hillslope_max_length,
        self.state['hillslope.length']
    )

    # subnetwork channel length [m]
    # tlen
    self.state['subnetwork.length'] = da.where(
        self.grid[self.config.get('grid.channel_length')] > 0,
        da.where(
            self.grid[self.config.get('grid.channel_length')] < channel_length_minimum,
            da.subtract(
                da.true_divide(
                    self.state['area'],
                    channel_length_minimum
                ) / 2,
                self.state['hillslope.length']
            ),
            da.subtract(
                da.true_divide(
                    self.state['area'],
                    self.grid[self.config.get('grid.channel_length')]
                ) / 2,
                self.state['hillslope.length']
            )
        ),
        0
    )

    # subnetwork channel width (adjusted from input file) [m]
    # twidth
    c_twid = 1 # TODO what is this
    self.state['subnetwork.width'] = da.where(
        self.grid[self.config.get('grid.channel_length')] > 0,
        da.where(
            self.grid[self.config.get('grid.subnetwork_width')] < 0,
            0,
            da.where(
                self.state['subnetwork.length'] > 0,
                da.where(
                    da.subtract(
                        total_channel_length,
                        da.divide(
                            self.grid[self.config.get('grid.channel_length')],
                            self.state['subnetwork.length']
                        )
                    ) > 1,
                    da.subtract(
                        da.multiply(
                            c_twid * self.grid[self.config.get('grid.subnetwork_width')],
                            total_channel_length
                        ),
                        da.divide(
                            self.grid[self.config.get('grid.channel_length')],
                            self.state['subnetwork.length']
                        )
                    ),
                    self.grid[self.config.get('grid.subnetwork_width')]
                ),
                0
            )
        ),
        0
    )
    self.state['subnetwork.width'] = da.where(
        da.logical_and(self.state['subnetwork.length'] > 0, self.state['subnetwork.width'] < 0),
        0,
        self.state['subnetwork.width'],
    )

    # sub timestep indicator for subnetwork
    # phi_t
    phi_sub = da.where(
        self.state['subnetwork.length'] > 0,
        da.true_divide(
            da.multiply(
                self.state['area'],
                da.sqrt(self.grid[self.config.get('grid.subnetwork_slope')])
            ),
            da.multiply(
                self.state['subnetwork.length'],
                self.state['subnetwork.width']
            )
        ),
        0,
    )

    # sub timesteps needed for subnetwork
    # numDT_t
    self.state['subtimesteps.subnetwork'] = da.where(
        self.state['subnetwork.length'] > 0,
        da.where(
            phi_sub > 10,
            1 + self.state['dlevelr'] * da.log10(phi_sub),
            da.where(
                self.state['subnetwork.length'] > 0,
                1 + self.state['dlevelr'],
                1
            )
        ),
        1,
    )
    self.state['subtimesteps.subnetwork'] = da.ceil(da.where(
        self.state['subtimesteps.subnetwork'] < 1,
        1,
        self.state['subtimesteps.subnetwork']
    ))

    return