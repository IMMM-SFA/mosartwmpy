import dask.array as da
import dask.dataframe as dd
import logging
import numpy as np
from datetime import datetime, time
from xarray import open_dataset

from ._update import update_hillslope_state, update_subnetwork_state, update_main_channel_state

def _initialize_state(self):

    # some constants used throughout the code
    # TODO better document what these are used for and what they should be and maybe they should be part of config?
    # TINYVALUE
    self.parameters['tiny_value'] = 1.0e-14
    # a small value in order to avoid abrupt change of hydraulic radius
    self.parameters['slope_1_def'] = 0.1
    self.parameters['sin_atan_slope_1_def'] = 1.0 / (da.sin(da.arctan(self.parameters['slope_1_def'])))
    # flood threshold - excess water will be sent back to ocean
    self.parameters['flood_threshold'] = 1.0e36 # [m3]?
    # liquid/ice effective velocity
    self.parameters['effective_tracer_velocity'] = 10.0 # [m/s]?
    # minimum river depth
    self.parameters['river_depth_minimum'] = 1.0e-4 # [m]?
    

    # restart file
    if self.config.get('simulation.restart_file') is not None and self.config.get('simulation.restart_file') != '':
        logging.info('Loading restart file.')
        self.restart = open_dataset(self.config.get('simulation.restart_file'), chunks={})
        # TODO set current timestep based on restart
        # TODO initialize state from restart file
        logging.error('Restart file not yet implemented. Aborting.')
        raise NotImplementedError

    # initialize all the state variables
    logging.info('Initializing state variables.')

    # current timestep
    self.current_time = datetime.combine(self.config.get('simulation.start_date'), time.min)

    # initialize state variables
    logging.debug(' - variables')
    state_dataframe = None
    for var in [
        # flow [m3/s]
        # flow
        'flow',
        # outflow into downstream links from previous timestep [m3/s]
        # eroup_lagi
        'outflow_downstream_previous_timestep',
        # outflow into downstream links from current timestep [m3/s]
        # eroup_lagf
        'outflow_downstream_current_timestep',
        # initial outflow before dam regulation at current timestep [m3/s]
        # erowm_regi
        'outflow_before_regulation',
        # final outflow after dam regulation at current timestep [m3/s]
        # erowm_regf
        'outflow_after_regulation',
        # outflow sum of upstream gridcells, average [m3/s]
        # eroutUp_avg
        'outflow_sum_upstream_average',
        # lateral flow from hillslope, including surface and subsurface runoff generation components, average [m3/s]
        # erlat_avg
        'lateral_flow_hillslope_average',
        # routing storage [m3]
        # volr
        'storage',
        # routing change in storage [m3/s]
        # dvolrdt
        'delta_storage',
        # routing change in storage masked for land [m3/s]
        # dvolrdtlnd
        'delta_storage_land',
        # routing change in storage masked for ocean [m3/s]
        # dvolrdtocn
        'delta_storage_ocean',
        # basin derived flow [m3/s]
        # runoff
        'runoff',
        # return direct flow [m3/s]
        # runofftot
        'runoff_total',
        # runoff masked for land [m3/s]
        # runofflnd
        'runoff_land',
        # runoff masked for ocean [m3/s]
        # runoffocn
        'runoff_ocean',
        # direct flow [m3/s]
        # direct
        'direct',
        # direct-to-ocean forcing [m3/s]
        # qdto
        'direct_to_ocean',
        # flood water [m3/s]
        # flood
        'flood',
        # hillslope surface water storage [m]
        # wh
        'hillslope_storage',
        # change of hillslope water storage [m/s]
        # dwh
        'hillslope_delta_storage',
        # depth of hillslope surface water [m]
        # yh
        'hillslope_depth',
        # surface runoff from hillslope [m/s]
        # qsur
        'hillslope_surface_runoff',
        # subsurface runoff from hillslope [m/s]
        # qsub
        'hillslope_subsurface_runoff',
        # runoff from glacier, wetlands, and lakes [m/s]
        # qgwl
        'hillslope_wetland_runoff',
        # overland flor from hillslope into subchannel (outflow is negative) [m/s]
        # ehout
        'hillslope_overland_flow',
        # subnetwork area of water surface [m2]
        # tarea
        'subnetwork_area',
        # subnetwork water storage [m3]
        # wt
        'subnetwork_storage',
        # subnetwork water storage at previous timestep [m3]
        # wt_last
        'subnetwork_storage_previous_timestep',
        # change of subnetwork water storage [m3]
        # dwt
        'subnetwork_delta_storage',
        # depth of subnetwork water [m]
        # yt
        'subnetwork_depth',
        # cross section area of subnetwork [m2]
        # mt
        'subnetwork_cross_section_area',
        # hydraulic radii of subnetwork [m]
        # rt
        'subnetwork_hydraulic_radii',
        # wetness perimeter of subnetwork [m]
        # pt
        'subnetwork_wetness_perimeter',
        # subnetwork flow velocity [m/s]
        # vt
        'subnetwork_flow_velocity',
        # subnetwork mean travel time of water within travel [s]
        # tt
        'subnetwork_mean_travel_time',
        # subnetwork evaporation [m/s]
        # tevap
        'subnetwork_evaporation',
        # subnetwork lateral inflow from hillslope [m3/s]
        # etin
        'subnetwork_lateral_inflow',
        # subnetwork discharge into main channel (outflow is negative) [m3/s]
        # etout
        'subnetwork_discharge',
        # irrigation demand [m/s]
        # qdem
        'subnetwork_irrigation_demand',
        # main channel area [m/2]
        # rarea
        'channel_area',
        # main channel storage [m3]
        # wr
        'channel_storage',
        # change in main channel storage [m3]
        # dwr
        'channel_delta_storage',
        # main channel storage at last timestep [m3]
        # wr_last
        'channel_storage_previous_timestep',
        # main channel water depth [m]
        # yr
        'channel_depth',
        # cross section area of main channel [m2]
        # mr
        'channel_cross_section_area',
        # hydraulic radii of main channel [m]
        # rr
        'channel_hydraulic_radii',
        # wetness perimeter of main channel[m]
        # pr
        'channel_wetness_perimeter',
        # main channel flow velocity [m/s]
        # vr
        'channel_flow_velocity',
        # main channel mean travel time of water within travel [s]
        # tr
        'channel_mean_travel_time',
        # main channel evaporation [m/s]
        # erlg
        'channel_evaporation',
        # lateral flow from hillslope [m3/s]
        # erlateral
        'channel_lateral_flow_hillslope',
        # inflow from upstream links [m3/s]
        # erin
        'channel_inflow_upstream',
        # outflow into downstream links [m3/s]
        # erout
        'channel_outflow_downstream',
        # outflow into downstream links from previous timestep [m3/s]
        # TRunoff%eroup_lagi
        'channel_outflow_downstream_previous_timestep',
        # outflow into downstream links from current timestep [m3/s]
        # TRunoff%eroup_lagf
        'channel_outflow_downstream_current_timestep',
        # initial outflow before dam regulation at current timestep [m3/s]
        # TRunoff%erowm_regi
        'channel_outflow_before_regulation',
        # final outflow after dam regulation at current timestep [m3/s]
        # TRunoff%erowm_regf
        'channel_outflow_after_regulation',
        # outflow sum of upstream gridcells, instantaneous [m3/s]
        # eroutUp
        'channel_outflow_sum_upstream_instant',
        # outflow sum of upstream gridcells, average [m3/s]
        # TRunoff%eroutUp_avg
        'channel_outflow_sum_upstream_average',
        # lateral flow from hillslope, including surface and subsurface runoff generation components, average [m3/s]
        # TRunoff%erlat_avg
        'channel_lateral_flow_hillslope_average',
        # flux for adjustment of water balance residual in glacier, wetlands, and lakes [m3/s]
        # ergwl
        'channel_wetland_flux',
        # streamflow from outlet, positive is out [m3/s]
        # flow
        'channel_flow',
        # a column of always all zeros, to use as a utility
        'zeros'
    ]:
        if state_dataframe is not None:
            state_dataframe = state_dataframe.join(dd.from_array(da.zeros(self.get_grid_size()), columns=[var])).persist()
        else:
            state_dataframe = dd.from_array(da.zeros(self.get_grid_size()), columns=[var]).persist()
    
    # tracers
    # TODO how to handle ice?
    self.tracers = (self.LIQUID_TRACER, self.ICE_TRACER) if self.config.get('water_management.ice_runoff_enabled') else (self.LIQUID_TRACER,)
    state_dataframe = state_dataframe.join(dd.from_array(
        da.full(self.get_grid_size(), self.LIQUID_TRACER), columns=['tracer']
    )).persist()

    # mask on whether or not to perform euler calculations
    state_dataframe = state_dataframe.join(dd.from_array(da.where(
        da.array(state_dataframe.tracer.eq(self.LIQUID_TRACER)).compute(),
        True,
        False
    ), columns=['euler_mask'])).persist()
    
    self.state = state_dataframe
    
    # initial conditions
    condition = self.state.tracer.eq(self.LIQUID_TRACER) & (self.grid.land_mask.eq(1) | self.grid.land_mask.eq(3))
    # assumed hillslope water depth
    self.state.hillslope_storage = self.state.zeros.mask(
        condition,
        0.001
    )
    # assumed subnetwork depth
    self.state.subnetwork_storage = self.state.zeros.mask(
        condition,
        1.0 * self.grid.subnetwork_length * self.grid.subnetwork_width
    )
    # assumed channel depth
    channel_depth = 0.9 * self.grid.channel_depth
    self.state.channel_storage = self.state.zeros.mask(
        condition,
        channel_depth * self.grid.channel_width * self.grid.channel_length
    )
    # hydraulic radius
    hydraulic_radius = self.grid.channel_width * channel_depth / ( self.grid.channel_width + 2.0 * channel_depth )
    # main channel outflow using manning equation # TODO consolidate the manning equations
    channel_velocity = self.state.zeros.mask(
        hydraulic_radius.gt(0) & self.grid.channel_slope.ne(0),
        ((hydraulic_radius ** (1/3)) * (self.grid.channel_slope ** (1/2)) / self.grid.channel_manning).mask(
            self.grid.channel_slope.gt(0),
            -((hydraulic_radius ** (2/3)) * (-self.grid.channel_slope ** (1/2)) / self.grid.channel_manning)
        )
    )
    self.state.channel_outflow_downstream = self.state.zeros.mask(
        condition,
        - channel_velocity * channel_depth * self.grid.channel_width
    )
    
    # update physical parameters based on initial conditions
    condition = self.state.zeros.eq(0)
    update_hillslope_state(self, condition)
    update_subnetwork_state(self, condition)
    update_main_channel_state(self, condition)
    self.state.storage = self.state.channel_storage + self.state.subnetwork_storage + self.state.hillslope_storage * self.grid.area
    
    self.state = state_dataframe.persist()