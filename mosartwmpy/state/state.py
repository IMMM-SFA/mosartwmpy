import logging
import numpy as np
import pandas as pd
from datetime import datetime, time
from xarray import open_dataset

from mosartwmpy.reservoirs.reservoirs import initialize_reservoir_state

class State():
    """Data structure for storing state variables."""
    
    def get_state_for_process(self, process_logical):
        """Get a copy of a State instance subsetted by a process number

        Args:
            process_logical (ndarray): a logical array indicating the slice to take (i.e. pass in `grid.process == 0` to get the first process)

        Returns:
            State: a State instance that is a subset of the original
        """

        sub_state = State(empty=True)
        for attribute in dir(self):
            if not attribute.startswith('_'):
                # TODO for now just copying the numpy arrays -- need something better for WM
                if isinstance(getattr(self, attribute), np.ndarray): 
                    setattr(sub_state, attribute, getattr(self, attribute)[process_logical])
        return sub_state
    
    def recombine_sub_states(self, processes, sub_states):
        for n, sub_state in enumerate(sub_states):
            for attribute in dir(self):
                if not attribute.startswith('_'):
                    # TODO for now just copying the numpy arrays -- need something better for WM
                    if isinstance(getattr(self, attribute), np.ndarray):
                        getattr(self, attribute)[processes==n] = getattr(sub_state, attribute)
    
    def __init__(self, grid=None, config=None, parameters=None, grid_size=None, empty=False):
        """Initialize the state.

        Args:
            grid (Grid): Model grid instance
            config (Config): Model configuration instance
            parameters (Parameters): Model parameters instance
            grid_size (int): Flattened grid size
            empty (bool): Set to True to return an empty instance
        """
        
        # shortcut to get an empty state instance
        if empty:
            return
        
        # initialize all the state variables
        logging.info('Initializing state variables.')
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
            # tracer, i.e. liquid or ice - TODO ice not implemented yet
            'tracer',
            # euler mask - which cells to perform the euler calculation on
            'euler_mask',
            # a column of always all zeros, to use as a utility
            'zeros'
        ]:
            setattr(self, var, np.zeros(grid_size))
        
        # set tracer to liquid everywhere... TODO ice is not implemented
        self.tracer = np.full(grid_size, parameters.LIQUID_TRACER)
        
        # set euler_mask to 1 everywhere... TODO not really necessary without ice implemented
        self.euler_mask = np.where(
            self.tracer == parameters.LIQUID_TRACER,
            True,
            False
        )
    
        if config.get('water_management.enabled', False):
            logging.debug(' - reservoirs')
            for var in [
                # reservoir streamflow schedule
                'reservoir_streamflow',
                # StorMthStOp
                'reservoir_storage_operation_year_start',
                # storage [m3]
                'reservoir_storage',
                # MthStOp,
                'reservoir_month_start_operations',
                # MthStFC
                'reservoir_month_flood_control_start',
                # MthNdFC
                'reservoir_month_flood_control_end',
                # release [m3/s]
                'reservoir_release',
                # supply [m3/s]
                'reservoir_supply',
                # monthly demand [m3/s] (demand0)
                'reservoir_monthly_demand',
                # unmet demand volume within sub timestep (deficit) [m3]
                'reservoir_demand',
                # unmet demand over whole timestep [m3]
                'reservoir_deficit',
                # potential evaporation [mm/s] # TODO this doesn't appear to be initialized anywhere currently
                'reservoir_potential_evaporation'
            ]:
                setattr(self, var, np.zeros(grid_size))
            initialize_reservoir_state(self, grid, config, parameters)