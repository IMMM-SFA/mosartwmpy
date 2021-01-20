import datetime
import numpy as np
import pandas as pd

from epiweeks import Week
from multiprocessing import Pool

from mosart.direct_to_ocean.direct_to_ocean import direct_to_ocean
from mosart.flood.flood import flood
from mosart.hillslope.routing import hillslope_routing
from mosart.input.runoff import load_runoff
from mosart.input.demand import load_demand
from mosart.main_channel.irrigation import  main_channel_irrigation
from mosart.main_channel.routing import main_channel_routing
from mosart.reservoirs.regulation import extraction_regulated_flow, regulation
from mosart.reservoirs.reservoirs import reservoir_release
from mosart.subnetwork.irrigation import subnetwork_irrigation
from mosart.subnetwork.routing import subnetwork_routing

def update(self):
    
    # read runoff
    if self.config.get('runoff.enabled', False):
        self.state = load_runoff(self.state, self.grid, self.parameters, self.config, self.current_time)
    
    # advance timestep
    self.current_time += datetime.timedelta(seconds=self.config.get('simulation.timestep'))
    
    # read demand
    if self.config.get('water_management.enabled', False):
        # only read new demand and compute new release if it's the very start of simulation or new month # TODO this is currently adjusted to try to match fortran mosart
        if self.current_time == datetime.datetime.combine(self.config.get('simulation.start_date'), datetime.time(3)) or self.current_time == datetime.datetime(self.current_time.year, self.current_time.month, 1):
            self.state = load_demand(self.state, self.grid, self.parameters, self.config, self.current_time)
            reservoir_release(self)
        # zero supply and demand
        self.state.reservoir_supply = self.state.zeros
        self.state.reservoir_demand = self.state.zeros
        self.state.reservoir_deficit = self.state.zeros
        # TODO this is still written assuming monthly, but here's the epiweek for when that is relevant
        epiweek = Week.fromdate(self.current_time).week
        month = self.current_time.month
        streamflow_time_name = self.config.get('water_management.reservoirs.streamflow_time_resolution')
        self.state.reservoir_streamflow[:] = self.reservoir_streamflow_schedule.sel({streamflow_time_name: month}).values
    
    # drop cells that won't be relevant, i.e. we only really care about land and outlet cells
    state = self.state[self.grid.mosart_mask.gt(0)]
    grid = self.grid[self.grid.mosart_mask.gt(0)]
    
    # if using multiple cores, split into roughly equal sized group based on outlet_id, since currently no math acts outside of the basin
    # TODO actually the reservoir extraction can act outside of basin :(
    if self.config.get('multiprocessing.enabled', False) and self.cores > 1:
        jobs = self.cores
        # group cells based on outlet_id bins
        grid['core'] = pd.cut(grid.outlet_id, bins=jobs, labels=False)
        with Pool(processes=jobs) as pool:
            state = pd.concat(pool.starmap(_update, [(state[grid.core.eq(n)], grid[grid.core.eq(n)], self.parameters, self.config, self.current_time) for n in np.arange(jobs)]))
    else:
        state = _update(self.state, self.grid, self.parameters, self.config, self.current_time)
    
    self.state = state.combine_first(self.state)

def _update(state, grid, parameters, config, current_time):
    # perform one timestep
    
    ###
    ### Reset certain state variables
    ###
    state.flow = state.zeros
    state.outflow_downstream_previous_timestep = state.zeros
    state.outflow_downstream_current_timestep = state.zeros
    state.outflow_before_regulation = state.zeros
    state.outflow_after_regulation = state.zeros
    state.outflow_sum_upstream_average = state.zeros
    state.lateral_flow_hillslope_average = state.zeros
    state.runoff = state.zeros
    state.direct = state.zeros
    state.flood = state.zeros
    state.runoff_land = state.zeros
    state.runoff_ocean = state.zeros
    state.delta_storage = state.zeros
    state.delta_storage_land = state.zeros
    state.delta_storage_ocean = state.zeros

    # flood
    state = flood(state, grid, parameters, config)

    # direct to ocean
    state = direct_to_ocean(state, grid, parameters, config)
    
    ###
    ### Subcycling
    ###
    
    # convert runoff to m/s
    state.hillslope_surface_runoff[:] = state.hillslope_surface_runoff.values / grid.area.values
    state.hillslope_subsurface_runoff[:] = state.hillslope_subsurface_runoff.values / grid.area.values
    state.hillslope_wetland_runoff[:] = state.hillslope_wetland_runoff.values / grid.area.values

    # subcycle timestep
    delta_t =  config.get('simulation.timestep') / config.get('simulation.subcycles')
    
    for _ in np.arange(config.get('simulation.subcycles')):
        
        ###
        ### hillslope routing
        ###
        state = hillslope_routing(state, grid, parameters, config, delta_t)
        
        # zero relevant state variables
        state.channel_flow = state.zeros
        state.channel_outflow_downstream_previous_timestep = state.zeros
        state.channel_outflow_downstream_current_timestep = state.zeros
        state.channel_outflow_sum_upstream_average = state.zeros
        state.channel_lateral_flow_hillslope_average = state.zeros
        
        # get the demand volume for this substep
        if config.get('water_management.enabled'):
            state.reservoir_demand[:] = state.reservoir_monthly_demand.values * delta_t
        
        # iterate substeps for remaining routing
        for __ in np.arange(config.get('simulation.routing_iterations')):
        
            ###
            ### subnetwork
            ###
            if config.get('water_management.enabled', False) and config.get('water_management.extraction_enabled', False):
                state = subnetwork_irrigation(state, grid, parameters, config)
            state = subnetwork_routing(state, grid, parameters, config, delta_t)
            
            ###
            ### upstream interactions
            ###
            state.channel_outflow_downstream_previous_timestep[:] = state.channel_outflow_downstream_previous_timestep.values - state.channel_outflow_downstream.values
            state.channel_outflow_sum_upstream_instant = state.zeros
            
            # send channel downstream outflow to downstream cells
            state.channel_outflow_sum_upstream_instant = grid[['downstream_id']].join(state[['channel_outflow_downstream']].join(grid.downstream_id).groupby('downstream_id').sum(), how='left').channel_outflow_downstream.fillna(0.0)
            state.channel_outflow_sum_upstream_average[:] = state.channel_outflow_sum_upstream_average.values + state.channel_outflow_sum_upstream_instant.values
            state.channel_lateral_flow_hillslope_average[:] = state.channel_lateral_flow_hillslope_average.values + state.channel_lateral_flow_hillslope.values
            
            ###
            ### main channel
            ###
            state = main_channel_routing(state, grid, parameters, config, delta_t)
            if config.get('water_management.enabled', False) and config.get('water_management.extraction_enabled', False) and config.get('water_management.extraction_main_channel_enabled', False):
                state = main_channel_irrigation(state, grid, parameters, config)
            
            ###
            ### regulation
            ###
            if config.get('water_management.enabled', False) and config.get('water_management.regulation_enabled', False):
                state = regulation(state, grid, parameters, config, delta_t / config.get('simulation.routing_iterations'))
            
            # update flow
            base_condition = (grid.mosart_mask.values > 0) & state.euler_mask.values
            state.channel_outflow_downstream_current_timestep[:] = np.where(
                base_condition,
                state.channel_outflow_downstream_current_timestep.values - state.channel_outflow_downstream.values,
                state.channel_outflow_downstream_current_timestep.values
            )
            state.channel_flow[:] = np.where(
                base_condition,
                state.channel_flow.values - state.channel_outflow_downstream.values,
                state.channel_flow.values
            )
        
        # average state values over dlevelh2r
        state.channel_flow[:] = state.channel_flow.values / config.get('simulation.routing_iterations')
        state.channel_outflow_downstream_previous_timestep[:] = state.channel_outflow_downstream_previous_timestep.values / config.get('simulation.routing_iterations')
        state.channel_outflow_downstream_current_timestep[:] = state.channel_outflow_downstream_current_timestep.values / config.get('simulation.routing_iterations')
        state.channel_outflow_sum_upstream_average[:] = state.channel_outflow_sum_upstream_average.values / config.get('simulation.routing_iterations')
        state.channel_lateral_flow_hillslope_average[:] = state.channel_lateral_flow_hillslope_average.values / config.get('simulation.routing_iterations')
        
        # regulation extraction
        if config.get('water_management.enabled', False) and config.get('water_management.regulation_enabled', False):
            state.outflow_before_regulation[:] = -state.channel_outflow_downstream.values
            state.channel_flow[:] = state.channel_flow.values + state.channel_outflow_downstream.values
            if config.get('water_management.extraction_enabled', False):
                state = extraction_regulated_flow(state, grid, parameters, config, delta_t)
            state.outflow_after_regulation[:] = -state.channel_outflow_downstream.values
            state.channel_flow[:] = state.channel_flow.values - state.channel_outflow_downstream.values
        
        # accumulate local flow field
        state.flow[:] = state.flow.values + state.channel_flow.values
        state.outflow_downstream_previous_timestep[:] = state.outflow_downstream_previous_timestep.values + state.channel_outflow_downstream_previous_timestep.values
        state.outflow_downstream_current_timestep[:] = state.outflow_downstream_current_timestep.values + state.channel_outflow_downstream_current_timestep.values
        state.outflow_before_regulation[:] = state.outflow_before_regulation.values + state.channel_outflow_before_regulation.values
        state.outflow_after_regulation[:] = state.outflow_after_regulation.values + state.channel_outflow_after_regulation.values
        state.outflow_sum_upstream_average[:] = state.outflow_sum_upstream_average.values + state.channel_outflow_sum_upstream_average.values
        state.lateral_flow_hillslope_average[:] = state.lateral_flow_hillslope_average.values + state.channel_lateral_flow_hillslope_average.values
        
        # current_time += datetime.timedelta(seconds=delta_t)
    
    if config.get('water_management.enabled'):
        # convert supply to flux
        state.reservoir_supply[:] = state.reservoir_supply.values / config.get('simulation.timestep')
    
    # convert runoff back to m3/s for output
    state.hillslope_surface_runoff[:] = state.hillslope_surface_runoff.values * grid.area.values
    state.hillslope_subsurface_runoff[:] = state.hillslope_subsurface_runoff.values * grid.area.values
    state.hillslope_wetland_runoff[:] = state.hillslope_wetland_runoff.values * grid.area.values
    
    # average state values over subcycles
    state.flow[:] = state.flow.values / config.get('simulation.subcycles')
    state.outflow_downstream_previous_timestep[:] = state.outflow_downstream_previous_timestep.values / config.get('simulation.subcycles')
    state.outflow_downstream_current_timestep[:] = state.outflow_downstream_current_timestep.values / config.get('simulation.subcycles')
    state.outflow_before_regulation[:] = state.outflow_before_regulation.values / config.get('simulation.subcycles')
    state.outflow_after_regulation[:] = state.outflow_after_regulation.values / config.get('simulation.subcycles')
    state.outflow_sum_upstream_average[:] = state.outflow_sum_upstream_average.values / config.get('simulation.subcycles')
    state.lateral_flow_hillslope_average[:] = state.lateral_flow_hillslope_average.values / config.get('simulation.subcycles')
    
    # update state values
    previous_storage = state.storage.values
    state.storage[:] = (state.channel_storage.values + state.subnetwork_storage.values + state.hillslope_storage.values * grid.area.values) * grid.drainage_fraction.values
    state.delta_storage[:] = (state.storage.values - previous_storage) / config.get('simulation.timestep')
    state.runoff = state.flow
    state.runoff_total = state.direct
    state.runoff_land[:] = np.where(
        grid.land_mask.values == 1,
        state.runoff.values,
        0
    )
    state.delta_storage_land[:] = np.where(
        grid.land_mask.values == 1,
        state.delta_storage.values,
        0
    )
    state.runoff_ocean[:] = np.where(
        grid.land_mask.values >= 2,
        state.runoff.values,
        0
    )
    state.runoff_total[:] = np.where(
        grid.land_mask.values >= 2,
        state.runoff_total.values + state.runoff.values,
        state.runoff_total
    )
    state.delta_storage_ocean[:] = np.where(
        grid.land_mask.values >= 2,
        state.delta_storage.values,
        0
    )
    
    # TODO budget checks?
    
    return state
