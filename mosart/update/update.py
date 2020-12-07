import datetime
import numpy as np

from multiprocessing import Pool

from mosart.direct_to_ocean.direct_to_ocean import direct_to_ocean
from mosart.flood.flood import flood
from mosart.hillslope.routing import hillslope_routing
from mosart.input.runoff import load_runoff
from mosart.main_channel.routing import main_channel_routing
from mosart.subnetwork.routing import subnetwork_routing

def update(self):
    # perform one timestep
    
    ###
    ### Reset certain state variables
    ###
    self.state.flow = self.state.zeros
    self.state.outflow_downstream_previous_timestep = self.state.zeros
    self.state.outflow_downstream_current_timestep = self.state.zeros
    self.state.outflow_before_regulation = self.state.zeros
    self.state.outflow_after_regulation = self.state.zeros
    self.state.outflow_sum_upstream_average = self.state.zeros
    self.state.lateral_flow_hillslope_average = self.state.zeros
    self.state.runoff = self.state.zeros
    self.state.direct = self.state.zeros
    self.state.flood = self.state.zeros
    self.state.runoff_land = self.state.zeros
    self.state.runoff_ocean = self.state.zeros
    self.state.delta_storage = self.state.zeros
    self.state.delta_storage_land = self.state.zeros
    self.state.delta_storage_ocean = self.state.zeros
    
    # read runoff
    if self.config.get('runoff.enabled', False):
        self.state = load_runoff(self.state, self.grid, self.parameters, self.config, self.current_time)

    # flood
    self.state = flood(self.state, self.grid, self.parameters, self.config)

    # direct to ocean
    self.state = direct_to_ocean(self.state, self.grid, self.parameters, self.config)
    
    ###
    ### Subcycling
    ###
    
    # convert runoff to m/s
    self.state.hillslope_surface_runoff = self.state.hillslope_surface_runoff / self.grid.area
    self.state.hillslope_subsurface_runoff = self.state.hillslope_subsurface_runoff / self.grid.area
    self.state.hillslope_wetland_runoff = self.state.hillslope_wetland_runoff / self.grid.area

    # subcycle timestep
    delta_t =  self.config.get('simulation.timestep') / self.config.get('simulation.subcycles')
    
    for _ in np.arange(self.config.get('simulation.subcycles')):
        
        ###
        ### hillslope routing
        ###
        self.state = hillslope_routing(self.state, self.grid, self.parameters, self.config, delta_t)
        
        # zero relevant state variables
        self.state.channel_flow = self.state.zeros
        self.state.channel_outflow_downstream_previous_timestep = self.state.zeros
        self.state.channel_outflow_downstream_current_timestep = self.state.zeros
        self.state.channel_outflow_sum_upstream_average = self.state.zeros
        self.state.channel_lateral_flow_hillslope_average = self.state.zeros
        
        # iterate substeps for remaining routing
        for __ in np.arange(self.config.get('simulation.routing_iterations')):
        
            ###
            ### subnetwork routing
            ###
            self.state = subnetwork_routing(self.state, self.grid, self.parameters, self.config, delta_t)
            
            ###
            ### upstream interactions
            ###
            self.state.channel_outflow_downstream_previous_timestep = self.state.channel_outflow_downstream_previous_timestep - self.state.channel_outflow_downstream
            self.state.channel_outflow_sum_upstream_instant = self.state.zeros
            
            # send channel downstream outflow to downstream cells
            self.state.channel_outflow_sum_upstream_instant = self.grid[['downstream_id']].join(self.state[['channel_outflow_downstream']].join(self.grid.downstream_id).groupby('downstream_id').sum(), how='left').channel_outflow_downstream.fillna(0.0)
            self.state.channel_outflow_sum_upstream_average = self.state.channel_outflow_sum_upstream_average + self.state.channel_outflow_sum_upstream_instant
            self.state.channel_lateral_flow_hillslope_average = self.state.channel_lateral_flow_hillslope_average + self.state.channel_lateral_flow_hillslope
            
            ###
            ### channel routing
            ###
            self.state = main_channel_routing(self.state, self.grid, self.parameters, self.config, delta_t)
        
        # average state values over dlevelh2r
        self.state.channel_flow = self.state.channel_flow / self.config.get('simulation.routing_iterations')
        self.state.channel_outflow_downstream_previous_timestep = self.state.channel_outflow_downstream_previous_timestep / self.config.get('simulation.routing_iterations')
        self.state.channel_outflow_downstream_current_timestep = self.state.channel_outflow_downstream_current_timestep / self.config.get('simulation.routing_iterations')
        self.state.channel_outflow_sum_upstream_average = self.state.channel_outflow_sum_upstream_average / self.config.get('simulation.routing_iterations')
        self.state.channel_lateral_flow_hillslope_average = self.state.channel_lateral_flow_hillslope_average / self.config.get('simulation.routing_iterations')
        
        # accumulate local flow field
        self.state.flow = self.state.flow + self.state.channel_flow
        self.state.outflow_downstream_previous_timestep = self.state.outflow_downstream_previous_timestep + self.state.channel_outflow_downstream_previous_timestep
        self.state.outflow_downstream_current_timestep = self.state.outflow_downstream_current_timestep + self.state.channel_outflow_downstream_current_timestep
        self.state.outflow_before_regulation = self.state.outflow_before_regulation + self.state.channel_outflow_before_regulation
        self.state.outflow_after_regulation = self.state.outflow_after_regulation + self.state.channel_outflow_after_regulation
        self.state.outflow_sum_upstream_average = self.state.outflow_sum_upstream_average + self.state.channel_outflow_sum_upstream_average
        self.state.lateral_flow_hillslope_average = self.state.lateral_flow_hillslope_average + self.state.channel_lateral_flow_hillslope_average
        
        self.current_time += datetime.timedelta(seconds=delta_t)
    
    # average state values over subcycles
    self.state.flow = self.state.flow / self.config.get('simulation.subcycles')
    self.state.outflow_downstream_previous_timestep = self.state.outflow_downstream_previous_timestep / self.config.get('simulation.subcycles')
    self.state.outflow_downstream_current_timestep = self.state.outflow_downstream_current_timestep / self.config.get('simulation.subcycles')
    self.state.outflow_before_regulation = self.state.outflow_before_regulation / self.config.get('simulation.subcycles')
    self.state.outflow_after_regulation = self.state.outflow_after_regulation / self.config.get('simulation.subcycles')
    self.state.outflow_sum_upstream_average = self.state.outflow_sum_upstream_average / self.config.get('simulation.subcycles')
    self.state.lateral_flow_hillslope_average = self.state.lateral_flow_hillslope_average / self.config.get('simulation.subcycles')
    
    # update state values
    previous_storage = self.state.storage
    self.state.storage = (self.state.channel_storage + self.state.subnetwork_storage + self.state.hillslope_storage * self.grid.area) * self.grid.drainage_fraction
    self.state.delta_storage = (self.state.storage - previous_storage) / self.config.get('simulation.timestep')
    self.state.runoff = self.state.flow
    self.state.runoff_total = self.state.direct
    self.state.runoff_land = self.state.runoff.where(self.grid.land_mask.eq(1), 0)
    self.state.delta_storage_land = self.state.delta_storage.where(self.grid.land_mask.eq(1), 0)
    self.state.runoff_ocean = self.state.runoff.where(self.grid.land_mask.ge(2), 0)
    self.state.runoff_total = self.state.runoff_total.mask(self.grid.land_mask.ge(2), self.state.runoff_total + self.state.runoff)
    self.state.delta_storage_ocean = self.state.delta_storage.where(self.grid.land_mask.ge(2), 0)
    
    # TODO negative storage checks etc
    # check for negative storage
    # if self.state.subnetwork_storage.lt(-self.parameters.tiny_value).any().compute():
    #     raise Exception('Negative subnetwork storage.')
    # if self.state.channel_storage.lt(-self.parameters.tiny_value).any().compute():
    #     raise Exception('Negative channel storage.')
    
    # TODO budget checks
