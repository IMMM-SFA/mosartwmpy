import datetime
import numpy as np
import pandas as pd

from multiprocessing import Pool

from mosart.direct_to_ocean.direct_to_ocean import direct_to_ocean
from mosart.flood.flood import flood
from mosart.hillslope.routing import hillslope_routing
from mosart.input.runoff import load_runoff
from mosart.main_channel.routing import main_channel_routing
from mosart.subnetwork.routing import subnetwork_routing

def update(self):
    
    # read runoff
    if self.config.get('runoff.enabled', False):
        self.state = load_runoff(self.state, self.grid, self.parameters, self.config, self.current_time)
    
    # drop cells that won't be relevant, i.e. we only really care about land and outlet cells
    state = self.state[self.grid.mosart_mask.gt(0)]
    grid = self.grid[self.grid.mosart_mask.gt(0)]
    
    # if using multiple cores, split into roughly equal sized group based on outlet_id, since currently no math acts outside of the basin
    if self.config.get('multiprocessing.enabled', False) and self.cores > 1:
        jobs = self.cores
        # group cells based on outlet_id bins
        grid['core'] = pd.cut(grid.outlet_id, bins=jobs, labels=False)
        #with Pool(processes=self.cores) as pool:
            #state = pd.concat(pool.starmap(_update, [(state[grid.core.eq(n)], grid[grid.core.eq(n)], self.parameters, self.config, self.current_time) for n in np.arange(self.cores)]))
        futures = []
        for n in np.arange(jobs):
            futures.append(self.client.submit(_update, state[grid.core.eq(n)], grid[grid.core.eq(n)], self.parameters, self.config, self.current_time))
        state = pd.concat(self.client.gather(futures))
                
    else:
        state = _update(self.state, self.grid, self.parameters, self.config, self.current_time)
    
    self.state = state.combine_first(self.state)
    self.current_time += datetime.timedelta(seconds=self.config.get('simulation.timestep'))

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
    state.hillslope_surface_runoff = pd.DataFrame(state.hillslope_surface_runoff.values / grid.area.values)
    state.hillslope_subsurface_runoff = pd.DataFrame(state.hillslope_subsurface_runoff.values / grid.area.values)
    state.hillslope_wetland_runoff = pd.DataFrame(state.hillslope_wetland_runoff.values / grid.area.values)

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
        
        # iterate substeps for remaining routing
        for __ in np.arange(config.get('simulation.routing_iterations')):
        
            ###
            ### subnetwork routing
            ###
            state = subnetwork_routing(state, grid, parameters, config, delta_t)
            
            ###
            ### upstream interactions
            ###
            state.channel_outflow_downstream_previous_timestep = pd.DataFrame(state.channel_outflow_downstream_previous_timestep.values - state.channel_outflow_downstream.values)
            state.channel_outflow_sum_upstream_instant = state.zeros
            
            # send channel downstream outflow to downstream cells
            state.channel_outflow_sum_upstream_instant = grid[['downstream_id']].join(state[['channel_outflow_downstream']].join(grid.downstream_id).groupby('downstream_id').sum(), how='left').channel_outflow_downstream.fillna(0.0)
            state.channel_outflow_sum_upstream_average = pd.DataFrame(state.channel_outflow_sum_upstream_average.values + state.channel_outflow_sum_upstream_instant.values)
            state.channel_lateral_flow_hillslope_average = pd.DataFrame(state.channel_lateral_flow_hillslope_average.values + state.channel_lateral_flow_hillslope.values)
            
            ###
            ### channel routing
            ###
            state = main_channel_routing(state, grid, parameters, config, delta_t)
        
        # average state values over dlevelh2r
        state.channel_flow = pd.DataFrame(state.channel_flow.values / config.get('simulation.routing_iterations'))
        state.channel_outflow_downstream_previous_timestep = pd.DataFrame(state.channel_outflow_downstream_previous_timestep.values / config.get('simulation.routing_iterations'))
        state.channel_outflow_downstream_current_timestep = pd.DataFrame(state.channel_outflow_downstream_current_timestep.values / config.get('simulation.routing_iterations'))
        state.channel_outflow_sum_upstream_average = pd.DataFrame(state.channel_outflow_sum_upstream_average.values / config.get('simulation.routing_iterations'))
        state.channel_lateral_flow_hillslope_average = pd.DataFrame(state.channel_lateral_flow_hillslope_average.values / config.get('simulation.routing_iterations'))
        
        # accumulate local flow field
        state.flow = pd.DataFrame(state.flow.values + state.channel_flow.values)
        state.outflow_downstream_previous_timestep = pd.DataFrame(state.outflow_downstream_previous_timestep.values + state.channel_outflow_downstream_previous_timestep.values)
        state.outflow_downstream_current_timestep = pd.DataFrame(state.outflow_downstream_current_timestep.values + state.channel_outflow_downstream_current_timestep.values)
        state.outflow_before_regulation = pd.DataFrame(state.outflow_before_regulation.values + state.channel_outflow_before_regulation.values)
        state.outflow_after_regulation = pd.DataFrame(state.outflow_after_regulation.values + state.channel_outflow_after_regulation.values)
        state.outflow_sum_upstream_average = pd.DataFrame(state.outflow_sum_upstream_average.values + state.channel_outflow_sum_upstream_average.values)
        state.lateral_flow_hillslope_average = pd.DataFrame(state.lateral_flow_hillslope_average.values + state.channel_lateral_flow_hillslope_average.values)
        
        current_time += datetime.timedelta(seconds=delta_t)
    
    # average state values over subcycles
    state.flow = pd.DataFrame(state.flow / config.get('simulation.subcycles'))
    state.outflow_downstream_previous_timestep = pd.DataFrame(state.outflow_downstream_previous_timestep.values / config.get('simulation.subcycles'))
    state.outflow_downstream_current_timestep = pd.DataFrame(state.outflow_downstream_current_timestep.values / config.get('simulation.subcycles'))
    state.outflow_before_regulation = pd.DataFrame(state.outflow_before_regulation.values / config.get('simulation.subcycles'))
    state.outflow_after_regulation = pd.DataFrame(state.outflow_after_regulation.values / config.get('simulation.subcycles'))
    state.outflow_sum_upstream_average = pd.DataFrame(state.outflow_sum_upstream_average.values / config.get('simulation.subcycles'))
    state.lateral_flow_hillslope_average = pd.DataFrame(state.lateral_flow_hillslope_average.values / config.get('simulation.subcycles'))
    
    # update state values
    previous_storage = state.storage.values
    state.storage = pd.DataFrame((state.channel_storage.values + state.subnetwork_storage.values + state.hillslope_storage.values * grid.area.values) * grid.drainage_fraction.values)
    state.delta_storage = pd.DataFrame((state.storage.values - previous_storage) / config.get('simulation.timestep'))
    state.runoff = state.flow
    state.runoff_total = state.direct
    state.runoff_land = pd.DataFrame(np.where(
        grid.land_mask.values == 1,
        state.runoff.values,
        0
    ))
    state.delta_storage_land = pd.DataFrame(np.where(
        grid.land_mask.values == 1,
        state.delta_storage.values,
        0
    ))
    state.runoff_ocean = pd.DataFrame(np.where(
        grid.land_mask.values >= 2,
        state.runoff.values,
        0
    ))
    state.runoff_total = pd.DataFrame(np.where(
        grid.land_mask.values >= 2,
        state.runoff_total.values + state.runoff.values,
        state.runoff_total
    ))
    state.delta_storage_ocean = pd.DataFrame(np.where(
        grid.land_mask.values >= 2,
        state.delta_storage.values,
        0
    ))
    
    # TODO budget checks?
    
    return state
