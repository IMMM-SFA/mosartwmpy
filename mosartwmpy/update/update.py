import datetime
import numpy as np
import pandas as pd

from epiweeks import Week
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool

from multiprocessing.shared_memory import SharedMemory

from mosartwmpy.direct_to_ocean.direct_to_ocean import direct_to_ocean
from mosartwmpy.flood.flood import flood
from mosartwmpy.hillslope.routing import hillslope_routing
from mosartwmpy.input.runoff import load_runoff
from mosartwmpy.input.demand import load_demand
from mosartwmpy.main_channel.irrigation import  main_channel_irrigation
from mosartwmpy.main_channel.routing import main_channel_routing
from mosartwmpy.reservoirs.regulation import extraction_regulated_flow, regulation
from mosartwmpy.reservoirs.reservoirs import reservoir_release
from mosartwmpy.subnetwork.irrigation import subnetwork_irrigation
from mosartwmpy.subnetwork.routing import subnetwork_routing

# ignore numpy NaN and invalid warnings
# (i.e. divide by zero and NaN logicals -- in Pandas/Dask, these simply remain NaN instead of becoming infinite)
np.seterr(all='ignore')
# np.seterr(all='raise')
# catch pandas assignment warnings
pd.options.mode.chained_assignment = 'raise'

# TODO add docstrings to each method/class
def update(self):
    
    # read runoff
    if self.config.get('runoff.enabled', False):
        load_runoff(self.state, self.grid, self.config, self.current_time)
    
    # advance timestep
    self.current_time += datetime.timedelta(seconds=self.config.get('simulation.timestep'))
    
    # read demand
    if self.config.get('water_management.enabled', False):
        # only read new demand and compute new release if it's the very start of simulation or new month
        # TODO this is currently adjusted to try to match fortran mosart
        if self.current_time == datetime.datetime.combine(self.config.get('simulation.start_date'), datetime.time(3)) or self.current_time == datetime.datetime(self.current_time.year, self.current_time.month, 1):
            load_demand(self.state, self.config, self.current_time)
            reservoir_release(self.state, self.grid, self.config, self.parameters, self.current_time)
        # zero supply and demand
        self.state.reservoir_supply[:] = self.state.zeros
        self.state.reservoir_demand[:] = self.state.zeros
        self.state.reservoir_deficit[:] = self.state.zeros
        # TODO this is still written assuming monthly, but here's the epiweek for when that is relevant
        epiweek = Week.fromdate(self.current_time).week
        month = self.current_time.month
        streamflow_time_name = self.config.get('water_management.reservoirs.streamflow_time_resolution')
        self.state.reservoir_streamflow[:] = self.grid.reservoir_streamflow_schedule.sel({streamflow_time_name: month}).values
    
    # drop cells that won't be relevant, i.e. we only really care about land and outlet cells
    # state = self.state[self.grid.mosart_mask > 0]
    # grid = self.grid[self.grid.mosart_mask > 0]
    
    # if using multiple cores, split into roughly equal sized group based on outlet_id, since currently no math acts outside of the basin
    # TODO actually the reservoir extraction can act outside of basin :(
    if self.config.get('multiprocessing.enabled', False) and self.cores > 1:
        pass
        # TODO joblib, pathos, ray?
        # with Pool(processes=self.cores) as pool:
        #     state = pd.concat(pool.starmap(_update, [(self.state.loc[grid.core.eq(n)], grid.loc[grid.core.eq(n)], self.parameters, self.config, self.current_time) for n in np.arange(self.cores)]))
        
    else:
        _update(self.state, self.grid, self.parameters, self.config, self.current_time)
    
    # self.state = state.combine_first(self.state)

def _update(state, grid, parameters, config, current_time):
    # perform one timestep
    
    ###
    ### Reset certain state variables
    ###
    state.flow[:] = state.zeros
    state.outflow_downstream_previous_timestep[:] = state.zeros
    state.outflow_downstream_current_timestep[:] = state.zeros
    state.outflow_before_regulation[:] = state.zeros
    state.outflow_after_regulation[:] = state.zeros
    state.outflow_sum_upstream_average[:] = state.zeros
    state.lateral_flow_hillslope_average[:] = state.zeros
    state.runoff[:] = state.zeros
    state.direct[:] = state.zeros
    state.flood[:] = state.zeros
    state.runoff_land[:] = state.zeros
    state.runoff_ocean[:] = state.zeros
    state.delta_storage[:] = state.zeros
    state.delta_storage_land[:] = state.zeros
    state.delta_storage_ocean[:] = state.zeros

    # flood
    flood(state, grid, parameters, config)

    # direct to ocean
    direct_to_ocean(state, grid, parameters, config)
    
    ###
    ### Subcycling
    ###
    
    # convert runoff to m/s
    state.hillslope_surface_runoff = state.hillslope_surface_runoff / grid.area
    state.hillslope_subsurface_runoff = state.hillslope_subsurface_runoff / grid.area
    state.hillslope_wetland_runoff = state.hillslope_wetland_runoff / grid.area

    # subcycle timestep
    delta_t =  config.get('simulation.timestep') / config.get('simulation.subcycles')
    
    for _ in np.arange(config.get('simulation.subcycles')):
        
        ###
        ### hillslope routing
        ###
        hillslope_routing(state, grid, parameters, delta_t)
        
        # zero relevant state variables
        state.channel_flow[:] = state.zeros
        state.channel_outflow_downstream_previous_timestep[:] = state.zeros
        state.channel_outflow_downstream_current_timestep[:] = state.zeros
        state.channel_outflow_sum_upstream_average[:] = state.zeros
        state.channel_lateral_flow_hillslope_average[:] = state.zeros
        
        # get the demand volume for this substep
        if config.get('water_management.enabled'):
            state.reservoir_demand = state.reservoir_monthly_demand * delta_t
        
        # iterate substeps for remaining routing
        for __ in np.arange(config.get('simulation.routing_iterations')):
        
            ###
            ### subnetwork
            ###
            if config.get('water_management.enabled', False) and config.get('water_management.extraction_enabled', False):
                subnetwork_irrigation(state, grid, parameters)
            subnetwork_routing(state, grid, parameters, config, delta_t)
            
            ###
            ### upstream interactions
            ###
            state.channel_outflow_downstream_previous_timestep = state.channel_outflow_downstream_previous_timestep - state.channel_outflow_downstream
            state.channel_outflow_sum_upstream_instant[:] = state.zeros
            
            # send channel downstream outflow to downstream cells
            state.channel_outflow_sum_upstream_instant[:] = pd.DataFrame(
                grid.id, columns=['id']
            ).merge(
                pd.DataFrame(
                    state.channel_outflow_downstream, columns=['channel_outflow_downstream']
                ).join(
                    pd.DataFrame(
                        grid.downstream_id, columns=['downstream_id']
                    )
                ).groupby('downstream_id').sum(),
                how='left',
                left_on='id',
                right_index=True
            ).channel_outflow_downstream.fillna(0.0).values
            
            state.channel_outflow_sum_upstream_average = state.channel_outflow_sum_upstream_average + state.channel_outflow_sum_upstream_instant
            state.channel_lateral_flow_hillslope_average = state.channel_lateral_flow_hillslope_average + state.channel_lateral_flow_hillslope
            
            ###
            ### main channel
            ###
            main_channel_routing(state, grid, parameters, config, delta_t)
            if config.get('water_management.enabled', False) and config.get('water_management.extraction_enabled', False) and config.get('water_management.extraction_main_channel_enabled', False):
                main_channel_irrigation(state, grid, parameters)
            
            ###
            ### regulation
            ###
            if config.get('water_management.enabled', False) and config.get('water_management.regulation_enabled', False):
                regulation(state, grid, parameters, delta_t / config.get('simulation.routing_iterations'))
            
            # update flow
            base_condition = (grid.mosart_mask > 0) & state.euler_mask
            state.channel_outflow_downstream_current_timestep = np.where(
                base_condition,
                state.channel_outflow_downstream_current_timestep - state.channel_outflow_downstream,
                state.channel_outflow_downstream_current_timestep
            )
            state.channel_flow = np.where(
                base_condition,
                state.channel_flow - state.channel_outflow_downstream,
                state.channel_flow
            )
        
        # average state values over dlevelh2r
        state.channel_flow = state.channel_flow / config.get('simulation.routing_iterations')
        state.channel_outflow_downstream_previous_timestep = state.channel_outflow_downstream_previous_timestep / config.get('simulation.routing_iterations')
        state.channel_outflow_downstream_current_timestep = state.channel_outflow_downstream_current_timestep / config.get('simulation.routing_iterations')
        state.channel_outflow_sum_upstream_average = state.channel_outflow_sum_upstream_average / config.get('simulation.routing_iterations')
        state.channel_lateral_flow_hillslope_average = state.channel_lateral_flow_hillslope_average / config.get('simulation.routing_iterations')
        
        # regulation extraction
        if config.get('water_management.enabled', False) and config.get('water_management.regulation_enabled', False):
            state.outflow_before_regulation = -state.channel_outflow_downstream
            state.channel_flow = state.channel_flow + state.channel_outflow_downstream
            if config.get('water_management.extraction_enabled', False):
                extraction_regulated_flow(state, grid, parameters, config, delta_t)
            state.outflow_after_regulation = -state.channel_outflow_downstream
            state.channel_flow = state.channel_flow - state.channel_outflow_downstream
        
        # accumulate local flow field
        state.flow = state.flow + state.channel_flow
        state.outflow_downstream_previous_timestep = state.outflow_downstream_previous_timestep + state.channel_outflow_downstream_previous_timestep
        state.outflow_downstream_current_timestep = state.outflow_downstream_current_timestep + state.channel_outflow_downstream_current_timestep
        state.outflow_before_regulation = state.outflow_before_regulation + state.channel_outflow_before_regulation
        state.outflow_after_regulation = state.outflow_after_regulation + state.channel_outflow_after_regulation
        state.outflow_sum_upstream_average = state.outflow_sum_upstream_average + state.channel_outflow_sum_upstream_average
        state.lateral_flow_hillslope_average = state.lateral_flow_hillslope_average + state.channel_lateral_flow_hillslope_average
        
        # current_time += datetime.timedelta(seconds=delta_t)
    
    if config.get('water_management.enabled'):
        # convert supply to flux
        state.reservoir_supply = state.reservoir_supply / config.get('simulation.timestep')
    
    # convert runoff back to m3/s for output
    state.hillslope_surface_runoff = state.hillslope_surface_runoff * grid.area
    state.hillslope_subsurface_runoff = state.hillslope_subsurface_runoff * grid.area
    state.hillslope_wetland_runoff = state.hillslope_wetland_runoff * grid.area
    
    # average state values over subcycles
    state.flow = state.flow / config.get('simulation.subcycles')
    state.outflow_downstream_previous_timestep = state.outflow_downstream_previous_timestep / config.get('simulation.subcycles')
    state.outflow_downstream_current_timestep = state.outflow_downstream_current_timestep / config.get('simulation.subcycles')
    state.outflow_before_regulation = state.outflow_before_regulation / config.get('simulation.subcycles')
    state.outflow_after_regulation = state.outflow_after_regulation / config.get('simulation.subcycles')
    state.outflow_sum_upstream_average = state.outflow_sum_upstream_average / config.get('simulation.subcycles')
    state.lateral_flow_hillslope_average = state.lateral_flow_hillslope_average / config.get('simulation.subcycles')
    
    # update state values
    previous_storage = 1.0 * state.storage
    state.storage = (state.channel_storage + state.subnetwork_storage + state.hillslope_storage * grid.area) * grid.drainage_fraction
    state.delta_storage = (state.storage - previous_storage) / config.get('simulation.timestep')
    state.runoff = 1.0 * state.flow
    state.runoff_total = 1.0 * state.direct
    state.runoff_land = np.where(
        grid.land_mask == 1,
        state.runoff,
        0
    )
    state.delta_storage_land = np.where(
        grid.land_mask == 1,
        state.delta_storage,
        0
    )
    state.runoff_ocean = np.where(
        grid.land_mask >= 2,
        state.runoff,
        0
    )
    state.runoff_total = np.where(
        grid.land_mask >= 2,
        state.runoff_total + state.runoff,
        state.runoff_total
    )
    state.delta_storage_ocean = np.where(
        grid.land_mask >= 2,
        state.delta_storage,
        0
    )
