import numpy as np
import pandas as pd
import logging

from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State

from mosartwmpy.direct_to_ocean.direct_to_ocean import direct_to_ocean
from mosartwmpy.flood.flood import flood
from mosartwmpy.hillslope.routing import hillslope_routing
from mosartwmpy.main_channel.irrigation import  main_channel_irrigation
from mosartwmpy.main_channel.routing import main_channel_routing
from mosartwmpy.reservoirs.regulation import extraction_regulated_flow, regulation
from mosartwmpy.subnetwork.irrigation import subnetwork_irrigation
from mosartwmpy.subnetwork.routing import subnetwork_routing

# ignore numpy NaN and invalid warnings
# (i.e. divide by zero and NaN logicals -- in Pandas/Dask, these simply remain NaN instead of becoming infinite)
np.seterr(all='ignore')
# filter pandas chained assignment warnings -- pretty sure handling this correctly
pd.options.mode.chained_assignment = None


def update(state: State, grid: Grid, parameters: Parameters, config: Benedict) -> None:
    """Advance the simulation one timestamp.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        parameters (Parameters): the model parameters
        config (Benedict): the model configuration
    """
    # perform one timestep
    
    # ignore nan, overflow, underflow, and div by 0 warnings, since they are handled correctly
    np.seterr(all='ignore')
    
    ###
    ### Reset certain state variables
    ###
    state.flow[:] = 0
    state.outflow_downstream_previous_timestep[:] = 0
    state.outflow_downstream_current_timestep[:] = 0
    state.outflow_before_regulation[:] = 0
    state.outflow_after_regulation[:] = 0
    state.outflow_sum_upstream_average[:] = 0
    state.lateral_flow_hillslope_average[:] = 0
    state.runoff[:] = 0
    state.direct[:] = 0
    state.flood[:] = 0
    state.runoff_land[:] = 0
    state.runoff_ocean[:] = 0
    state.delta_storage[:] = 0
    state.delta_storage_land[:] = 0
    state.delta_storage_ocean[:] = 0
    if config.get('water_management.enabled', False):
        state.grid_cell_deficit[:] = 0

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
        state.channel_flow[:] = 0
        state.channel_outflow_downstream_previous_timestep[:] = 0
        state.channel_outflow_downstream_current_timestep[:] = 0
        state.channel_outflow_sum_upstream_average[:] = 0
        state.channel_lateral_flow_hillslope_average[:] = 0
        
        # get the demand volume for this substep
        if config.get('water_management.enabled', False):
            state.grid_cell_unmet_demand = state.grid_cell_demand_rate * delta_t
        
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
            state.channel_outflow_sum_upstream_instant[:] = 0
            
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
            # aggregate deficit
            state.grid_cell_deficit = state.grid_cell_deficit + state.grid_cell_unmet_demand
        
        # accumulate local flow field
        state.flow = state.flow + state.channel_flow
        state.outflow_downstream_previous_timestep = state.outflow_downstream_previous_timestep + state.channel_outflow_downstream_previous_timestep
        state.outflow_downstream_current_timestep = state.outflow_downstream_current_timestep + state.channel_outflow_downstream_current_timestep
        state.outflow_before_regulation = state.outflow_before_regulation + state.channel_outflow_before_regulation
        state.outflow_after_regulation = state.outflow_after_regulation + state.channel_outflow_after_regulation
        state.outflow_sum_upstream_average = state.outflow_sum_upstream_average + state.channel_outflow_sum_upstream_average
        state.lateral_flow_hillslope_average = state.lateral_flow_hillslope_average + state.channel_lateral_flow_hillslope_average
    
    if config.get('water_management.enabled'):
        # convert supply to flux
        state.grid_cell_supply = state.grid_cell_supply / config.get('simulation.timestep')
    
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
    state.runoff[:] = state.flow
    state.runoff_total[:] = state.direct
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