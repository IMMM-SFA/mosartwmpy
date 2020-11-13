import dask.array as da
import datetime
import logging
import numpy as np

def _update(self):
    # perform one timestep

    ###
    ### Compute Flood
    ### Remove excess liquid water from land
    ###
    ### TODO tcraig leaves a comment here concerning surface_runoff in fortran mosart:
    ### "This seems like an odd approach, you
    ### might create negative forcing.  why not take it out of
    ### the volr directly?  it's also odd to compute this
    ### at the initial time of the time loop.  why not do
    ### it at the end or even during the run loop as the
    ### new volume is computed.  fluxout depends on volr, so
    ### how this is implemented does impact the solution."
    ###
    logging.debug(' - flood')

    # flux sent back to land
    self.state.flood = self.state.zeros.mask(
        self.grid.land_mask.eq(1) & self.state.storage.gt(self.parameters['flood_threshold']) & self.state.tracer.eq(self.LIQUID_TRACER),
        (self.state.storage - self.parameters['flood_threshold']) / self.config.get('simulation.timestep')
    )
    # remove this flux from the input runoff from land
    self.state.hillslope_surface_runoff = self.state.hillslope_surface_runoff.mask(
        self.state.tracer.eq(self.LIQUID_TRACER),
        self.state.hillslope_surface_runoff - self.state.flood
    )


    ###
    ### Direct transfer to outlet point
    ###
    logging.debug(' - direct to outlet')

    # direct to ocean
    # note - in fortran mosart this direct_to_ocean forcing could be provided from LND component, but we don't seem to be using it
    source_direct = self.state.direct_to_ocean
    
    # wetland runoff
    wetland_runoff_volume = self.state.hillslope_wetland_runoff * self.config.get('simulation.timestep') / self.config.get('simulation.subcycles')
    river_volume_minimum = self.parameters['river_depth_minimum'] * self.grid.area

    # if wetland runoff is negative and it would bring main channel storage below the minimum, send it directly to ocean
    condition = (self.state.channel_storage + wetland_runoff_volume).lt(river_volume_minimum) & self.state.hillslope_wetland_runoff.lt(0)
    source_direct = source_direct.mask(condition, source_direct + self.state.hillslope_wetland_runoff)
    self.state.hillslope_wetland_runoff = self.state.hillslope_wetland_runoff.mask(condition, 0)
    # remove remaining wetland runoff (negative and positive)
    source_direct = source_direct + self.state.hillslope_wetland_runoff
    self.state.hillslope_wetland_runoff = self.state.hillslope_wetland_runoff.mask(self.state.hillslope_wetland_runoff.ne(0), 0).persist()
    
    # runoff from hillslope
    # remove negative subsurface water
    condition = self.state.hillslope_subsurface_runoff.lt(0)
    source_direct = source_direct.mask(condition, source_direct + self.state.hillslope_subsurface_runoff)
    self.state.hillslope_subsurface_runoff = self.state.hillslope_subsurface_runoff.mask(condition, 0)
    # remove negative surface water
    condition = self.state.hillslope_surface_runoff.lt(0)
    source_direct = source_direct.mask(condition, source_direct + self.state.hillslope_surface_runoff)
    self.state.hillslope_surface_runoff = self.state.hillslope_surface_runoff.mask(condition, 0)

    # if ocean cell or ice tracer, remove the rest of the sub and surface water
    # other cells will be handled by mosart euler
    condition = self.grid.mosart_mask.eq(0) | self.state.tracer.eq(self.ICE_TRACER)
    source_direct = source_direct.mask(condition, source_direct + self.state.hillslope_subsurface_runoff + self.state.hillslope_surface_runoff)
    self.state.hillslope_subsurface_runoff = self.state.hillslope_subsurface_runoff.mask(condition, 0)
    self.state.hillslope_surface_runoff = self.state.hillslope_surface_runoff.mask(condition, 0)

    # send the direct water to outlet for each tracer
    self.state.direct = self.grid[['outlet_id']].merge(self.state[['direct']].join(self.grid[['outlet_id']]).groupby('outlet_id').sum(), how='left')['direct'].fillna(0.0)
    
    ###
    ### Subcycling
    ###
    logging.debug(' - subcycling')

    # subcycle timestep
    delta_t =  self.config.get('simulation.timestep') / self.config.get('simulation.subcycles')
    
    # convert runoff from m3/s to m/s
    self.state.hillslope_surface_runoff = self.state.hillslope_surface_runoff / self.grid.area
    self.state.hillslope_subsurface_runoff = self.state.hillslope_subsurface_runoff / self.grid.area
    self.state.hillslope_wetland_runoff = self.state.hillslope_wetland_runoff / self.grid.area
    self.state.subnetwork_irrigation_demand = self.state.subnetwork_irrigation_demand / self.grid.area
    
    self.state = self.state.persist()
    
    for _ in np.arange(self.config.get('simulation.subcycles')):
        logging.debug(f' - subcycle {int(_)}')
        
        ###
        ### hillslope routing
        ###
        logging.debug(' - hillslope')
        hillslope_routing(self, delta_t)
        
        # zero relevant state variables
        self.state.channel_flow = self.state.zeros
        self.state.channel_outlow_downstream_previous_timestep = self.state.zeros
        self.state.channel_outflow_downstream_current_timestep = self.state.zeros
        self.state.channel_outflow_sum_upstream_average = self.state.zeros
        self.state.channel_lateral_flow_hillslope_average = self.state.zeros
        
        # iterate substeps for remaining routing
        for __ in np.arange(self.config.get('simulation.routing_iterations')):
            logging.debug(f' - routing iteration {int(__)}')
        
            ###
            ### subnetwork routing
            ###
            logging.debug(' - subnetwork routing')
            subnetwork_routing(self, delta_t)
            self.state = self.state.persist()
            
            ###
            ### upstream interactions
            ###
            logging.debug(' - upstream interactions')
            self.state.channel_outlow_downstream_previous_timestep = self.state.channel_outlow_downstream_previous_timestep - self.state.channel_outflow_downstream
            self.state.channel_outflow_sum_upstream_instant = self.state.zeros
            
            # send channel downstream outflow to downstream cells
            self.state.channel_outflow_sum_upstream_instant = self.grid[['downstream_id']].merge(self.state[['channel_outflow_downstream']].join(self.grid[['downstream_id']]).groupby('downstream_id').sum(), how='left')['channel_outflow_downstream'].fillna(0.0)
            self.state.channel_outflow_sum_upstream_average = self.state.channel_outflow_sum_upstream_average + self.state.channel_outflow_sum_upstream_instant
            self.state.channel_lateral_flow_hillslope_average = self.state.channel_lateral_flow_hillslope_average + self.state.channel_lateral_flow_hillslope
            self.state = self.state.persist()
            
            ###
            ### channel routing
            ###
            logging.debug(' - main channel routing')
            channel_routing(self, delta_t)
            self.state = self.state.persist()
        
        # check for negative storage
        logging.debug(' - checking for negative storage')
        if self.state.channel_storage.lt(-1.0e-10).any().compute():
            raise Exception("Error - Negative channel storage found!")
        
        # average state values over dlevelh2r
        logging.debug(' - averaging state values over dlevelh2r')
        self.state.channel_flow = self.state.channel_flow / self.config.get('simulation.routing_iterations')
        self.state.channel_outlow_downstream_previous_timestep = self.state.channel_outlow_downstream_previous_timestep / self.config.get('simulation.routing_iterations')
        self.state.channel_outflow_downstream_current_timestep = self.state.channel_outflow_downstream_current_timestep / self.config.get('simulation.routing_iterations')
        self.state.channel_outflow_sum_upstream_average = self.state.channel_outflow_sum_upstream_average / self.config.get('simulation.routing_iterations')
        self.state.channel_lateral_flow_hillslope_average = self.state.channel_lateral_flow_hillslope_average / self.config.get('simulation.routing_iterations')
        
        # accumulate local flow field
        logging.debug(' - accumulating local flow field')
        self.state.flow = self.state.flow + self.state.channel_flow
        self.state.outlow_downstream_previous_timestep = self.state.outlow_downstream_previous_timestep + self.state.channel_outlow_downstream_previous_timestep
        self.state.outflow_downstream_current_timestep = self.state.outflow_downstream_current_timestep + self.state.channel_outflow_downstream_current_timestep
        self.state.outflow_before_regulation = self.state.outflow_before_regulation + self.state.channel_outflow_before_regulation
        self.state.outflow_after_regulation = self.state.outflow_after_regulation + self.state.channel_outflow_after_regulation
        self.state.outflow_sum_upstream_average = self.state.outflow_sum_upstream_average + self.state.channel_outflow_sum_upstream_average
        self.state.lateral_flow_hillslope_average = self.state.lateral_flow_hillslope_average + self.state.channel_lateral_flow_hillslope_average
        
        self.state = self.state.persist()
        
        self.current_time += datetime.timedelta(seconds=delta_t)
    
    # average state values over subcycles
    logging.debug(' - averaging state values over subcycle')
    self.state.flow = self.state.flow / self.config.get('simulation.subcycles')
    self.state.outlow_downstream_previous_timestep = self.state.outlow_downstream_previous_timestep / self.config.get('simulation.subcycles')
    self.state.outflow_downstream_current_timestep = self.state.outflow_downstream_current_timestep / self.config.get('simulation.subcycles')
    self.state.outflow_before_regulation = self.state.outflow_before_regulation / self.config.get('simulation.subcycles')
    self.state.outflow_after_regulation = self.state.outflow_after_regulation / self.config.get('simulation.subcycles')
    self.state.outflow_sum_upstream_average = self.state.outflow_sum_upstream_average / self.config.get('simulation.subcycles')
    self.state.lateral_flow_hillslope_average = self.state.lateral_flow_hillslope_average / self.config.get('simulation.subcycles')
    
    # update state values
    logging.debug(' - updating state values')
    previous_storage = self.state.storage
    self.state.storage = (self.state.channel_storage + self.state.subnetwork_storage + self.state.hillslope_storage) * self.grid.area * self.grid.drainage_fraction
    self.state.delta_storage = (self.state.storage - previous_storage) / self.config.get('simulation.timestep')
    self.state.runoff = self.state.flow
    self.state.runoff_total = self.state.direct
    self.state.runoff_land = self.state.runoff.where(self.grid.land_mask.eq(1), 0)
    self.state.delta_storage_land = self.state.delta_storage.where(self.grid.land_mask.eq(1), 0)
    self.state.runoff_ocean = self.state.runoff.where(self.grid.land_mask.ge(2), 0)
    self.state.runoff_total = self.state.runoff_total.mask(self.grid.land_mask.ge(2), self.state.runoff_total + self.state.runoff)
    self.state.delta_storage_ocean = self.state.delta_storage.where(self.grid.land_mask.ge(2), 0)
    self.state = self.state.persist()
    
    # TODO budget checks
    
    # TODO write output file
    
    # TODO write restart file


def hillslope_routing(self, delta_t):
    # perform the hillslope routing for the whole grid
    # TODO describe what is happening heres
    
    base_condition = (self.grid.mosart_mask.gt(0) & self.state.euler_mask).persist()
    
    velocity_hillslope = self.state.zeros.mask(
        base_condition & self.state.hillslope_depth.gt(0),
        ((self.state.hillslope_depth ** 2) ** (1/3)) * (self.grid.hillslope ** (1/2)) / self.grid.hillslope_manning
    )
    
    self.state.hillslope_overland_flow = self.state.hillslope_overland_flow.mask(
        base_condition,
        self.state.hillslope_depth * velocity_hillslope * self.grid.drainage_density
    )
    self.state.hillslope_overland_flow = self.state.hillslope_overland_flow.mask(
        base_condition &
        self.state.hillslope_overland_flow.lt(0) &
        (self.state.hillslope_storage + delta_t * (self.state.hillslope_surface_runoff + self.state.hillslope_overland_flow)).lt(self.parameters['tiny_value']),
        -(self.state.hillslope_storage + self.state.hillslope_surface_runoff) / delta_t
    )
    
    self.state.hillslope_delta_storage = self.state.hillslope_delta_storage.mask(
        base_condition,
        self.state.hillslope_surface_runoff + self.state.hillslope_overland_flow
    )
    
    self.state.hillslope_storage = self.state.hillslope_storage.mask(
        base_condition,
        self.state.hillslope_storage + delta_t * self.state.hillslope_delta_storage
    )
    
    self.state.hillslope_depth = self.state.hillslope_depth.mask(
        base_condition,
        self.state.hillslope_storage
    )
    
    self.state.subnetwork_lateral_inflow = self.state.subnetwork_lateral_inflow.mask(
        base_condition,
        (self.state.hillslope_subsurface_runoff - self.state.hillslope_overland_flow) * self.grid.drainage_fraction * self.grid.area
    )


def subnetwork_routing(self, delta_t):
    # perform the subnetwork (tributary) routing
    # TODO describe what is happening here
    
    self.state.channel_lateral_flow_hillslope = self.state.zeros
    local_delta_t = (delta_t / self.config.get('simulation.routing_iterations') / self.grid.iterations_subnetwork).persist()
    
    # step through max iterations, masking out the unnecessary cells each time
    base_condition = (self.grid.mosart_mask.gt(0) & self.state.euler_mask).persist()
    for _ in np.arange(self.grid.iterations_subnetwork.max().compute()):
        logging.debug(f' - subnetwork iteration {int(_)}')
        iteration_condition = base_condition & self.grid.iterations_subnetwork.gt(_)

        self.state.subnetwork_flow_velocity = self.state.subnetwork_flow_velocity.mask(
            iteration_condition,
            self.state.zeros.mask(
                self.state.subnetwork_hydraulic_radii.gt(0),
                ((self.state.subnetwork_hydraulic_radii ** 2) ** (1/3)) * (self.grid.subnetwork_slope ** (1/2)) / self.grid.subnetwork_manning
            )
        )
        
        self.state.subnetwork_discharge = self.state.subnetwork_discharge.mask(
            iteration_condition,
            (-self.state.subnetwork_lateral_inflow).mask(
                self.grid.subnetwork_length.gt(self.grid.hillslope_length),
                -self.state.subnetwork_flow_velocity * self.state.subnetwork_cross_section_area
            )
        )
        condition = (
            iteration_condition &
            ((self.state.subnetwork_storage + self.state.subnetwork_lateral_inflow + self.state.subnetwork_discharge) * local_delta_t).lt(self.parameters['tiny_value'])
        )
        
        self.state.subnetwork_discharge = self.state.subnetwork_discharge.mask(
            condition,
            -(self.state.subnetwork_lateral_inflow + self.state.subnetwork_storage / local_delta_t)
        )
        
        self.state.subnetwork_flow_velocity = self.state.subnetwork_flow_velocity.mask(
            condition & self.state.subnetwork_cross_section_area.gt(0),
            -self.state.subnetwork_discharge / self.state.subnetwork_cross_section_area
        )
        
        self.state.subnetwork_delta_storage = self.state.subnetwork_delta_storage.mask(
            iteration_condition,
            self.state.subnetwork_lateral_inflow + self.state.subnetwork_discharge
        )
        
        # update storage
        self.state.subnetwork_storage_previous_timestep = self.state.subnetwork_storage_previous_timestep.mask(
            iteration_condition,
            self.state.subnetwork_storage
        )
        self.state.subnetwork_storage = self.state.subnetwork_storage.mask(
            iteration_condition,
            self.state.subnetwork_storage + self.state.subnetwork_delta_storage * local_delta_t
        )
        
        update_subnetwork_state(self, iteration_condition)
        
        self.state.channel_lateral_flow_hillslope = self.state.channel_lateral_flow_hillslope.mask(
            iteration_condition,
            self.state.channel_lateral_flow_hillslope - self.state.subnetwork_discharge
        )
    
    # average lateral flow over substeps
    self.state.channel_lateral_flow_hillslope = self.state.channel_lateral_flow_hillslope.mask(
        base_condition,
        self.state.channel_lateral_flow_hillslope / self.grid.iterations_subnetwork
    )


def update_subnetwork_state(self, base_condition):
    # update the physical properties of the subnetwork
        
    # update state variables
    condition = base_condition & self.grid.subnetwork_length.gt(0) & self.state.subnetwork_storage.gt(0)
    self.state.subnetwork_cross_section_area = self.state.zeros.mask(
        condition,
        self.state.subnetwork_storage / self.grid.subnetwork_length
    )
    self.state.subnetwork_depth = self.state.zeros.mask(
        condition & self.state.subnetwork_cross_section_area.gt(self.parameters['tiny_value']),
        self.state.subnetwork_cross_section_area / self.grid.subnetwork_width
    )
    self.state.subnetwork_wetness_perimeter = self.state.zeros.mask(
        condition & self.state.subnetwork_depth.gt(self.parameters['tiny_value']),
        self.grid.subnetwork_width + 2 * self.state.subnetwork_depth
    )
    self.state.subnetwork_hydraulic_radii = self.state.zeros.mask(
        condition & self.state.subnetwork_wetness_perimeter.gt(self.parameters['tiny_value'])
    )


def channel_routing(self, delta_t):
    # perform the main channel routing
    # TODO describe what is happening here
    
    tmp_outflow_downstream = self.state.zeros
    local_delta_t = (delta_t / self.config.get('simulation.routing_iterations') / self.grid.iterations_main_channel).persist()
    
    # step through max iterations, masking out the unnecessary cells each time
    base_condition = (self.grid.mosart_mask.gt(0) & self.state.euler_mask).persist()
    for _ in np.arange(self.grid.iterations_main_channel.max().compute()):
        logging.debug(f' - main channel iteration {int(_)}')
        iteration_condition = base_condition & self.grid.iterations_main_channel.gt(_)
    
        # routing
        routing_method = self.config.get('simulation.routing_method', 1)
        if routing_method == 1:
            kinematic_wave_routing(self, local_delta_t, iteration_condition)
        else:
            raise Exception(f"Error - Routing method {routing_method} not implemented.")
        
        # update storage
        self.state.channel_storage_previous_timestep = self.state.channel_storage
        self.state.channel_storage = self.state.channel_storage + self.state.channel_delta_storage * local_delta_t
        
        update_main_channel_state(self, iteration_condition)
        
        # update outflow tracking
        tmp_outflow_downstream = tmp_outflow_downstream + self.state.channel_outflow_downstream
    
    # update outflow
    self.state.channel_outflow_downstream = tmp_outflow_downstream / self.grid.iterations_main_channel
    self.state.channel_outflow_downstream_current_timestep = self.state.channel_outflow_downstream_current_timestep - self.state.channel_outflow_downstream
    self.state.channel_flow = self.state.channel_flow - self.state.channel_outflow_downstream


def update_main_channel_state(self, base_condition):
    # update the physical properties of the main channel
    condition = (
        base_condition &
        self.grid.channel_length.gt(0) &
        self.state.channel_storage.gt(0)
    )
    self.state.channel_cross_section_area = self.state.zeros.mask(
        condition,
        self.state.channel_storage / self.grid.channel_length
    )
    # Function for estimating maximum water depth assuming rectangular channel and tropezoidal flood plain
    # here assuming the channel cross-section consists of three parts, from bottom to up,
    # part 1 is a rectangular with bankfull depth (rdep) and bankfull width (rwid)
    # part 2 is a tropezoidal, bottom width rwid and top width rwid0, height 0.1*((rwid0-rwid)/2), assuming slope is 0.1
    # part 3 is a rectagular with the width rwid0
    not_flooded = (self.state.channel_cross_section_area - (self.grid.channel_depth * self.grid.channel_width)).le(self.parameters['tiny_value'])
    flooded_condition = self.state.channel_cross_section_area.gt(
        self.grid.channel_depth * self.grid.channel_width + self.parameters['slope_1_def'] * (self.grid.channel_width + self.grid.channel_floodplain_width) * (self.grid.channel_floodplain_width - self.grid.channel_width) / 2.0 / 2.0 + self.parameters['tiny_value']
    )
    self.state.channel_depth = self.state.zeros.mask(
        condition & self.state.channel_cross_section_area.gt(self.parameters['tiny_value']),
        (self.state.channel_cross_section_area / self.grid.channel_width).where(
            not_flooded,
            (self.grid.channel_depth + self.parameters['slope_1_def'] * (self.grid.channel_floodplain_width - self.grid.channel_width) / 2.0 + (self.state.channel_cross_section_area - self.grid.channel_depth * self.grid.channel_width - self.parameters['slope_1_def'] * (self.grid.channel_width - self.grid.channel_floodplain_width) * (self.grid.channel_floodplain_width - self.grid.channel_width) / 2.0 / 2.0) / self.grid.channel_floodplain_width).where(
                flooded_condition,
                self.grid.channel_depth + (-self.grid.channel_width + (((self.grid.channel_width ** 2) + 4 * (self.state.channel_cross_section_area - self.grid.channel_depth * self.grid.channel_width) / self.parameters['slope_1_def']) ** (1/2))) * self.parameters['slope_1_def'] / 2.0
            )
        )
    )
    # Function for estimating wetness perimeter based on same assumptions as above
    not_flooded = self.state.channel_depth.le(self.grid.channel_depth + self.parameters['tiny_value'])
    flooded_condition = self.state.channel_depth.gt(
        self.grid.channel_depth + (self.grid.channel_floodplain_width - self.grid.channel_width) / 2.0 * self.parameters['slope_1_def'] + self.parameters['tiny_value']
    )
    self.state.channel_wetness_perimeter = self.state.zeros.mask(
        condition & self.state.channel_depth.gt(self.parameters['tiny_value']),
        (self.grid.channel_width + 2 * self.state.channel_depth).where(
            not_flooded,
            (self.grid.channel_width + 2 * (self.grid.channel_depth + (self.grid.channel_floodplain_width - self.grid.channel_width) / 2.0) * self.parameters['slope_1_def'] * self.parameters['sin_atan_slope_1_def'] + (self.state.channel_depth - self.grid.channel_depth - (self.grid.channel_floodplain_width - self.grid.channel_width) / 2.0) * self.parameters['slope_1_def']).where(
                flooded_condition,
                self.grid.channel_width + 2 * (self.grid.channel_depth + (self.state.channel_depth - self.grid.channel_depth) * self.parameters['sin_atan_slope_1_def'])
            )
        )
    )
    self.state.channel_hydraulic_radii = self.state.zeros.mask(
        condition & self.state.channel_wetness_perimeter.gt(self.parameters['tiny_value']),
        self.state.channel_cross_section_area / self.state.channel_wetness_perimeter
    )

def kinematic_wave_routing(self, delta_t, base_condition):
    # classic kinematic wave routing method
    
    # estimation of inflow
    self.state.channel_inflow_upstream = self.state.zeros.mask(
        base_condition,
        -self.state.channel_outflow_sum_upstream_instant
    )
    
    # estimation of outflow
    self.state.channel_flow_velocity = self.state.zeros.mask(
        base_condition & self.grid.channel_length.gt(0) & self.state.channel_hydraulic_radii.gt(0),
        ((self.state.channel_hydraulic_radii ** 2) ** (1/3)) * (self.grid.channel_slope ** (1/2)) / self.grid.channel_manning
    )
    condition = (self.grid.total_drainage_area_single / self.grid.channel_width / self.grid.channel_length).gt(1.0e6)
    self.state.channel_outflow_downstream = self.state.channel_outflow_downstream.mask(
        base_condition,
        (-self.state.channel_flow_velocity * self.state.channel_cross_section_area).mask(
            condition | self.grid.channel_length.le(0),
            -self.state.channel_inflow_upstream - self.state.channel_lateral_flow_hillslope
        )
    )
    condition = (
        base_condition &
        da.logical_not(condition) &
        (-self.state.channel_outflow_downstream).gt(self.parameters['tiny_value']) &
        ((self.state.channel_storage + self.state.channel_lateral_flow_hillslope + self.state.channel_inflow_upstream + self.state.channel_outflow_downstream) * delta_t).lt(self.parameters['tiny_value'])
    )
    self.state.channel_outflow_downstream = self.state.channel_outflow_downstream.mask(
        condition,
        -(self.state.channel_lateral_flow_hillslope + self.state.channel_inflow_upstream + self.state.channel_storage) / delta_t
    )
    self.state.channel_flow_velocity = self.state.channel_flow_velocity.mask(
        condition & self.state.channel_cross_section_area.gt(0),
        -self.state.channel_outflow_downstream / self.state.channel_cross_section_area
    )
    
    # calculate change in storage, but first round small runoff to zero
    tmp_delta_runoff = self.state.hillslope_wetland_runoff * self.grid.area * self.grid.drainage_fraction
    tmp_delta_runoff = tmp_delta_runoff.mask(
        tmp_delta_runoff.abs().le(self.parameters['tiny_value']),
        0
    )
    self.state.channel_delta_storage = self.state.channel_lateral_flow_hillslope + self.state.channel_inflow_upstream + self.state.channel_outflow_downstream + tmp_delta_runoff
