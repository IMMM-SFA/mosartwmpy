import dask.array as da
import datetime
import logging
import numpy as np

def aggregate_values_to_indices(values, destination):
    # efficiently aggregates grid cell values to other indices
    # i.e. sending a value upstream, downstream, or directly to outlet
    # values is a 1d array-like of grid cells values
    # destination is a 1d array-like of the index that each cell should send its value to
    # returns a 1d dask array of the aggreated values
    dataframe = da.array(values).to_dask_dataframe(['value']).join(da.array(destination).to_dask_dataframe(['destination_index']))
    aggregation = dataframe.groupby('destination_index').aggregate({'value': 'sum'}).rename(columns={'value': 'aggregation'})
    return da.from_array(dataframe.merge(aggregation, how='left', left_index=True, right_on='destination_index')['aggregation'].to_dask_array().compute())


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
    self.state['flood'] = da.where(
        da.logical_and(
            self.state[['tracer', self.LIQUID_TRACER]],
            da.logical_and(
                self.state['mask'] == 1,
                self.state['storage'] > self.state['flood_threshold']
            )
        ),
        da.true_divide(
            da.subtract(
                self.state['storage'],
                self.state['flood_threshold']
            ),
            self.config.get('simulation.timestep')
        ),
        0
    )
    # remove this flux from the input runoff from land
    self.state['hillslope.surface_runoff'] = da.where(
        self.state[['tracer', self.LIQUID_TRACER]],
        da.subtract(
            self.state['hillslope.surface_runoff'],
            self.state['flood']
        ),
        self.state['hillslope.surface_runoff']
    )


    ###
    ### Direct transfer to outlet point
    ###
    logging.debug(' - direct to outlet')

    # direct to ocean
    # note - in fortran mosart this direct_to_ocean forcing could be provided from LND component, but we don't seem to be using it
    source_direct = da.add(
        0, # could be input ??
        self.state['direct_to_ocean']
    )
    
    # wetland runoff
    wetland_runoff_volume = self.state['hillslope.wetland_runoff'] * self.config.get('simulation.sub_timestep')
    river_depth_minimum = 1.0e-4 # [m]
    river_volume_minimum = da.atleast_3d(river_depth_minimum * self.state['area'])

    # if wetland runoff is negative and it would bring main channel storage below the minimum, send it directly to ocean
    condition_mask = da.logical_and(
        da.add(
            self.state['channel.storage'],
            wetland_runoff_volume
        ) < river_volume_minimum,
        self.state['hillslope.wetland_runoff'] < 0
    )
    source_direct = da.where(
        condition_mask,
        da.add(
            source_direct,
            self.state['hillslope.wetland_runoff']
        ),
        source_direct
    )
    self.state['hillslope.wetland_runoff'] = da.where(
        condition_mask,
        0,
        self.state['hillslope.wetland_runoff']
    )
    # remove remaining wetland runoff (negative and positive)
    source_direct = da.add(
        source_direct,
        self.state['hillslope.wetland_runoff']
    )
    self.state['hillslope.wetland_runoff'] = da.zeros(self.DATA_SHAPE)
    
    # runoff from hillslope
    # remove negative subsurface water
    condition_mask = self.state['hillslope.subsurface_runoff'] < 0
    source_direct = da.where(
        condition_mask,
        da.add(source_direct, self.state['hillslope.subsurface_runoff']),
        source_direct
    )
    self.state['hillslope.subsurface_runoff'] = da.where(
        condition_mask,
        0,
        self.state['hillslope.subsurface_runoff']
    )
    # remove negative surface water
    condition_mask = self.state['hillslope.surface_runoff'] < 0
    source_direct = da.where(
        condition_mask,
        da.add(
            source_direct,
            self.state['hillslope.surface_runoff']
        ),
        source_direct
    )
    self.state['hillslope.surface_runoff'] = da.where(
        condition_mask,
        0,
        self.state['hillslope.surface_runoff']
    )

    # if ocean cell or ice tracer, remove the rest of the sub and surface water
    # other cells will be handled by mosart euler
    condition_mask = da.logical_or(da.logical_not(self.state['mosart_mask'] > 0), self.state[['tracer', self.ICE_TRACER]])
    source_direct = da.where(
        condition_mask,
        da.add(
            source_direct,
            da.add(
                self.state['hillslope.subsurface_runoff'],
                self.state['hillslope.surface_runoff']
            )
        ),
        source_direct
    )
    self.state['hillslope.subsurface_runoff'] = da.where(
        condition_mask,
        0,
        self.state['hillslope.subsurface_runoff']
    )
    self.state['hillslope.surface_runoff'] = da.where(
        condition_mask,
        0,
        self.state['hillslope.surface_runoff']
    )

    # send the direct water to outlet for each tracer
    for i, tracer in enumerate(self.state['tracers']):
        self.state['direct'] = da.where(
            self.state[['tracer', tracer]],
            da.atleast_3d(aggregate_values_to_indices(source_direct[:,:,i].flatten(), self.state['outlet_id'].flatten()).reshape(self.get_grid_shape())),
            self.state['direct']
        )
    
    ###
    ### Subcycling
    ###
    logging.debug(' - subcycling')

    # number of euler subcycles
    # nsub
    number_subcycles = self.config.get('simulation.timestep') // self.config.get('simulation.sub_timestep')
    if number_subcycles * self.config.get('simulation.timestep'):
        number_subcycles += 1
    # corrected subtimestep seconds
    # delt
    delta_t =  self.config.get('simulation.timestep') / float(number_subcycles)
    
    # convert runoff from m3/s to m/s
    self.state['hillslope.surface_runoff'] = da.true_divide(self.state['hillslope.surface_runoff'], da.atleast_3d(self.state['area']))
    self.state['hillslope.subsurface_runoff'] = da.true_divide(self.state['hillslope.subsurface_runoff'], da.atleast_3d(self.state['area']))
    self.state['hillslope.wetland_runoff'] = da.true_divide(self.state['hillslope.wetland_runoff'], da.atleast_3d(self.state['area']))
    self.state['subnetwork.irrigation_demand'] = da.true_divide(self.state['subnetwork.irrigation_demand'], da.atleast_3d(self.state['area']))
    
    for _ in np.arange(number_subcycles):
        logging.debug(f' - subcycle {_}')
        
        ###
        ### hillslope routing
        ###
        logging.debug(' - hillslope')
        hillslope_routing(self, delta_t)
        
        # zero relevant state variables
        self.state['channel.flow'] = da.zeros(self.DATA_SHAPE)
        self.state['channel.outlow_downstream_previous_timestep'] = da.zeros(self.DATA_SHAPE)
        self.state['channel.outflow_downstream_current_timestep'] = da.zeros(self.DATA_SHAPE)
        self.state['channel.outflow_sum_upstream_average'] = da.zeros(self.DATA_SHAPE)
        self.state['channel.lateral_flow_hillslope_average'] = da.zeros(self.DATA_SHAPE)
        
        # iterate substeps for remaining routing
        for _ in np.arange(self.state['dlevelh2r']):
            logging.debug(f' - dlevelh2r {_}')
        
            ###
            ### subnetwork routing
            ###
            logging.debug(' - subnetwork routing')
            subnetwork_routing(self, delta_t)
            
            ###
            ### upstream interactions
            ###
            logging.debug(' - upstream interactions')
            self.state['channel.outlow_downstream_previous_timestep'] = self.state['channel.outlow_downstream_previous_timestep'] - self.state['channel.outflow_downstream']
            self.state['channel.outflow_sum_upstream_instant'] = da.zeros(self.DATA_SHAPE)
            
            # send channel downstream outflow to downstream cells
            for i, tracer in enumerate(self.state['tracers']):
                self.state['channel.outflow_sum_upstream_instant'] = da.where(
                    self.state[['tracer', tracer]],
                    da.atleast_3d(aggregate_values_to_indices(self.state['channel.outflow_downstream'][:,:,i].flatten(), self.state['downstream_id'].flatten()).reshape(self.get_grid_shape())),
                    self.state['channel.outflow_sum_upstream_instant']
                )
            self.state['channel.outflow_sum_upstream_average'] = self.state['channel.outflow_sum_upstream_average'] + self.state['channel.outflow_sum_upstream_instant']
            self.state['channel.lateral_flow_hillslope_average'] = self.state['channel.lateral_flow_hillslope_average'] + self.state['channel.lateral_flow_hillslope']
            
            ###
            ### channel routing
            ###
            logging.debug(' - main channel routing')
            channel_routing(self, delta_t)
        
        # compute intermediate values
        self.state['channel.storage'] = da.array(self.state['channel.storage']).compute()
        
        # check for negative storage
        logging.debug(' - checking for negative storage')
        if da.any(self.state['channel.storage'] < -1.0e-10):
            raise Exception("Error - Negative channel storage found!")
        
        # average state values over dlevelh2r
        logging.debug(' - averaging state values over dlevelh2r')
        self.state['channel.flow'] = self.state['channel.flow'] / self.state['dlevelh2r']
        self.state['channel.outlow_downstream_previous_timestep'] = self.state['channel.outlow_downstream_previous_timestep'] / self.state['dlevelh2r']
        self.state['channel.outflow_downstream_current_timestep'] = self.state['channel.outflow_downstream_current_timestep'] / self.state['dlevelh2r']
        self.state['channel.outflow_sum_upstream_average'] = self.state['channel.outflow_sum_upstream_average'] / self.state['dlevelh2r']
        self.state['channel.lateral_flow_hillslope_average'] = self.state['channel.lateral_flow_hillslope_average'] / self.state['dlevelh2r']
        
        # accumulate local flow field
        logging.debug(' - accumulating local flow field')
        self.state['flow'] = self.state['flow'] + self.state['channel.flow']
        self.state['outlow_downstream_previous_timestep'] = self.state['outlow_downstream_previous_timestep'] + self.state['channel.outlow_downstream_previous_timestep']
        self.state['outflow_downstream_current_timestep'] = self.state['outflow_downstream_current_timestep'] + self.state['channel.outflow_downstream_current_timestep']
        self.state['outflow_before_regulation'] = self.state['outflow_before_regulation'] + self.state['channel.outflow_before_regulation']
        self.state['outflow_after_regulation'] = self.state['outflow_after_regulation'] + self.state['channel.outflow_after_regulation']
        self.state['outflow_sum_upstream_average'] = self.state['outflow_sum_upstream_average'] + self.state['channel.outflow_sum_upstream_average']
        self.state['lateral_flow_hillslope_average'] = self.state['lateral_flow_hillslope_average'] + self.state['channel.lateral_flow_hillslope_average']
        
        self.current_time += datetime.timedelta(seconds=delta_t)
    
    # average state values over subcycles
    logging.debug(' - averaging state values over subcycle')
    self.state['flow'] = self.state['flow'] / number_subcycles
    self.state['outlow_downstream_previous_timestep'] = self.state['outlow_downstream_previous_timestep'] / number_subcycles
    self.state['outflow_downstream_current_timestep'] = self.state['outflow_downstream_current_timestep'] / number_subcycles
    self.state['outflow_before_regulation'] = self.state['outflow_before_regulation'] / number_subcycles
    self.state['outflow_after_regulation'] = self.state['outflow_after_regulation'] / number_subcycles
    self.state['outflow_sum_upstream_average'] = self.state['outflow_sum_upstream_average'] / number_subcycles
    self.state['lateral_flow_hillslope_average'] = self.state['lateral_flow_hillslope_average'] / number_subcycles
    
    # update state values
    logging.debug(' - updating state values')
    previous_storage = da.where(True, self.state['storage'], 0)
    self.state['storage'] = (self.state['channel.storage'] + self.state['subnetwork.storage'] + self.state['hillslope.storage']) * da.atleast_3d(self.state['area']) * da.atleast_3d(self.grid[self.config.get('grid.drainage_fraction')])
    self.state['delta_storage'] = (self.state['storage'] - previous_storage) / self.config.get('simulation.timestep')
    self.state['runoff'] = self.state['flow']
    self.state['runoff_total'] = self.state['direct']
    self.state['runoff_land'] = da.where(
        self.state['mask'] == 1,
        self.state['runoff'],
        0
    )
    self.state['delta_storage_land'] = da.where(
        self.state['mask'] == 1,
        self.state['delta_storage'],
        0
    )
    self.state['runoff_ocean'] = da.where(
        self.state['mask'] >= 2,
        self.state['runoff'],
        0
    )
    self.state['runoff_total'] = da.where(
        self.state['mask'] >= 2,
        self.state['runoff_total'] + self.state['runoff'],
        self.state['runoff_total']
    )
    self.state['delta_storage_ocean'] = da.where(
        self.state['mask'] >= 2,
        self.state['delta_storage'],
        0
    )
    
    # TODO budget checks
    
    # TODO write output file
    
    # TODO write restart file


def hillslope_routing(self, delta_t):
    # perform the hillslope routing for the whole grid
    
    velocity_hillslope = da.where(
        da.logical_and(
            self.state['hillslope.depth'] > 0,
            da.logical_and(
                self.state['mosart_mask'] > 0,
                self.state['euler_mask']
            )
        ),
        da.multiply(
            da.power(da.square(self.state['hillslope.depth']), 1/3),
            da.atleast_3d(da.true_divide(
                da.sqrt(self.grid[self.config.get('grid.hillslope')]),
                self.grid[self.config.get('grid.hillslope_manning')]
            ))
        ),
        0
    ).compute()
    self.state['hillslope.overland_flow'] = da.where(
        da.logical_and(
            self.state['mosart_mask'] > 0,
            self.state['euler_mask']
        ),
        da.multiply(
            da.multiply(
                self.state['hillslope.depth'],
                velocity_hillslope
            ),
            da.atleast_3d(self.grid[self.config.get('grid.drainage_density')])
        ),
        self.state['hillslope.overland_flow']
    )
    self.state['hillslope.overland_flow'] = da.where(
        da.logical_and(
            da.logical_and(
                self.state['mosart_mask'] > 0,
                self.state['euler_mask']
            ),
            da.logical_and(
                self.state['hillslope.overland_flow'] < 0,
                da.add(
                    self.state['hillslope.storage'],
                    delta_t * da.add(
                        self.state['hillslope.surface_runoff'],
                        self.state['hillslope.overland_flow']
                    )
                ) < self.state['tiny_value']
            )
        ),
        -da.add(
            self.state['hillslope.storage'],
            self.state['hillslope.surface_runoff']
        ) / delta_t,
        self.state['hillslope.overland_flow']
    ).compute()
    self.state['hillslope.delta_storage'] = da.where(
        da.logical_and(
            self.state['mosart_mask'] > 0,
            self.state['euler_mask']
        ),
        da.add(
            self.state['hillslope.surface_runoff'],
            self.state['hillslope.overland_flow']
        ),
        self.state['hillslope.delta_storage']
    ).compute()
    self.state['hillslope.storage'] = da.where(
        da.logical_and(
            self.state['mosart_mask'] > 0,
            self.state['euler_mask']
        ),
        da.add(
            self.state['hillslope.storage'],
            delta_t * self.state['hillslope.delta_storage']
        ),
        self.state['hillslope.storage']
    ).compute()
    self.state['hillslope.depth'] = da.where(
        da.logical_and(
            self.state['mosart_mask'] > 0,
            self.state['euler_mask']
        ),
        self.state['hillslope.storage'],
        self.state['hillslope.depth']
    ).compute()
    self.state['subnetwork.lateral_inflow'] = da.where(
        da.logical_and(
            self.state['mosart_mask'] > 0,
            self.state['euler_mask']
        ),
        da.multiply(
            da.subtract(
                self.state['hillslope.subsurface_runoff'],
                self.state['hillslope.overland_flow']
            ),
            da.atleast_3d(da.multiply(
                self.grid[self.config.get('grid.drainage_fraction')],
                self.state['area']
            ))
        ),
        self.state['subnetwork.lateral_inflow']
    ).compute()
    

def subnetwork_routing(self, delta_t):
    # perform the subnetwork (tributary) routing
    
    self.state['channel.lateral_flow_hillslope'] = da.zeros(self.DATA_SHAPE)
    local_delta_t = delta_t / self.state['dlevelh2r'] / self.state['subtimesteps.subnetwork']

    # TODO matrix solve seems faster even though many cells will have excessive steps
    if True:
        subnetwork_routing_by_matrix(self, delta_t)
    else:
        # since each grid cell has a different timestep, iterate over each cell
        # flatten the matrices and use dask blocks so that the iteration can be parallel
        # iterate over each tracer (liquid and ice)
        # (although technically euler solve is disabled for ice)
        chunks = self.get_grid_size() # TODO tune this? so far seems fastest at 1 chunk :\
        for i, tracer in enumerate(self.state['tracers']):
            subnetwork_results = da.map_blocks(
                subnetwork_routing_by_block,
                self.state['mosart_mask'].flatten().rechunk(chunks),
                self.state['euler_mask'][:,:,i].flatten().rechunk(chunks),
                local_delta_t.flatten().rechunk(chunks),
                self.state['subtimesteps.subnetwork'].flatten().rechunk(chunks),
                self.state['hillslope.length'].flatten().rechunk(chunks),
                self.state['subnetwork.length'].flatten().rechunk(chunks),
                self.grid[self.config.get('grid.subnetwork_slope')].data.flatten().rechunk(chunks),
                self.grid[self.config.get('grid.subnetwork_manning')].data.flatten().rechunk(chunks),
                self.state['subnetwork.width'].flatten().rechunk(chunks),
                self.state['subnetwork.cross_section_area'][:,:,i].flatten().rechunk(chunks),
                self.state['subnetwork.depth'][:,:,i].flatten().rechunk(chunks),
                self.state['subnetwork.wetness_perimeter'][:,:,i].flatten().rechunk(chunks),
                self.state['subnetwork.hydraulic_radii'][:,:,i].flatten().rechunk(chunks),
                self.state['subnetwork.storage'][:,:,i].flatten().rechunk(chunks),
                self.state['subnetwork.delta_storage'][:,:,i].flatten().rechunk(chunks),
                self.state['subnetwork.storage_previous_timestep'][:,:,i].flatten().rechunk(chunks),
                self.state['subnetwork.lateral_inflow'][:,:,i].flatten().rechunk(chunks),
                self.state['subnetwork.flow_velocity'][:,:,i].flatten().rechunk(chunks),
                self.state['subnetwork.discharge'][:,:,i].flatten().rechunk(chunks),
                self.state['channel.lateral_flow_hillslope'][:,:,i].flatten().rechunk(chunks),
                tiny_value=self.state['tiny_value']
            ).compute()
            # recombine results from each block into state variables
            for j, state in enumerate([
                'subnetwork.cross_section_area',
                'subnetwork.depth',
                'subnetwork.wetness_perimeter',
                'subnetwork.hydraulic_radii',
                'subnetwork.storage',
                'subnetwork.delta_storage',
                'subnetwork.storage_previous_timestep',
                'subnetwork.flow_velocity',
                'subnetwork.discharge',
                'channel.lateral_flow_hillslope',
            ]):
                self.state[state] = da.where(
                    self.state[['tracer', tracer]],
                    da.atleast_3d(subnetwork_results[:,j].reshape(self.get_grid_shape())),
                    self.state[state]
                )

def subnetwork_routing_by_block(
    mosart_mask,
    euler_mask,
    local_delta_t,
    subnetwork_subtimesteps,
    hillslope_length,
    subnetwork_length,
    subnetwork_slope,
    subnetwork_manning,
    subnetwork_width,
    subnetwork_cross_section_area,
    subnetwork_depth,
    subnetwork_wetness_perimeter,
    subnetwork_hydraulic_radii,
    subnetwork_storage,
    subnetwork_delta_storage,
    subnetwork_storage_previous_timestep,
    subnetwork_lateral_inflow,
    subnetwork_flow_velocity,
    subnetwork_discharge,
    channel_lateral_flow_hillslope,
    tiny_value = 0,
):
    # perform the subnetwork channel routing for one flattened block of the grid
    
    block_size = len(mosart_mask)
    
    # since dask arrays do not allow index assignment, let's reassign the outputs as numpy arrays
    subnetwork_cross_section_area = np.array(subnetwork_cross_section_area)
    subnetwork_depth = np.array(subnetwork_depth)
    subnetwork_wetness_perimeter = np.array(subnetwork_wetness_perimeter)
    subnetwork_hydraulic_radii = np.array(subnetwork_hydraulic_radii)
    subnetwork_storage = np.array(subnetwork_storage)
    subnetwork_delta_storage = np.array(subnetwork_delta_storage)
    subnetwork_storage_previous_timestep = np.array(subnetwork_storage_previous_timestep)
    subnetwork_flow_velocity = np.array(subnetwork_flow_velocity)
    subnetwork_discharge = np.array(subnetwork_discharge)
    channel_lateral_flow_hillslope = np.array(channel_lateral_flow_hillslope)
    
    for i in np.arange(block_size):
        # skip cells not needing euler and skip ocean cells
        if not euler_mask[i] or mosart_mask[i] < 1:
            continue
        
        # iterate over each subtimestep
        for _  in np.arange(subnetwork_subtimesteps[i]):
        
            if subnetwork_length[i] <= hillslope_length[i]:
                # if no tributaries, it is not subnetwork routing
                subnetwork_discharge[i] = -subnetwork_lateral_inflow[i]
                
            else:
                subnetwork_flow_velocity[i] = 0 if subnetwork_hydraulic_radii[i] <= 0 else ((subnetwork_hydraulic_radii[i] ** 2) ** (1 / 3)) * da.sqrt(subnetwork_slope[i]) / subnetwork_manning[i]
                subnetwork_discharge[i] = -subnetwork_flow_velocity[i] * subnetwork_cross_section_area[i]
                if (subnetwork_storage[i] + (subnetwork_lateral_inflow[i] + subnetwork_discharge[i]) * local_delta_t[i]) < tiny_value:
                    subnetwork_discharge[i] = -(subnetwork_lateral_inflow[i] + subnetwork_storage[i] / local_delta_t[i])
                    if subnetwork_cross_section_area[i] > 0:
                        subnetwork_flow_velocity[i] = -subnetwork_discharge[i] / subnetwork_cross_section_area[i]
            
            # update state values
            subnetwork_delta_storage[i] = subnetwork_lateral_inflow[i] + subnetwork_discharge[i]
            subnetwork_storage_previous_timestep[i] = subnetwork_storage[i] # TODO i don't think this is used anywhere
            subnetwork_storage[i] = subnetwork_storage[i] + subnetwork_delta_storage[i] * local_delta_t[i]
            
            if subnetwork_length[i] > 0 and subnetwork_storage[i] > 0:
                subnetwork_cross_section_area[i] = subnetwork_storage[i] / subnetwork_length[i]
                if subnetwork_cross_section_area[i] > tiny_value:
                    subnetwork_depth[i] = subnetwork_cross_section_area[i] / subnetwork_width[i]
                else:
                    subnetwork_depth[i] = 0
                if subnetwork_depth[i] > tiny_value:
                    subnetwork_wetness_perimeter[i] = subnetwork_width[i] + 2 * subnetwork_depth[i]
                else:
                    subnetwork_wetness_perimeter[i] = 0
                if subnetwork_wetness_perimeter[i] > tiny_value:
                    subnetwork_hydraulic_radii[i] = subnetwork_cross_section_area[i] / subnetwork_wetness_perimeter[i]
                else:
                    subnetwork_hydraulic_radii[i] = 0
            else:
                subnetwork_cross_section_area[i] = 0
                subnetwork_depth[i] = 0
                subnetwork_wetness_perimeter[i] = 0
                subnetwork_hydraulic_radii[i] = 0
            
            channel_lateral_flow_hillslope[i] =  channel_lateral_flow_hillslope[i] - subnetwork_discharge[i]
            
        channel_lateral_flow_hillslope[i] = channel_lateral_flow_hillslope[i] / subnetwork_subtimesteps[i]
        
    # return a stack of results for this block -- use to update the actual state variables outside of the block mapping
    return da.stack([
        subnetwork_cross_section_area,
        subnetwork_depth,
        subnetwork_wetness_perimeter,
        subnetwork_hydraulic_radii,
        subnetwork_storage,
        subnetwork_delta_storage,
        subnetwork_storage_previous_timestep,
        subnetwork_flow_velocity,
        subnetwork_discharge,
        channel_lateral_flow_hillslope
    ], axis=1)

def subnetwork_routing_by_matrix(self, delta_t):
    # use the max necessary substeps for all cells, so that all calculations can be matrix level
    
    substeps = da.max(self.state['subtimesteps.subnetwork']).compute()
    local_delta_t = delta_t / self.state['dlevelh2r'] / substeps
    base_condition = da.logical_and(
        self.state['euler_mask'],
        self.state['mosart_mask'] > 0
    )
    for _ in np.arange(substeps):
        logging.debug(f' - subnetwork step {_}')
        self.state['subnetwork.flow_velocity'] = da.where(
            base_condition,
            da.where(
                self.state['subnetwork.hydraulic_radii'] <= 0,
                0,
                ((self.state['subnetwork.hydraulic_radii'] ** 2) ** (1/3)) * da.atleast_3d(da.sqrt(self.grid[self.config.get('grid.subnetwork_slope')])) / da.atleast_3d(self.grid[self.config.get('grid.subnetwork_manning')])
            ),
            self.state['subnetwork.flow_velocity']
        ).compute()
        self.state['subnetwork.discharge'] = da.where(
            base_condition,
            da.where(
                da.atleast_3d(self.state['subnetwork.length'] <= self.state['hillslope.length']),
                -self.state['subnetwork.lateral_inflow'],
                -self.state['subnetwork.flow_velocity'] * self.state['subnetwork.cross_section_area']
            ),
            self.state['subnetwork.discharge']
        ).compute()
        condition = da.logical_and(
            base_condition,
            (self.state['subnetwork.storage'] + (self.state['subnetwork.lateral_inflow'] + self.state['subnetwork.discharge']) * local_delta_t) < self.state['tiny_value']
        )
        self.state['subnetwork.discharge'] = da.where(
            condition,
            -(self.state['subnetwork.lateral_inflow'] + self.state['subnetwork.storage'] / local_delta_t),
            self.state['subnetwork.discharge']
        ).compute()
        self.state['subnetwork.flow_velocity'] = da.where(
            da.logical_and(
                condition,
                self.state['subnetwork.cross_section_area'] > 0
            ),
            -self.state['subnetwork.discharge'] / self.state['subnetwork.cross_section_area'],
            self.state['subnetwork.flow_velocity']
        ).compute()
        self.state['subnetwork.delta_storage'] = da.where(
            base_condition,
            self.state['subnetwork.lateral_inflow'] + self.state['subnetwork.discharge'],
            self.state['subnetwork.delta_storage']
        ).compute()
        
        # update storage
        self.state['subnetwork.storage_previous_timestep'] = da.where(
            base_condition,
            self.state['subnetwork.storage'],
            self.state['subnetwork.storage_previous_timestep']
        ).compute()
        self.state['subnetwork.storage'] = da.where(
            base_condition,
            self.state['subnetwork.storage'] + self.state['subnetwork.delta_storage'] * local_delta_t,
            self.state['subnetwork.storage']
        ).compute()
        
        # update state variables
        condition = da.logical_and(
            base_condition,
            da.logical_and(
                da.atleast_3d(self.state['subnetwork.length'] > 0),
                self.state['subnetwork.storage'] > 0
            )
        ).compute()
        self.state['subnetwork.cross_section_area'] = da.where(
            condition,
            self.state['subnetwork.storage'] / da.atleast_3d(self.state['subnetwork.length']),
            0
        ).compute()
        self.state['subnetwork.depth'] = da.where(
            da.logical_and(
                condition,
                self.state['subnetwork.cross_section_area'] > self.state['tiny_value']
            ),
            self.state['subnetwork.cross_section_area'] / da.atleast_3d(self.state['subnetwork.width']),
            0
        ).compute()
        self.state['subnetwork.wetness_perimeter'] = da.where(
            da.logical_and(
                condition,
                self.state['subnetwork.depth'] > self.state['tiny_value']
            ),
            da.atleast_3d(self.state['subnetwork.width']) + 2 * self.state['subnetwork.depth'],
            0
        ).compute()
        self.state['subnetwork.hydraulic_radii'] = da.where(
            da.logical_and(
                condition,
                self.state['subnetwork.wetness_perimeter'] > self.state['tiny_value']
            ),
            self.state['subnetwork.cross_section_area'] / self.state['subnetwork.wetness_perimeter'],
            0
        ).compute()
        
        self.state['channel.lateral_flow_hillslope'] = da.where(
            base_condition,
            self.state['channel.lateral_flow_hillslope'] - self.state['subnetwork.discharge'],
            self.state['channel.lateral_flow_hillslope']
        ).compute()
    
    # average lateral flow over substeps
    self.state['channel.lateral_flow_hillslope'] = da.where(
        base_condition,
        self.state['channel.lateral_flow_hillslope'] / substeps,
        self.state['channel.lateral_flow_hillslope']
    ).compute()

def channel_routing(self, delta_t):
    # perform the main channel routing
    # use the max necessary substeps for all cells, so that all calculations can be matrix level
    
    substeps = da.max(self.state['subtimesteps.main']).compute()
    local_delta_t = delta_t / self.state['dlevelh2r'] / substeps
    tmp_outflow_downstream = da.zeros(self.DATA_SHAPE)
    for _ in np.arange(substeps):
        logging.debug(f' - main channel step {_}')
        # routing
        routing_method = self.config.get('simulation.routing_method', 1)
        if routing_method == 1:
            kinematic_wave_routing(self, local_delta_t)
        else:
            raise Exception(f"Error - Routing method {routing_method} not implemented.")
        
        # update storage
        self.state['channel.storage_previous_timestep'] = da.where(True, self.state['channel.storage'], 0).compute()
        self.state['channel.storage'] = self.state['channel.storage'] + self.state['channel.delta_storage'] * delta_t
    
        # update state variables
        condition = da.logical_and(
            da.logical_and(
                self.state['euler_mask'],
                self.state['mosart_mask'] > 0
            ),
            da.logical_and(
                da.atleast_3d(self.grid[self.config.get('grid.channel_length')] > 0),
                self.state['channel.storage'] > 0
            )
        )
        self.state['channel.cross_section_area'] = da.where(
            condition,
            self.state['channel.storage'] / da.atleast_3d(self.grid[self.config.get('grid.channel_length')]),
            0
        ).compute()
        # Function for estimating maximum water depth assuming rectangular channel and tropezoidal flood plain
        # here assuming the channel cross-section consists of three parts, from bottom to up,
        # part 1 is a rectangular with bankfull depth (rdep) and bankfull width (rwid)
        # part 2 is a tropezoidal, bottom width rwid and top width rwid0, height 0.1*((rwid0-rwid)/2), assuming slope is 0.1
        # part 3 is a rectagular with the width rwid0
        self.state['channel.depth'] = da.where(
            da.logical_and(
                condition,
                self.state['channel.cross_section_area'] > self.state['tiny_value']
            ),
            da.where(
                # not flooded
                self.state['channel.cross_section_area'] - da.atleast_3d(self.grid[self.config.get('grid.channel_depth')] * self.grid[self.config.get('grid.channel_width')]) <= self.state['tiny_value'],
                self.state['channel.cross_section_area'] / da.atleast_3d(self.grid[self.config.get('grid.channel_width')]),
                # flooded
                da.where(
                    self.state['channel.cross_section_area'] > (da.atleast_3d(self.grid[self.config.get('grid.channel_depth')] * self.grid[self.config.get('grid.channel_width')]) + da.atleast_3d(self.grid[self.config.get('grid.channel_width')] + self.grid[self.config.get('grid.channel_floodplain_width')]) * self.state['slope_1_def'] * da.atleast_3d((self.grid[self.config.get('grid.channel_floodplain_width')] - self.grid[self.config.get('grid.channel_width')]) / 2.0) / 2.0 + self.state['tiny_value']),
                    da.atleast_3d(self.grid[self.config.get('grid.channel_depth')]) + self.state['slope_1_def'] * da.atleast_3d((self.grid[self.config.get('grid.channel_floodplain_width')] - self.grid[self.config.get('grid.channel_width')]) / 2.0) + (self.state['channel.cross_section_area'] - da.atleast_3d(self.grid[self.config.get('grid.channel_depth')] * self.grid[self.config.get('grid.channel_width')]) + da.atleast_3d(self.grid[self.config.get('grid.channel_width')] + self.grid[self.config.get('grid.channel_floodplain_width')]) * self.state['slope_1_def'] * da.atleast_3d((self.grid[self.config.get('grid.channel_floodplain_width')] - self.grid[self.config.get('grid.channel_width')]) / 2.0) / 2.0) / da.atleast_3d(self.grid[self.config.get('grid.channel_floodplain_width')]),
                    da.atleast_3d(self.grid[self.config.get('grid.channel_depth')]) + (-da.atleast_3d(self.grid[self.config.get('grid.channel_width')]) + da.sqrt(da.atleast_3d(self.grid[self.config.get('grid.channel_width')] ** 2) + 4.0 * (self.state['channel.cross_section_area'] - da.atleast_3d(self.grid[self.config.get('grid.channel_depth')] * self.grid[self.config.get('grid.channel_width')])) / self.state['slope_1_def'])) * self.state['slope_1_def'] / 2.0
                )
            ),
            0
        ).compute()
        # Function for estimating wetness perimeter based on same assumptions as above
        self.state['channel.wetness_perimeter'] = da.where(
            da.logical_and(
                condition,
                self.state['channel.depth'] > self.state['tiny_value']
            ),
            da.where(
                # not flooded
                self.state['channel.depth'] <= da.atleast_3d(self.grid[self.config.get('grid.channel_depth')]) + self.state['tiny_value'],
                da.atleast_3d(self.grid[self.config.get('grid.channel_width')]) + 2 * self.state['channel.depth'],
                # flooded
                da.where(
                    self.state['channel.depth'] > da.atleast_3d(self.grid[self.config.get('grid.channel_depth')]) + da.atleast_3d((self.grid[self.config.get('grid.channel_floodplain_width')] - self.grid[self.config.get('grid.channel_width')]) / 2) * self.state['slope_1_def'] + self.state['tiny_value'],
                    da.atleast_3d(self.grid[self.config.get('grid.channel_width')]) + 2.0 * (da.atleast_3d(self.grid[self.config.get('grid.channel_depth')]) + da.atleast_3d((self.grid[self.config.get('grid.channel_floodplain_width')] - self.grid[self.config.get('grid.channel_width')]) / 2.0) * self.state['slope_1_def'] * self.state['sin_atan_slope_1_def'] + (self.state['channel.depth'] - da.atleast_3d(self.grid[self.config.get('grid.channel_depth')]) - da.atleast_3d((self.grid[self.config.get('grid.channel_floodplain_width')] - self.grid[self.config.get('grid.channel_width')]) / 2) * self.state['slope_1_def'])),
                    da.atleast_3d(self.grid[self.config.get('grid.channel_width')]) + 2.0 * (da.atleast_3d(self.grid[self.config.get('grid.channel_depth')]) + (self.state['channel.depth'] - da.atleast_3d(self.grid[self.config.get('grid.channel_depth')])) * self.state['sin_atan_slope_1_def'])
                )
            ),
            0
        ).compute()
        self.state['channel.hydraulic_radii'] = da.where(
            da.logical_and(
                condition,
                self.state['channel.wetness_perimeter'] > self.state['tiny_value']
            ),
            self.state['channel.cross_section_area'] / self.state['channel.wetness_perimeter'],
            0
        ).compute()
        
        # update outflow tracking
        tmp_outflow_downstream = tmp_outflow_downstream + self.state['channel.outflow_downstream']
    
    # update outflow
    self.state['channel.outflow_downstream'] = (tmp_outflow_downstream / substeps).compute()
    self.state['channel.outflow_downstream_current_timestep'] = da.array(self.state['channel.outflow_downstream_current_timestep'] - self.state['channel.outflow_downstream']).compute()
    self.state['channel.flow'] = da.array(self.state['channel.flow'] - self.state['channel.outflow_downstream']).compute()

def kinematic_wave_routing(self, delta_t):
    # classic kinematic wave routing method
    
    base_condition = da.logical_and(
        self.state['euler_mask'],
        self.state['mosart_mask'] > 0
    )
    
    # estimation of inflow
    self.state['channel.inflow_upstream'] = da.where(
        base_condition,
        -self.state['channel.outflow_sum_upstream_instant'],
        0
    ).compute()
    
    # estimation of outflow
    self.state['channel.flow_velocity'] = da.where(
        da.logical_or(
            da.logical_not(base_condition),
            da.logical_and(
                da.atleast_3d(self.grid[self.config.get('grid.channel_length')] <= 0),
                self.state['channel.hydraulic_radii'] <= 0
            )
        ),
        0,
        ((self.state['channel.hydraulic_radii'] ** 2) ** (1/3)) * da.atleast_3d(da.sqrt(self.grid[self.config.get('grid.channel_slope')])) / da.atleast_3d(self.grid[self.config.get('grid.channel_manning')])
    ).compute()
    condition = da.atleast_3d(self.grid[self.config.get('grid.total_drainage_area_single')] / self.grid[self.config.get('grid.channel_width')] / self.grid[self.config.get('grid.channel_length')] > 1.0e6)
    self.state['channel.outflow_downstream'] = da.where(
        da.logical_not(base_condition),
        self.state['channel.outflow_downstream'],
        da.where(
            da.logical_or(
                da.atleast_3d(self.grid[self.config.get('grid.channel_length')] <= 0),
                condition
            ),
            -self.state['channel.inflow_upstream'] - self.state['channel.lateral_flow_hillslope'],
            -self.state['channel.flow_velocity'] * self.state['channel.cross_section_area']
        )
    ).compute()
    condition = da.logical_and(
        base_condition,
        da.logical_and(
            da.logical_not(condition),
            da.logical_and(
                -self.state['channel.outflow_downstream'] > self.state['tiny_value'],
                (self.state['channel.storage'] + (self.state['channel.lateral_flow_hillslope'] + self.state['channel.inflow_upstream'] + self.state['channel.outflow_downstream']) * delta_t) < self.state['tiny_value']
            )
        )
    )
    self.state['channel.outflow_downstream'] = da.where(
        condition,
        -(self.state['channel.lateral_flow_hillslope'] + self.state['channel.inflow_upstream'] + self.state['channel.storage']) / delta_t,
        self.state['channel.outflow_downstream']
    ).compute()
    self.state['channel.flow_velocity'] = da.where(
        da.logical_and(
            condition,
            self.state['channel.cross_section_area'] > 0
        ),
        -self.state['channel.outflow_downstream'] / self.state['channel.cross_section_area'],
        self.state['channel.flow_velocity']
    ).compute()
    
    # calculate change in storage, but first round small runoff to zero
    tmp_delta_runoff = self.state['hillslope.wetland_runoff'] * da.atleast_3d(self.state['area']) * da.atleast_3d(self.grid[self.config.get('grid.drainage_fraction')])
    tmp_delta_runoff = da.where(
        da.absolute(tmp_delta_runoff) <= self.state['tiny_value'],
        0,
        tmp_delta_runoff
    )
    self.state['channel.delta_storage'] = (self.state['channel.lateral_flow_hillslope'] + self.state['channel.inflow_upstream'] + self.state['channel.outflow_downstream'] + tmp_delta_runoff).compute()
