import numba as nb
import numpy as np
import pandas as pd

from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State

from mosartwmpy.hillslope.routing import hillslope_routing
from mosartwmpy.main_channel.irrigation import main_channel_irrigation
from mosartwmpy.main_channel.routing import main_channel_routing
from mosartwmpy.reservoirs.regulation import extraction_regulated_flow, regulation
from mosartwmpy.subnetwork.irrigation import subnetwork_irrigation
from mosartwmpy.subnetwork.routing import subnetwork_routing


def update(state: State, grid: Grid, parameters: Parameters, config: Benedict) -> None:
    """Advance the simulation one timestamp.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        parameters (Parameters): the model parameters
        config (Benedict): the model configuration
    """

    # size of flattened grid
    n = len(grid.id)

    _prepare(
        n,
        config.get('water_management.enabled', False),
        config.get('simulation.timestep'),
        grid.land_mask,
        grid.mosart_mask,
        grid.area,
        state.flow,
        state.direct_to_ocean,
        state.outflow_downstream_previous_timestep,
        state.outflow_downstream_current_timestep,
        state.outflow_before_regulation,
        state.outflow_after_regulation,
        state.outflow_sum_upstream_average,
        state.lateral_flow_hillslope_average,
        state.runoff,
        state.direct,
        state.flood,
        state.runoff_land,
        state.runoff_ocean,
        state.delta_storage,
        state.delta_storage_land,
        state.delta_storage_ocean,
        state.grid_cell_deficit,
        state.storage,
        state.channel_storage,
        state.hillslope_wetland_runoff,
        state.hillslope_subsurface_runoff,
        state.hillslope_surface_runoff,
        parameters.flood_threshold,
        parameters.river_depth_minimum,
    )
    # send the direct water to outlet
    state.direct[:] = pd.DataFrame(grid.id, columns=['id']).merge(
        pd.DataFrame(state.direct, columns=['direct']).join(
            pd.DataFrame(grid.outlet_id, columns=['outlet_id'])
        ).groupby('outlet_id').sum(),
        how='left',
        left_on='id',
        right_index=True
    ).direct.fillna(0.0).values
    subcycle_delta_t = config.get('simulation.timestep') / config.get('simulation.subcycles')
    for _ in np.arange(config.get('simulation.subcycles')):
        _subcycle(
            n,
            config.get('water_management.enabled', False),
            subcycle_delta_t,
            grid.mosart_mask,
            grid.hillslope,
            grid.hillslope_manning,
            grid.drainage_density,
            grid.drainage_fraction,
            grid.area,
            state.euler_mask,
            state.hillslope_depth,
            state.hillslope_overland_flow,
            state.hillslope_storage,
            state.hillslope_surface_runoff,
            state.hillslope_delta_storage,
            state.hillslope_subsurface_runoff,
            state.subnetwork_lateral_inflow,
            state.channel_flow,
            state.channel_outflow_downstream_previous_timestep,
            state.channel_outflow_downstream_current_timestep,
            state.channel_outflow_sum_upstream_average,
            state.channel_lateral_flow_hillslope_average,
            state.grid_cell_unmet_demand,
            state.grid_cell_demand_rate,
            parameters.tiny_value,
        )
        for __ in np.arange(config.get('simulation.routing_iterations')):
            if config.get('water_management.enabled', False):
                subnetwork_irrigation(
                    n,
                    grid.mosart_mask,
                    grid.subnetwork_length,
                    grid.subnetwork_width,
                    state.euler_mask,
                    state.subnetwork_depth,
                    state.subnetwork_storage,
                    state.grid_cell_unmet_demand,
                    state.grid_cell_supply,
                    state.subnetwork_cross_section_area,
                    state.subnetwork_wetness_perimeter,
                    state.subnetwork_hydraulic_radii,
                    parameters.irrigation_extraction_parameter,
                    parameters.tiny_value,
                )
            subnetwork_routing(
                n,
                subcycle_delta_t,
                config.get('simulation.routing_iterations'),
                np.nanmax(grid.iterations_subnetwork),
                grid.iterations_subnetwork,
                grid.mosart_mask,
                grid.subnetwork_slope,
                grid.subnetwork_manning,
                grid.subnetwork_length,
                grid.subnetwork_width,
                grid.hillslope_length,
                state.euler_mask,
                state.channel_lateral_flow_hillslope,
                state.subnetwork_flow_velocity,
                state.subnetwork_discharge,
                state.subnetwork_lateral_inflow,
                state.subnetwork_storage,
                state.subnetwork_storage_previous_timestep,
                state.subnetwork_delta_storage,
                state.subnetwork_depth,
                state.subnetwork_cross_section_area,
                state.subnetwork_wetness_perimeter,
                state.subnetwork_hydraulic_radii,
                parameters.tiny_value,
            )

            # upstream interactions
            state.channel_outflow_downstream_previous_timestep = state.channel_outflow_downstream_previous_timestep - state.channel_outflow_downstream
            state.channel_outflow_sum_upstream_instant[:] = 0.0
            # send channel downstream outflow to downstream cells
            state.channel_outflow_sum_upstream_instant[:] = pd.DataFrame(grid.id, columns=['id']).merge(
                pd.DataFrame(state.channel_outflow_downstream, columns=['channel_outflow_downstream']).join(
                    pd.DataFrame(grid.downstream_id, columns=['downstream_id'])
                ).groupby('downstream_id').sum(),
                how='left',
                left_on='id',
                right_index=True
            ).channel_outflow_downstream.fillna(0.0).values
            state.channel_outflow_sum_upstream_average = state.channel_outflow_sum_upstream_average + state.channel_outflow_sum_upstream_instant
            state.channel_lateral_flow_hillslope_average = state.channel_lateral_flow_hillslope_average + state.channel_lateral_flow_hillslope

            main_channel_routing(
                n,
                subcycle_delta_t,
                config.get('simulation.routing_iterations'),
                np.nanmax(grid.iterations_main_channel),
                grid.iterations_main_channel,
                grid.mosart_mask,
                grid.channel_length,
                grid.channel_slope,
                grid.channel_manning,
                grid.total_drainage_area_single,
                grid.channel_width,
                grid.area,
                grid.drainage_fraction,
                grid.grid_channel_depth,
                grid.channel_floodplain_width,
                state.euler_mask,
                state.channel_inflow_upstream,
                state.channel_outflow_sum_upstream_instant,
                state.channel_hydraulic_radii,
                state.channel_flow_velocity,
                state.channel_outflow_downstream,
                state.channel_cross_section_area,
                state.channel_lateral_flow_hillslope,
                state.channel_storage,
                state.channel_storage_previous_timestep,
                state.hillslope_wetland_runoff,
                state.channel_delta_storage,
                state.channel_depth,
                state.channel_wetness_perimeter,
                parameters.slope_1_def,
                parameters.inverse_sin_atan_slope_1_def,
                parameters.tiny_value,
                parameters.kinematic_wave_parameter,
            )
            if config.get('water_management.enabled', False):
                main_channel_irrigation(
                    n,
                    grid.mosart_mask,
                    grid.channel_length,
                    grid.grid_channel_depth,
                    grid.channel_width,
                    grid.channel_floodplain_width,
                    state.euler_mask,
                    state.channel_depth,
                    state.channel_storage,
                    state.grid_cell_unmet_demand,
                    state.grid_cell_supply,
                    state.channel_cross_section_area,
                    state.channel_wetness_perimeter,
                    state.channel_hydraulic_radii,
                    parameters.tiny_value,
                    parameters.tinier_value,
                    parameters.slope_1_def,
                    parameters.inverse_sin_atan_slope_1_def,
                    parameters.irrigation_extraction_parameter,
                    parameters.irrigation_extraction_maximum_fraction,
                )
                regulation(
                    n,
                    subcycle_delta_t / config.get('simulation.routing_iterations'),
                    grid.mosart_mask,
                    grid.reservoir_id,
                    grid.reservoir_surface_area,
                    grid.reservoir_storage_capacity,
                    state.euler_mask,
                    state.channel_outflow_downstream,
                    state.reservoir_release,
                    state.reservoir_potential_evaporation,
                    state.reservoir_streamflow,
                    state.reservoir_storage,
                    parameters.reservoir_runoff_capacity_parameter,
                )

            # update flow
            mask = np.logical_and(state.euler_mask, grid.mosart_mask > 0)
            state.channel_outflow_downstream_current_timestep = np.where(
                mask,
                state.channel_outflow_downstream_current_timestep - state.channel_outflow_downstream,
                state.channel_outflow_downstream_current_timestep
            )
            state.channel_flow = np.where(
                mask,
                state.channel_flow - state.channel_outflow_downstream,
                state.channel_flow
            )

        _average_over_routing_iterations(
            n,
            config.get('simulation.routing_iterations'),
            state.channel_flow,
            state.channel_outflow_downstream_previous_timestep,
            state.channel_outflow_downstream_current_timestep,
            state.channel_outflow_sum_upstream_average,
            state.channel_lateral_flow_hillslope_average
        )

        if config.get('water_management.enabled', False):
            extraction_regulated_flow(
                n,
                int(np.nanmax(grid.reservoir_id) - 1),
                subcycle_delta_t,
                grid.id,
                grid.reservoir_id,
                grid.reservoir_to_grid_map,
                state.channel_outflow_downstream,
                state.outflow_before_regulation,
                state.outflow_after_regulation,
                state.channel_flow,
                state.grid_cell_unmet_demand,
                state.grid_cell_supply,
                state.grid_cell_deficit,
                parameters.reservoir_supply_iterations,
                parameters.reservoir_flow_volume_ratio,
            )

        _accumulate_flow_field(
            n,
            state.flow,
            state.channel_flow,
            state.outflow_downstream_previous_timestep,
            state.outflow_downstream_current_timestep,
            state.channel_outflow_downstream_previous_timestep,
            state.channel_outflow_downstream_current_timestep,
            state.outflow_before_regulation,
            state.outflow_after_regulation,
            state.channel_outflow_before_regulation,
            state.channel_outflow_after_regulation,
            state.outflow_sum_upstream_average,
            state.channel_outflow_sum_upstream_average,
            state.lateral_flow_hillslope_average,
            state.channel_lateral_flow_hillslope_average,
        )

    _finalize(
        n,
        config.get('water_management.enabled', False),
        config.get('simulation.timestep'),
        config.get('simulation.subcycles'),
        grid.land_mask,
        grid.area,
        grid.drainage_fraction,
        state.grid_cell_supply,
        state.hillslope_surface_runoff,
        state.hillslope_subsurface_runoff,
        state.hillslope_wetland_runoff,
        state.flow,
        state.outflow_downstream_previous_timestep,
        state.outflow_downstream_current_timestep,
        state.outflow_before_regulation,
        state.outflow_after_regulation,
        state.outflow_sum_upstream_average,
        state.lateral_flow_hillslope_average,
        state.storage,
        state.channel_storage,
        state.subnetwork_storage,
        state.hillslope_storage,
        state.delta_storage,
        state.direct,
        state.runoff,
        state.runoff_total,
        state.runoff_land,
        state.delta_storage_land,
        state.runoff_ocean,
        state.delta_storage_ocean,
    )


@nb.jit(
    "void("
        "int64, boolean, int64, int64[:], int64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        " float64[:], float64, float64"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def _prepare(
    n,
    water_management_enabled,
    timestep,
    land_mask,
    mosart_mask,
    area,
    flow,
    direct_to_ocean,
    outflow_downstream_previous_timestep,
    outflow_downstream_current_timestep,
    outflow_before_regulation,
    outflow_after_regulation,
    outflow_sum_upstream_average,
    lateral_flow_hillslope_average,
    runoff,
    direct,
    flood,
    runoff_land,
    runoff_ocean,
    delta_storage,
    delta_storage_land,
    delta_storage_ocean,
    grid_cell_deficit,
    storage,
    channel_storage,
    hillslope_wetland_runoff,
    hillslope_subsurface_runoff,
    hillslope_surface_runoff,
    flood_threshold,
    river_depth_minimum,
):
    for i in nb.prange(n):
        # Reset certain state variables
        flow[i] = 0.0
        outflow_downstream_previous_timestep[i] = 0.0
        outflow_downstream_current_timestep[i] = 0.0
        outflow_before_regulation[i] = 0.0
        outflow_after_regulation[i] = 0.0
        outflow_sum_upstream_average[i] = 0.0
        lateral_flow_hillslope_average[i] = 0.0
        runoff[i] = 0.0
        direct[i] = 0.0
        flood[i] = 0.0
        runoff_land[i] = 0.0
        runoff_ocean[i] = 0.0
        delta_storage[i] = 0.0
        delta_storage_land[i] = 0.0
        delta_storage_ocean[i] = 0.0
        if water_management_enabled:
            grid_cell_deficit[i] = 0.0

        ###
        ### flood
        ###
        if (land_mask[i] == 1) and (storage[i] > flood_threshold):
            flood[i] = (storage[i] - flood_threshold) / timestep
            # remove this flux from the input runoff from land
            hillslope_surface_runoff[i] = hillslope_surface_runoff[i] - flood[i]
        else:
            flood[i] = 0.0

        ###
        ### direct to ocean
        ###
        # note - in fortran mosart this direct_to_ocean forcing could be provided from LND component, but we don't seem to be using it
        source_direct = 1.0 * direct_to_ocean[i]

        # wetland runoff
        wetland_runoff_volume = hillslope_wetland_runoff[i] * timestep
        river_volume_minimum = river_depth_minimum * area[i]

        # if wetland runoff is negative and it would bring main channel storage below the minimum, send it directly to ocean
        if ((channel_storage[i] + wetland_runoff_volume) < river_volume_minimum) and (hillslope_wetland_runoff[i] < 0.0):
            source_direct = source_direct + hillslope_wetland_runoff[i]
            hillslope_wetland_runoff[i] = 0.0

        # remove remaining wetland runoff (negative and positive)
        source_direct = source_direct + hillslope_wetland_runoff[i]
        hillslope_wetland_runoff[i] = 0.0

        # runoff from hillslope
        # remove negative subsurface water
        if hillslope_subsurface_runoff[i] < 0.0:
            source_direct = source_direct + hillslope_subsurface_runoff[i]
            hillslope_subsurface_runoff[i] = 0.0

        # remove negative surface water
        if hillslope_surface_runoff[i] < 0.0:
            source_direct = source_direct + hillslope_surface_runoff[i]
            hillslope_surface_runoff[i] = 0.0

        # if ocean cell remove the rest of the sub and surface water
        # other cells will be handled by mosart euler
        if mosart_mask[i] == 0:
            source_direct = source_direct + hillslope_subsurface_runoff[i] + hillslope_surface_runoff[i]
            hillslope_subsurface_runoff[i] = 0.0
            hillslope_surface_runoff[i] = 0.0

        direct[i] = source_direct

        # convert runoff from m3/s to m/s
        hillslope_surface_runoff[i] = hillslope_surface_runoff[i] / area[i]
        hillslope_subsurface_runoff[i] = hillslope_subsurface_runoff[i] / area[i]
        hillslope_wetland_runoff[i] = hillslope_wetland_runoff[i] / area[i]


@nb.jit(
    "void("
        "int64, boolean, float64, int64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "boolean[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def _subcycle(
    n,
    water_management_enabled,
    delta_t,
    mosart_mask,
    hillslope,
    hillslope_manning,
    drainage_density,
    drainage_fraction,
    area,
    euler_mask,
    hillslope_depth,
    hillslope_overland_flow,
    hillslope_storage,
    hillslope_surface_runoff,
    hillslope_delta_storage,
    hillslope_subsurface_runoff,
    subnetwork_lateral_inflow,
    channel_flow,
    channel_outflow_downstream_previous_timestep,
    channel_outflow_downstream_current_timestep,
    channel_outflow_sum_upstream_average,
    channel_lateral_flow_hillslope_average,
    grid_cell_unmet_demand,
    grid_cell_demand_rate,
    tiny_value,
):
    for i in nb.prange(n):

        ###
        ### hillslope routing
        ###
        hillslope_routing(
            i,
            delta_t,
            mosart_mask,
            hillslope,
            hillslope_manning,
            drainage_density,
            drainage_fraction,
            area,
            euler_mask,
            hillslope_depth,
            hillslope_overland_flow,
            hillslope_storage,
            hillslope_surface_runoff,
            hillslope_delta_storage,
            hillslope_subsurface_runoff,
            subnetwork_lateral_inflow,
            tiny_value,
        )

        # zero relevant state variables
        channel_flow[i] = 0.0
        channel_outflow_downstream_previous_timestep[i] = 0.0
        channel_outflow_downstream_current_timestep[i] = 0.0
        channel_outflow_sum_upstream_average[i] = 0.0
        channel_lateral_flow_hillslope_average[i] = 0.0

        # get the demand volume for this substep
        if water_management_enabled:
            grid_cell_unmet_demand[i] = grid_cell_demand_rate[i] * delta_t


@nb.jit(
    "void("
        "int64, int64,"
        "float64[:], float64[:], float64[:], float64[:], float64[:]"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def _average_over_routing_iterations(
    n,
    routing_iterations,
    channel_flow,
    channel_outflow_downstream_previous_timestep,
    channel_outflow_downstream_current_timestep,
    channel_outflow_sum_upstream_average,
    channel_lateral_flow_hillslope_average,
):
    for i in nb.prange(n):
        # average state values over the routing iterations
        channel_flow[i] = channel_flow[i] / routing_iterations
        channel_outflow_downstream_previous_timestep[i] = channel_outflow_downstream_previous_timestep[i] / routing_iterations
        channel_outflow_downstream_current_timestep[i] = channel_outflow_downstream_current_timestep[i] / routing_iterations
        channel_outflow_sum_upstream_average[i] = channel_outflow_sum_upstream_average[i] / routing_iterations
        channel_lateral_flow_hillslope_average[i] = channel_lateral_flow_hillslope_average[i] / routing_iterations


@nb.jit(
    "void("
        "int64, float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def _accumulate_flow_field(
    n,
    flow,
    channel_flow,
    outflow_downstream_previous_timestep,
    outflow_downstream_current_timestep,
    channel_outflow_downstream_previous_timestep,
    channel_outflow_downstream_current_timestep,
    outflow_before_regulation,
    outflow_after_regulation,
    channel_outflow_before_regulation,
    channel_outflow_after_regulation,
    outflow_sum_upstream_average,
    channel_outflow_sum_upstream_average,
    lateral_flow_hillslope_average,
    channel_lateral_flow_hillslope_average,
):
    for i in nb.prange(n):
        # accumulate local flow field
        flow[i] = flow[i] + channel_flow[i]
        outflow_downstream_previous_timestep[i] = outflow_downstream_previous_timestep[i] + channel_outflow_downstream_previous_timestep[i]
        outflow_downstream_current_timestep[i] = outflow_downstream_current_timestep[i] + channel_outflow_downstream_current_timestep[i]
        outflow_before_regulation[i] = outflow_before_regulation[i] + channel_outflow_before_regulation[i]
        outflow_after_regulation[i] = outflow_after_regulation[i] + channel_outflow_after_regulation[i]
        outflow_sum_upstream_average[i] = outflow_sum_upstream_average[i] + channel_outflow_sum_upstream_average[i]
        lateral_flow_hillslope_average[i] = lateral_flow_hillslope_average[i] + channel_lateral_flow_hillslope_average[i]


@nb.jit(
    "void("
        "int64, boolean, int64, int64, int64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def _finalize(
    n,
    water_management_enabled,
    timestep,
    subcycles,
    land_mask,
    area,
    drainage_fraction,
    grid_cell_supply,
    hillslope_surface_runoff,
    hillslope_subsurface_runoff,
    hillslope_wetland_runoff,
    flow,
    outflow_downstream_previous_timestep,
    outflow_downstream_current_timestep,
    outflow_before_regulation,
    outflow_after_regulation,
    outflow_sum_upstream_average,
    lateral_flow_hillslope_average,
    storage,
    channel_storage,
    subnetwork_storage,
    hillslope_storage,
    delta_storage,
    direct,
    runoff,
    runoff_total,
    runoff_land,
    delta_storage_land,
    runoff_ocean,
    delta_storage_ocean,
):
    for i in nb.prange(n):
        if water_management_enabled:
            # convert supply to flux
            grid_cell_supply[i] = grid_cell_supply[i] / timestep

        # convert runoff back to m3/s for output
        hillslope_surface_runoff[i] = hillslope_surface_runoff[i] * area[i]
        hillslope_subsurface_runoff[i] = hillslope_subsurface_runoff[i] * area[i]
        hillslope_wetland_runoff[i] = hillslope_wetland_runoff[i] * area[i]

        # average state values over the subcycles
        flow[i] = flow[i] / subcycles
        outflow_downstream_previous_timestep[i] = outflow_downstream_previous_timestep[i] / subcycles
        outflow_downstream_current_timestep[i] = outflow_downstream_current_timestep[i] / subcycles
        outflow_before_regulation[i] = outflow_before_regulation[i] / subcycles
        outflow_after_regulation[i] = outflow_after_regulation[i] / subcycles
        outflow_sum_upstream_average[i] = outflow_sum_upstream_average[i] / subcycles
        lateral_flow_hillslope_average[i] = lateral_flow_hillslope_average[i] / subcycles

        # update state values
        previous_storage = 1.0 * storage[i]
        storage[i] = (channel_storage[i] + subnetwork_storage[i] + hillslope_storage[i] * area[i]) * drainage_fraction[i]
        delta_storage[i] = (storage[i] - previous_storage) / timestep
        runoff[i] = 1.0 * flow[i]
        runoff_total[i] = 1.0 * direct[i]
        if land_mask[i] == 1:
            runoff_land[i] = 1.0 * runoff[i]
            delta_storage_land[i] = 1.0 * delta_storage[i]
        else:
            runoff_land[i] = 0.0
            delta_storage_land[i] = 0.0
        if land_mask[i] >= 2:
            runoff_ocean[i] = 1.0 * runoff[i]
            runoff_total[i] = runoff_total[i] + runoff[i]
            delta_storage_ocean[i] = 1.0 * delta_storage[i]
        else:
            runoff_ocean[i] = 0.0
            delta_storage_ocean[i] = 0.0
