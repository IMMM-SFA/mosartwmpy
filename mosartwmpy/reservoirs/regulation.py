import numba as nb
import numpy as np

from numba.typed import List


@nb.jit(
    "void("
        "int64, float64,"
        "int64[:], float64[:], float64[:], float64[:],"
        "boolean[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64"
    ")",
    nopython=True,
    nogil=True,
    cache=True,
)
def regulation(
    n,
    delta_t,
    mosart_mask,
    reservoir_id,
    reservoir_surface_area,
    reservoir_storage_capacity,
    euler_mask,
    channel_outflow_downstream,
    reservoir_release,
    reservoir_potential_evaporation,
    reservoir_streamflow,
    reservoir_storage,
    reservoir_runoff_capacity_parameter,
):
    """Regulates the flow across the reservoirs."""

    for i in nb.prange(n):

        if euler_mask[i] and (mosart_mask[i] > 0) and np.isfinite(reservoir_id[i]):

            flow_volume = -channel_outflow_downstream[i] * delta_t

            flow_reservoir = reservoir_release[i] * delta_t

            evaporation = 1e6 * reservoir_potential_evaporation[i] * delta_t * reservoir_surface_area[i]

            minimum_flow = reservoir_runoff_capacity_parameter * reservoir_streamflow[i] * delta_t
            minimum_storage = reservoir_runoff_capacity_parameter * reservoir_storage_capacity[i]
            maximum_storage = reservoir_storage_capacity[i]

            condition_max = (flow_volume + reservoir_storage[i] - flow_reservoir - evaporation) >= maximum_storage
            condition_min = (flow_volume + reservoir_storage[i] - flow_reservoir - evaporation) < minimum_storage
            condition_min_one = flow_reservoir <= (flow_volume - evaporation)
            condition_min_two = (flow_volume - evaporation) >= minimum_flow

            if condition_max:
                flow_reservoir = flow_volume + reservoir_storage[i] - maximum_storage - evaporation
                reservoir_storage[i] = maximum_storage
            else:
                if condition_min:
                    if condition_min_one:
                        reservoir_storage[i] = reservoir_storage[i] + flow_volume - flow_reservoir - evaporation
                    else:
                        if condition_min_two:
                            flow_reservoir = flow_volume - evaporation
                        else:
                            flow_reservoir = flow_volume
                            reservoir_storage[i] = max(0, reservoir_storage[i] - flow_reservoir + flow_volume - evaporation)
                else:
                    reservoir_storage[i] = reservoir_storage[i] + flow_volume - flow_reservoir - evaporation

            channel_outflow_downstream[i] = -flow_reservoir / delta_t


@nb.jit(
    "void("
        "int64, int64, int64, int64[:], float64[:], DictType(int64, int64[:]),"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "int64, float64"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def extraction_regulated_flow(
    n,
    n_reservoir,
    delta_t,
    grid_id,
    reservoir_id,
    reservoir_to_grid_map,
    channel_outflow_downstream,
    outflow_before_regulation,
    outflow_after_regulation,
    channel_flow,
    grid_cell_unmet_demand,
    grid_cell_supply,
    grid_cell_deficit,
    reservoir_supply_iterations,
    reservoir_flow_volume_ratio,
):
    """Tracks the supply of water extracted from the reservoirs to fulfill demand from dependent grid cells."""

    # notes from fortran mosart:
    # This is an iterative algorithm that converts main channel flow
    # at each dam into gridcell supply based on the demand of each
    # gridcell.
    # The basic algorithm is as follows
    # - Compute flow_vol at each dam based on the main channel flow at the gridcell
    # - Compute the demand at each dam based on the demand at each gridcell and the
    #   gridcell/dam dependency.  This dependency is stored in the sparse matrix
    #   SMatP_g2d.  The demand on each dam is the sum of the demand of all the gridcells
    #   that depend on that dam.
    # - Covert dam flow_vol to gridcell supply.  In doing so, reduce the flow_vol
    #   at the dam, reduce the demand at the gridcell, and increase the supply at
    #   the gridcell by the same amount.  There are three conditions for this conversion
    #   to occur and these are carried out in the following order.  dam fraction
    #   is the ratio of the dam flow_vol over the total dam demand.
    #   1. if any dam fraction >= 1.0 for a gridcell, then provide full demand to gridcell
    #      prorated by the number of dams that can provide all the water.
    #   2. if any sum of dam fraction >= 1.0 for a gridcell, then provide full demand to
    #      gridcell prorated by the dam fraction of each dam.
    #   3. if any sum of dam fraction < 1.0 for a gridcell, then provide fraction of 
    #      demand to gridcell prorated by the dam fraction of each dam.
    # - Once the iterative solution has converged, convert the residual flow_vol
    #   back into main channel flow.
    #
    # This implementation assumes several things
    # - Each dam is associated with a particular gridcell and each gridcell has
    #   either 0 or 1 dam associated with it.
    # - The local dam decomposition
    #
    # convert reservoir flow volume to grid cell supply
    # notes from fortran mosart:
    # Covert dam flow_vol to gridcell supply.  In doing so, reduce the flow_vol
    # at the dam, reduce the demand at the gridcell, and increase the supply at
    # the gridcell by the same amount.  There are three conditions for this conversion
    # to occur and these are carried out in the following order.  dam fraction
    # is the ratio of the dam flow_vol over the total dam demand.
    # 1. if any dam fraction >= 1.0 for a gridcell, then provide full demand to gridcell
    #    prorated by the number of dams that can provide all the water.
    # 2. if any sum of dam fraction >= 1.0 for a gridcell, then provide full demand to
    #    gridcell prorated by the dam fraction of each dam.
    # 3. if any sum of dam fraction < 1.0 for a gridcell, then provide fraction of 
    #    demand to gridcell prorated by the dam fraction of each dam.
    #

    # note that these reservoir arrays start at zero index but reservoir IDs start at one index

    reservoir_flow_volume = np.full(n_reservoir, 0.0)

    for i in nb.prange(n):

        outflow_before_regulation[i] = -channel_outflow_downstream[i]
        channel_flow[i] = channel_flow[i] + channel_outflow_downstream[i]

        has_reservoir = np.isfinite(reservoir_id[i])

        if has_reservoir:
            reservoir_flow_volume[int(reservoir_id[i]) - 1] = -(reservoir_flow_volume_ratio * delta_t * channel_outflow_downstream[i])
            channel_outflow_downstream[i] = channel_outflow_downstream[i] + reservoir_flow_volume[int(reservoir_id[i]) - 1] / delta_t

    for _ in np.arange(reservoir_supply_iterations):
        # total potential water demand on a reservoir
        reservoir_demand = np.full(n_reservoir, 0.0)
        # ratio of available water to total potential demand on a reservoir
        demand_fraction = np.full(n_reservoir, 0.0)
        for i in nb.prange(n):
            if grid_cell_unmet_demand[i] > 0:
                # assign this grid cell's demand to each available reservoir
                if grid_id[i] in reservoir_to_grid_map:
                    for r in reservoir_to_grid_map[grid_id[i]]:
                        reservoir_demand[r - 1] = reservoir_demand[r - 1] + grid_cell_unmet_demand[i]
        for r in nb.prange(n_reservoir):
            # ratio of available water to total demand on a reservoir
            if (reservoir_demand[r] > 0.0) and (reservoir_flow_volume[r] > 0.0):
                demand_fraction[r] = reservoir_flow_volume[r] / reservoir_demand[r]
            else:
                demand_fraction[r] = 0.0

        if np.max(demand_fraction) >= 1.0:
            # case 1 - provide all water to grid cell split from all available reservoirs with demand_fraction > 1
            for i in nb.prange(n):
                if grid_id[i] in reservoir_to_grid_map:
                    available_reservoirs = List()
                    for r in reservoir_to_grid_map[grid_id[i]]:
                        if demand_fraction[r - 1] >= 1.0:
                            available_reservoirs.append(r - 1)
                    # take equally from each reservoir
                    if len(available_reservoirs) > 0:
                        for r in available_reservoirs:
                            reservoir_flow_volume[r] = reservoir_flow_volume[r] - grid_cell_unmet_demand[i] / float(len(available_reservoirs))
                        grid_cell_supply[i] = grid_cell_supply[i] + grid_cell_unmet_demand[i]
                        grid_cell_unmet_demand[i] = 0.0

        else:
            sum_demand_fraction = np.full(n, 0.0)
            for i in nb.prange(n):
                if grid_id[i] in reservoir_to_grid_map:
                    for r in reservoir_to_grid_map[grid_id[i]]:
                        sum_demand_fraction[i] = sum_demand_fraction[i] + demand_fraction[r - 1]

            if np.any(sum_demand_fraction >= 1.0):
                # case 2 - provide all water to grid cell prorated from all available reservoirs
                for i in nb.prange(n):
                    if sum_demand_fraction[i] >= 1.0:
                        for r in reservoir_to_grid_map[grid_id[i]]:
                            reservoir_flow_volume[r - 1] = reservoir_flow_volume[r - 1] - grid_cell_unmet_demand[i] * demand_fraction[r - 1] / sum_demand_fraction[i]
                        grid_cell_supply[i] = grid_cell_supply[i] + grid_cell_unmet_demand[i]
                        grid_cell_unmet_demand[i] = 0.0

            else:
                # case 3 - provide fraction of water to grid cell prorated from all available reservoirs
                for i in nb.prange(n):
                    if sum_demand_fraction[i] > 0.0:
                        total_take = 0.0
                        for r in reservoir_to_grid_map[grid_id[i]]:
                            take = min(grid_cell_unmet_demand[i] * demand_fraction[r - 1], grid_cell_unmet_demand[i])
                            total_take = total_take + take
                            reservoir_flow_volume[r - 1] = reservoir_flow_volume[r - 1] - take
                        grid_cell_supply[i] = grid_cell_supply[i] + total_take
                        grid_cell_unmet_demand[i] = grid_cell_unmet_demand[i] - total_take

    for i in nb.prange(n):
        has_reservoir = np.isfinite(reservoir_id[i])
        if has_reservoir:
            # add the residual flow volume back to channel
            channel_outflow_downstream[i] = channel_outflow_downstream[i] - reservoir_flow_volume[int(reservoir_id[i]) - 1] / delta_t
        outflow_after_regulation[i] = -channel_outflow_downstream[i]
        channel_flow[i] = channel_flow[i] - channel_outflow_downstream[i]
        # aggregate deficit
        grid_cell_deficit[i] = grid_cell_deficit[i] + grid_cell_unmet_demand[i]
