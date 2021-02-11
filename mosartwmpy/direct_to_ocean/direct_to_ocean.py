import numpy as np
import numexpr as ne
import pandas as pd

from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State

def direct_to_ocean(state: State, grid: Grid, parameters: Parameters, config: Benedict) -> None:
    """Direct transfer to outlet point; mutates state.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        parameters (Parameters): the model parameters
        config (Benedict): the model configuration
    """

    # direct to ocean
    # note - in fortran mosart this direct_to_ocean forcing could be provided from LND component, but we don't seem to be using it
    source_direct = 1.0 * state.direct_to_ocean
    
    # wetland runoff
    wetland_runoff_volume = state.hillslope_wetland_runoff * config.get('simulation.timestep') / config.get('simulation.subcycles')
    river_volume_minimum = parameters.river_depth_minimum * grid.area

    # if wetland runoff is negative and it would bring main channel storage below the minimum, send it directly to ocean
    condition = ((state.channel_storage + wetland_runoff_volume) < river_volume_minimum) & (state.hillslope_wetland_runoff < 0)
    source_direct = np.where(
        condition,
        source_direct + state.hillslope_wetland_runoff,
        source_direct
    )
    state.hillslope_wetland_runoff = np.where(
        condition,
        0,
        state.hillslope_wetland_runoff
    )
    # remove remaining wetland runoff (negative and positive)
    source_direct = source_direct + state.hillslope_wetland_runoff
    state.hillslope_wetland_runoff = 0.0 * state.zeros
    
    # runoff from hillslope
    # remove negative subsurface water
    condition = state.hillslope_subsurface_runoff < 0
    source_direct = np.where(
        condition,
        source_direct + state.hillslope_subsurface_runoff,
        source_direct
    )
    state.hillslope_subsurface_runoff = np.where(
        condition,
        0,
        state.hillslope_subsurface_runoff
    )
    # remove negative surface water
    condition = state.hillslope_surface_runoff < 0
    source_direct = np.where(
        condition,
        source_direct + state.hillslope_surface_runoff,
        source_direct
    )
    state.hillslope_surface_runoff = np.where(
        condition,
        0,
        state.hillslope_surface_runoff
    )

    # if ocean cell or ice tracer, remove the rest of the sub and surface water
    # other cells will be handled by mosart euler
    condition = (grid.mosart_mask == 0) | (state.tracer == parameters.ICE_TRACER)
    source_direct = np.where(
        condition,
        source_direct + state.hillslope_subsurface_runoff + state.hillslope_surface_runoff,
        source_direct
    )
    state.hillslope_subsurface_runoff = np.where(
        condition,
        0,
        state.hillslope_subsurface_runoff
    )
    state.hillslope_surface_runoff = np.where(
        condition,
        0,
        state.hillslope_surface_runoff
    )
    
    state.direct[:] = source_direct

    # send the direct water to outlet for each tracer
    state.direct[:] = pd.DataFrame(
        grid.id, columns=['id']
    ).merge(
        pd.DataFrame(
            state.direct, columns=['direct']
        ).join(
            pd.DataFrame(
                grid.outlet_id, columns=['outlet_id']
            )
        ).groupby('outlet_id').sum(),
        how='left',
        left_on='id',
        right_index=True
    ).direct.fillna(0.0).values