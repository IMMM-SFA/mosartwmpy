import numpy as np

from benedict.dicts import benedict as Benedict
from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State

def flood(state: State, grid: Grid, parameters: Parameters, config: Benedict) -> None:
    """Excess runoff is removed from the available groundwater; mutates state.

    Args:
        state (State): the current model state; will be mutated
        grid (Grid): the model grid
        parameters (Parameters): the model parameters
        config (Benedict): the model configuration
    """
    
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

    # flux sent back to land
    state.flood = np.where(
        (grid.land_mask == 1) & (state.storage > parameters.flood_threshold) & (state.tracer == parameters.LIQUID_TRACER),
        (state.storage - parameters.flood_threshold) / config.get('simulation.timestep'),
        0
    )
    # remove this flux from the input runoff from land
    state.hillslope_surface_runoff = np.where(
        state.tracer == parameters.LIQUID_TRACER,
        state.hillslope_surface_runoff - state.flood,
        state.hillslope_surface_runoff
    )