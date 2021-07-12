import numpy as np
import pandas as pd

from benedict.dicts import benedict as Benedict

from mosartwmpy.config.parameters import Parameters
from mosartwmpy.grid.grid import Grid
from mosartwmpy.reservoirs.state import initialize_reservoir_state


class State:

    def __init__(self, grid: Grid = None, config: Benedict = None, parameters: Parameters = None, grid_size: int = None, empty: bool = False):
        """Initialize the model state.

        Args:
            grid (Grid): the model grid
            config (Config): the model configuration
            parameters (Parameters): the model parameters
            grid_size (int): size of the flattened grid
            empty (bool): if True, return an empty instance
        """
    
        # flow [m3/s]
        # flow
        self.flow: np.ndarray = np.empty(0)
        # outflow into downstream links from previous timestep [m3/s]
        # eroup_lagi
        self.outflow_downstream_previous_timestep: np.ndarray = np.empty(0)
        # outflow into downstream links from current timestep [m3/s]
        # eroup_lagf
        self.outflow_downstream_current_timestep: np.ndarray = np.empty(0)
        # initial outflow before dam regulation at current timestep [m3/s]
        # erowm_regi
        self.outflow_before_regulation: np.ndarray = np.empty(0)
        # final outflow after dam regulation at current timestep [m3/s]
        # erowm_regf
        self.outflow_after_regulation: np.ndarray = np.empty(0)
        # outflow sum of upstream gridcells, average [m3/s]
        # eroutUp_avg
        self.outflow_sum_upstream_average: np.ndarray = np.empty(0)
        # lateral flow from hillslope, including surface and subsurface runoff generation components, average [m3/s]
        # erlat_avg
        self.lateral_flow_hillslope_average: np.ndarray = np.empty(0)
        # routing storage [m3]
        # volr
        self.storage: np.ndarray = np.empty(0)
        # routing change in storage [m3/s]
        # dvolrdt
        self.delta_storage: np.ndarray = np.empty(0)
        # routing change in storage masked for land [m3/s]
        # dvolrdtlnd
        self.delta_storage_land: np.ndarray = np.empty(0)
        # routing change in storage masked for ocean [m3/s]
        # dvolrdtocn
        self.delta_storage_ocean: np.ndarray = np.empty(0)
        # basin derived flow [m3/s]
        # runoff
        self.runoff: np.ndarray = np.empty(0)
        # return direct flow [m3/s]
        # runofftot
        self.runoff_total: np.ndarray = np.empty(0)
        # runoff masked for land [m3/s]
        # runofflnd
        self.runoff_land: np.ndarray = np.empty(0)
        # runoff masked for ocean [m3/s]
        # runoffocn
        self.runoff_ocean: np.ndarray = np.empty(0)
        # direct flow [m3/s]
        # direct
        self.direct: np.ndarray = np.empty(0)
        # direct-to-ocean forcing [m3/s]
        # qdto
        self.direct_to_ocean: np.ndarray = np.empty(0)
        # flood water [m3/s]
        # flood
        self.flood: np.ndarray = np.empty(0)
        # hillslope surface water storage [m]
        # wh
        self.hillslope_storage: np.ndarray = np.empty(0)
        # change of hillslope water storage [m/s]
        # dwh
        self.hillslope_delta_storage: np.ndarray = np.empty(0)
        # depth of hillslope surface water [m]
        # yh
        self.hillslope_depth: np.ndarray = np.empty(0)
        # surface runoff from hillslope [m/s]
        # qsur
        self.hillslope_surface_runoff: np.ndarray = np.empty(0)
        # subsurface runoff from hillslope [m/s]
        # qsub
        self.hillslope_subsurface_runoff: np.ndarray = np.empty(0)
        # runoff from glacier, wetlands, and lakes [m/s]
        # qgwl
        self.hillslope_wetland_runoff: np.ndarray = np.empty(0)
        # overland flor from hillslope into subchannel (outflow is negative) [m/s]
        # ehout
        self.hillslope_overland_flow: np.ndarray = np.empty(0)
        # subnetwork water storage [m3]
        # wt
        self.subnetwork_storage: np.ndarray = np.empty(0)
        # subnetwork water storage at previous timestep [m3]
        # wt_last
        self.subnetwork_storage_previous_timestep: np.ndarray = np.empty(0)
        # change of subnetwork water storage [m3]
        # dwt
        self.subnetwork_delta_storage: np.ndarray = np.empty(0)
        # depth of subnetwork water [m]
        # yt
        self.subnetwork_depth: np.ndarray = np.empty(0)
        # cross section area of subnetwork [m2]
        # mt
        self.subnetwork_cross_section_area: np.ndarray = np.empty(0)
        # hydraulic radii of subnetwork [m]
        # rt
        self.subnetwork_hydraulic_radii: np.ndarray = np.empty(0)
        # wetness perimeter of subnetwork [m]
        # pt
        self.subnetwork_wetness_perimeter: np.ndarray = np.empty(0)
        # subnetwork flow velocity [m/s]
        # vt
        self.subnetwork_flow_velocity: np.ndarray = np.empty(0)
        # subnetwork mean travel time of water within travel [s]
        # tt
        self.subnetwork_mean_travel_time: np.ndarray = np.empty(0)
        # subnetwork evaporation [m/s]
        # tevap
        self.subnetwork_evaporation: np.ndarray = np.empty(0)
        # subnetwork lateral inflow from hillslope [m3/s]
        # etin
        self.subnetwork_lateral_inflow: np.ndarray = np.empty(0)
        # subnetwork discharge into main channel (outflow is negative) [m3/s]
        # etout
        self.subnetwork_discharge: np.ndarray = np.empty(0)
        # main channel storage [m3]
        # wr
        self.channel_storage: np.ndarray = np.empty(0)
        # change in main channel storage [m3]
        # dwr
        self.channel_delta_storage: np.ndarray = np.empty(0)
        # main channel storage at last timestep [m3]
        # wr_last
        self.channel_storage_previous_timestep: np.ndarray = np.empty(0)
        # main channel water depth [m]
        # yr
        self.channel_depth: np.ndarray = np.empty(0)
        # cross section area of main channel [m2]
        # mr
        self.channel_cross_section_area: np.ndarray = np.empty(0)
        # hydraulic radii of main channel [m]
        # rr
        self.channel_hydraulic_radii: np.ndarray = np.empty(0)
        # wetness perimeter of main channel[m]
        # pr
        self.channel_wetness_perimeter: np.ndarray = np.empty(0)
        # main channel flow velocity [m/s]
        # vr
        self.channel_flow_velocity: np.ndarray = np.empty(0)
        # main channel evaporation [m/s]
        # erlg
        self.channel_evaporation: np.ndarray = np.empty(0)
        # lateral flow from hillslope [m3/s]
        # erlateral
        self.channel_lateral_flow_hillslope: np.ndarray = np.empty(0)
        # inflow from upstream links [m3/s]
        # erin
        self.channel_inflow_upstream: np.ndarray = np.empty(0)
        # outflow into downstream links [m3/s]
        # erout
        self.channel_outflow_downstream: np.ndarray = np.empty(0)
        # outflow into downstream links from previous timestep [m3/s]
        # TRunoff%eroup_lagi
        self.channel_outflow_downstream_previous_timestep: np.ndarray = np.empty(0)
        # outflow into downstream links from current timestep [m3/s]
        # TRunoff%eroup_lagf
        self.channel_outflow_downstream_current_timestep: np.ndarray = np.empty(0)
        # initial outflow before dam regulation at current timestep [m3/s]
        # TRunoff%erowm_regi
        self.channel_outflow_before_regulation: np.ndarray = np.empty(0)
        # final outflow after dam regulation at current timestep [m3/s]
        # TRunoff%erowm_regf
        self.channel_outflow_after_regulation: np.ndarray = np.empty(0)
        # outflow sum of upstream gridcells, instantaneous [m3/s]
        # eroutUp
        self.channel_outflow_sum_upstream_instant: np.ndarray = np.empty(0)
        # outflow sum of upstream gridcells, average [m3/s]
        # TRunoff%eroutUp_avg
        self.channel_outflow_sum_upstream_average: np.ndarray = np.empty(0)
        # lateral flow from hillslope, including surface and subsurface runoff generation components, average [m3/s]
        # TRunoff%erlat_avg
        self.channel_lateral_flow_hillslope_average: np.ndarray = np.empty(0)
        # flux for adjustment of water balance residual in glacier, wetlands, and lakes [m3/s]
        # ergwl
        self.channel_wetland_flux: np.ndarray = np.empty(0)
        # streamflow from outlet, positive is out [m3/s]
        # flow
        self.channel_flow: np.ndarray = np.empty(0)
        # tracer, i.e. liquid or ice - TODO ice not implemented yet
        self.tracer: np.ndarray = np.empty(0)
        # euler mask - which cells to perform the euler calculation on
        self.euler_mask: np.ndarray = np.empty(0)
        # a column of always all zeros, to use as a utility
        self.zeros: np.ndarray = np.empty(0)

        # Reservoir related state variables

        # reservoir streamflow schedule
        self.reservoir_streamflow: np.ndarray = np.empty(0)
        # StorMthStOp
        self.reservoir_storage_operation_year_start: np.ndarray = np.empty(0)
        # storage [m3]
        self.reservoir_storage: np.ndarray = np.empty(0)
        # MthStOp,
        self.reservoir_month_start_operations: np.ndarray = np.empty(0)
        # MthStFC
        self.reservoir_month_flood_control_start: np.ndarray = np.empty(0)
        # MthNdFC
        self.reservoir_month_flood_control_end: np.ndarray = np.empty(0)
        # release [m3/s]
        self.reservoir_release: np.ndarray = np.empty(0)
        # supply [m3/s]
        self.grid_cell_supply: np.ndarray = np.empty(0)
        # demand rate [m3/s] (demand0)
        self.grid_cell_demand_rate: np.ndarray = np.empty(0)
        # unmet demand volume within sub timestep [m3]
        self.grid_cell_unmet_demand: np.ndarray = np.empty(0)
        # unmet demand over whole timestep [m3]
        self.grid_cell_deficit: np.ndarray = np.empty(0)
        # potential evaporation [mm/s] # TODO this doesn't appear to be initialized anywhere currently
        self.reservoir_potential_evaporation: np.ndarray = np.empty(0)

        # shortcut to get an empty state instance
        if empty:
            return
        
        # initialize all the state variables
        for key in [key for key in dir(self) if isinstance(getattr(self, key), np.ndarray)]:
            setattr(self, key, np.zeros(grid_size))
        
        # set tracer to liquid everywhere... TODO ice is not implemented
        self.tracer = np.full(grid_size, parameters.LIQUID_TRACER)
        
        # set euler_mask to 1 everywhere... TODO not really necessary without ice implemented
        self.euler_mask = np.where(
            self.tracer == parameters.LIQUID_TRACER,
            True,
            False
        )
    
        if config.get('water_management.enabled', False):
            initialize_reservoir_state(self, grid, config, parameters)

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dataframe(self) -> pd.DataFrame:
        """Builds a dataframe from all the state values.

        Returns:
            pd.DataFrame: a dataframe with all the state values as columns
        """
        keys = [key for key in dir(self) if isinstance(getattr(self, key), np.ndarray)]
        df = pd.DataFrame()
        for key in keys:
            df[key] = getattr(self, key)
        return df
    
    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> 'State':
        """Creates a State instance from columns in a dataframe.

        Args:
            df (pd.DataFrame): the dataframe from which to build the state

        Returns:
            State: a State instance populated with the columns from the dataframe
        """
        state = State(empty=True)
        for key in df.columns:
            setattr(state, key, df[key].values)
        return state
