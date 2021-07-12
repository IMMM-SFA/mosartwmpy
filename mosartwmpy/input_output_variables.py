from typing import List


class IO:
    """Tracks supported model input and output variables."""

    FLOAT = 'float'
    FLOAT64 = 8
    STATE = 'state'
    GRID = 'grid'

    class Variable:
        """Represents a model input/output variable and associated metadata."""

        def __init__(
            self,
            standard_name: str,
            variable: str,
            variable_type: str,
            variable_item_size: int,
            variable_class: str,
            units: str
        ):
            self.standard_name: str = standard_name
            self.variable: str = variable
            self.variable_type: str = variable_type
            self.variable_item_size: int = variable_item_size
            self.variable_class: str = variable_class
            self.units: str = units

    inputs: List[Variable] = [
        Variable(
            standard_name='surface_runoff_flux',
            variable='hillslope_surface_runoff',
            variable_type=FLOAT,
            variable_item_size=FLOAT64,
            variable_class=STATE,
            units='mm s-1',
        ),
        Variable(
            standard_name='subsurface_runoff_flux',
            variable='hillslope_subsurface_runoff',
            variable_type=FLOAT,
            variable_item_size=FLOAT64,
            variable_class=STATE,
            units='mm s-1',
        ),
        Variable(
            standard_name='demand_flux',
            variable='grid_cell_demand_rate',
            variable_type=FLOAT,
            variable_item_size=FLOAT64,
            variable_class=STATE,
            units='m3 s-1',
        ),
    ]

    outputs: List[Variable] = [
        Variable(
            standard_name='outgoing_water_volume_transport_along_river_channel',
            variable='runoff_land',
            variable_type=FLOAT,
            variable_item_size=FLOAT64,
            variable_class=STATE,
            units='m3 s-1'
        ),
        Variable(
            standard_name='incoming_water_volume_transport_along_river_channel',
            variable='channel_inflow_upstream',
            variable_type=FLOAT,
            variable_item_size=FLOAT64,
            variable_class=STATE,
            units='m3 s-1'
        ),
        Variable(
            standard_name='surface_water_amount',
            variable='storage',
            variable_type=FLOAT,
            variable_item_size=FLOAT64,
            variable_class=STATE,
            units='m3'
        ),
        Variable(
            standard_name='reservoir_water_amount',
            variable='reservoir_storage',
            variable_type=FLOAT,
            variable_item_size=FLOAT64,
            variable_class=STATE,
            units='m3'
        ),
        Variable(
            standard_name='supply_water_amount',
            variable='grid_cell_supply',
            variable_type=FLOAT,
            variable_item_size=FLOAT64,
            variable_class=STATE,
            units='m3'
        ),
        Variable(
            standard_name='deficit_water_amount',
            variable='grid_cell_deficit',
            variable_type=FLOAT,
            variable_item_size=FLOAT64,
            variable_class=STATE,
            units='m3'
        ),
    ]
