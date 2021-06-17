import numpy as np


class Parameters:
    """Constant parameters used in the model."""
    
    def __init__(self):
        """Initialize the constants."""
        
        # TODO better document what these are used for and what they should be and maybe they should be part of config?
        
        # TINYVALUE
        self.tiny_value = 1.0e-14
        # new and improved even tinier value, MYTINYVALUE
        self.tinier_value = 1.0e-50
        # small value, for less precise arithmetic
        self.small_value = 1.0e-10
        # radius of the earth [m]
        self.radius_earth = 6.37122e6
        # a small value in order to avoid abrupt change of hydraulic radius
        self.slope_1_def = 0.1
        self.inverse_sin_atan_slope_1_def = 1.0 / (np.sin(np.arctan(self.slope_1_def)))
        # flood threshold - excess water will be sent back to ocean
        self.flood_threshold = 1.0e36 # [m3]?
        # liquid/ice effective velocity # TODO is this used anywhere
        self.effective_tracer_velocity = 10.0 # [m/s]?
        # minimum river depth
        self.river_depth_minimum = 1.0e-4 # [m]?
        # coefficient to adjust the width of the subnetwork channel
        self.subnetwork_width_parameter = 1.0
        # minimum hillslope (replaces 0s from grid file)
        self.hillslope_minimum = 0.005
        # minimum subnetwork slope (replaces 0s from grid file)
        self.subnetwork_slope_minimum = 0.0001
        # minimum main channel slope (replaces 0s from grid file)
        self.channel_slope_minimum = 0.0001
        # kinematic wave condition # TODO what is it?
        self.kinematic_wave_parameter = 1.0e6
        
        # reservoir parameters # TODO better describe
        self.reservoir_minimum_flow_condition = 0.05
        self.reservoir_flood_control_condition = 1.0
        self.reservoir_small_magnitude_difference = 0.01
        self.reservoir_regulation_release_parameter = 0.85
        self.reservoir_runoff_capacity_parameter = 0.1
        self.reservoir_flow_volume_ratio = 0.9
        
        # number of supply iterations
        self.reservoir_supply_iterations = 3
        
        # minimum depth to perform irrigation extraction [m]
        self.irrigation_extraction_parameter = 0.1
        # maximum fraction of flow that can be extracted from main channel
        self.irrigation_extraction_maximum_fraction = 0.5
        
        # TODO probably can dispose of these if we never do ICE separately
        self.LIQUID_TRACER = 0
        self.ICE_TRACER = 1
