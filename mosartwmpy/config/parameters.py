import numpy as np


class Parameters:
    """Constant physical, numerical, and operational parameters used across the model."""

    def __init__(self):
        """Initialize the model constants."""

        # Numerical guards for division and floating-point comparisons [-]
        self.tiny_value = 1.0e-14
        self.tinier_value = 1.0e-50

        # Mean radius of the Earth [m]
        self.radius_earth = 6.37122e6

        # Bank side-slope threshold used to avoid an abrupt jump in hydraulic radius [-]
        self.slope_1_def = 0.1
        self.inverse_sin_atan_slope_1_def = 1.0 / (np.sin(np.arctan(self.slope_1_def)))

        # Flood storage threshold above which excess water is routed directly to the ocean [m3]
        self.flood_threshold = 1.0e36

        # Minimum river depth required before negative wetland runoff is routed to the ocean [m]
        self.river_depth_minimum = 1.0e-4

        # Multiplier applied to the subnetwork channel width [-]
        self.subnetwork_width_parameter = 1.0

        # Minimum hillslope gradient (replaces 0s from the grid file) [m/m]
        self.hillslope_minimum = 0.005
        # Minimum subnetwork channel slope (replaces 0s from the grid file) [m/m]
        self.subnetwork_slope_minimum = 0.0001
        # Minimum main channel slope (replaces 0s from the grid file) [m/m]
        self.channel_slope_minimum = 0.0001

        # Upper threshold for the kinematic wave condition [-]
        self.kinematic_wave_parameter = 1.0e6

        # Reservoir operation and regulation parameters
        # Minimum monthly mean inflow required when identifying the start of the operational year [m3/s]
        self.reservoir_minimum_flow_condition = 0.05
        # Maximum monthly storage-drop-to-capacity ratio enabling flood-control release adjustments [-]
        self.reservoir_flood_control_condition = 1.0
        # Tolerance for small differences in monthly flow vs. demand schedules [m3/s]
        self.reservoir_small_magnitude_difference = 0.01
        # Reference fraction of storage capacity in the Biemans (2011) k-factor denominator [-]
        self.reservoir_regulation_release_parameter = 0.85
        # Cold-start fill fraction of storage capacity for operation-year-start storage [-]
        self.reservoir_initial_op_year_storage_fraction = 0.85
        # Default initial reservoir storage fraction of capacity when no CAP_INIT value is provided [-]
        self.reservoir_default_initial_storage_fraction = 0.9
        # Default minimum (dead) storage fraction of capacity when CAP_MIN is missing or non-positive [-]
        self.reservoir_default_minimum_storage_fraction = 0.1
        # Minimum runoff-to-capacity ratio in Biemans release rules and minimum-inflow fraction in regulation [-]
        self.reservoir_runoff_capacity_parameter = 0.1
        # Maximum fraction of regulated reservoir flow volume available for extraction [-]
        self.reservoir_flow_volume_ratio = 0.9

        # Number of iterative passes used to allocate reservoir supply to dependent cells [-]
        self.reservoir_supply_iterations = 3

        # Minimum channel/subnetwork depth required to perform local irrigation extraction [m]
        self.irrigation_extraction_parameter = 0.1
        # Maximum fraction of main-channel storage that can be extracted in a single substep [-]
        self.irrigation_extraction_maximum_fraction = 0.5

        # Liquid water tracer index [-]
        self.LIQUID_TRACER = 0
