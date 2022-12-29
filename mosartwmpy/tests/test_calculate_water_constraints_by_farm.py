import unittest
import numpy as np
import pandas as pd
import pkg_resources

from mosartwmpy import Model
from mosartwmpy.farmer_abm.farmer_abm import FarmerABM
from mosartwmpy.grid.grid import Grid


class CalculateWaterConstraintsByFarmTest(unittest.TestCase):

    GRID_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/grid.zip')
    OUTPUT_PATH = pkg_resources.resource_filename('mosartwmpy', 'tests')
    CONFIG_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/test_config.yaml')
    RUNOFF_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/runoff_1981_01_01.nc')
    DEMAND_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/demand_1981_01_01.nc')
    RESERVOIRS_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/reservoirs.nc')
    DEPENDENCY_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/dependency_database.parquet')
    MEAN_FLOW_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/mean_flow.parquet')
    MEAN_DEMAND_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/mean_demand.parquet')
    CONSTRAINTS_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/test_land_water_constraints_by_farm.parquet')
    LIVE_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/test_land_water_constraints_by_farm_live.parquet')
    BIAS_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/test_historic_storage_supply_bias.parquet')
    CROP_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/test_crop_prices_by_nldas_id.parquet')

    def test_calculate_water_constraints_by_farm(self):
        model = Model()
        grid = Grid.from_files(self.GRID_FILE)
        model.initialize(self.CONFIG_FILE, grid=grid)

        # set paths for runoff data relative to package
        model.config['simulation.output_path'] = self.OUTPUT_PATH
        model.config['runoff.path'] = self.RUNOFF_FILE
        model.config['water_management.demand.path'] = self.DEMAND_FILE
        model.config['water_management.reservoirs.parameters.path'] = self.RESERVOIRS_FILE
        model.config['water_management.reservoirs.dependencies.path'] = self.DEPENDENCY_FILE
        model.config['water_management.reservoirs.streamflow.path'] = self.MEAN_DEMAND_FILE
        model.config['water_management.reservoirs.demand.path'] = self.MEAN_FLOW_FILE
        model.config['water_management.demand.farmer_abm.land_water_constraints.path'] = self.CONSTRAINTS_FILE
        model.config['water_management.demand.farmer_abm.land_water_constraints_live.path'] = self.LIVE_FILE
        model.config['water_management.demand.farmer_abm.historic_storage_supply.path'] = self.BIAS_FILE
        model.config['water_management.demand.farmer_abm.crop_prices_by_nldas_id.path'] = self.CROP_FILE

        farmerABM = FarmerABM(model)

        land_water_constraints_by_farm_path = model.config.get('water_management.demand.farmer_abm.land_water_constraints.path')
        land_water_constraints_by_farm = pd.read_parquet(land_water_constraints_by_farm_path)
        water_constraints_by_farm = farmerABM.calculate_water_constraints_by_farm(land_water_constraints_by_farm)

        expected = {0: 30.72285118452137, 1: 56.40536982827646, 2: 3.0684983145957405, 3: 37.66540204352916, 4: 83.92334320542051, 5: 16.471781747342227, 
        6: 233.10350292402126, 7: 2552.1110474347847, 8: 39.77842459395363, 9: 1794.121981735524, 10: 222.93698006044025, 11: 5816.8735676312635, 
        12: 384.71756436000385, 13: 33.75622675565492, 14: 8.992966440755907, 15: 13915.186859239666, 16: 5332.013002733427, 17: 9.440670507910413, 
        18: 19.680088488121392, 19: 2.593803835847434, 20: 226.0076990680527, 21: 21.23900104164071, 22: 12.338676766223513, 23: 74.46348552407855, 
        24: 654.7202788173175, 25: 326.74447842874287, 26: 49.27986148382904, 27: 15.594722020424333, 28: 2.042073646707058, 29: 10.87679754091625, 
        30: 0.0, 31: 9.101107376292541, 32: 85.6742114585276, 33: 211.35930661507126, 34: 607.7330637330299, 35: 483.5908697772146, 
        36: 1176.3042993778172, 37: 2615.216044397955, 38: 3299.4129275649575, 39: 3288.416218192872, 40: 5645.160759609735, 41: 3111.019269761427, 
        42: 1694.3293926337408, 43: 1613.3569833770598, 44: 10813.511979632387, 45: 12215.247525518405, 46: 25394.201806792233, 47: 14588.669019674044, 
        48: 369.8737720511687, 49: 122.81606333922728, 50: 8.323512859757498, 51: 3652.4126529340183, 52: 18174.9365444444, 53: 19144.127546798303, 
        54: 1534.553615542034, 55: 220.74229489581606, 56: 277.4710702842115, 57: 0.09653429406007573, 58: 565.8425932097445, 59: 12.728978339437774, 
        60: 0.772274352480606, 61: 1551.6497118991267, 62: 1392.7267419231105, 63: 7221.506155647856, 64: 828.13864024198, 65: 0.0, 
        66: 0.6118340282182333, 67: 0.0, 68: 58.32028963190562, 69: 232.47165527754677, 70: 0.734200833460332, 71: 5.059971435408126, 
        72: 3.4965916238134627, 73: 1431.8118059436317, 74: 1485.4959445184363, 75: 2.584332432101624, 76: 6.6368652768786855, 77: 6.409825493793453, 
        78: 1.4802583101439315, 79: 42.93789870356224, 80: 478.64376563589775, 81: 320.83639169291024, 82: 115.03324054540802, 83: 983.5804008315356, 
        84: 1203.645132678576, 85: 498.9473799328982, 86: 2571.661013189811, 87: 2306.8185854646376, 88: 257.65481107310165, 89: 1426.1978236790142, 
        90: 1867.5493916929483, 91: 871.2380665376529, 92: 551.7301491571899, 93: 12.085666626862842, 94: 0.0, 95: 0.0, 
        96: 2.757034402743735, 97: 2.059543992483289, 98: 0.0, 99: 0.0}

        np.testing.assert_almost_equal(
            np.array(list(water_constraints_by_farm.values())),
            np.array(list(expected.values())),
            2
        )


if __name__ == '__main__':
    unittest.main()
