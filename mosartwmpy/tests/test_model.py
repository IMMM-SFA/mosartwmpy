import numpy as np
from pathlib import Path
import unittest

import pkg_resources

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid


class ModelTest(unittest.TestCase):
    """Test that the model initializes and runs with the default settings."""

    # package data
    GRID_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/grid.zip')
    CONFIG_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/test_config.yaml')
    RUNOFF_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/runoff_1981_01_01.nc')
    DEMAND_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/demand_1981_01_01.nc')
    RESERVOIRS_FILE = pkg_resources.resource_filename('mosartwmpy', 'tests/reservoirs.nc')

    @classmethod
    def setUpClass(self):
        self.model = Model()
        self.grid = Grid.from_files(self.GRID_FILE)
        self.model.initialize(self.CONFIG_FILE, grid=self.grid)

        # set paths for runoff data relative to package
        self.model.config['runoff.path'] = self.RUNOFF_FILE
        self.model.config['water_management.demand.path'] = self.DEMAND_FILE
        self.model.config['water_management.reservoirs.path'] = self.RESERVOIRS_FILE

    @classmethod
    def tearDownClass(self):
        self.model.finalize()

    def test_can_run(self):
        self.model.update()
        self.assertTrue(True, "model initializes and updates")

    def test_input_io(self):
        self.assertGreater(self.model.get_input_item_count(), 0, "model can count input")
        input_vars = self.model.get_input_var_names()
        self.assertGreater(len(input_vars), 0, "model can get input names")
        self.assertIsInstance(self.model.get_var_units(input_vars[0]), str, "model can get units of input variable")
        self.assertIsInstance(self.model.get_var_type(input_vars[0]), str, "model can get type of input variable")
        self.assertGreater(self.model.get_var_itemsize(input_vars[0]), 0, "model can get item size of input variable")
        self.assertGreater(self.model.get_var_nbytes(input_vars[0]), 0, "model can get nbytes of input variable")
        single_index_value = np.full(1, -1234.56)
        single_index_destination = np.empty(1)
        single_index_position = np.full(1, 0)
        self.model.set_value_at_indices(input_vars[0], single_index_position, single_index_value)
        self.model.get_value_at_indices(input_vars[0], single_index_destination, single_index_position)
        np.testing.assert_array_equal(single_index_destination, single_index_value, "model can get and set input at index")
        full_value = np.full(self.model.get_grid_size(), -1234.56)
        full_destination = np.empty_like(full_value)
        pointer = self.model.get_value_ptr(input_vars[0])
        self.model.set_value(input_vars[0], full_value)
        self.model.get_value(input_vars[0], full_destination)
        np.testing.assert_array_equal(full_destination, full_value, "model can get and set input")
        np.testing.assert_array_equal(pointer, full_value, "model can get pointer for input variable")

    def test_output_io(self):
        self.assertGreater(self.model.get_output_item_count(), 0, "model can count output")
        output_vars = self.model.get_output_var_names()
        self.assertGreater(len(output_vars), 0, "model can get output names")
        self.assertIsInstance(self.model.get_var_units(output_vars[0]), str, "model can get units of output variable")
        self.assertIsInstance(self.model.get_var_type(output_vars[0]), str, "model can get type of output variable")
        self.assertGreater(self.model.get_var_itemsize(output_vars[0]), 0, "model can get item size of output variable")
        self.assertGreater(self.model.get_var_nbytes(output_vars[0]), 0, "model can get nbytes of output variable")
        single_index_value = np.full(1, -1234.56)
        single_index_destination = np.empty(1)
        single_index_position = np.full(1, 0)
        self.model.set_value_at_indices(output_vars[0], single_index_position, single_index_value)
        self.model.get_value_at_indices(output_vars[0], single_index_destination, single_index_position)
        np.testing.assert_array_equal(single_index_destination, single_index_value, "model can get and set output at index")
        full_value = np.full(self.model.get_grid_size(), -1234.56)
        full_destination = np.empty_like(full_value)
        pointer = self.model.get_value_ptr(output_vars[0])
        self.model.set_value(output_vars[0], full_value)
        self.model.get_value(output_vars[0], full_destination)
        np.testing.assert_array_equal(full_destination, full_value, "model can get and set output")
        np.testing.assert_array_equal(pointer, full_value, "model can get pointer for output variable")


if __name__ == '__main__':
    unittest.main()
