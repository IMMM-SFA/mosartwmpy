import numpy as np
import unittest

from mosartwmpy import Model
from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State


class ModelTest(unittest.TestCase):
    """Test that the model initializes and runs with the default settings."""
    
    def setUp(self):
        self.model = Model()
        self.grid = Grid.from_files('./mosartwmpy/tests/grid.zip')

    def test_can_initialize_and_run(self):
        self.model.initialize('./mosartwmpy/tests/test_config.yaml', grid = self.grid)
        self.model.update()
        self.assertTrue(True, "model initializes and updates")

if __name__ == '__main__':
    unittest.main()