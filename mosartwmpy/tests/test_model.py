import unittest

from mosartwmpy import Model


class ModelTest(unittest.TestCase):
    """Test that the model initializes and runs with the default settings."""
    
    def setUp(self):
        self.model = Model()

    def test_can_initialize_and_run(self):
        self.model.initialize('./mosartwmpy/tests/test_config.yaml')
        self.model.update()
        self.assertTrue(True, "model initializes and updates")

if __name__ == '__main__':
    unittest.main()