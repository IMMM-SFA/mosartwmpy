from affine import Affine
from dataclasses import dataclass
from bil_to_parquet import avg_resample, return_in_memory
import numpy as np
from rasterio.crs import CRS
import unittest

class TestAvgResample(unittest.TestCase):
    def setUp(self):
        self.crs = CRS.from_epsg(4326)
        self.transform = Affine.identity()
    
    def test_avg_resample(self):
        @dataclass
        class TestCase:
            name: str
            scale: float
            input: np.ndarray
            expected: np.ndarray

        testcases = [
            TestCase(name="rescale_to_same_size", scale=1, input=np.zeros((1, 4, 4)), expected=np.zeros((1, 4, 4))),
            TestCase(name="rescale_double", scale=.5, input=np.zeros((1, 4, 4)), expected=np.zeros((1, 8, 8))),
            TestCase(name="rescale_half", scale=2, input=np.zeros((1, 4, 4)), expected=np.zeros((1, 2, 2))),
            TestCase(name="rescale_test_average_positive", scale=2, input=np.array([[[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]]], dtype=float), expected=np.array([[[1.5, 1.5, 1.5]]])),
            TestCase(name="rescale_test_average_negative", scale=2, input=np.array([[[-1, -2, -1, -2, -1, -2], [-1, -2, -1, -2, -1, -2]]], dtype=float), expected=np.array([[[-1.5, -1.5, -1.5]]])),
            TestCase(name="rescale_test_average_zero", scale=2, input=np.array([[[-1, -2, -1, -2, -1, -2], [1, 2, 1, 2, 1, 2]]], dtype=float), expected=np.array([[[0, 0, 0]]])),
        ]

        for case in testcases:
            bil = return_in_memory(case.input, self.crs, self.transform)
            actual = avg_resample(bil, case.scale).read()
            self.assertTrue(np.array_equiv(actual, case.expected))

if __name__ == "__main__":
    unittest.main()