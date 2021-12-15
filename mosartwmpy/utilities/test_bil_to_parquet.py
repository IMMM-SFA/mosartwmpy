from affine import Affine
from dataclasses import dataclass
from mosartwmpy.utilities.bil_to_parquet import avg_resample, crop_to_domain, return_in_memory
import numpy as np
from rasterio.crs import CRS
from shapely.geometry import box
import unittest
import xarray as xr

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

class TestCropToDomain(unittest.TestCase):
    def setUp(self):
        self.crs = CRS.from_epsg(4326)
        self.transform = Affine.identity()
        self.grid_latitude_key = 'lat'
        self.grid_longitude_key = 'lon'
        self.grid_resolution = .125

    def test_crop_to_domain(self):
        @dataclass
        class TestCase:
            name: str
            resolution: float
            input_bounds: list
            crop_bounds: list
            expected: list

        testcases = [
            TestCase(name='identity', resolution=1, input_bounds=[0, 0, 4, 4], crop_bounds=[0, 0, 4, 4], expected=[0, 4, 4 ,0]),
            TestCase(name='crop_smaller', resolution=1, input_bounds=[0, 0, 20, 20], crop_bounds=[5, 5, 10, 10], expected=[5, 11, 11, 5]),
            TestCase(name='smaller_than_crop', resolution=1, input_bounds=[0, 0, 1, 1], crop_bounds=[0, 0, 5, 5], expected=[0, 1, 1, 0]),
        ]

        for case in testcases:
            bil = return_in_memory(np.zeros((1, case.input_bounds[2], case.input_bounds[3])), self.crs, self.transform)
            lon = [case.crop_bounds[0], case.crop_bounds[2]]
            lat = [case.crop_bounds[1], case.crop_bounds[3]]
            domain = xr.Dataset(
                coords=dict(
                    lon=(np.linspace(min(lon), max(lon), int((max(lon)-min(lon))/case.resolution))),
                    lat=(np.linspace(min(lat), max(lat), int((max(lat)-min(lat))/case.resolution))),
                )
            )

            actual = crop_to_domain(bil, domain, self.grid_latitude_key, self.grid_longitude_key, case.resolution)
            self.assertEqual([actual.bounds[0], actual.bounds[1], actual.bounds[2], actual.bounds[3]], case.expected)
            
if __name__ == "__main__":
    unittest.main()