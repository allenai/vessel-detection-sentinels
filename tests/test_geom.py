import unittest
import math

from src.utils import geom


class TestDegreesLonPerMeter(unittest.TestCase):

    def test_zero_latitude(self):
        self.assertAlmostEqual(geom.degrees_lon_per_meter(0), geom.LON_RANGE/(2*math.pi*geom.R_EARTH_M))

    def test_negative_latitude(self):
        lat = -math.pi/4  # -45 degrees in radians
        self.assertAlmostEqual(geom.degrees_lon_per_meter(lat), geom.degrees_lon_per_meter(-lat))
