from datetime import datetime
import unittest

from mosartwmpy.utilities.epiweek import get_epiweek_from_datetime


class EpiweekTest(unittest.TestCase):

    def test_epiweek(self):
        self.assertEqual(get_epiweek_from_datetime(datetime(2017, 1, 1)), 1)
        self.assertEqual(get_epiweek_from_datetime(datetime(2015, 1, 1)), 53)
        self.assertEqual(get_epiweek_from_datetime(datetime(2016, 1, 1)), 52)
        self.assertEqual(get_epiweek_from_datetime(datetime(2017, 3, 15)), 11)
        self.assertEqual(get_epiweek_from_datetime(datetime(2017, 3, 31)), 13)


if __name__ == '__main__':
    unittest.main()
