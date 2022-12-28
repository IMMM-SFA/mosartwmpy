import numpy as np
import unittest

from datetime import date

from mosartwmpy.reservoirs.istarf import compute_istarf_release
from mosartwmpy.utilities.epiweek import get_epiweek_from_datetime


class IstarfTest(unittest.TestCase):

    def test_is_correct(self):

        epiweek = np.minimum(float(get_epiweek_from_datetime(date(1994, 10, 1))), 52.0)
        uses_istarf = np.array([True])
        reservoir_id = np.array([0.0])
        upper_min = np.array([-np.Inf])
        upper_max = np.array([np.Inf])
        upper_alpha = np.array([-1.74])
        upper_beta = np.array([-13.62])
        upper_mu = np.array([78.17])
        lower_min = np.array([48.02])
        lower_max = np.array([np.Inf])
        lower_alpha = np.array([0.64])
        lower_beta = np.array([-3.34])
        lower_mu = np.array([50.12])
        release_min_parameter = np.array([-0.992])
        release_max_parameter = np.array([3.312])
        release_alpha_one = np.array([0.0836])
        release_alpha_two = np.array([0.2017])
        release_beta_one = np.array([-0.1011])
        release_beta_two = np.array([-0.0162])
        release_p_one = np.array([1.502])
        release_p_two = np.array([0.006])
        release_c = np.array([-0.617])
        capacity = np.array([6670.7*1.0e6])
        inflow_mean = np.array([176.297])
        storage = np.array([3493.943 * 1.0e6])
        inflow = np.array([10.512893 * 1.0e6 / (24.0 * 60.0 * 60.0)])
        release = np.array([0.0])

        compute_istarf_release(
            epiweek,
            uses_istarf,
            reservoir_id,
            upper_min,
            upper_max,
            upper_alpha,
            upper_beta,
            upper_mu,
            lower_min,
            lower_max,
            lower_alpha,
            lower_beta,
            lower_mu,
            release_min_parameter,
            release_max_parameter,
            release_alpha_one,
            release_alpha_two,
            release_beta_one,
            release_beta_two,
            release_p_one,
            release_p_two,
            release_c,
            capacity,
            inflow_mean,
            storage,
            inflow,
            release,
        )

        self.assertAlmostEqual(release[0], 6957353.29, 2)


if __name__ == '__main__':
    unittest.main()
