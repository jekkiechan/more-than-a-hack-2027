import unittest
import numpy as np

from experiments.stress import mask_ball


class ObservationStressTests(unittest.TestCase):
    def test_mask_is_reproducible_and_target_independent(self):
        data = {"ball": np.ones((10000, 2)), "fps": np.array(25.),
                "period": np.ones(10000), "labels": np.full(10000, "H")}
        first = mask_ball(data)
        changed = dict(data, labels=np.full(10000, "D"))
        second = mask_ball(changed)
        np.testing.assert_array_equal(np.isnan(first["ball"]), np.isnan(second["ball"]))
        self.assertTrue(np.isfinite(data["ball"]).all())
        rate = np.isnan(first["ball"]).all(axis=1).mean()
        self.assertGreater(rate, .30)
        self.assertLess(rate, .52)

    def test_zero_rate_preserves_existing_missingness(self):
        data = {"ball": np.array([[1., 2.], [np.nan, np.nan]]),
                "fps": 25., "period": np.ones(2)}
        result = mask_ball(data, missing_rate=0)
        np.testing.assert_array_equal(result["ball"], data["ball"])
        np.testing.assert_array_equal(result["ball_observed"], [True, False])


if __name__ == "__main__":
    unittest.main()
