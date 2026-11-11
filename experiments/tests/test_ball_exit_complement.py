import unittest
import numpy as np

from experiments import ball_exit_complement as study


class ComplementContracts(unittest.TestCase):
    def test_team_change_preserves_corrected_dead_mask(self):
        cell = {"model_period": np.zeros(60, dtype=int), "fps": 10.}
        mask = np.full(60, "H")
        mask[20:40] = "D"
        home = np.tile([.99, .01], (60, 1))
        away = np.tile([.01, .99], (60, 1))
        a, b = study.predict(home, mask, cell), study.predict(away, mask, cell)
        np.testing.assert_array_equal(a == "D", mask == "D")
        np.testing.assert_array_equal(b == "D", mask == "D")
        self.assertTrue((a[mask != "D"] == "H").all())
        self.assertTrue((b[mask != "D"] == "A").all())

    def test_team_swap_symmetry_and_target_independence(self):
        cell = {"model_period": np.zeros(60, dtype=int), "fps": 10.,
                "labels": object()}
        mask = np.full(60, "H")
        mask[20:40] = "D"
        prob = np.tile([.99, .01], (60, 1))
        original = study.predict(prob, mask, cell)
        swapped = study.predict(prob[:, ::-1], mask, cell)
        expected = np.where(original == "H", "A", np.where(original == "A", "H", original))
        np.testing.assert_array_equal(swapped, expected)

    def test_factorial_budget_contains_only_archived_endpoints_and_retained_mixture(self):
        self.assertEqual(study.TEAMS, ("frontier", "fresh", "hard", "mix075"))
        self.assertEqual(len(study.ARMS), 8)
        self.assertEqual(study.CONTROL, "snap__frontier")


if __name__ == "__main__":
    unittest.main()
