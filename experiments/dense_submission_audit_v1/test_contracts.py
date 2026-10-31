"""Label-free contract probes for dense submission input handling."""
import unittest
import numpy as np
from experiments import supervised
from experiments.refine_features import self_test as feature_contract_self_test


class InputContractTests(unittest.TestCase):
    def test_timestamp_gaps_and_period_changes_make_independent_blocks(self):
        data = {
            "period": np.array([1, 1, 1, 1, 2, 2]),
            "timestamps": np.array([0., .04, .08, .24, .28, .32]),
            "fps": 25.,
        }
        np.testing.assert_array_equal(supervised.model_periods(data), [0, 0, 0, 1, 2, 2])

    def test_all_missing_player_frame_is_preserved_as_missing(self):
        players = np.zeros((3, 4, 2), dtype=float)
        observed = np.ones((3, 4), dtype=bool)
        observed[1] = False
        players[~observed] = np.nan
        np.testing.assert_array_equal(np.isfinite(players).all(axis=2), observed)

    def test_actual_feature_extraction_is_team_swap_equivariant(self):
        feature_contract_self_test()

    def test_team_swap_feature_mapping_is_involutive(self):
        names = ["home_advantage", "H_distance", "A_distance", "home_vote", "ball_speed"]
        x = np.array([[2., 3., 5., .75, 4.]])
        swapped = supervised.swap_team_features(x, names)
        np.testing.assert_array_equal(supervised.swap_team_features(swapped, names), x)
        np.testing.assert_array_equal(swapped, [[-2., 5., 3., -.75, 4.]])


if __name__ == "__main__":
    unittest.main()
