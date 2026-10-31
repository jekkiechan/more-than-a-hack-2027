import unittest

import numpy as np

from experiments import supervised, turnover_decoder as td


def record_fixture(fps=25.0, n=4, indices=None, periods=None, y=None):
    if indices is None:
        stride = int(round(fps / 5))
        indices = np.arange(n, dtype=np.int64) * stride
    if periods is None:
        periods = np.zeros(n, dtype=np.int64)
    if y is None:
        y = np.array([0, 1, 1, 0][:n], dtype=np.int8)
    names = [
        "ball_speed", "ball_available",
        "H_physical_control_evidence", "A_physical_control_evidence",
        "H_physical_contact_impulse", "A_physical_contact_impulse",
        "H_distance1", "A_distance1", "H_within5m", "A_within5m",
        "H_visible_players", "A_visible_players",
    ]
    x = np.zeros((n, len(names)), dtype=np.float64)
    for i in range(n):
        x[i] = [4 + i, 1, .2 + .1*i, .7 - .1*i,
                .3 + .2*i, .6 - .1*i, 5 + i, 8 - i,
                3 + i, 8 - i, 9 + i, 10 - i]
    return {"x": x, "names": names, "indices": np.asarray(indices, dtype=np.int64),
            "period": np.asarray(periods), "y": np.asarray(y, dtype=np.int8),
            "fps": float(fps), "native_n": int(np.max(indices) + 1)}


def probabilities_fixture(n):
    p = np.zeros((n, 3), dtype=np.float64)
    p[:, 0] = np.linspace(.65, .2, n)
    p[:, 1] = 1 - p[:, 0] - .15
    p[:, 2] = .15
    return p


class TurnoverDecoderTests(unittest.TestCase):
    def test_cost3_matches_existing_decoder_bit_exact_at_multiple_fps_and_breaks(self):
        rng = np.random.default_rng(41)
        for fps in (5., 10., 25.):
            p = rng.uniform(.01, 1., size=(67, 3))
            p /= p.sum(axis=1, keepdims=True)
            periods = np.zeros(len(p), dtype=np.int64)
            periods[23:39] = 1
            periods[39:] = 2
            costs = np.full((len(p), 2), td.BASE_COST, dtype=np.float64)
            actual = td.decode_dynamic(p, periods, fps, costs)
            expected = supervised.decode(p, periods, fps, td.BASE_COST)
            np.testing.assert_array_equal(actual, expected)

    def test_transition_cost_units_leave_d_live_edges_exactly_at_cost3(self):
        p = np.ones((3, 3), dtype=np.float64) / 3
        edge = np.array([[1.5, 4.5], [4.5, 1.5], [2., 4.]], dtype=np.float64)
        penalty = td.transition_penalties(p, edge, fps=25.)
        self.assertEqual(penalty[0, 0, 1], 1.5 * 2.5)
        self.assertEqual(penalty[0, 1, 0], 4.5 * 2.5)
        for h_a in (0, 1):
            dead = 2
            self.assertEqual(penalty[0, h_a, dead], 3. * 2.5)
            self.assertEqual(penalty[0, dead, h_a], 3. * 2.5)

    def test_neutral_llr_is_exact_cost3_and_bounds_are_fixed(self):
        np.testing.assert_array_equal(td.cost_from_llr(np.array([0., 0.])), [3., 3.])
        np.testing.assert_array_equal(td.cost_from_llr(np.array([-100., 100.])), [4.5, 1.5])

    def test_team_swap_maps_oriented_features_to_opposite_edge(self):
        record = record_fixture(n=4)
        p = probabilities_fixture(4)
        original = td.edge_features(record, p)
        swapped_record = dict(record, x=supervised.swap_team_features(record["x"], record["names"]))
        swapped_p = p[:, [1, 0, 2]]
        swapped = td.edge_features(swapped_record, swapped_p)
        edge_count = len(td._edge_pairs(record))
        np.testing.assert_allclose(swapped[:edge_count], original[edge_count:], equal_nan=True)
        np.testing.assert_allclose(swapped[edge_count:], original[:edge_count], equal_nan=True)

    def test_native_cost_mapping_holds_each_5hz_llr_over_its_native_interval(self):
        record = record_fixture(n=4, indices=np.array([0, 5, 10, 15]),
                                periods=np.array([0, 0, 1, 1]))
        event = {"pairs": td._edge_pairs(record),
                 "cost_home_away": np.array([1.5, 4.5]),
                 "cost_away_home": np.array([4.5, 1.5])}
        costs = td.native_edge_costs(record, 20, event)
        np.testing.assert_array_equal(costs[0], [3., 3.])
        np.testing.assert_array_equal(costs[1:6, 0], np.full(5, 1.5))
        np.testing.assert_array_equal(costs[1:6, 1], np.full(5, 4.5))
        np.testing.assert_array_equal(costs[6], [3., 3.])
        np.testing.assert_array_equal(costs[11:16, 0], np.full(5, 4.5))
        np.testing.assert_array_equal(costs[11:16, 1], np.full(5, 1.5))
        np.testing.assert_array_equal(costs[16:], np.full((4, 2), 3.))

    def test_native_unknown_interval_is_excluded_without_creating_a_timebreak(self):
        record = record_fixture(n=4, indices=np.array([0, 5, 10, 15]),
                                periods=np.zeros(4, dtype=np.int64),
                                y=np.array([0, 1, 1, 0], dtype=np.int8))
        full_y = np.repeat(record["y"], 5)
        full_y[7] = -1
        full_period = np.zeros_like(full_y)
        all_tracking_pairs = td._edge_pairs(record)
        pairs, targets = td.eligible_training_events(record, full_y, full_period)
        self.assertEqual(len(all_tracking_pairs), 3)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(targets), 4)
        # Pair 5->10 crosses native X and is omitted; following time-contiguous
        # 10->15 remains an eligible event pair, so X did not reset the block.
        np.testing.assert_array_equal(pairs, [[0, 1], [2, 3]])

    def test_directed_targets_follow_feature_orientation_order(self):
        record = record_fixture(n=4, indices=np.array([0, 5, 10, 15]),
                                periods=np.zeros(4, dtype=np.int64),
                                y=np.array([0, 1, 0, 1], dtype=np.int8))
        full_y = np.repeat(record["y"], 5)
        pairs, target = td.eligible_training_events(record, full_y, np.zeros_like(full_y))
        self.assertEqual(len(pairs), 3)
        # edge_features emits every H->A row followed by every A->H row.
        np.testing.assert_array_equal(target, [1, 0, 1, 0, 1, 0])

    def test_feature_inference_does_not_depend_on_record_targets(self):
        record = record_fixture(n=4)
        p = probabilities_fixture(4)
        actual = td.edge_features(record, p)
        changed = dict(record, y=np.array([1, 0, 0, 1], dtype=np.int8))
        np.testing.assert_array_equal(actual, td.edge_features(changed, p))

    def test_tiny_fit_is_deterministic_natural_prevalence_and_bounded(self):
        # Two source groups and two matches exercise the declared source/match
        # weighting without using external labels or a learned backbone.
        rng = np.random.default_rng(13)
        blocks = []
        for source in ("sportec", "metrica"):
            for match_i in range(2):
                target = np.tile(np.array([0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8), 8)
                features = rng.normal(0, .25, size=(len(target), len(td.FEATURE_NAMES)))
                features[:, 0] += target * 3
                blocks.append({"source": source, "match": f"{source}_{match_i}",
                               "target": target, "features": features})
        first = td.fit_head(blocks, "skillcorner")
        second = td.fit_head(blocks, "skillcorner")
        np.testing.assert_array_equal(first["estimator"].coef_, second["estimator"].coef_)
        np.testing.assert_array_equal(first["transform"]["mean"], second["transform"]["mean"])
        self.assertFalse(first["training"]["class_balancing"])
        self.assertAlmostEqual(first["natural_event_prevalence"], .25)
        sample_record = record_fixture(n=4)
        event = td.predict_edge_events(first, sample_record, probabilities_fixture(4))
        self.assertTrue(np.all(event["cost_home_away"] >= td.COST_MIN))
        self.assertTrue(np.all(event["cost_home_away"] <= td.COST_MAX))
        self.assertTrue(np.all(event["cost_away_home"] >= td.COST_MIN))
        self.assertTrue(np.all(event["cost_away_home"] <= td.COST_MAX))
        swapped_record = dict(sample_record, x=supervised.swap_team_features(
            sample_record["x"], sample_record["names"]))
        swapped_event = td.predict_edge_events(
            first, swapped_record, probabilities_fixture(4)[:, [1, 0, 2]])
        np.testing.assert_allclose(swapped_event["q_home_away"], event["q_away_home"])
        np.testing.assert_allclose(swapped_event["q_away_home"], event["q_home_away"])

    def test_decoder_can_keep_a_real_one_frame_possession(self):
        p = np.array([[.9999, .00005, .00005],
                      [.00005, .9999, .00005],
                      [.9999, .00005, .00005]], dtype=np.float64)
        periods = np.zeros(3, dtype=np.int64)
        baseline = supervised.decode(p, periods, 25., 3.)
        candidate = td.decode_dynamic(p, periods, 25., np.full((3, 2), 1.5))
        np.testing.assert_array_equal(baseline, np.array(["H", "H", "H"]))
        np.testing.assert_array_equal(candidate, np.array(["H", "A", "H"]))


if __name__ == "__main__":
    unittest.main()
