"""Contracts for sparse observed-contact event memory."""
import unittest

import numpy as np

from experiments import contact_memory as cm


def fixture(n=250, fps=25):
    timestamps = np.arange(n) / fps
    ball = np.column_stack((np.full(n, 20.0), np.full(n, 30.0)))
    players = np.stack(
        [ball + [0.5, 0], ball + [2, 0], ball + [5, 0], ball + [7, 0]],
        axis=1,
    )
    return dict(
        ball=ball,
        players=players,
        teams=np.array([0, 0, 1, 1]),
        period=np.ones(n, dtype=np.int32),
        timestamps=timestamps,
        fps=float(fps),
        pitch=np.array([105.0, 68.0]),
        ball_observed=np.ones(n, dtype=bool),
        player_observed=np.ones((n, 4), dtype=bool),
    )


def bracketed_turnover():
    data = fixture()
    data["ball_observed"][75:175] = False
    data["players"][175:, 0:2] = data["ball"][175:, None, :] + [5, 0]
    data["players"][175:, 2:4] = data["ball"][175:, None, :] + [0.5, 0]
    return data


class ContactMemoryTests(unittest.TestCase):
    def test_sparse_memory_bridges_two_observed_contact_events(self):
        data = bracketed_turnover()
        memory = cm.memory_features(data, np.array([125], dtype=np.int64))
        cm.validate_memory(memory, 1)
        self.assertTrue(memory["valid"][0].all())
        self.assertGreater(memory["tokens"][0, 0, 2], 0.2)
        self.assertLess(memory["tokens"][0, 1, 2], -0.2)
        self.assertGreater(memory["ages"][0, 0], 1.9)
        self.assertGreater(memory["ages"][0, 1], 1.9)

    def test_same_long_contact_run_is_never_both_memory_sides(self):
        data = fixture()
        memory = cm.memory_features(data, np.array([125], dtype=np.int64))
        np.testing.assert_array_equal(memory["valid"], [[False, False]])

    def test_team_swap_transforms_evidence_and_slot_order_is_stable(self):
        data = bracketed_turnover()
        memory = cm.memory_features(data, np.array([125], dtype=np.int64))
        swapped_raw = dict(data, teams=1 - data["teams"])
        actual = cm.memory_features(swapped_raw, np.array([125], dtype=np.int64))
        expected = cm.swap_memory(memory)
        self.assertEqual(memory["valid"].tolist(), actual["valid"].tolist())
        self.assertEqual(memory["ages"].tolist(), actual["ages"].tolist())
        np.testing.assert_allclose(actual["tokens"], expected["tokens"], atol=1e-6)

    def test_unknown_labels_and_event_metadata_do_not_change_features(self):
        data = bracketed_turnover()
        expected = cm.memory_features(data, np.array([125], dtype=np.int64))
        changed = dict(data, labels=np.full(len(data["ball"]), "D"),
                       targets=np.zeros(len(data["ball"])),
                       events=np.array(["turnover"] * len(data["ball"])))
        actual = cm.memory_features(changed, np.array([125], dtype=np.int64))
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_finite_unobserved_coordinates_cannot_create_or_change_memory(self):
        data = bracketed_turnover()
        expected = cm.memory_features(data, np.array([125], dtype=np.int64))
        changed = dict(data, ball=data["ball"].copy(), players=data["players"].copy())
        changed["ball"][75:175] = 1e6
        changed["players"][75:175] = -1e6
        actual = cm.memory_features(changed, np.array([125], dtype=np.int64))
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        hidden = fixture()
        hidden["ball_observed"][:] = False
        hidden["player_observed"][:] = False
        empty = cm.memory_features(hidden, np.array([125], dtype=np.int64))
        np.testing.assert_array_equal(empty["valid"], False)

    def test_memory_uses_no_samples_outside_its_declared_context(self):
        data = fixture(600)
        data["ball_observed"][75:425] = False
        data["players"][425:580, 0:2] = data["ball"][425:580, None, :] + [5, 0]
        data["players"][425:580, 2:4] = data["ball"][425:580, None, :] + [0.5, 0]
        query = np.array([300], dtype=np.int64)  # t=12s; context is [2, 22]s.
        expected = cm.memory_features(data, query)
        self.assertTrue(expected["valid"].all())

        changed = dict(data, ball=data["ball"].copy(), players=data["players"].copy())
        # Change the prior run before the query-local token crop begins, and
        # split the future run after the crop ends. A two-frame derivative
        # halo is reserved on each edge.
        changed["players"][:35, 0:2] = changed["ball"][:35, None, :] + [8, 0]
        changed["players"][570:, 0:2] = changed["ball"][570:, None, :] + [8, 0]
        actual = cm.memory_features(changed, query)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_contact_memory_does_not_cross_timestamp_breaks(self):
        data = fixture()
        data["timestamps"][125:] += 20.0
        memory = cm.memory_features(data, np.array([120, 125], dtype=np.int64))
        # The otherwise continuous contact run is split at the input break and
        # neither partial run can be used as an anchor from the other block.
        np.testing.assert_array_equal(memory["valid"], np.zeros((2, 2), dtype=bool))

    def test_invalid_slots_are_zero_and_age_limit_is_enforced(self):
        data = bracketed_turnover()
        memory = cm.memory_features(data, np.array([125], dtype=np.int64))
        cm.validate_memory(memory, 1)
        invalid = dict(tokens=memory["tokens"].copy(), ages=memory["ages"].copy(),
                       valid=memory["valid"].copy())
        invalid["valid"][0, 0] = False
        invalid["tokens"][0, 0, 0] = 1
        with self.assertRaises(ValueError):
            cm.validate_memory(invalid, 1)


if __name__ == "__main__":
    unittest.main()
