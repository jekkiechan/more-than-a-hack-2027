import unittest

import numpy as np

from experiments import ball_exit_dmask_v1 as study


class BallStateContracts(unittest.TestCase):
    def test_unobserved_ball_is_neither_inside_nor_outside(self):
        ball = np.array([[50.0, 30.0], [np.nan, np.nan], [-0.2, 30.0], [105.3, 68.4], [50.0, 68.0]])
        inside, outside = study.ball_state(ball, (105.0, 68.0), 0.0)
        np.testing.assert_array_equal(inside, [True, False, False, False, True])
        np.testing.assert_array_equal(outside, [False, False, True, True, False])
        _, strict = study.ball_state(ball, (105.0, 68.0), 0.5)
        np.testing.assert_array_equal(strict, [False, False, False, False, False])


class EditContracts(unittest.TestCase):
    def setUp(self):
        self.fps = 10.0
        self.period = np.zeros(100, dtype=np.int64)
        self.dead = np.zeros(100, dtype=bool); self.dead[40:70] = True
        self.inside = np.ones(100, dtype=bool); self.outside = np.zeros(100, dtype=bool)

    def test_delay_moves_start_to_first_outside_observation_after_an_inside_one(self):
        self.inside[52:70] = False; self.outside[52:60] = True
        edited = study.edit_mask(self.dead, self.inside, self.outside, self.period, self.fps, "delay")
        self.assertFalse(edited[40:52].any()); self.assertTrue(edited[52:70].all())
        self.assertEqual(edited.sum(), 18)

    def test_delay_needs_an_observed_exit_inside_the_window(self):
        np.testing.assert_array_equal(
            study.edit_mask(self.dead, self.inside, self.outside, self.period, self.fps, "delay"), self.dead)
        self.inside[40:70] = False; self.outside[45:50] = True          # outside, but never seen inside first
        np.testing.assert_array_equal(
            study.edit_mask(self.dead, self.inside, self.outside, self.period, self.fps, "delay"), self.dead)
        self.inside[:] = True; self.outside[:] = False; self.inside[95:] = False
        late = self.dead.copy(); self.outside[69] = False
        np.testing.assert_array_equal(
            study.edit_mask(late, self.inside, self.outside, self.period, self.fps, "delay"), self.dead)

    def test_force_marks_outside_frames_dead_until_ball_returns_or_cap(self):
        self.inside[20:26] = False; self.outside[20:23] = True           # unseen 23-25, inside again at 26
        edited = study.edit_mask(self.dead, self.inside, self.outside, self.period, self.fps, "force")
        self.assertTrue(edited[20:26].all()); self.assertFalse(edited[26:40].any()); self.assertFalse(edited[:20].any())
        self.inside[:] = False; self.outside[:] = False; self.outside[0] = True
        capped = study.edit_mask(np.zeros(100, dtype=bool), self.inside, self.outside, self.period, self.fps, "force")
        self.assertEqual(capped.sum(), int(study.HOLD_S * self.fps))

    def test_edits_never_cross_period_blocks_and_only_use_declared_rules(self):
        period = np.r_[np.zeros(50, dtype=np.int64), np.ones(50, dtype=np.int64)]
        inside = np.zeros(100, dtype=bool); outside = np.zeros(100, dtype=bool); outside[45] = True
        edited = study.edit_mask(np.zeros(100, dtype=bool), inside, outside, period, self.fps, "force")
        self.assertTrue(edited[45:50].all()); self.assertFalse(edited[50:].any())
        with self.assertRaises(ValueError):
            study.edit_mask(self.dead, self.inside, self.outside, self.period, self.fps, "trim")


if __name__ == "__main__":
    unittest.main()
