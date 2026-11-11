import unittest
from unittest.mock import patch

import numpy as np

from experiments import ball_exit_crossing as crossing, ball_exit_runtime as runtime
from experiments import ball_exit_dmask_v1 as fable, supervised


class CrossingContracts(unittest.TestCase):
    def setUp(self):
        self.ball = np.full((80, 2), np.nan)
        self.ball[20] = [95, 34]
        self.ball[40] = [110, 34]
        self.dead = np.zeros(80, dtype=bool)
        self.dead[40:65] = True
        self.period = np.zeros(80, dtype=int)

    def run_rule(self, method="linear"):
        return crossing.refine_start(self.dead, self.ball, (105, 68), self.period, 10., method)

    def test_geometric_crossing_and_midpoint_are_distinct(self):
        linear = self.run_rule()
        midpoint = self.run_rule("midpoint")
        self.assertEqual(np.flatnonzero(linear)[0], 34)
        self.assertEqual(np.flatnonzero(midpoint)[0], 30)
        self.assertEqual(np.flatnonzero(linear)[-1], 64)
        self.assertTrue(self.dead[40:65].all())

    def test_first_rectangle_intersection_at_corner(self):
        self.ball[20] = [95, 60]
        self.ball[40] = [110, 100]
        self.assertEqual(np.flatnonzero(self.run_rule())[0], 24)

    def test_no_cross_block_or_unobserved_inference(self):
        self.period[30:] = 1
        np.testing.assert_array_equal(self.run_rule(), self.dead)
        self.period[:] = 0
        self.ball[20] = np.nan
        np.testing.assert_array_equal(self.run_rule(), self.dead)

    def test_previous_outside_and_long_gaps_do_not_supply_an_inside_bracket(self):
        self.ball[30] = [108, 34]
        np.testing.assert_array_equal(self.run_rule(), self.dead)
        self.ball[30] = np.nan
        self.ball[0] = self.ball[20]
        self.ball[20] = np.nan
        result = crossing.refine_start(self.dead, self.ball, (105, 68), self.period, 5., "linear")
        np.testing.assert_array_equal(result, self.dead)

    def test_rotation_symmetry(self):
        rotated = np.asarray([105., 68.]) - self.ball
        result = crossing.refine_start(self.dead, rotated, (105, 68), self.period, 10., "linear")
        np.testing.assert_array_equal(result, self.run_rule())


class DeploymentContracts(unittest.TestCase):
    def inputs(self):
        return dict(ball=np.array([[50., 34.]] * 100), pitch=np.array([105., 68.]),
                    period=np.ones(100), timestamps=np.arange(100) / 10, fps=10.)

    def test_exact_frozen_parity_and_observation_masks(self):
        inputs = self.inputs()
        inputs["ball"][52:70, 0] = 110.
        dead = np.zeros(100, dtype=bool)
        dead[40:70] = True
        inside, outside = fable.ball_state(inputs["ball"], inputs["pitch"], 0.)
        expected = fable.edit_mask(dead, inside, outside, supervised.model_periods(inputs), 10., "snap")
        np.testing.assert_array_equal(runtime.corrected_mask(inputs, dead), expected)
        inputs["ball_observed"] = np.ones(100, dtype=bool)
        inputs["ball_observed"][52:70] = False
        np.testing.assert_array_equal(runtime.corrected_mask(inputs, dead), dead)

    def test_time_discontinuity_blocks_correction(self):
        inputs = self.inputs()
        inputs["ball"][55:, 0] = 110.
        inputs["timestamps"][50:] += 100
        dead = np.zeros(100, dtype=bool)
        dead[40:70] = True
        np.testing.assert_array_equal(runtime.corrected_mask(inputs, dead), dead)

    def test_targets_cannot_affect_runtime(self):
        inputs = self.inputs()
        dead = np.zeros(100, dtype=bool)
        expected = runtime.corrected_mask(inputs, dead)
        inputs["labels"] = np.full(100, "D")
        np.testing.assert_array_equal(runtime.corrected_mask(inputs, dead), expected)

    def test_manifest_cannot_change_frozen_constants(self):
        manifest = {"architecture": {"kind": runtime.KIND, "ball_exit": dict(runtime.CORRECTION)}}
        self.assertEqual(runtime.frontier_manifest(manifest)["architecture"]["kind"], runtime.frontier.KIND)
        manifest["architecture"]["ball_exit"]["margin_m"] = 0.5
        with self.assertRaises(ValueError):
            runtime.frontier_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
