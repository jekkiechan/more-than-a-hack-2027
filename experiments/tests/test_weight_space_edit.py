import unittest
import importlib.util

import numpy as np

from experiments.weight_space_edit import (
    delayed_transition_activations,
    edit_conditional_ha_state_projection,
    edit_had_state_ha_projection,
    hidden_failure_direction,
    shift_gru_update_bias,
    subtract_output_direction,
)


class ShiftGruUpdateBiasTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "sequence.bias_ih_l0": np.arange(12, dtype=np.float32),
            "sequence.bias_hh_l0": np.full(12, 4, dtype=np.float32),
            "sequence.bias_ih_l0_reverse": np.arange(12, dtype=np.float32) + 20,
            "sequence.weight_ih_l0": np.arange(36, dtype=np.float32).reshape(12, 3),
            "output.weight": np.arange(8, dtype=np.float32).reshape(2, 4),
            "d_state": np.array([3.0, 7.0], dtype=np.float32),
        }

    def test_negative_shift_changes_only_update_input_bias_for_both_directions(self):
        edited = shift_gru_update_bias(self.state, hidden_size=4, delta=-0.25)
        self.assertIsNot(edited, self.state)
        self.assertIsNot(edited["sequence.bias_ih_l0"], self.state["sequence.bias_ih_l0"])
        np.testing.assert_array_equal(edited["sequence.bias_ih_l0"][:4], self.state["sequence.bias_ih_l0"][:4])
        np.testing.assert_allclose(edited["sequence.bias_ih_l0"][4:8], self.state["sequence.bias_ih_l0"][4:8] - 0.25)
        np.testing.assert_array_equal(edited["sequence.bias_ih_l0"][8:], self.state["sequence.bias_ih_l0"][8:])
        np.testing.assert_allclose(edited["sequence.bias_ih_l0_reverse"][4:8], self.state["sequence.bias_ih_l0_reverse"][4:8] - 0.25)
        for key in ("sequence.bias_hh_l0", "sequence.weight_ih_l0", "output.weight", "d_state"):
            np.testing.assert_array_equal(edited[key], self.state[key])

    def test_zero_delta_is_exact_identity_by_value(self):
        edited = shift_gru_update_bias(self.state, hidden_size=4, delta=0.0)
        for key in self.state:
            np.testing.assert_array_equal(edited[key], self.state[key])

    def test_positive_delta_is_an_opposite_sign_control(self):
        edited = shift_gru_update_bias(self.state, hidden_size=4, delta=0.5)
        np.testing.assert_allclose(edited["sequence.bias_ih_l0"][4:8], self.state["sequence.bias_ih_l0"][4:8] + 0.5)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed in this interpreter")
    def test_negative_update_bias_reduces_actual_gru_hidden_retention(self):
        import torch
        from torch import nn

        torch.manual_seed(0)
        base = nn.GRU(input_size=1, hidden_size=1, batch_first=True)
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.zero_()
        h0 = torch.ones(1, 1, 1)
        x = torch.zeros(1, 1, 1)
        original, _ = base(x, h0)
        edited = nn.GRU(input_size=1, hidden_size=1, batch_first=True)
        edited.load_state_dict({k: torch.as_tensor(v) for k, v in shift_gru_update_bias(
            {k: v.detach().numpy() for k, v in base.state_dict().items()},
            hidden_size=1, delta=-0.5, module_prefix="",
            directions=("",),
        ).items()})
        shifted, _ = edited(x, h0)
        self.assertLess(float(shifted.item()), float(original.item()))

    def test_bad_architecture_or_nonfinite_delta_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            shift_gru_update_bias(self.state, hidden_size=3, delta=-0.25)
        with self.assertRaisesRegex(ValueError, "finite"):
            shift_gru_update_bias(self.state, hidden_size=4, delta=float("nan"))
        one_direction = {k: v for k, v in self.state.items() if not k.endswith("_reverse")}
        with self.assertRaisesRegex(ValueError, "missing"):
            shift_gru_update_bias(one_direction, hidden_size=4, delta=-0.25)
        bad_dtype = dict(self.state, **{"sequence.bias_ih_l0": np.zeros(12, dtype=np.int64)})
        with self.assertRaisesRegex(ValueError, "floating-point"):
            shift_gru_update_bias(bad_dtype, hidden_size=4, delta=-0.25)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            shift_gru_update_bias(self.state, hidden_size=True, delta=-0.25)


class DirectProjectionEditTests(unittest.TestCase):
    def test_conditional_model_edit_preserves_every_non_output_tensor_and_d_state(self):
        state = {
            "output.weight": np.arange(10, dtype=np.float32).reshape(2, 5),
            "output.bias": np.array([1., 2.], dtype=np.float32),
            "sequence.bias_ih_l0": np.arange(12, dtype=np.float32),
        }
        d_state = {"d_head": np.array([9., 8.], dtype=np.float32)}
        vector = np.array([1., 0., 0., 0., 0.], dtype=np.float32)
        changed = edit_conditional_ha_state_projection(state, vector, alpha=.25)
        self.assertFalse(np.array_equal(changed["output.weight"], state["output.weight"]))
        for key in ("output.bias", "sequence.bias_ih_l0"):
            np.testing.assert_array_equal(changed[key], state[key])
        np.testing.assert_array_equal(d_state["d_head"], [9., 8.])

    def test_full_state_edit_preserves_d_row_and_separate_d_state(self):
        state = {
            "output.weight": np.arange(15, dtype=np.float32).reshape(3, 5),
            "output.bias": np.array([1., 2., 3.], dtype=np.float32),
            "sequence.bias_ih_l0": np.arange(12, dtype=np.float32),
        }
        direction = np.array([1., 0., 0., 0., 0.], dtype=np.float32)
        d_state = {"d_head": np.array([9., 8.], dtype=np.float32)}
        edited = edit_had_state_ha_projection(state, direction, alpha=.5)
        np.testing.assert_array_equal(edited["output.weight"][2], state["output.weight"][2])
        np.testing.assert_array_equal(edited["output.bias"], state["output.bias"])
        np.testing.assert_array_equal(edited["sequence.bias_ih_l0"], state["sequence.bias_ih_l0"])
        np.testing.assert_array_equal(d_state["d_head"], [9., 8.])

    def test_delayed_transition_diagnostic_obeys_sequence_breaks_and_known_ha_only(self):
        hidden = np.arange(30, dtype=np.float32).reshape(15, 2)
        # Correct, one-frame-late, already-early, unknown/D interruption, and
        # sequence-break cases. Only the first two yield boundary examples.
        target = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, -1, 1, 0, 1, 1])
        predicted = np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        groups = np.array([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4, 5, 5])
        late, controls = delayed_transition_activations(hidden, target, predicted, groups)
        np.testing.assert_array_equal(late, hidden[[4]])
        np.testing.assert_array_equal(controls, hidden[[1]])

    def test_train_activation_contrast_is_normalized_and_rank_one_edit_preserves_common_mode(self):
        errors = np.array([[3, 0, 0], [5, 0, 0]], dtype=np.float32)
        controls = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
        direction = hidden_failure_direction(errors, controls)
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0)

        weight = np.array([[1, 2, 3], [-2, 1, 4]], dtype=np.float32)
        edited = subtract_output_direction(weight, direction, alpha=0.5)
        self.assertFalse(np.array_equal(edited, weight))
        projected = weight.astype(np.float64) @ direction.astype(np.float64)
        projected -= projected.mean()
        expected = weight - 0.5 * np.outer(projected, direction)
        np.testing.assert_allclose(edited, expected, atol=1e-6)
        # The change is rank one, and its H/A rows sum to zero.
        change = edited - weight
        self.assertLessEqual(np.linalg.matrix_rank(change, tol=1e-6), 1)
        np.testing.assert_allclose(change.sum(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose((edited @ direction).sum(), (weight @ direction).sum(), atol=1e-6)

    def test_edit_is_not_scalar_temperature_and_zero_is_identity(self):
        weight = np.array([[1, 0], [0, 2]], dtype=np.float64)
        direction = np.array([1, 0], dtype=np.float32)
        edited = subtract_output_direction(weight, direction, alpha=0.5)
        # A scalar temperature scales every direction equally; this edit only
        # removes half the H/A contrast on the selected hidden axis.
        np.testing.assert_allclose(edited[:, 1], weight[:, 1])
        self.assertFalse(np.allclose(edited, 0.5 * weight))
        np.testing.assert_array_equal(subtract_output_direction(weight, direction, alpha=0), weight)

    def test_failure_direction_requires_distinct_nonempty_finite_groups(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            hidden_failure_direction(np.empty((0, 2)), np.ones((1, 2)))
        with self.assertRaisesRegex(ValueError, "do not define"):
            hidden_failure_direction(np.ones((2, 2)), np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, "normalized"):
            subtract_output_direction(np.eye(2), np.array([2., 0.]), alpha=0.5)
        with self.assertRaisesRegex(ValueError, "two-logit"):
            subtract_output_direction(np.ones((3, 2)), np.array([1., 0.]), alpha=0.5)

    def test_direction_weights_matches_equally_instead_of_pooling_frames(self):
        errors = np.array([[2., 0.], [2., 0.], [0., 2.]])
        controls = np.array([[0., 0.], [0., 0.], [0., 0.]])
        direction = hidden_failure_direction(
            errors, controls,
            error_match_ids=np.array(["long", "long", "short"]),
            control_match_ids=np.array(["long", "long", "short"]))
        # Equal match means are [1,1], while pooled rows would be [4/3,2/3].
        np.testing.assert_allclose(direction, np.array([1., 1.]) / np.sqrt(2), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
