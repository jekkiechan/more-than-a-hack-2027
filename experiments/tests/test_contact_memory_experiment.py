"""Zero-fit contracts for the frozen-backbone contact-memory adapter."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import contact_memory as cm
from experiments import contact_memory_experiment as experiment
from experiments import factorized_had as fh, learned_temporal as lt


def base_bundle():
    torch.manual_seed(11)
    names = ["H_gap", "A_gap", "D_player"]
    model = fh.network(6, [2, 5], player_only=True)
    fitted = {
        "kind": "factorized_had",
        "player_only": True,
        "player_indices": [2, 5],
        "input_features": 6,
        "feature_names": names,
        "median": np.zeros(3, dtype=np.float32),
        "scale": np.ones(3, dtype=np.float32),
        "state": {key: value.detach().numpy()
                  for key, value in model.state_dict().items()},
        "_checkpoint_sha256": "synthetic-frozen-checkpoint",
    }
    model.eval()
    return fitted, model


def example_record(n=100):
    rng = np.random.default_rng(34)
    return dict(
        x=rng.normal(size=(n, 3)).astype(np.float32),
        y=np.full(n, -1, dtype=np.int8),
        period=np.zeros(n, dtype=np.int32),
        names=["H_gap", "A_gap", "D_player"],
        indices=np.arange(n, dtype=np.int64),
        source="synthetic",
        match="synthetic",
    )


def no_memory(n):
    return {
        "tokens": np.zeros((n, 2, len(cm.EVENT_FIELDS)), dtype=np.float32),
        "ages": np.zeros((n, 2), dtype=np.float32),
        "valid": np.zeros((n, 2), dtype=bool),
    }


class ContactMemoryAdapterTests(unittest.TestCase):
    def test_zero_residual_reproduces_frozen_backbone_predictions(self):
        fitted, base_model = base_bundle()
        record = example_record()
        adapter = experiment.build_adapter()
        probability = experiment.predict(
            record, no_memory(len(record["x"])), fitted, base_model, adapter)
        control = fh.predict(record, fitted)
        np.testing.assert_array_equal(probability, control)

    def test_nonzero_memory_residual_changes_only_conditional_team_probabilities(self):
        fitted, base_model = base_bundle()
        record = example_record(n=8)
        adapter = experiment.build_adapter()
        with torch.no_grad():
            adapter.residual[-1].weight.fill_(0.1)
            adapter.residual[-1].bias.copy_(torch.tensor([0.1, -0.1]))
        x = torch.as_tensor(lt.normalize(record["x"], fitted["median"], fitted["scale"]))
        base_probability = base_model(x.unsqueeze(0)).exp()
        tokens = torch.zeros((1, len(record["x"]), 2, len(cm.EVENT_FIELDS)))
        tokens[:, :, 0, 0] = 0.8
        tokens[:, :, 0, 2] = 0.7
        ages = torch.ones((1, len(record["x"]), 2))
        valid = torch.zeros((1, len(record["x"]), 2), dtype=torch.bool)
        valid[:, :, 0] = True
        adapted = experiment.adapted_log_probabilities(
            base_model, adapter, x.unsqueeze(0), tokens, ages, valid).exp()
        torch.testing.assert_close(adapted[..., 2], base_probability[..., 2],
                                   rtol=0, atol=0)
        difference = (adapted[..., :2] - base_probability[..., :2]).abs().sum().detach()
        self.assertGreater(float(difference), 0)

    def test_adapter_has_exact_zero_effect_when_no_contact_event_is_available(self):
        _, base_model = base_bundle()
        adapter = experiment.build_adapter()
        with torch.no_grad():
            adapter.residual[-1].weight.normal_()
            adapter.residual[-1].bias.normal_()
        x = torch.randn(2, 13, 6)
        tokens = torch.randn(2, 13, 2, len(cm.EVENT_FIELDS))
        ages = torch.ones(2, 13, 2)
        valid = torch.zeros(2, 13, 2, dtype=torch.bool)
        adapted = experiment.adapted_log_probabilities(
            base_model, adapter, x, tokens, ages, valid)
        control = base_model(x)
        torch.testing.assert_close(adapted, control, rtol=0, atol=0)

    def test_training_routine_fits_adapter_and_never_updates_base_weights(self):
        fitted, base_model = base_bundle()
        before = {key: value.detach().clone()
                  for key, value in base_model.state_dict().items()}
        count = 1
        hidden = np.zeros((count, lt.CORE + 1, 48), dtype=np.float32)
        logits = np.zeros((count, lt.CORE + 1, 2), dtype=np.float32)
        tokens = np.zeros((count, lt.CORE + 1, 2, len(cm.EVENT_FIELDS)),
                          dtype=np.float32)
        tokens[:, :8, 0, 0] = .8
        tokens[:, :8, 0, 2] = .7
        ages = np.zeros((count, lt.CORE + 1, 2), dtype=np.float32)
        ages[:, :8, 0] = 1.0
        valid = np.zeros((count, lt.CORE + 1, 2), dtype=bool)
        valid[:, :8, 0] = True
        targets = np.zeros((count, lt.CORE), dtype=np.int64)
        frame_weights = np.zeros((count, lt.CORE), dtype=np.float32)
        frame_weights[:, :8] = 1.0
        batches = {
            "hidden": hidden,
            "base_logits": logits,
            "tokens": tokens,
            "ages": ages,
            "valid": valid,
            "targets": targets,
            "frame_weights": frame_weights,
            "spell_weights": np.ones((count, lt.CORE), dtype=np.float32),
            "live_targets": np.zeros((count, lt.CORE), dtype=np.float32),
            "live_masks": np.zeros((count, lt.CORE), dtype=bool),
            "live_weights": np.zeros((count, lt.CORE), dtype=np.float32),
            "windows": np.asarray(count, dtype=np.int64),
        }
        with patch.object(experiment, "_fit_samples", return_value=batches):
            result = experiment.fit_adapter([], fitted, base_model, seed=71)
        self.assertGreater(result["optimizer_updates"], 0)
        self.assertEqual(result["windows_per_epoch"], 1)
        self.assertGreater(float(np.linalg.norm(
            result["state"]["residual.2.bias"])), 0.0)
        for key, value in base_model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_archived_dead_probability_is_copied_exactly_while_live_odds_are_kept(self):
        probability = np.array([[.62, .28, .10], [.1, .2, .7]], dtype=np.float64)
        archived_dead = np.array([.1000000000001, .7], dtype=np.float64)
        result = experiment.retain_dead_probability(probability, archived_dead)
        np.testing.assert_array_equal(result[:, 2], archived_dead)
        np.testing.assert_allclose(result[:, 0] / result[:, :2].sum(axis=1),
                                   probability[:, 0] / probability[:, :2].sum(axis=1),
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=0, atol=1e-15)

    def test_25_fps_sampled_predictions_upsample_to_native_grid(self):
        fitted, base_model = base_bundle()
        record = example_record(n=100)
        record.update(indices=np.arange(0, 500, 5, dtype=np.int64),
                      native_n=496, fps=25.0)
        memory = no_memory(len(record["x"]))
        adapter = experiment.build_adapter()
        sampled = experiment.predict(record, memory, fitted, base_model, adapter)
        native = experiment.upsample_probabilities(
            sampled, record, np.zeros(record["native_n"], dtype=np.int32))
        self.assertEqual(native.shape, (496, 3))
        self.assertTrue(np.isfinite(native).all())
        np.testing.assert_allclose(native.sum(axis=1), 1.0, rtol=0, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
