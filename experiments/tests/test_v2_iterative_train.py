import unittest

import numpy as np

from experiments import v2_iterative_train as study
from experiments.tests.test_sn_dense_train import old_record, sn_record, teacher as factorized_teacher


class WarmStartContracts(unittest.TestCase):
    def test_adopted_ha_and_d_states_are_copied_and_teacher_hash_is_preserved(self):
        teacher = dict(kind="teacher", feature_names=["a", "b"],
                       median=np.array([1.0, 2.0]), scale=np.array([3.0, 4.0]),
                       state={"w": np.array([[1.0, 2.0]])})
        checkpoint = dict(kind="soccernet_capacity", hidden=96,
                          feature_names=["a", "b"],
                          median=np.array([1.0, 2.0]), scale=np.array([3.0, 4.0]),
                          state={"ha": np.array([5.0, 6.0])},
                          d_state={"d": np.array([7.0, 8.0])})
        teacher_hash = study.cap._checkpoint_sha(teacher)

        initialized = study.warmstart_state(teacher, checkpoint)

        self.assertEqual(initialized["_original_teacher_sha256"], teacher_hash)
        np.testing.assert_array_equal(initialized["state"]["ha"], [5.0, 6.0])
        np.testing.assert_array_equal(initialized["d_state"]["d"], [7.0, 8.0])
        initialized["d_state"]["d"][0] = 99
        self.assertEqual(checkpoint["d_state"]["d"][0], 7.0)

    def test_rejects_normalizer_or_schema_drift(self):
        teacher = dict(kind="teacher", feature_names=["a"],
                       median=np.array([1.0]), scale=np.array([2.0]),
                       state={"w": np.array([[1.0]])})
        checkpoint = dict(kind="soccernet_capacity", hidden=96,
                          feature_names=["a"], median=np.array([1.0]),
                          scale=np.array([2.0]), state={"ha": np.array([3.0])})
        bad = dict(checkpoint, scale=np.array([3.0]))
        with self.assertRaisesRegex(ValueError, "scale"):
            study.warmstart_state(teacher, bad)


class BoundaryMaskContracts(unittest.TestCase):
    def test_only_known_h_a_transition_neighborhoods_are_selected(self):
        y = np.array([0, 0, 1, 1, 2, 0, -1, 1], dtype=np.int64)
        np.testing.assert_array_equal(
            study.boundary_rows(y),
            np.array([True, True, True, True, False, False, False, False]))

    def test_neighborhood_does_not_cross_time_period_break(self):
        y = np.array([0, 1, 0, 0], dtype=np.int64)
        periods = np.array([10, 10, 20, 20], dtype=np.int64)
        np.testing.assert_array_equal(
            study.boundary_rows(y, periods=periods),
            np.array([True, True, False, False]))

    def test_shared_valid_pair_mask_gates_boundary_edges(self):
        y = np.array([0, 1, 1], dtype=np.int64)
        valid_edges = np.array([False, True])
        self.assertFalse(study.boundary_rows(y, valid_edges=valid_edges).any())

    def test_h_a_edge_separated_by_unknown_or_d_is_not_a_transition(self):
        y = np.array([0, 2, 1, -1, 0], dtype=np.int64)
        self.assertFalse(study.boundary_rows(y).any())

    def test_boundary_loss_is_differentiable_and_zero_without_valid_edges(self):
        import torch

        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [0.0, 2.0]]],
                              requires_grad=True)
        labels = torch.tensor([[0, 1, 1]])
        loss = study.boundary_margin_loss(logits, labels)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

        no_edge = study.boundary_margin_loss(torch.zeros((1, 3, 2)),
                                             torch.tensor([[0, 0, 1]]),
                                             periods=torch.tensor([[1, 1, 2]]),
                                             valid_edges=torch.tensor([[True, True]]))
        self.assertEqual(float(no_edge), 0.0)


class TrainingIntegrationSmoke(unittest.TestCase):
    def test_control_matches_shared_dense_trainer_after_identical_warmstart(self):
        import torch
        from unittest.mock import patch
        from experiments import sn_dense_train as sn

        torch.set_num_threads(1)
        teacher = factorized_teacher(seed=17)
        ha = study.cap.network(teacher["input_features"], 96)
        prior = dict(kind="soccernet_capacity", hidden=96,
                     feature_names=list(teacher["feature_names"]),
                     median=teacher["median"].copy(), scale=teacher["scale"].copy(),
                     state={key: value.detach().cpu().numpy().copy()
                            for key, value in ha.state_dict().items()},
                     d_state={key: value.copy() for key, value in teacher["state"].items()})
        old = [old_record("old-a", 3), old_record("old-b", 4)]
        gar = [sn_record("game-a", 5), sn_record("game-b", 6)]
        actual = study._fit_loop(study.warmstart_state(teacher, prior), old, gar,
                                 seed=2026, arm="ce_control", steps=2, verbose=False)
        def load_warmstart(_teacher, target):
            target.load_state_dict({k: torch.as_tensor(v) for k, v in prior["state"].items()})
        with patch.object(study.cap, "_copy_teacher_function", load_warmstart):
            expected = sn.train(teacher, old, gar, width=96, lr=study.LR, seed=2026,
                                steps=2, fraction=.5, batch_size=32, ha_target="dense",
                                train_d=False, require_full_native=True, verbose=False)
        for key, value in expected["model"]["state"].items():
            np.testing.assert_allclose(actual["model"]["state"][key], value,
                                       rtol=0.0, atol=3e-7)

    def test_control_and_treatment_train_and_keep_adopted_d_state(self):
        import torch
        from experiments import sn_dense_train as sn

        torch.set_num_threads(1)
        teacher = factorized_teacher(seed=17)
        ha = study.cap.network(teacher["input_features"], 96)
        prior = dict(kind="soccernet_capacity", hidden=96,
                     feature_names=list(teacher["feature_names"]),
                     median=teacher["median"].copy(), scale=teacher["scale"].copy(),
                     state={key: value.detach().cpu().numpy().copy()
                            for key, value in ha.state_dict().items()},
                     d_state={key: value.copy() for key, value in teacher["state"].items()})
        old = [old_record("old-a", 3), old_record("old-b", 4)]
        gar = [sn_record("game-a", 5), sn_record("game-b", 6)]
        initial = study.warmstart_state(teacher, prior)

        for arm in study.ARMS:
            with self.subTest(arm=arm):
                result = study._fit_loop(initial, old, gar, seed=2026, arm=arm,
                                          steps=2, verbose=False)
                model = result["model"]
                self.assertEqual(model["source"]["checkpoint_sha256"],
                                 study.cap._checkpoint_sha(teacher))
                self.assertEqual(model["training"]["input_dropout"], study.cap.INPUT_DROPOUT)
                for key, expected in prior["d_state"].items():
                    np.testing.assert_array_equal(model["d_state"][key], expected)
                probabilities = sn.predict(sn_record("eval-smoke", 7), model, teacher)
                self.assertEqual(probabilities.shape, (160, 3))
                self.assertTrue(np.isfinite(probabilities).all())


if __name__ == "__main__":
    unittest.main()
