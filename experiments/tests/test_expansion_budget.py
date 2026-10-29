"""Contracts for matched, no-replacement expansion window budgets."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from experiments import learned_temporal as lt
from experiments import objective_losses as ol
from experiments.expansion_budget import matched_window_budget


def synthetic_record(n=100, seed=31):
    rng = np.random.default_rng(seed)
    y = np.resize(np.array([0, 0, 1, 1, 2, 2, -1, 1], dtype=np.int8), n)
    period = np.zeros(n, dtype=np.int32)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    names = ["H_gap", "A_gap", "D_speed", "D_accel"]
    chunks = list(lt.chunks(dict(period=period)))
    return dict(source="Synthetic", match=f"synthetic-{n}", x=x, alternate=x + .1,
                y=y, period=period, names=names,
                replace_rows=np.zeros(n, dtype=bool),
                replace_windows=np.zeros(len(chunks), dtype=bool),
                full_y=np.repeat(y, 5), full_period=np.repeat(period, 5),
                indices=np.arange(n, dtype=np.int64) * 5, fps=25.0)


def dataloader(dataset, *, batch_size, generator, shuffle=True, num_workers=0):
    # The local import is the same resolution path used by objective_losses.fit.
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=generator)


def epoch_indices(loader, epochs):
    result = []
    for _ in range(epochs):
        result.append(torch.cat([batch[0] for batch in loader]).tolist())
    return result


class MatchedWindowBudgetContracts(unittest.TestCase):
    def test_equal_length_preserves_two_epoch_order_and_generator_state(self):
        dataset = TensorDataset(torch.arange(13))
        reference_generator = torch.Generator().manual_seed(20261029)
        budgeted_generator = torch.Generator().manual_seed(20261029)
        reference = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=True, num_workers=0,
            generator=reference_generator)
        expected = epoch_indices(reference, 2)
        expected_generator_state = reference_generator.get_state().clone()

        with patch.object(lt, "EPOCHS", 2):
            with matched_window_budget(len(dataset)) as audit:
                candidate = dataloader(dataset, batch_size=4,
                                       generator=budgeted_generator)
                actual = epoch_indices(candidate, 2)
                actual_generator_state = budgeted_generator.get_state().clone()

        self.assertEqual(expected, actual)
        self.assertTrue(torch.equal(expected_generator_state, actual_generator_state))
        self.assertEqual(audit.sampling, "unchanged")
        self.assertEqual(audit.dataset_windows, 13)
        self.assertEqual(audit.requested_windows_per_epoch, 13)
        self.assertEqual(audit.actual_windows_per_epoch, 13)
        self.assertEqual(audit.requested_batches_per_epoch, 4)
        self.assertEqual(audit.actual_batches_per_epoch, 4)
        self.assertEqual(audit.expected_optimizer_updates, 8)

    def test_expanded_dataset_draws_budget_without_replacement_each_epoch(self):
        dataset = TensorDataset(torch.arange(20))
        generator = torch.Generator().manual_seed(20261029)

        with patch.object(lt, "EPOCHS", 2):
            with matched_window_budget(7) as audit:
                loader = dataloader(dataset, batch_size=4, generator=generator)
                drawn = epoch_indices(loader, 2)

        self.assertEqual(audit.sampling, "uniform_without_replacement")
        self.assertEqual(audit.dataset_windows, 20)
        self.assertEqual(audit.requested_windows_per_epoch, 7)
        self.assertEqual(audit.actual_windows_per_epoch, 7)
        self.assertEqual(audit.requested_batches_per_epoch, 2)
        self.assertEqual(audit.actual_batches_per_epoch, 2)
        self.assertEqual(audit.expected_optimizer_updates, 4)
        for epoch in drawn:
            self.assertEqual(len(epoch), 7)
            self.assertEqual(len(set(epoch)), 7)
            self.assertTrue(set(epoch).issubset(set(range(20))))
        self.assertNotEqual(set(drawn[0]), set(drawn[1]))

    def test_objective_fit_parity_and_expanded_fit_update_budget(self):
        short = synthetic_record(100)
        expanded = synthetic_record(200, seed=32)
        with patch.object(lt, "EPOCHS", 2):
            reference = ol.fit([short], player_only=True, seed=2026,
                               objective="combined")
            with matched_window_budget(reference["windows_per_epoch"]) as parity_audit:
                parity = ol.fit([short], player_only=True, seed=2026,
                                objective="combined")

            self.assertEqual(parity_audit.dataset_windows,
                             reference["windows_per_epoch"])
            self.assertEqual(parity_audit.sampling, "unchanged")
            self.assertEqual(parity_audit.expected_optimizer_updates,
                             reference["optimizer_updates"])
            self.assertEqual(parity["history"], reference["history"])
            self.assertEqual(parity["nll_history"], reference["nll_history"])
            self.assertEqual(parity["optimizer_updates"], reference["optimizer_updates"])
            for key, value in reference["state"].items():
                np.testing.assert_array_equal(parity["state"][key], value)
            np.testing.assert_array_equal(
                ol.predict(short, parity), ol.predict(short, reference))

            with matched_window_budget(reference["windows_per_epoch"]) as expanded_audit:
                capped = ol.fit([expanded], player_only=True, seed=2026,
                                objective="combined")

        self.assertGreater(expanded_audit.dataset_windows,
                           expanded_audit.requested_windows_per_epoch)
        self.assertEqual(expanded_audit.actual_windows_per_epoch,
                         reference["windows_per_epoch"])
        self.assertEqual(expanded_audit.expected_optimizer_updates,
                         reference["optimizer_updates"])
        self.assertEqual(capped["optimizer_updates"], reference["optimizer_updates"])

    def test_invalid_budgets_and_undersized_dataset_are_rejected(self):
        original_loader = torch.utils.data.DataLoader
        for invalid in (0, -1, True, 1.5, None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    with matched_window_budget(invalid):
                        pass
                self.assertIs(torch.utils.data.DataLoader, original_loader)

        small_dataset = TensorDataset(torch.arange(3))
        with self.assertRaisesRegex(ValueError, "below matched budget"):
            with matched_window_budget(4):
                dataloader(small_dataset, batch_size=2,
                           generator=torch.Generator().manual_seed(1))
        self.assertIs(torch.utils.data.DataLoader, original_loader)

    def test_patch_restores_on_error_and_rejects_nested_context(self):
        dataset = TensorDataset(torch.arange(8))
        original_loader = torch.utils.data.DataLoader
        with self.assertRaisesRegex(RuntimeError, "intentional test failure"):
            with matched_window_budget(8):
                dataloader(dataset, batch_size=4,
                           generator=torch.Generator().manual_seed(2))
                with self.assertRaisesRegex(RuntimeError, "nested or concurrent"):
                    with matched_window_budget(8):
                        pass
                raise RuntimeError("intentional test failure")
        self.assertIs(torch.utils.data.DataLoader, original_loader)

    def test_success_without_fit_dataloader_fails_closed_and_restores(self):
        original_loader = torch.utils.data.DataLoader
        with self.assertRaisesRegex(RuntimeError, "without one objective DataLoader"):
            with matched_window_budget(8):
                pass
        self.assertIs(torch.utils.data.DataLoader, original_loader)


if __name__ == "__main__":
    unittest.main()
