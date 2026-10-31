"""Focused contracts for the bounded SkillCorner-removal sampler."""
from __future__ import annotations

from collections import Counter
import unittest


class RemovalBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - the pinned runner has CPU torch
            raise unittest.SkipTest("CPU PyTorch is required for the sampler contract") from exc
        cls.torch = torch
        cls.data = torch.utils.data

    def _orders(self, n, budget, seed, *, batch_size=4):
        from experiments import removal_budget

        torch, data = self.torch, self.data
        dataset = data.TensorDataset(torch.arange(n))
        orders = []
        with removal_budget.matched_window_budget(budget) as audit:
            loader = data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                generator=torch.Generator().manual_seed(seed))
            for _ in range(audit.epochs):
                orders.append([int(row) for (batch,) in loader for row in batch.tolist()])
        return orders, audit

    def test_small_dataset_uses_full_permutations_and_shuffled_remainder(self):
        orders, audit = self._orders(5, 12, 2026)
        self.assertEqual(len(orders), 8)
        for order in orders:
            self.assertEqual(len(order), 12)
            self.assertEqual(set(order), set(range(5)))
            counts = Counter(order)
            self.assertEqual(min(counts.values()), 2)
            self.assertEqual(max(counts.values()), 3)
        self.assertEqual(audit.dataset_windows, 5)
        self.assertEqual(audit.dataset_unique_windows, 5)
        self.assertEqual(audit.windows_drawn_per_epoch, 12)
        self.assertEqual(audit.full_cycles, 2)
        self.assertEqual(audit.remainder, 2)
        self.assertEqual(audit.sampling, "random_sampler_permutations_with_remainder")

    def test_seed_repeats_entire_epoch_order_sequence(self):
        first, _ = self._orders(7, 19, 2027)
        second, _ = self._orders(7, 19, 2027)
        other_seed, _ = self._orders(7, 19, 2028)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_seed)

    def test_equal_budget_keeps_exact_loader_rng_and_epoch_parity(self):
        from experiments import removal_budget

        torch, data = self.torch, self.data
        dataset = data.TensorDataset(torch.arange(13))
        original_loader = data.DataLoader

        def run():
            generator = torch.Generator().manual_seed(91)
            loader = data.DataLoader(dataset, batch_size=4, shuffle=True,
                                     num_workers=0, generator=generator)
            orders = [[int(row) for (batch,) in loader for row in batch.tolist()]
                      for _ in range(8)]
            return orders, generator.get_state().clone()

        expected = run()
        with removal_budget.matched_window_budget(13) as audit:
            actual = run()
        self.assertIs(data.DataLoader, original_loader)
        self.assertEqual(expected[0], actual[0])
        self.assertTrue(torch.equal(expected[1], actual[1]))
        self.assertEqual(audit.sampling, "unchanged")
        self.assertEqual(audit.full_cycles, 1)
        self.assertEqual(audit.remainder, 0)
        self.assertEqual(audit.windows_drawn_per_epoch, 13)

    def test_update_budget_is_batches_per_epoch_times_frozen_epochs(self):
        orders, audit = self._orders(7, 129, 99, batch_size=64)
        updates = sum((len(order) + 63) // 64 for order in orders)
        self.assertEqual(audit.batches_per_epoch, 3)
        self.assertEqual(audit.expected_optimizer_updates, 3 * 8)
        self.assertEqual(updates, audit.expected_optimizer_updates)
        self.assertEqual(audit.windows_drawn_per_epoch, 129)
        self.assertEqual(audit.full_cycles, 18)
        self.assertEqual(audit.remainder, 3)

    def test_custom_sampler_rejected_and_patch_restored_after_errors(self):
        from experiments import removal_budget

        torch, data = self.torch, self.data
        dataset = data.TensorDataset(torch.arange(5))
        original_loader = data.DataLoader
        with self.assertRaisesRegex(ValueError, "sampler"):
            with removal_budget.matched_window_budget(12):
                data.DataLoader(
                    dataset, batch_size=4, shuffle=True,
                    sampler=data.RandomSampler(dataset), num_workers=0,
                    generator=torch.Generator().manual_seed(4))
        self.assertIs(data.DataLoader, original_loader)

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with removal_budget.matched_window_budget(12):
                raise RuntimeError("sentinel")
        self.assertIs(data.DataLoader, original_loader)

    def test_context_requires_exactly_one_loader(self):
        from experiments import removal_budget

        original_loader = self.data.DataLoader
        with self.assertRaisesRegex(RuntimeError, "without one DataLoader"):
            with removal_budget.matched_window_budget(12):
                pass
        self.assertIs(self.data.DataLoader, original_loader)

    def test_inference_record_drops_all_target_fields(self):
        import numpy as np
        from experiments.skillcorner_removal import inference_record

        record = {
            "x": np.zeros((3, 4), np.float32),
            "names": ["f0", "f1", "f2", "f3"],
            "period": np.array([0, 0, 1]),
            "indices": np.array([0, 2, 5]),
            "native_n": 8,
            "y": np.array([0, 1, 2]),
            "labels": np.array(["H", "A", "D"]),
            "full_y": np.arange(8),
            "native_labels": np.array(["H"] * 8),
        }
        projected = inference_record(record)
        self.assertEqual(set(projected), {"x", "names", "period", "indices", "native_n"})
        self.assertEqual(projected["native_n"], 8)
        self.assertNotIn("y", projected)
        self.assertNotIn("labels", projected)


if __name__ == "__main__":
    unittest.main()
