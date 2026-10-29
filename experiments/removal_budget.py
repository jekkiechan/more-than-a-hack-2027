"""Matched per-epoch window draws for the SkillCorner-removal fits.

The active thirteen-match controls have fixed window and optimizer-update
budgets. A removal arm can have fewer unique windows than that budget, so this
scoped adapter uses PyTorch's seeded ``RandomSampler`` without replacement for
each full permutation and its final shuffled remainder. Existing-size fits
pass through the original ``DataLoader`` constructor unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from inspect import signature
from numbers import Integral
from threading import Lock
from typing import Any, Iterator


_PATCH_LOCK = Lock()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _batch_count(windows: int, batch_size: int, drop_last: bool) -> int:
    if drop_last:
        return windows // batch_size
    return (windows + batch_size - 1) // batch_size


@dataclass
class RemovalBudgetAudit:
    """Observed loader settings and the requested per-fit compute budget."""

    requested_windows_per_epoch: int
    epochs: int
    dataset_windows: int | None = None
    dataset_unique_windows: int | None = None
    windows_drawn_per_epoch: int | None = None
    full_cycles: int | None = None
    remainder: int | None = None
    batch_size: int | None = None
    batches_per_epoch: int | None = None
    expected_optimizer_updates: int | None = None
    sampling: str | None = None
    loader_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return the exact JSON-safe draw/update accounting."""
        return asdict(self)


@contextmanager
def matched_window_budget(windows_per_epoch: int) -> Iterator[RemovalBudgetAudit]:
    """Temporarily emit the control number of training windows each epoch.

    For ``N`` dataset windows and control budget ``B``, PyTorch's
    ``RandomSampler(replacement=False, num_samples=B)`` emits
    ``B // N`` independent full permutations and a fresh permutation prefix of
    length ``B % N``. Thus each unique window is drawn ``k`` or ``k+1`` times
    when the no-SkillCorner dataset is smaller than the matched budget.

    This patch is process-wide and limited to the exact shuffled, fixed-size,
    single-worker training loader used by ``objective_losses.fit``. Nested or
    concurrent contexts, custom samplers, invalid settings, and multiple loader
    constructions fail closed. Any error restores the original DataLoader.
    """
    budget = _positive_integer(windows_per_epoch, "windows_per_epoch")
    if not _PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("matched_window_budget contexts cannot be nested or concurrent")

    original_loader = None
    torch_data = None
    try:
        import torch
        import torch.utils.data as torch_data_module
        from experiments import learned_temporal as lt

        torch_data = torch_data_module
        original_loader = torch_data.DataLoader
        epochs = _positive_integer(lt.EPOCHS, "learned_temporal.EPOCHS")
        audit = RemovalBudgetAudit(requested_windows_per_epoch=budget, epochs=epochs)

        def budgeted_dataloader(dataset, *args, **kwargs):
            if audit.loader_calls:
                raise RuntimeError("matched_window_budget expects one DataLoader per fit")

            bound = signature(original_loader).bind(dataset, *args, **kwargs)
            bound.apply_defaults()
            parameters = bound.arguments
            dataset_windows = len(dataset)
            if (isinstance(dataset_windows, bool) or
                    not isinstance(dataset_windows, Integral) or dataset_windows <= 0):
                raise ValueError("the training dataset must contain at least one window")
            dataset_windows = int(dataset_windows)

            if parameters["shuffle"] is not True:
                raise ValueError("matched removal fits require the original shuffled DataLoader")
            if parameters["sampler"] is not None or parameters["batch_sampler"] is not None:
                raise ValueError("matched removal fits cannot replace a sampler or batch sampler")
            if parameters["drop_last"] is not False:
                raise ValueError("matched removal fits require drop_last=False")
            if parameters["num_workers"] != 0:
                raise ValueError("matched removal fits require the original num_workers=0")
            batch_size = _positive_integer(parameters["batch_size"], "DataLoader batch_size")
            generator = parameters["generator"]
            if not isinstance(generator, torch.Generator):
                raise ValueError("matched removal fits require a seeded torch.Generator")

            audit.loader_calls = 1
            audit.dataset_windows = dataset_windows
            audit.dataset_unique_windows = min(dataset_windows, budget)
            audit.windows_drawn_per_epoch = budget
            audit.full_cycles, audit.remainder = divmod(budget, dataset_windows)
            audit.batch_size = batch_size
            audit.batches_per_epoch = _batch_count(budget, batch_size, drop_last=False)
            audit.expected_optimizer_updates = audit.batches_per_epoch * epochs

            if dataset_windows == budget:
                # Preserve original constructor call and RNG use exactly.
                audit.sampling = "unchanged"
                return original_loader(dataset, *args, **kwargs)

            sampler = torch_data.RandomSampler(
                dataset, replacement=False, num_samples=budget, generator=generator)
            parameters["shuffle"] = False
            parameters["sampler"] = sampler
            audit.sampling = "random_sampler_permutations_with_remainder"
            loader = original_loader(**parameters)
            if len(loader) != audit.batches_per_epoch:
                raise RuntimeError("DataLoader batch count differs from the frozen budget")
            return loader

        torch_data.DataLoader = budgeted_dataloader
        yield audit
        if audit.loader_calls != 1:
            raise RuntimeError("matched_window_budget completed without one DataLoader")
    finally:
        if original_loader is not None:
            torch_data.DataLoader = original_loader
        _PATCH_LOCK.release()
