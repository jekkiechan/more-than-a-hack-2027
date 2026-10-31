"""Matched per-epoch window sampling for expanded objective-loss fits.

The patch is intentionally scoped to ``torch.utils.data.DataLoader`` because
``objective_losses.fit`` imports that symbol inside ``fit``. Existing-size fits
use the original DataLoader constructor unchanged; expanded datasets use a
uniform, seeded sample without replacement to preserve the frozen optimizer
budget.
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
class ExpansionBudgetAudit:
    """The planned window and optimizer budget for one fit invocation.

    ``expected_optimizer_updates`` is the nominal batch count times epochs.
    ``objective_losses.fit`` can skip an all-zero-weight batch, so callers must
    still compare the fit receipt's actual ``optimizer_updates`` against the
    matched baseline for each seed.
    """

    requested_windows_per_epoch: int
    epochs: int
    dataset_windows: int | None = None
    actual_windows_per_epoch: int | None = None
    batch_size: int | None = None
    requested_batches_per_epoch: int | None = None
    actual_batches_per_epoch: int | None = None
    expected_optimizer_updates: int | None = None
    sampling: str | None = None
    loader_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly audit record."""
        return asdict(self)


@contextmanager
def matched_window_budget(windows_per_epoch: int) -> Iterator[ExpansionBudgetAudit]:
    """Temporarily cap an objective-loss DataLoader to a matched window budget.

    The training dataset must contain at least ``windows_per_epoch`` windows.
    If its length is exactly the budget, its DataLoader constructor is called
    with the original arguments so the existing shuffle order and RNG usage are
    unchanged. If it is larger, each epoch draws a uniform, seeded sample
    without replacement while retaining the original batch size, worker count,
    generator, and other loader settings.

    Only one DataLoader may be constructed inside a context. Nested or
    concurrent contexts are rejected because this scoped patch is process-wide.
    """
    budget = _positive_integer(windows_per_epoch, "windows_per_epoch")
    if not _PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("matched_window_budget contexts cannot be nested or concurrent")

    original_loader = None
    torch_data = None
    try:
        import torch.utils.data as torch_data_module
        from experiments import learned_temporal as lt

        torch_data = torch_data_module
        original_loader = torch_data.DataLoader
        epochs = _positive_integer(lt.EPOCHS, "learned_temporal.EPOCHS")
        audit = ExpansionBudgetAudit(
            requested_windows_per_epoch=budget,
            epochs=epochs,
        )

        def budgeted_dataloader(dataset, *args, **kwargs):
            if audit.loader_calls:
                raise RuntimeError("matched_window_budget expects one DataLoader per fit")

            bound = signature(original_loader).bind(dataset, *args, **kwargs)
            bound.apply_defaults()
            parameters = bound.arguments
            dataset_windows = len(dataset)
            audit.loader_calls = 1
            audit.dataset_windows = dataset_windows

            if dataset_windows < budget:
                raise ValueError(
                    f"candidate has {dataset_windows} windows, below matched budget {budget}"
                )

            batch_size = parameters["batch_size"]
            if batch_size is None:
                raise ValueError("matched window budgets require a fixed batch_size")
            batch_size = _positive_integer(batch_size, "DataLoader batch_size")
            drop_last = bool(parameters["drop_last"])
            audit.batch_size = batch_size
            audit.requested_batches_per_epoch = _batch_count(
                budget, batch_size, drop_last
            )
            audit.actual_windows_per_epoch = budget

            if dataset_windows == budget:
                audit.sampling = "unchanged"
                loader = original_loader(dataset, *args, **kwargs)
            else:
                if parameters["shuffle"] is not True:
                    raise ValueError(
                        "expanded fits must use the original shuffled DataLoader"
                    )
                if parameters["sampler"] is not None or parameters["batch_sampler"] is not None:
                    raise ValueError(
                        "expanded fits cannot replace an existing sampler or batch sampler"
                    )

                generator = parameters["generator"]
                sampler = torch_data.RandomSampler(
                    dataset,
                    replacement=False,
                    num_samples=budget,
                    generator=generator,
                )
                parameters["shuffle"] = False
                parameters["sampler"] = sampler
                audit.sampling = "uniform_without_replacement"
                loader = original_loader(**parameters)

            audit.actual_batches_per_epoch = len(loader)
            audit.expected_optimizer_updates = (
                audit.actual_batches_per_epoch * audit.epochs
            )
            return loader

        torch_data.DataLoader = budgeted_dataloader
        yield audit
        if audit.loader_calls != 1:
            raise RuntimeError(
                "matched_window_budget completed without one objective DataLoader"
            )
    finally:
        if original_loader is not None:
            torch_data.DataLoader = original_loader
        _PATCH_LOCK.release()
