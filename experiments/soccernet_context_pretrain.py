"""Training-step utilities for masked context reconstruction on the H/A GRU.

The caller owns record validation, train-only normalization and checkpointing.
The auxiliary decoder is disposable: save only the unchanged capacity model
state so its inference API and separate frozen D model stay untouched.
"""
from __future__ import annotations


MASK_FRACTION = 0.15
MASK_SPAN = 3


def reconstruction_head(hidden: int, n_features: int):
    """Create the disposable auxiliary decoder for a capacity GRU."""
    import torch
    return torch.nn.Linear(2 * int(hidden), int(n_features))


def mask_context(normalized_x, valid_time_mask, *, seed: int,
                 mask_fraction: float = MASK_FRACTION, span: int = MASK_SPAN):
    """Mask exactly ``mask_fraction`` of valid rows in short temporal spans.

    Input layout is ``[normalized values, finite-value indicators]`` as emitted
    by ``learned_temporal.normalize``. Spans stop at every invalid time row.
    Returns corrupted input, original normalized value targets, and a loss mask
    restricted to naturally observed feature values at masked valid rows.
    """
    import torch
    if normalized_x.ndim != 3 or normalized_x.shape[-1] % 2:
        raise ValueError("Expected batch x time [values, observation masks]")
    if valid_time_mask.shape != normalized_x.shape[:2]:
        raise ValueError("Time mask shape differs from context batch")
    if not 0 < float(mask_fraction) <= 1 or isinstance(span, bool) or int(span) < 1:
        raise ValueError("Invalid temporal mask fraction/span")
    width = normalized_x.shape[-1] // 2
    generator = torch.Generator(device=normalized_x.device).manual_seed(int(seed))
    observed = (normalized_x[..., width:] > 0.5) & valid_time_mask.unsqueeze(-1)
    selected = torch.zeros_like(valid_time_mask, dtype=torch.bool)
    for bi in range(len(normalized_x)):
        valid_indices = torch.nonzero(valid_time_mask[bi], as_tuple=False).flatten()
        budget = min(len(valid_indices), max(1, int(round(len(valid_indices) * float(mask_fraction)))))
        remaining = int(budget)
        if not remaining:
            continue
        order = torch.randperm(len(valid_indices), generator=generator,
                               device=normalized_x.device)
        for choice in order:
            start = int(valid_indices[choice])
            if selected[bi, start]:
                continue
            stop = start
            while (stop < normalized_x.shape[1] and remaining > 0
                   and bool(valid_time_mask[bi, stop])
                   and not bool(selected[bi, stop])
                   and stop - start < int(span)):
                selected[bi, stop] = True
                remaining -= 1
                stop += 1
            if not remaining:
                break
        if remaining:
            # Random span starts can overlap; fill any remaining budget with
            # isolated valid rows while retaining the exact requested count.
            for choice in order:
                index = int(valid_indices[choice])
                if not bool(selected[bi, index]):
                    selected[bi, index] = True
                    remaining -= 1
                    if not remaining:
                        break
    masked_x = normalized_x.clone()
    masked_x[..., :width] = torch.where(selected.unsqueeze(-1),
                                         torch.zeros_like(masked_x[..., :width]),
                                         masked_x[..., :width])
    masked_x[..., width:] = torch.where(selected.unsqueeze(-1),
                                         torch.zeros_like(masked_x[..., width:]),
                                         masked_x[..., width:])
    loss_mask = observed & selected.unsqueeze(-1)
    return masked_x, normalized_x[..., :width].detach(), loss_mask


def masked_reconstruction_loss(net, head, masked_x, target, loss_mask):
    """Decode recurrent representations; average over observed masked values."""
    import torch
    if target.shape != loss_mask.shape or target.shape[:2] != masked_x.shape[:2]:
        raise ValueError("Reconstruction target/mask dimensions differ")
    width = masked_x.shape[-1] // 2
    if target.shape[-1] != width:
        raise ValueError("Reconstruction target width differs from feature schema")
    hidden = net.sequence(net.input_dropout(masked_x))[0]
    prediction = head(hidden)
    count = loss_mask.sum()
    if int(count.detach()) == 0:
        return prediction.sum() * 0.0
    return ((prediction - target).square() * loss_mask).sum() / count
