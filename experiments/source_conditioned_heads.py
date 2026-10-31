"""Tensor-level prototype for source-conditioned dense-SoccerNet heads.

H/A and player-only D have separate recurrent encoders. Each encoder is shared
across target and SoccerNet supervision, which route to separate projections.
This module deliberately contains no data loading, full inference, or runner.
"""
from __future__ import annotations

import numpy as np


TARGET = "target"
SOCCERNET = "soccernet"
SOURCES = (TARGET, SOCCERNET)


def player_columns(names: list[str]) -> list[int]:
    """Return indices for both teams' player-only features in the joint schema."""
    base = [i for i, name in enumerate(names) if str(name).startswith("D_")]
    if not base or len(base) == len(names):
        raise ValueError("Expected separate team and player-only D feature blocks")
    return base + [i + len(names) for i in base]


def network(n_features: int, ha_hidden: int, player_indices: list[int], *,
            d_hidden: int = 24, separate_d: bool = True):
    """Separate H/A and player-only D BiGRUs with source-specific projections.

    Input layout follows factorized_had: full team features followed by the
    player-only feature block. ``player_indices`` selects the D encoder input.
    ``ha_hidden`` and ``d_hidden`` are separate to permit baseline-aligned H/D
    and H+ D configurations.
    """
    import torch
    from torch import nn

    if n_features <= 0 or ha_hidden <= 0 or d_hidden <= 0:
        raise ValueError("feature count and hidden widths must be positive")
    if not player_indices or min(player_indices) < 0 or max(player_indices) >= n_features:
        raise ValueError("player_indices must be nonempty valid input columns")

    class SourceConditionedNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.ha_sequence = nn.GRU(n_features, ha_hidden, batch_first=True,
                                      bidirectional=True)
            self.dead_sequence = nn.GRU(len(player_indices), d_hidden, batch_first=True,
                                        bidirectional=True)
            self.ha = nn.ModuleDict({s: nn.Linear(2 * ha_hidden, 2) for s in SOURCES})
            self.dead = nn.ModuleDict({s: nn.Linear(2 * d_hidden, 1)
                                       for s in (SOURCES if separate_d else (TARGET,))})
            self.register_buffer("player_indices", torch.as_tensor(player_indices, dtype=torch.long))
            # Preserve matched initialization: both source routes start identical.
            self.ha[SOCCERNET].load_state_dict(self.ha[TARGET].state_dict())
            if separate_d:
                self.dead[SOCCERNET].load_state_dict(self.dead[TARGET].state_dict())

        def forward(self, x, source: str):
            if source not in SOURCES:
                raise ValueError(f"Unknown training head: {source}")
            if x.ndim != 3 or x.shape[-1] != n_features:
                raise ValueError("x must have shape [batch, time, n_features]")
            features = self.ha_sequence(x)[0]
            player_x = x.index_select(-1, self.player_indices)
            dead_features = self.dead_sequence(player_x)[0]
            d_source = source if source in self.dead else TARGET
            return {"ha": self.ha[source](features),
                    "d": self.dead[d_source](dead_features).squeeze(-1)}

    return SourceConditionedNetwork()


def routed_loss(outputs: dict, target, *, source: str, mask=None):
    """Categorical H/A/D loss for one source; unknown targets are ignored.

    This utility makes supervision routing explicit and testable. ``target``
    uses -1 unknown, 0 H, 1 A, 2 D, matching prepared records.
    """
    import torch
    import torch.nn.functional as F

    if source not in SOURCES:
        raise ValueError(f"Unknown training head: {source}")
    valid = target >= 0
    if mask is not None:
        valid = valid & mask.bool()
    if not torch.any(valid):
        return outputs["ha"].sum() * 0 + outputs["d"].sum() * 0
    dead = outputs["d"]
    ha = outputs["ha"]
    log_live = F.logsigmoid(-dead)
    log_probs = torch.cat((log_live[..., None] + F.log_softmax(ha, -1),
                           F.logsigmoid(dead)[..., None]), dim=-1)
    return F.nll_loss(log_probs[valid], target[valid].long())


def target_probabilities(net, x):
    """Return H/A/D probabilities using only the target-domain projections."""
    import torch
    import torch.nn.functional as F

    outputs = net(x, TARGET)
    dead = torch.sigmoid(outputs["d"])
    live = (1 - dead)[..., None] * F.softmax(outputs["ha"], -1)
    return torch.cat((live, dead[..., None]), -1)


def export_state(net) -> dict[str, np.ndarray]:
    """Copy a state dict into pickle-free NumPy arrays for model bundles."""
    return {key: value.detach().cpu().numpy().copy() for key, value in net.state_dict().items()}
