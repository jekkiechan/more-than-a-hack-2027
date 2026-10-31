"""Tensor-level prototype for source-conditioned dense-SoccerNet heads.

H/A and player-only D have separate recurrent encoders. Each encoder is shared
across target and SoccerNet supervision, which route to separate projections.
This module deliberately contains no data loading, full inference, or runner.
"""
from __future__ import annotations

import copy
import numpy as np


TARGET = "target"
SOCCERNET = "soccernet"
SOURCES = (TARGET, SOCCERNET)
SOURCE_CONDITIONED_HA = {"C": False, "H": True, "HD": True}
SOURCE_CONDITIONED_D = {"C": False, "H": False, "HD": True}


def player_columns(names: list[str]) -> list[int]:
    """Return indices for both teams' player-only features in the joint schema."""
    base = [i for i, name in enumerate(names) if str(name).startswith("D_")]
    if not base or len(base) == len(names):
        raise ValueError("Expected separate team and player-only D feature blocks")
    return base + [i + len(names) for i in base]


def network(n_features: int, ha_hidden: int, player_indices: list[int], *,
            d_hidden: int = 24, input_dropout: float = 0.1,
            separate_ha: bool = True, separate_d: bool = True):
    """Separate H/A and player-only D BiGRUs with source-specific projections.

    Input layout follows factorized_had: full team features followed by the
    player-only feature block. D uses the original full-width GRU with
    non-player inputs masked to zero, matching the frozen teacher path.
    ``ha_hidden`` and ``d_hidden`` are separate to permit baseline-aligned H/D
    and H+ D configurations.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    if n_features <= 0 or ha_hidden <= 0 or d_hidden <= 0:
        raise ValueError("feature count and hidden widths must be positive")
    if not 0 <= float(input_dropout) < 1:
        raise ValueError("input_dropout must be in [0, 1)")
    if not player_indices or min(player_indices) < 0 or max(player_indices) >= n_features:
        raise ValueError("player_indices must be nonempty valid input columns")

    class SourceConditionedNetwork(nn.Module):
        class DeadProjection(nn.Module):
            def __init__(self, width):
                super().__init__()
                # Consume the same init draws as factorized_had's 3-logit
                # layer, retaining only the live/dead third row as D2 does.
                initialized = nn.Linear(width, 3)
                self.weight = nn.Parameter(initialized.weight[2:3].detach().clone())
                self.bias = nn.Parameter(initialized.bias[2:3].detach().clone())

            def forward(self, features):
                return F.linear(features, self.weight, self.bias)

        def __init__(self):
            super().__init__()
            self.ha_input_dropout = nn.Dropout(float(input_dropout))
            self.ha_sequence = nn.GRU(n_features, ha_hidden, batch_first=True,
                                      bidirectional=True)
            self.ha = nn.ModuleDict({TARGET: nn.Linear(2 * ha_hidden, 2)})
            self.dead_sequence = nn.GRU(n_features, d_hidden, batch_first=True,
                                        bidirectional=True)
            self.dead = nn.ModuleDict({TARGET: self.DeadProjection(2 * d_hidden)})
            mask = torch.zeros(n_features)
            mask[player_indices] = 1
            self.register_buffer("dead_input_mask", mask)
            # Deep copies add no random draws, preserving the D2 dropout stream.
            if separate_ha:
                self.ha[SOCCERNET] = copy.deepcopy(self.ha[TARGET])
            if separate_d:
                self.dead[SOCCERNET] = copy.deepcopy(self.dead[TARGET])

        def forward(self, x, source):
            if x.ndim != 3 or x.shape[-1] != n_features:
                raise ValueError("x must have shape [batch, time, n_features]")
            features = self.ha_sequence(self.ha_input_dropout(x))[0]
            dead_features = self.dead_sequence(x * self.dead_input_mask)[0]
            if isinstance(source, str):
                if source not in SOURCES:
                    raise ValueError(f"Unknown training head: {source}")
                ha = self.ha[source if source in self.ha else TARGET](features)
                dead = self.dead[source if source in self.dead else TARGET](dead_features)
            else:
                gar_rows = torch.as_tensor(source, dtype=torch.bool, device=x.device)
                if gar_rows.shape != (len(x),):
                    raise ValueError("source mask must have one boolean per batch row")
                if SOCCERNET in self.ha:
                    target_rows = torch.nonzero(~gar_rows, as_tuple=True)[0]
                    gar_row_ids = torch.nonzero(gar_rows, as_tuple=True)[0]
                    ha = features.new_zeros((len(x), features.shape[1], 2))
                    if len(target_rows):
                        ha = ha.index_copy(0, target_rows,
                                           self.ha[TARGET](features.index_select(0, target_rows)))
                    if len(gar_row_ids):
                        ha = ha.index_copy(0, gar_row_ids,
                                           self.ha[SOCCERNET](features.index_select(0, gar_row_ids)))
                else:
                    ha = self.ha[TARGET](features)
                if SOCCERNET in self.dead:
                    target_rows = torch.nonzero(~gar_rows, as_tuple=True)[0]
                    gar_row_ids = torch.nonzero(gar_rows, as_tuple=True)[0]
                    dead = dead_features.new_zeros((len(x), dead_features.shape[1], 1))
                    if len(target_rows):
                        dead = dead.index_copy(0, target_rows,
                            self.dead[TARGET](dead_features.index_select(0, target_rows)))
                    if len(gar_row_ids):
                        dead = dead.index_copy(0, gar_row_ids,
                            self.dead[SOCCERNET](dead_features.index_select(0, gar_row_ids)))
                else:
                    dead = self.dead[TARGET](dead_features)
            return {"ha": ha, "d": dead.squeeze(-1)}

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


def initialize_from_teacher(net, teacher: dict) -> None:
    """Copy the exact widened H/A and player-only D teacher paths into a model.

    The target projections receive the teacher's weights; enabled GAR
    projections are cloned from those initialized target projections.
    """
    import torch
    from experiments import factorized_had as fh
    from experiments import soccernet_capacity_model as cap
    from torch import nn

    n_features = int(teacher["input_features"])
    ha_hidden = int(net.ha_sequence.hidden_size)
    d_hidden = int(net.dead_sequence.hidden_size)
    if n_features != net.ha_sequence.input_size or d_hidden != 24:
        raise ValueError("Teacher feature schema or D width is incompatible")
    # Wrap the candidate's already initialized modules. The widening helper
    # preserves random extra-width units, so it must operate on this exact
    # instance rather than on a newly initialized GRU.
    with torch.random.fork_rng(devices=[]):
        class HAAdapter(nn.Module):
            def __init__(self, sequence, output):
                super().__init__()
                self.sequence = sequence
                self.output = output

        cap._copy_teacher_function(teacher,
            HAAdapter(net.ha_sequence, net.ha[TARGET]))

        dead = fh.network(n_features, list(teacher["player_indices"]), True)
        dead.load_state_dict({k: torch.as_tensor(v) for k, v in teacher["state"].items()})
        net.dead_sequence.load_state_dict(dead.sequence.state_dict())
        net.dead_input_mask.copy_(dead.dead_input_mask)
        net.dead[TARGET].weight.data.copy_(dead.output.weight.data[2:3])
        net.dead[TARGET].bias.data.copy_(dead.output.bias.data[2:3])
    if SOCCERNET in net.ha:
        net.ha[SOCCERNET].load_state_dict(net.ha[TARGET].state_dict())
    if SOCCERNET in net.dead:
        net.dead[SOCCERNET].load_state_dict(net.dead[TARGET].state_dict())
