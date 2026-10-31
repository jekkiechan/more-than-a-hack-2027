"""Training harness for the dense-supervision SoccerNet-GAR study (``sn_dense_v1``).

Generalizes ``soccernet_capacity_model.train`` (a frozen module; not edited) with:

* ``ha_target``: ``"anchors"`` (sparse event-team labels; GAR windows get neutral
  spell weights and no transition term -- the historical behaviour) or
  ``"dense"`` (dense H/A/D silver labels, ``y`` in {0:H, 1:A, 2:D, -1:unknown}; GAR
  windows receive the same complete-spell weighting and live-transition surrogate
  as original windows, using ``full_y/full_period/indices`` and the *native* fps).
* ``train_d``: ``False`` copies the teacher P(D) (historical); ``True`` also
  fine-tunes a separate copy of the teacher's factorized network through its
  player-only dead-ball path (binary dead/live loss), starting from the teacher
  weights. Inference keeps the factorization P(H/A) = q * (1 - P(D)).
  With ``ha_target="anchors"`` GAR windows carry no D labels, so D is trained on
  original windows only (anchors would otherwise teach "never dead").
* ``fraction`` in [0, 1] (share of each batch drawn from SoccerNet). 0.0 trains
  on original data only; 1.0 uses no original windows and the original-data
  spell/transition terms drop out.
* ``soccernet_games``: deterministic subset of GAR games (see ``nested_games``).
* ``eval_every`` / ``validate`` / ``patience``: early stopping on a callback.
* ``ball_dropout``: probability that a GAR window uses a ball-gap view built from
  the existing ``tracking_corruption.corrupt`` gap process (feature-level
  approximation; pass ``ball_view_fn`` for an exact raw-tracking recompute).

With defaults (anchors, train_d False, no dropout, no eval) and identical
inputs the result reproduces ``soccernet_capacity_model.train`` bit-for-bit
(RNG draw order is identical; extras use separate generators).

Only code lives here; no SoccerNet data are read at import.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments import (factorized_had as fh, learned_temporal as lt,
                         objective_losses as ol, soccernet_capacity_model as cap,
                         supervised, tracking_corruption as tc)

CORE, CONTEXT = cap.CORE, cap.CONTEXT
ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_SOURCE = ROOT / "experiments/results/confidence_abstention_declaration.json"


# --------------------------------------------------------------------------- games

def rank_games(game_ids) -> list[str]:
    """Fixed SHA-256 ranking of game ids (ties impossible for distinct ids)."""
    ids = sorted({str(g) for g in game_ids})
    return sorted(ids, key=lambda g: (hashlib.sha256(g.encode("utf-8")).hexdigest(), g))


def nested_games(game_ids, n: int) -> list[str]:
    """First ``n`` games of the fixed ranking; subsets are nested across n."""
    ranked = rank_games(game_ids)
    if isinstance(n, bool) or int(n) != n or not 1 <= n <= len(ranked):
        raise ValueError(f"n must be an integer in [1, {len(ranked)}]")
    return ranked[:int(n)]


# ------------------------------------------------------------------ ball dropout view

_BALL_DEPENDENT = (
    r"^(ball_speed|ball_pitch_edge_distance|[HA]_distance[123]|home_distance_advantage|"
    r"home_relative_advantage|[HA]_distance_mean_|home_advantage_mean_|home_nearest_vote_|"
    r"speed_mean_|physical_ball_|boundary_outward_ball_speed|"
    r"[HA]_physical_(nearest[12]_|control_evidence|contact_impulse|within))")


def ball_dependent_columns(names) -> list[int]:
    import re
    pattern = re.compile(_BALL_DEPENDENT)
    return [i for i, n in enumerate(names) if pattern.match(str(n))]


def default_ball_calibration() -> dict:
    """Label-free organizer-tracking gap calibration already used by feed_like."""
    return json.loads(CALIBRATION_SOURCE.read_text())["calibration"]


def _sampled_rate(record: dict) -> float:
    native = float(np.asarray(record.get("native_fps", record["fps"])))
    if "indices" in record and len(record["indices"]) > 1:
        stride = float(np.median(np.diff(np.asarray(record["indices"]))))
        return native / max(stride, 1.0)
    return float(np.asarray(record["fps"]))


def ball_dropout_view(record: dict, calibration: dict, seed: int) -> np.ndarray:
    """Return a copy of ``record['x']`` with extra ball gaps.

    The gap process is the unchanged ``tracking_corruption.corrupt`` renewal
    process run on a placeholder track at the record's sampled rate. Rows that
    become unobserved get NaN in ball-dependent features, ball_available /
    ball_observed = 0 and recomputed missing_before_s / missing_after_s. Contact
    memory and player-only features are unchanged. This is an approximation of
    recomputing features from corrupted raw tracking.
    """
    names = list(record["names"])
    for needed in ("ball_observed", "ball_available", "missing_before_s", "missing_after_s"):
        if needed not in names:
            raise ValueError(f"Ball dropout needs feature {needed}")
    x = np.array(record["x"], copy=True)
    n = len(x)
    rate = _sampled_rate(record)
    _, period = np.unique(np.asarray(record["period"]), return_inverse=True)
    fake = dict(ball=np.full((n, 2), 50.0), players=np.full((n, 1, 2), 50.0),
                teams=np.zeros(1, dtype=int), period=period + 1, fps=rate,
                pitch=np.array([105.0, 68.0]))
    kept = np.asarray(tc.corrupt(fake, calibration, int(seed))["ball_observed"], dtype=bool)
    ob = names.index("ball_observed")
    was_observed = np.nan_to_num(x[:, ob], nan=0.0) > 0.5
    dropped = was_observed & ~kept
    cols = ball_dependent_columns(names)
    x[np.ix_(dropped, cols)] = np.nan
    for name in ("ball_observed", "ball_available"):
        x[dropped, names.index(name)] = 0.0
    now = was_observed & kept
    mb, ma = names.index("missing_before_s"), names.index("missing_after_s")
    for a, b in supervised.blocks(period):
        seg = now[a:b]
        pos = np.arange(b - a)
        last = np.maximum.accumulate(np.where(seg, pos, -1))
        nxt = np.minimum.accumulate(np.where(seg, pos, 10 ** 9)[::-1])[::-1]
        before = np.where(last >= 0, (pos - last) / rate, (pos + 1) / rate)
        after = np.where(nxt < 10 ** 9, (nxt - pos) / rate, (b - a - pos) / rate)
        rows = np.flatnonzero(dropped[a:b]) + a
        x[rows, mb] = before[rows - a]
        x[rows, ma] = after[rows - a]
    return x


# --------------------------------------------------------------- objective targets

def _native_fps(record: dict) -> float:
    return float(np.asarray(record.get("native_fps", record["fps"])))


def objective_targets(records: list[dict], *, require_full_native: bool):
    """Complete-spell factors and transition targets, one entry per record.

    Same helpers as the historical trainer; SoccerNet records carry sampled
    ``fps`` plus ``native_fps`` and the spell duration must use the latter.
    """
    fields = ("full_y", "full_period", "indices", "fps")
    presence = [all(k in r for k in fields) for r in records]
    if any(presence) and not all(presence):
        raise ValueError("Records must consistently include full-native objective fields")
    has_native = bool(presence and all(presence))
    if require_full_native and not has_native:
        raise ValueError("full_y/full_period/indices/fps are required on every record")
    spell, pairs = [], []
    for r in records:
        if has_native:
            spell.append(ol.spell_duration_factors(r["full_y"], r["full_period"],
                         r["indices"], r["y"], _native_fps(r)))
            pairs.append(ol.transition_targets(r["y"], r["period"], core_size=len(r["y"]),
                         full_y=r["full_y"], full_period=r["full_period"], indices=r["indices"]))
        else:
            spell.append(np.ones(len(r["y"]), dtype=np.float32))
            pairs.append(ol.transition_targets(r["y"], r["period"], core_size=len(r["y"])))
    weights = [p["live_mask"].astype(np.float32) for p in pairs]
    return spell, pairs, weights, has_native


# ------------------------------------------------------------------------ model I/O

def _dead_logit(net, x):
    dead_x = x * net.dead_input_mask if bool(net_player_only(net)) else x
    return net.output(net.sequence(dead_x)[0])[..., 2]


def net_player_only(net) -> bool:
    return bool(getattr(net, "_sn_player_only", True))


def _export(model, d_net, *, base: dict, teacher_sha: str, config: dict, step: int) -> dict:
    import torch  # noqa: F401
    out = dict(kind="soccernet_capacity", hidden=int(model.sequence.hidden_size),
        state={k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()},
        median=base["median"].copy(), scale=base["scale"].copy(),
        feature_names=list(base["feature_names"]), input_features=base["input_features"],
        source=dict(kind="factorized_had", checkpoint_sha256=teacher_sha),
        step=int(step), training=copy.deepcopy(config), sn_dense=True)
    if d_net is not None:
        out["d_state"] = {k: v.detach().cpu().numpy().copy() for k, v in d_net.state_dict().items()}
    return out


def predict(record: dict, model: dict, original: dict) -> np.ndarray:
    """H/A/D probabilities. P(D) comes from the fine-tuned D path when present,
    otherwise from the teacher; H/A is q * (1 - P(D)) as in the historical model."""
    feature = {k: record[k] for k in ("x", "names", "period")}
    teacher = dict(original, state=model["d_state"]) if "d_state" in model else original
    baseline = fh.predict(feature, teacher)
    return cap.predict(record, model, original, original_probability=baseline)


# ----------------------------------------------------------------------------- train

def train(initial_model: dict, old_records: list[dict], new_records: list[dict], *,
          width: int = 96, lr: float = 3e-4, seed: int = 2026, steps: int = 2048,
          fraction: float = 0.5, batch_size: int = 32,
          ha_target: str = "anchors", train_d: bool = False,
          d_lr: float = 1e-4, d_weight: float = 1.0,
          input_dropout: float = cap.INPUT_DROPOUT,
          soccernet_games: list[str] | None = None,
          ball_dropout: float = 0.0, ball_calibration: dict | None = None,
          ball_view_fn: Callable[[dict, int], np.ndarray] | None = None,
          eval_every: int = 0, validate: Callable[[dict], dict] | None = None,
          score_key: str = "score", patience: int = 0, eval_at_start: bool = False,
          require_full_native: bool = False, log_every: int = 128,
          verbose: bool = True) -> dict:
    """Train and return ``dict(model, final, history, best_step, config)``.

    ``model`` is the best validated snapshot (or the final one when no
    ``validate``); it is accepted by ``predict`` here and, when ``train_d`` is
    False, by ``soccernet_capacity_model.predict``.
    ``validate(snapshot) -> dict`` must return ``score_key`` (higher is better).
    ``patience`` counts evaluations without improvement (0 disables stopping).
    """
    if ha_target not in ("anchors", "dense"):
        raise ValueError("ha_target must be 'anchors' or 'dense'")
    dense = ha_target == "dense"
    if int(width) not in (24, 48, 96) or isinstance(width, bool):
        raise ValueError("width must be 24, 48, or 96")
    if int(steps) != steps or steps <= 0 or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive integers")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive")
    if not 0 <= float(fraction) <= 1:
        raise ValueError("fraction must be in [0, 1]")
    if not 0 <= float(ball_dropout) <= 1:
        raise ValueError("ball_dropout must be in [0, 1]")
    new_count = int(round(batch_size * float(fraction)))
    if not np.isclose(new_count / batch_size, fraction, atol=1e-12, rtol=0):
        raise ValueError("batch_size * fraction must be an integer")
    old_count = int(batch_size) - new_count
    if eval_every and validate is None:
        raise ValueError("eval_every requires validate")
    if new_count and not new_records:
        raise ValueError("fraction > 0 requires SoccerNet records")
    if old_count and not old_records:
        raise ValueError("fraction < 1 requires original records")

    if soccernet_games is not None:
        wanted = {str(g) for g in soccernet_games}
        have = {str(np.asarray(r["match"]).reshape(()).item()) for r in new_records}
        if not wanted <= have:
            raise ValueError(f"Unknown SoccerNet games: {sorted(wanted - have)[:5]}")
        new_records = [r for r in new_records
                       if str(np.asarray(r["match"]).reshape(()).item()) in wanted]

    names, median, scale, input_features = cap._model_input_width(initial_model)
    old_groups, old_ids, old_spell, old_pairs, old_weights, old_native = {}, [], [], [], [], False
    if old_count:
        old_groups, old_ids = cap._window_index(old_records, names, old=True)
        if not old_groups:
            raise ValueError("Original records contain no H/A-supervised windows")
        old_spell, old_pairs, old_weights, old_native = cap._old_objective_targets(
            old_records, require_full_native=require_full_native)
    new_groups, new_ids, new_spell, new_pairs, new_weights = {}, [], [], [], []
    if new_count:
        new_groups, new_ids = cap._window_index(new_records, names, old=False)
        if not new_groups:
            raise ValueError("SoccerNet records contain no H/A-supervised windows")
        if dense:
            new_spell, new_pairs, new_weights, _ = objective_targets(
                new_records, require_full_native=True)
    use_d_old = bool(train_d and old_count)
    use_d_new = bool(train_d and dense and new_count)
    if train_d and not (use_d_old or use_d_new):
        raise ValueError("train_d needs D labels: original windows or dense SoccerNet windows")

    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    teacher_sha = cap._checkpoint_sha(initial_model)
    torch.manual_seed(int(seed))
    model = cap.network(input_features, int(width), float(input_dropout))
    cap._copy_teacher_function(initial_model, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=cap.WEIGHT_DECAY)
    d_net = d_optimizer = None
    if train_d:
        d_net = fh.network(input_features, list(initial_model["player_indices"]),
                           bool(initial_model["player_only"]))
        d_net._sn_player_only = bool(initial_model["player_only"])
        d_net.load_state_dict({k: torch.as_tensor(v) for k, v in initial_model["state"].items()})
        d_optimizer = torch.optim.AdamW(d_net.parameters(), lr=float(d_lr),
                                        weight_decay=cap.WEIGHT_DECAY)
    rng = random.Random(int(seed))
    brng = np.random.default_rng(int(seed) + 7919)  # ball-dropout draws only
    if ball_dropout > 0 and ball_view_fn is None:
        ball_calibration = ball_calibration or default_ball_calibration()
    ball_cache: dict[int, np.ndarray] = {}

    def ball_x(index: int) -> np.ndarray:
        if index not in ball_cache:
            record = new_records[index]
            vseed = int(brng.integers(0, 2 ** 31 - 1))
            ball_cache[index] = (ball_view_fn(record, vseed) if ball_view_fn
                                 else ball_dropout_view(record, ball_calibration, vseed))
        return ball_cache[index]

    config = dict(width=int(width), lr=float(lr), seed=int(seed), steps=int(steps),
        fraction=float(fraction), batch_size=int(batch_size), input_dropout=float(input_dropout),
        weight_decay=cap.WEIGHT_DECAY, gradient_clip=cap.GRADIENT_CLIP,
        ha_target=ha_target, train_d=bool(train_d), d_lr=float(d_lr), d_weight=float(d_weight),
        ball_dropout=float(ball_dropout), eval_every=int(eval_every), patience=int(patience),
        soccernet_games=sorted({i[1] for i in new_ids}) if new_ids else [],
        old_windows=sum(map(len, old_groups.values())),
        new_windows=sum(map(len, new_groups.values())),
        old_windows_per_batch=old_count, new_windows_per_batch=new_count,
        spell_coefficient=cap.SPELL_COEFFICIENT,
        live_transition_coefficient=cap.LIVE_TRANSITION_COEFFICIENT,
        teacher_sha256=teacher_sha)
    base = dict(median=median, scale=scale, feature_names=names, input_features=input_features)

    def snapshot(step):
        return _export(model, d_net, base=base, teacher_sha=teacher_sha, config=config, step=step)

    history: list[dict] = []
    best = dict(score=-np.inf, step=None, model=None)
    stale = 0

    def evaluate(step) -> bool:
        nonlocal stale
        was_training = model.training
        model.eval()
        if d_net is not None:
            d_net.eval()
        snap = snapshot(step)
        result = dict(validate(snap))
        score = float(result[score_key])
        history.append(dict(step=step, eval=result))
        if score > best["score"]:
            best.update(score=score, step=step, model=snap)
            stale = 0
        else:
            stale += 1
        if was_training:
            model.train()
            if d_net is not None:
                d_net.train()
        if verbose:
            print(f"SN_DENSE eval step={step} {score_key}={score:.5f} best={best['score']:.5f}@{best['step']}",
                  flush=True)
        return bool(patience) and stale >= patience

    model.train()
    if d_net is not None:
        d_net.train()
    if eval_at_start and eval_every:
        evaluate(0)
    exposures: dict[str, int] = defaultdict(int)
    loss_log: list[dict] = []
    accum: dict[str, list[float]] = defaultdict(list)
    last_step = 0
    for step in range(1, int(steps) + 1):
        bx, by, bg, bs, bt, bm, bw = [], [], [], [], [], [], []
        for group, count in ((0, old_count), (1, new_count)):
            for _ in range(count):
                if group == 0:
                    ri, start, size, full, alt = cap._draw_old(rng, old_groups, old_records)
                    record, swapped = old_records[ri], False
                else:
                    ri, start, size, full, alt = cap._draw_new(rng, new_groups)
                    record = new_records[ri]
                    swapped = rng.random() < 0.5
                    if ball_dropout > 0 and brng.random() < ball_dropout:
                        record = dict(record, alternate=ball_x(ri)); alt = True
                x, y = cap._prepare_window(record, start, size, full, alt,
                    median=median, scale=scale, swap=swapped)
                if not np.isin(y, (0, 1)).any():
                    raise RuntimeError("Sampler emitted a window without an H/A anchor")
                spell = np.ones(CORE, dtype=np.float32)
                tt = np.zeros(CORE, dtype=np.float32)
                tm = np.zeros(CORE, dtype=bool)
                tw = np.zeros(CORE, dtype=np.float32)
                if group == 0:
                    spell[:size] = old_spell[ri][start:start + size]
                    pr = old_pairs[ri]
                    tt[:size] = pr["live_target"][start:start + size]
                    tm[:size] = pr["live_mask"][start:start + size]
                    tw[:size] = old_weights[ri][start:start + size]
                    exposures["old/" + "/".join(old_ids[ri])] += 1
                else:
                    if dense:
                        spell[:size] = new_spell[ri][start:start + size]
                        pr = new_pairs[ri]
                        tt[:size] = pr["live_target"][start:start + size]
                        tm[:size] = pr["live_mask"][start:start + size]
                        tw[:size] = new_weights[ri][start:start + size]
                    exposures["new/" + new_ids[ri][1]] += 1
                bx.append(x); by.append(y); bg.append(group); bs.append(spell)
                bt.append(tt); bm.append(tm); bw.append(tw)
        order = list(range(batch_size)); rng.shuffle(order)
        xt = torch.as_tensor(np.stack([bx[i] for i in order]), dtype=torch.float32)
        yt = torch.as_tensor(np.stack([by[i] for i in order]), dtype=torch.long)
        gt = torch.as_tensor([bg[i] for i in order], dtype=torch.int64)
        st = torch.as_tensor(np.stack([bs[i] for i in order]), dtype=torch.float32)
        ttar = torch.as_tensor(np.stack([bt[i] for i in order]), dtype=torch.float32)
        tmask = torch.as_tensor(np.stack([bm[i] for i in order]), dtype=torch.bool)
        twt = torch.as_tensor(np.stack([bw[i] for i in order]), dtype=torch.float32)

        all_logits = model(xt)
        logits = all_logits[:, CONTEXT:CONTEXT + CORE]
        labelled = (yt == 0) | (yt == 1)
        safe_y = torch.where(labelled, yt, torch.zeros_like(yt))
        row_loss = F.cross_entropy(logits.transpose(1, 2), safe_y, reduction="none")
        count = labelled.sum(dim=1)
        if torch.any(count == 0):
            raise RuntimeError("Every sampled window must contain a known H/A label")
        factors = 1.0 + cap.SPELL_COEFFICIENT * (st - 1.0)
        per_window = (row_loss * labelled * factors).sum(dim=1) / count
        old_mask, new_mask = gt == 0, gt == 1
        terms = []
        old_loss = per_window[old_mask].mean() if old_count else None
        new_loss = per_window[new_mask].mean() if new_count else None
        if old_loss is not None:
            terms.append((1.0 - fraction) * old_loss if new_count else old_loss)
        if new_loss is not None:
            terms.append(fraction * new_loss if old_count else new_loss)
        loss = terms[0] if len(terms) == 1 else terms[0] + terms[1]

        q_left = torch.softmax(all_logits[:, CONTEXT:CONTEXT + CORE], dim=-1)
        q_right = torch.softmax(all_logits[:, CONTEXT + 1:CONTEXT + CORE + 1], dim=-1)
        p_change = q_left[..., 0] * q_right[..., 1] + q_left[..., 1] * q_right[..., 0]
        edge = F.binary_cross_entropy(p_change.clamp(1e-6, 1 - 1e-6), ttar, reduction="none")
        t_old = t_new = None
        if old_count:
            t_old = ol.group_balanced_binary_loss(edge, ttar, tmask & old_mask[:, None], twt)
            if t_old is not None:
                loss = loss + (1.0 - fraction) * cap.LIVE_TRANSITION_COEFFICIENT * t_old
        if dense and new_count:
            t_new = ol.group_balanced_binary_loss(edge, ttar, tmask & new_mask[:, None], twt)
            if t_new is not None:
                loss = loss + fraction * cap.LIVE_TRANSITION_COEFFICIENT * t_new

        d_value = None
        if train_d:
            dead = _dead_logit(d_net, xt)[:, CONTEXT:CONTEXT + CORE]
            known = yt >= 0
            bce = F.binary_cross_entropy_with_logits(dead, (yt == 2).float(), reduction="none")
            per_d = (bce * known).sum(dim=1) / known.sum(dim=1).clamp(min=1)
            has = known.any(dim=1)
            d_terms = []
            if use_d_old and (old_mask & has).any():
                d_terms.append(((1.0 - fraction) if new_count else 1.0) * per_d[old_mask & has].mean())
            if use_d_new and (new_mask & has).any():
                d_terms.append((fraction if old_count else 1.0) * per_d[new_mask & has].mean())
            if d_terms:
                d_value = sum(d_terms)
                loss = loss + float(d_weight) * d_value

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        if d_optimizer is not None:
            d_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cap.GRADIENT_CLIP)
        optimizer.step()
        if d_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(d_net.parameters(), cap.GRADIENT_CLIP)
            d_optimizer.step()
        accum["loss"].append(float(loss.detach()))
        if old_loss is not None:
            accum["old_loss"].append(float(old_loss.detach()))
        if new_loss is not None:
            accum["new_loss"].append(float(new_loss.detach()))
        if d_value is not None:
            accum["d_loss"].append(float(d_value.detach()))
        last_step = step
        if verbose and log_every and step % log_every == 0:
            print(f"SN_DENSE width={width} step={step}/{steps} loss={accum['loss'][-1]:.5g}", flush=True)
        if step % 32 == 0 or step == steps:
            loss_log.append(dict(step=step, **{k: float(np.mean(v)) for k, v in accum.items()}))
            accum.clear()
        if eval_every and (step % eval_every == 0 or step == steps):
            if evaluate(step):
                break
    final = snapshot(last_step)
    final["metadata"] = dict(exposures=dict(sorted(exposures.items())), loss_intervals=loss_log)
    chosen = best["model"] if best["model"] is not None else final
    return dict(model=chosen, final=final, history=history, best_step=best["step"],
                best_score=None if best["model"] is None else best["score"],
                config=config, loss_intervals=loss_log, exposures=dict(exposures))


# ------------------------------------------------------------------------ validation

def make_gar_validator(valid_records: list[dict], teacher: dict, *,
                       recipe: str = "gar_valid", decode_cost: int = lt.COST,
                       score: str = "spell_f1") -> Callable[[dict], dict]:
    """Validation callback scoring a model on dense (silver) GAR-valid records.

    Each record: x, names, period, y, indices, fps (sampled), native_fps
    (optional), full_y, full_period, optional timestamps/full_timestamps.
    Sampled probabilities are interpolated to the native grid inside each
    block, Viterbi-decoded with the baseline switch cost, and scored with the
    official metric code (experiments.metrics via objective_ensemble). Records
    are concatenated with unique period ids so no span crosses a record.
    Returns spell_f1, macro_f1, d_f1, short_spell_recall, ... and ``score``.
    Never evaluate the sealed GAR test split with this.
    """
    from experiments import objective_ensemble as scoring

    labels_text = np.array(["H", "A", "D", "X"])

    def validate(model: dict) -> dict:
        truth, pred, period, stamps = [], [], [], []
        offset = 0
        native_fps = None
        for r in valid_records:
            prob = predict(r, model, teacher)
            full_period = np.asarray(r["full_period"])
            n_native = len(r["full_y"])
            info = dict(indices=np.asarray(r["indices"]), native_n=n_native)
            native = np.column_stack([lt.upsample(prob[:, k], info, full_period) for k in range(3)])
            fps = _native_fps(r)
            native_fps = fps if native_fps is None else native_fps
            if fps != native_fps:
                raise ValueError("Validation records must share one native fps")
            guess = supervised.decode(native, full_period, fps, decode_cost)
            truth.append(labels_text[np.where(np.asarray(r["full_y"]) < 0, 3, r["full_y"])])
            pred.append(np.asarray(guess).astype(str))
            period.append(full_period - full_period.min() + offset)
            offset = int(period[-1].max()) + 1
            stamps.append(np.asarray(r["full_timestamps"], dtype=float) if "full_timestamps" in r
                          else np.arange(n_native) / fps)
        truth, pred = np.concatenate(truth), np.concatenate(pred)
        inputs = dict(period=np.concatenate(period), fps=native_fps,
                      timestamps=np.concatenate(stamps))
        row = scoring.score_prediction(truth, pred, inputs, recipe, {}, "n/a")
        out = {k: row[k] for k in ("spell_f1", "macro_f1", "weighted_70_30", "d_f1", "d_precision",
               "d_recall", "short_spell_recall", "short_reference_spells", "n_gt_spells",
               "n_pred_spells", "graded_frames") if k in row}
        out["score"] = float(row[score])
        return out

    return validate
