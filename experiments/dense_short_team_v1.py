"""Training-only short-possession H/A weighting for dense supervision.

This module defines the one fixed candidate recipe and its declaration. It
does not fit models or score evaluation data. The frame weights are built from
TRAIN labels only and are consumed by a study-specific trainer path.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MTAH_DATA_DIR", str(ROOT.parent / "mtah-data"))

from experiments import objective_losses as ol
from experiments import factorized_had as fh, sn_dense_train as sn
from experiments import soccernet_capacity as sc
from experiments import soccernet_capacity_model as cap

CORE, CONTEXT = cap.CORE, cap.CONTEXT

STUDY_ID = "dense_short_team_v1"
SEEDS = (2026, 2027)
FOLDS = ("leave_out_metrica", "leave_out_skillcorner", "leave_out_sportec")
SHORT_SPELL_SECONDS = 1.0
SHORT_SPELL_MULTIPLIER = 2.0
EXPECTED_CONTROL = {"clean": 0.6472145590743718, "feed_like": 0.5058107894247519}
CONTROL_TOLERANCE = 0.000051
OUT = ROOT / "data/experiments" / STUDY_ID
MODELS = OUT / "models"
FITS = OUT / "fits"
PREDICTIONS = OUT / "predictions"
CONTROL_OUT = ROOT / "data/experiments/dense_label_weight_v1"


def _native_fps(record: dict) -> float:
    """Return the actual label rate for old-provider or SoccerNet records."""
    value = record.get("native_fps", record.get("fps"))
    if value is None:
        raise ValueError("Record requires native_fps or fps")
    fps = float(np.asarray(value))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Native frame rate must be positive and finite")
    return fps


def complete_short_spell_mask(full_y, full_period, fps: float,
                              threshold_seconds: float = SHORT_SPELL_SECONDS) -> np.ndarray:
    """Mark native H/A frames in complete known spells of at most 1 second.

    Runs are bounded by the supplied time-block ids. A run touching a block
    edge or an unknown/X frame is censored and excluded. A known D, H, or A
    neighbor establishes a real observed endpoint. No run can cross a period,
    match/time break, or unknown target.
    """
    y = ol._labels(full_y, "full_y")
    period = np.asarray(full_period)
    if period.ndim != 1 or len(period) != len(y):
        raise ValueError("full_period must match full_y")
    fps = float(fps)
    threshold_seconds = float(threshold_seconds)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not np.isfinite(threshold_seconds) or threshold_seconds <= 0:
        raise ValueError("threshold_seconds must be positive and finite")

    selected = np.zeros(len(y), dtype=bool)
    if not len(y):
        return selected
    starts = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1]
    stops = np.r_[starts[1:], len(y)]
    max_frames = threshold_seconds * fps
    for block_start, block_stop in zip(starts, stops):
        at = int(block_start)
        while at < block_stop:
            label = int(y[at])
            if label not in (0, 1):
                at += 1
                continue
            stop = at + 1
            while stop < block_stop and y[stop] == label:
                stop += 1
            complete = (
                at > block_start and stop < block_stop
                and y[at - 1] >= 0 and y[stop] >= 0
            )
            if complete and (stop - at) <= max_frames:
                selected[at:stop] = True
            at = stop
    return selected


def short_spell_frame_weights(record: dict, *, multiplier: float = SHORT_SPELL_MULTIPLIER,
                              threshold_seconds: float = SHORT_SPELL_SECONDS) -> np.ndarray:
    """Return raw 2x/1x native-frame H/A weights mapped to sampled rows.

    H/A rows in selected spells receive ``multiplier``; all other rows have
    weight one. No corpus-level normalization is applied. Callers normalize
    each sampled-window loss by the combined weight mass on its known H/A
    rows, including the existing frozen complete-spell factors.
    """
    for field in ("full_y", "full_period", "indices", "y"):
        if field not in record:
            raise ValueError(f"Record is missing {field}")
    multiplier = float(multiplier)
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("multiplier must be finite and at least one")
    full_y = ol._labels(record["full_y"], "full_y")
    sampled_y = ol._labels(record["y"], "y")
    indices = np.asarray(record["indices"])
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array")
    if len(indices) != len(sampled_y) or (len(indices) and
            (indices.min() < 0 or indices.max() >= len(full_y))):
        raise ValueError("indices must align with sampled labels and full_y")
    if len(indices) > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError("indices must be strictly increasing")
    if not np.array_equal(sampled_y >= 0, full_y[indices] >= 0):
        raise ValueError("sampled and full-resolution known-label masks disagree")
    if not np.array_equal(sampled_y < 2, full_y[indices] < 2):
        raise ValueError("sampled and full-resolution H/A masks disagree")

    selected = complete_short_spell_mask(record["full_y"], record["full_period"],
                                         _native_fps(record), threshold_seconds)
    native = np.ones(len(full_y), dtype=np.float64)
    native[selected] = multiplier
    return native[indices].astype(np.float32)


def eligibility_summary(records: list[dict]) -> dict:
    """Count native/sampled short-spell mass and windows containing it.

    This is a label-only TRAIN diagnostic. It reports counts, not scores, and
    uses the same 10-second core windows as the shared temporal trainer.
    """
    from experiments import learned_temporal as lt

    result = {
        "records": len(records), "native_known_ha_frames": 0,
        "native_eligible_short_ha_frames": 0, "sampled_known_ha_frames": 0,
        "sampled_eligible_short_ha_frames": 0, "training_windows": 0,
        "windows_with_eligible_short_ha": 0,
    }
    for record in records:
        full_y = ol._labels(record["full_y"], "full_y")
        eligible = complete_short_spell_mask(record["full_y"], record["full_period"],
                                             _native_fps(record))
        sampled_y = ol._labels(record["y"], "y")
        indices = np.asarray(record["indices"], dtype=np.int64)
        sampled_eligible = eligible[indices] & ((sampled_y == 0) | (sampled_y == 1))
        result["native_known_ha_frames"] += int(np.isin(full_y, (0, 1)).sum())
        result["native_eligible_short_ha_frames"] += int(eligible.sum())
        result["sampled_known_ha_frames"] += int(np.isin(sampled_y, (0, 1)).sum())
        result["sampled_eligible_short_ha_frames"] += int(sampled_eligible.sum())
        for start, size, _ in lt.chunks(record):
            result["training_windows"] += 1
            if sampled_eligible[start:start + size].any():
                result["windows_with_eligible_short_ha"] += 1
    return result


def weighted_window_ha_loss(logits, targets, weights, base_factors=None):
    """Per-window conditional H/A cross-entropy with weighted-mass normalization.

    ``logits`` has shape (batch, time, 2); unknown and D targets are masked.
    ``weights`` are only the short-spell multiplier. Existing complete-spell
    factors multiply the numerator separately and keep their frozen baseline
    definition. The denominator is the short-spell weight mass, so weights of
    one reproduce ordinary D2 per-window CE exactly.
    """
    import torch
    from torch.nn import functional as F

    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("logits must have shape (batch, time, 2)")
    if targets.shape != logits.shape[:2] or weights.shape != targets.shape:
        raise ValueError("targets and weights must match logits batch/time dimensions")
    if base_factors is None:
        base_factors = torch.ones_like(weights)
    elif base_factors.shape != targets.shape:
        raise ValueError("base_factors must match targets")
    known = (targets == 0) | (targets == 1)
    safe_targets = torch.where(known, targets, torch.zeros_like(targets))
    row = F.cross_entropy(logits.transpose(1, 2), safe_targets, reduction="none")
    active_weights = weights * known.to(weights.dtype)
    denom = active_weights.sum(dim=1)
    if torch.any(denom <= 0):
        raise ValueError("Every sampled window must contain a known H/A target")
    return (row * active_weights * base_factors).sum(dim=1) / denom


def train_candidate(initial_model: dict, old_records: list[dict], new_records: list[dict], *,
                    seed: int, steps: int = 2048,
                    short_multiplier: float = SHORT_SPELL_MULTIPLIER,
                    verbose: bool = False) -> dict:
    """Fit one candidate with shared D2 sampling/model helpers.

    The loop is deliberately study-specific: only the H/A CE reduction differs
    from ``sn.train``. The D2 reference is reproduced when multiplier=1.
    """
    if seed not in SEEDS or int(steps) != steps or steps <= 0:
        raise ValueError("seed and steps must match the bounded study budget")
    if not np.isfinite(short_multiplier) or short_multiplier < 1:
        raise ValueError("short_multiplier must be finite and at least one")
    names, median, scale, input_features = cap._model_input_width(initial_model)
    old_groups, old_ids = cap._window_index(old_records, names, old=True)
    new_groups, new_ids = cap._window_index(new_records, names, old=False)
    if not old_groups or not new_groups:
        raise ValueError("Both old-provider and SoccerNet TRAIN windows are required")
    old_spell, old_pairs, old_transition_weights, _ = cap._old_objective_targets(
        old_records, require_full_native=True)
    new_spell, new_pairs, new_transition_weights, _ = sn.objective_targets(
        new_records, require_full_native=True)
    old_short = [short_spell_frame_weights(r, multiplier=short_multiplier) for r in old_records]
    new_short = [short_spell_frame_weights(r, multiplier=short_multiplier) for r in new_records]

    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    model = cap.network(input_features, 96, cap.INPUT_DROPOUT)
    cap._copy_teacher_function(initial_model, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=cap.WEIGHT_DECAY)
    d_net = fh.network(input_features, list(initial_model["player_indices"]),
                        bool(initial_model["player_only"]))
    d_net._sn_player_only = bool(initial_model["player_only"])
    d_net.load_state_dict({k: torch.as_tensor(v) for k, v in initial_model["state"].items()})
    d_optimizer = torch.optim.AdamW(d_net.parameters(), lr=1e-4,
                                    weight_decay=cap.WEIGHT_DECAY)
    rng = random.Random(int(seed))
    teacher_sha = cap._checkpoint_sha(initial_model)
    base = dict(median=median, scale=scale, feature_names=names,
                input_features=input_features)
    config = dict(width=96, lr=3e-4, seed=int(seed), steps=int(steps), fraction=0.5,
                  batch_size=32, input_dropout=cap.INPUT_DROPOUT,
                  train_d=True, d_lr=1e-4, d_weight=1.0,
                  spell_coefficient=cap.SPELL_COEFFICIENT,
                  live_transition_coefficient=cap.LIVE_TRANSITION_COEFFICIENT,
                  short_spell_seconds=SHORT_SPELL_SECONDS,
                  short_spell_multiplier=float(short_multiplier),
                  teacher_sha256=teacher_sha)
    exposures: dict[str, int] = defaultdict(int)
    history = []
    model.train(); d_net.train()
    for step in range(1, int(steps) + 1):
        bx, by, bg, bs, bt, bm, bw, shw = [], [], [], [], [], [], [], []
        for group, count in ((0, 16), (1, 16)):
            for _ in range(count):
                if group == 0:
                    ri, start, size, full, alt = cap._draw_old(rng, old_groups, old_records)
                    record = old_records[ri]
                    spell, pairs, tr_weights, short = old_spell, old_pairs, old_transition_weights, old_short
                    identity = old_ids[ri]
                else:
                    ri, start, size, full, alt = cap._draw_new(rng, new_groups)
                    record = new_records[ri]
                    spell, pairs, tr_weights, short = new_spell, new_pairs, new_transition_weights, new_short
                    identity = new_ids[ri]
                    # Match sn_dense_train's 50% team-swap augmentation draw.
                swapped = group == 1 and rng.random() < 0.5
                x, y = cap._prepare_window(record, start, size, full, alt,
                    median=median, scale=scale, swap=swapped)
                if not np.isin(y, (0, 1)).any():
                    raise RuntimeError("Sampler emitted a window without a known H/A target")
                s = np.ones(CORE, dtype=np.float32)
                short_w = np.ones(CORE, dtype=np.float32)
                tt = np.zeros(CORE, dtype=np.float32)
                tm = np.zeros(CORE, dtype=bool)
                tw = np.zeros(CORE, dtype=np.float32)
                s[:size] = spell[ri][start:start + size]
                short_w[:size] = short[ri][start:start + size]
                tt[:size] = pairs[ri]["live_target"][start:start + size]
                tm[:size] = pairs[ri]["live_mask"][start:start + size]
                tw[:size] = tr_weights[ri][start:start + size]
                bx.append(x); by.append(y); bg.append(group); bs.append(s)
                bt.append(tt); bm.append(tm); bw.append(tw); shw.append(short_w)
                prefix = "old/" if group == 0 else "new/"
                exposures[prefix + "/".join(identity)] += 1
        order = list(range(32)); rng.shuffle(order)
        xt = torch.as_tensor(np.stack([bx[i] for i in order]), dtype=torch.float32)
        yt = torch.as_tensor(np.stack([by[i] for i in order]), dtype=torch.long)
        gt = torch.as_tensor([bg[i] for i in order], dtype=torch.int64)
        st = torch.as_tensor(np.stack([bs[i] for i in order]), dtype=torch.float32)
        ttar = torch.as_tensor(np.stack([bt[i] for i in order]), dtype=torch.float32)
        tmask = torch.as_tensor(np.stack([bm[i] for i in order]), dtype=torch.bool)
        twt = torch.as_tensor(np.stack([bw[i] for i in order]), dtype=torch.float32)
        short_t = torch.as_tensor(np.stack([shw[i] for i in order]), dtype=torch.float32)

        all_logits = model(xt)
        logits = all_logits[:, CONTEXT:CONTEXT + CORE]
        base_factors = 1.0 + cap.SPELL_COEFFICIENT * (st - 1.0)
        per_window = weighted_window_ha_loss(logits, yt, short_t, base_factors)
        old_mask, new_mask = gt == 0, gt == 1
        loss = 0.5 * per_window[old_mask].mean() + 0.5 * per_window[new_mask].mean()

        q_left = torch.softmax(all_logits[:, CONTEXT:CONTEXT + CORE], dim=-1)
        q_right = torch.softmax(all_logits[:, CONTEXT + 1:CONTEXT + CORE + 1], dim=-1)
        p_change = q_left[..., 0] * q_right[..., 1] + q_left[..., 1] * q_right[..., 0]
        edge = F.binary_cross_entropy(p_change.clamp(1e-6, 1 - 1e-6), ttar, reduction="none")
        t_old = ol.group_balanced_binary_loss(edge, ttar, tmask & old_mask[:, None], twt)
        t_new = ol.group_balanced_binary_loss(edge, ttar, tmask & new_mask[:, None], twt)
        if t_old is not None: loss = loss + 0.5 * cap.LIVE_TRANSITION_COEFFICIENT * t_old
        if t_new is not None: loss = loss + 0.5 * cap.LIVE_TRANSITION_COEFFICIENT * t_new

        dead = sn._dead_logit(d_net, xt)[:, CONTEXT:CONTEXT + CORE]
        targets = yt
        known = targets >= 0
        bce = F.binary_cross_entropy_with_logits(dead, (targets == 2).float(), reduction="none")
        per_d = (bce * known).sum(dim=1) / known.sum(dim=1).clamp(min=1)
        has = known.any(dim=1)
        d_value = 0.5 * per_d[old_mask & has].mean() + 0.5 * per_d[new_mask & has].mean()
        loss = loss + d_value
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); d_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cap.GRADIENT_CLIP)
        torch.nn.utils.clip_grad_norm_(d_net.parameters(), cap.GRADIENT_CLIP)
        optimizer.step(); d_optimizer.step()
        if step % 32 == 0 or step == steps:
            history.append(dict(step=step, loss=float(loss.detach())))
        if verbose and step % 128 == 0:
            print(f"dense_short_team_v1 seed={seed} step={step}/{steps} loss={float(loss.detach()):.5g}", flush=True)
    final = sn._export(model, d_net, base=base, teacher_sha=teacher_sha,
                       config=config, step=int(steps))
    final["metadata"] = dict(exposures=dict(sorted(exposures.items())), loss_intervals=history)
    return dict(model=final, config=config, history=history, exposures=dict(exposures))


def declaration() -> dict:
    """Return the fixed pre-fit recipe; artifact hashes are added after restore."""
    return {
        "schema": 1,
        "study": STUDY_ID,
        "status": "proposal_unfrozen",
        "question": "Does emphasizing complete dense H/A possessions lasting at most one second improve full-provider transfer team attribution?",
        "arms": {
            "control_D2": "The reconstructed adopted dense D2 training recipe, reused from dense_label_weight_v1; no control refits.",
            "short_team_2x": "Same D2 training recipe and sampled windows; H/A CE frames in complete known spells <=1.0s receive multiplier 2.0; all other H/A frames receive 1.0; normalize weighted CE by known H/A weight mass in each sampled window.",
        },
        "common": {
            "folds": list(FOLDS), "seeds": list(SEEDS), "candidate_fits": 6,
            "updates_per_fit": 2048, "batch_size": 32, "soccernet_fraction": 0.5,
            "width": 96, "learning_rate": 0.0003, "d_learning_rate": 0.0001,
            "input_dropout": 0.1, "initialization": "matching full-provider fold teacher",
            "train_sources": "the two non-held-out old providers plus 45 SoccerNet TRAIN games; apply the same short-spell weighting rule to both sources using each record's native FPS",
            "evaluation": "reuse dense_label_weight_v1 reconstructed controls and paired 5-match x clean/corrupt_1/corrupt_2 cohort; no control refits",
            "scoring": "official spell-F1; equal-provider means; paired by source/match/view; individual members and native-probability average ensemble",
        },
        "candidate_weight": {
            "threshold_seconds": SHORT_SPELL_SECONDS,
            "multiplier": SHORT_SPELL_MULTIPLIER,
            "complete": "known H/A run has a known H/A/D neighbor on both sides within the same full_period block; block-edge and X/unknown-censored runs are excluded",
            "application": "training-only conditional H/A cross-entropy; same team-symmetric rule for H and A; D loss, transition loss, inputs, sampler, optimizer, model and inference unchanged",
            "normalization": "sum(short_weight * existing_D2_complete_spell_factor * H/A CE) / sum(short_weight over known H/A rows) independently for each sampled window; this reproduces the D2 numerator/denominator exactly when short_weight=1",
            "inference_targets": "none; labels and eligibility masks are never available at evaluation or inference",
        },
        "diagnostics_and_guards": {
            "primary": "official spell_f1_mean: global one-to-one possession spell-F1 averaged over IoU thresholds 0.3, 0.4, 0.5, 0.6 and 0.7",
            "diagnostic": "wrong-team frame-error rate inside reference possessions <=1s, with the audit's fixed grading population",
            "control_replay": "reused reconstructed D2 clean/feed-like aggregates must match dense_label_weight_v1 within 0.000051 before candidate scoring",
            "expected_control": EXPECTED_CONTROL,
            "provider_exclusion": "held-out provider is excluded from every learned component and normalization in its fold",
            "existing_guards_unchanged": True,
            "no_public_or_gar_valid_test_labels": True,
            "no_promotion": True,
        },
        "artifacts": {
            "candidate_models": "data/experiments/dense_short_team_v1/models/short_team_2x__{provider}__s{seed}.joblib",
            "fit_receipts": "data/experiments/dense_short_team_v1/fits/short_team_2x__{provider}__s{seed}.json",
            "candidate_predictions": "data/experiments/dense_short_team_v1/predictions/short_team_2x/{fold}/",
            "reused_controls": "data/experiments/dense_label_weight_v1/{models,predictions}/control_D2/",
        },
    }


def fit_id(fold: str, seed: int) -> str:
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Unknown fold or seed")
    return f"short_team_2x__{fold.removeprefix('leave_out_')}__s{seed}"


def candidate_model_path(fold: str, seed: int) -> Path:
    return MODELS / f"{fit_id(fold, seed)}.joblib"


def candidate_fit_receipt_path(fold: str, seed: int) -> Path:
    return FITS / f"{fit_id(fold, seed)}.json"


def control_model_path(fold: str, seed: int) -> Path:
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Unknown fold or seed")
    return CONTROL_OUT / "models" / f"control_D2__{fold.removeprefix('leave_out_')}__s{seed}.joblib"


def candidate_prediction_dir(fold: str) -> Path:
    if fold not in FOLDS:
        raise ValueError("Unknown fold")
    return PREDICTIONS / "short_team_2x" / fold


def load_training(fold: str, seed: int) -> tuple[dict, list[dict], list[dict]]:
    """Load the provider-excluded teacher, old TRAIN records and dense GAR TRAIN."""
    import joblib
    from experiments import dense_label_weight_v1 as prior
    from experiments import soccernet_dense_labels as dl
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Unknown fold or seed")
    prior.verify_preparation_reconstruction()
    teacher_path = sc.initial_path(fold, seed)
    teacher = joblib.load(teacher_path)
    old_records = sc.old.load_old(fold)
    new_records = dl.load_dense_records("train", output=prior.RECORDS)
    if len({str(np.asarray(r["match"]).reshape(()).item()) for r in new_records}) != 45:
        raise ValueError("Expected the restored 45-game SoccerNet TRAIN cohort")
    for record in (*old_records, *new_records):
        required = ("full_y", "full_period", "indices", "y", "period")
        if not all(field in record for field in required):
            raise ValueError("Every training record must carry full-native labels and time breaks")
    return teacher, old_records, new_records


def _training_input_hashes(fold: str, seed: int) -> dict:
    """Bind the exact training records and provider-excluded teacher inputs."""
    from experiments import dense_label_weight_v1 as prior
    old = sc.old.old_spec()
    plan = sc.prep.split_plan(fold=fold)
    paths: set[Path] = {
        prior.RECORDS / "train/manifest.json",
        sc.old.DATA / "experiments/objective_alignment_v1/predeclared.json",
    }
    for item in plan["train"]:
        entry = old["prepared_training_caches"][item["match"]]
        for field in ("cache", "receipt"):
            rel = Path(entry[field]).relative_to("data")
            paths.add(sc.old.DATA / rel)
    manifest = json.loads((prior.RECORDS / "train/manifest.json").read_text())
    paths.update(prior.RECORDS / "train/games" / name
                 for name in manifest["game_files"])
    return {str(path): sha256(path) for path in sorted(paths)}


def eligibility_by_fold(fold: str, seed: int) -> dict:
    """Return eligible label/window mass for the exact fold training sources."""
    _, old_records, new_records = load_training(fold, seed)
    return {"old_provider": eligibility_summary(old_records),
            "soccernet": eligibility_summary(new_records)}


def preflight() -> dict:
    """Verify restored inputs, matched control gate and candidate training mass."""
    from experiments import dense_label_weight_v1 as prior

    prior.verify_declared_inputs()
    replay = json.loads((ROOT / "experiments/results/dense_label_weight_v1_control_replay.json").read_text())
    if replay.get("passed") is not True:
        raise RuntimeError("Matched D2 control replay gate is missing or failed")
    control_models = {}
    for fold in FOLDS:
        for seed in SEEDS:
            model_path = control_model_path(fold, seed)
            fit_path = prior.FITS / f"control_D2__{fold.removeprefix('leave_out_')}__s{seed}.json"
            if not model_path.is_file() or not fit_path.is_file():
                raise FileNotFoundError(f"Restored D2 control model/fit receipt missing: {model_path}")
            control_models[str(model_path.relative_to(ROOT))] = sha256(model_path)
            control_models[str(fit_path.relative_to(ROOT))] = sha256(fit_path)

    training_inputs = {fold: _training_input_hashes(fold, 2026) for fold in FOLDS}
    teacher_inputs = {
        f"{fold}/{seed}": {
            str(sc.initial_path(fold, seed)): sha256(sc.initial_path(fold, seed)),
            str(sc.initial_path(fold, seed).with_suffix(".json")):
                sha256(sc.initial_path(fold, seed).with_suffix(".json")),
        }
        for fold in FOLDS for seed in SEEDS
    }
    eligibility = {fold: eligibility_by_fold(fold, 2026) for fold in FOLDS}
    evaluation_inputs = {}
    for item in sc.old.selection_items():
        for view, replicate in sc.old.VIEWS:
            path = sc.old.eval_path(item, view, replicate)
            evaluation_inputs[str(path)] = sha256(path)
    control_predictions = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted((CONTROL_OUT / "predictions" / "control_D2").rglob("*.npz"))
    }
    if not control_predictions:
        raise FileNotFoundError("Exact restored D2 control prediction arrays are required")

    source_paths = [
        ROOT / "experiments/dense_short_team_v1.py",
        ROOT / "experiments/tests/test_dense_short_team_v1.py",
        ROOT / "experiments/sn_dense_train.py",
        ROOT / "experiments/soccernet_capacity.py",
        ROOT / "experiments/soccernet_capacity_model.py",
        ROOT / "experiments/soccernet_extension.py",
        ROOT / "experiments/soccernet_dense_labels.py",
        ROOT / "experiments/objective_losses.py",
        ROOT / "experiments/learned_temporal.py",
        ROOT / "experiments/factorized_had.py",
        ROOT / "experiments/supervised.py",
        ROOT / "experiments/dense_label_weight_v1.py",
        ROOT / "evaluation/possession_metrics.py",
    ]
    for path in (ROOT / "experiments/tests/test_dense_short_team_eval_v1.py",
                 ROOT / "experiments/dense_short_team_eval_v1.py"):
        if path.is_file():
            source_paths.append(path)
    evaluator = ROOT / "experiments/dense_short_team_eval_v1.py"
    if evaluator.is_file():
        source_paths.append(evaluator)
    source_sha256 = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    return {
        "restore_verification": "dense_label_weight_v1.verify_declared_inputs passed",
        "control_replay": replay,
        "control_models_and_fit_receipts_sha256": control_models,
        "training_inputs_sha256_by_fold": training_inputs,
        "teacher_sha256_by_fit": teacher_inputs,
        "training_eligibility_by_fold": eligibility,
        "evaluation_inputs_sha256": evaluation_inputs,
        "control_prediction_archives_sha256": control_predictions,
        "source_sha256": source_sha256,
        "active_baseline_and_policy_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (ROOT / "experiments/baseline_manifest.json",
                         ROOT / "experiments/model_decision_policy.json")
        },
    }


def write_declaration(path: Path | None = None, *, preflight_receipt: dict | None = None) -> Path:
    """Freeze the recipe with verified source/input hashes; write once only."""
    path = Path(path or ROOT / "experiments/results/dense_short_team_v1_declaration.json")
    checked = preflight_receipt if preflight_receipt is not None else preflight()
    value = declaration()
    value["status"] = "frozen_pre_fit"
    value["restore_and_mass_audit"] = checked
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Refusing to overwrite frozen declaration: {path}")
    else:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return path


def verify_declaration() -> dict:
    path = ROOT / "experiments/results/dense_short_team_v1_declaration.json"
    if not path.is_file():
        raise RuntimeError("Freeze and review the declaration before fitting")
    value = json.loads(path.read_text())
    if value.get("study") != STUDY_ID or value.get("status") != "frozen_pre_fit":
        raise RuntimeError("Declaration is not the frozen dense_short_team_v1 recipe")
    for rel, expected in value["restore_and_mass_audit"]["source_sha256"].items():
        actual = sha256(ROOT / rel)
        if actual != expected:
            raise RuntimeError(f"Declared source changed after freeze: {rel}")
    replay = json.loads((ROOT / "experiments/results/dense_label_weight_v1_control_replay.json").read_text())
    if replay != value["restore_and_mass_audit"]["control_replay"]:
        raise RuntimeError("Matched D2 control replay receipt changed after freeze")
    return value


def run_fit(fold: str, seed: int, *, verbose: bool = True) -> dict:
    """Fit, seal and return one of the six predeclared candidate models."""
    import joblib
    from experiments import dense_label_weight_v1 as prior

    declared = verify_declaration()
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Fit identity is outside the six declared fold/seed pairs")
    prior.verify_declared_inputs()
    path = candidate_model_path(fold, seed)
    receipt_path = candidate_fit_receipt_path(fold, seed)
    if path.exists() or receipt_path.exists():
        raise FileExistsError("Candidate fit already exists; refusing duplicate training")
    input_hashes = _training_input_hashes(fold, seed)
    declared_inputs = declared["restore_and_mass_audit"]["training_inputs_sha256_by_fold"][fold]
    if input_hashes != declared_inputs:
        raise RuntimeError("Training inputs changed after declaration")
    teacher_hashes = declared["restore_and_mass_audit"]["teacher_sha256_by_fit"][f"{fold}/{seed}"]
    for teacher_path, expected_hash in teacher_hashes.items():
        if sha256(Path(teacher_path)) != expected_hash:
            raise RuntimeError(f"Provider-excluded teacher checkpoint/receipt changed after declaration: {teacher_path}")
    teacher, old_records, new_records = load_training(fold, seed)
    result = train_candidate(teacher, old_records, new_records, seed=seed, steps=2048,
                             short_multiplier=SHORT_SPELL_MULTIPLIER, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(result["model"], temporary)
    model_hash = sha256(temporary)
    os.replace(temporary, path)
    receipt = {
        "study": STUDY_ID, "arm": "short_team_2x", "fold": fold,
        "heldout": fold.removeprefix("leave_out_"), "seed": seed,
        "updates": 2048, "model_path": str(path.relative_to(ROOT)),
        "model_sha256": model_hash,
        "teacher_path": str(sc.initial_path(fold, seed)),
        "teacher_sha256": sha256(sc.initial_path(fold, seed)),
        "training_config": result["config"],
        "eligibility": declared["restore_and_mass_audit"]["training_eligibility_by_fold"][fold],
        "train_input_sha256": input_hashes,
        "declaration_sha256": sha256(ROOT / "experiments/results/dense_short_team_v1_declaration.json"),
        "source_sha256": declared["restore_and_mass_audit"]["source_sha256"],
        "exposures": result["exposures"], "loss_intervals": result["history"],
    }
    write_once(receipt_path, receipt)
    return receipt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Refusing to overwrite sealed artifact: {path}")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="verify restored inputs and report training mass")
    sub.add_parser("freeze", help="write the immutable pre-fit declaration")
    fit = sub.add_parser("fit", help="fit one predeclared provider-excluded member")
    fit.add_argument("--fold", choices=FOLDS, required=True)
    fit.add_argument("--seed", choices=SEEDS, type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(preflight(), indent=2, sort_keys=True, allow_nan=False))
    elif args.command == "freeze":
        path = write_declaration()
        print(json.dumps({"declaration": str(path), "sha256": sha256(path)}, indent=2))
    elif args.command == "fit":
        print(json.dumps(run_fit(args.fold, args.seed), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
