"""Small, direct edits to recurrent weights inspired by weight-space programming.

This module does not fit parameters. It provides two declared tensor edits: a
rank-one removal from a conditional H/A output projection along a diagnosed
hidden activation direction, and a fixed additive shift to PyTorch GRU update
biases (gate order reset, update, new). The former is closer to the cited Qwen
edit; the latter tests whether excess recurrent persistence delays H/A
transitions.

The source article's Qwen edit targets a diagnosed repetition-associated
activation direction in selected feed-forward output columns.  GRUs have no
matching feed-forward block, so this is a bounded architecture-specific
analogue, not a reproduction of that edit.  Callers must derive any diagnosis
from eligible training records and freeze the edit before evaluating held-out
providers.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def shift_gru_update_bias(
    state: Mapping[str, Any],
    *,
    hidden_size: int,
    delta: float,
    module_prefix: str = "sequence",
    directions: tuple[str, ...] = ("", "_reverse"),
) -> dict[str, np.ndarray]:
    """Return a copied state dict with a fixed shift to GRU update-gate biases.

    PyTorch stores the three gate blocks in ``reset, update, new`` order.  The
    update-gate block therefore occupies ``[hidden_size:2*hidden_size]`` in
    each ``bias_ih_l0{suffix}`` tensor.  Only the input bias is changed; PyTorch
    sums the input and recurrent biases during inference, so this yields the
    exact requested logit shift without doubling it.  No output projection,
    recurrent matrix, reset/new gate, normalization value, or D model changes.

    ``delta < 0`` lowers update-gate logits and hence reduces retention;
    ``delta > 0`` is a useful opposite-sign control.  This function is
    deterministic and performs no gradient or optimizer operations.
    """
    if isinstance(hidden_size, (bool, np.bool_)) or not isinstance(hidden_size, (int, np.integer)) or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer")
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    if not isinstance(module_prefix, str):
        raise ValueError("module_prefix must be a string")
    if not directions or any(s not in ("", "_reverse") for s in directions):
        raise ValueError("directions must contain GRU direction suffixes")
    if len(set(directions)) != len(directions):
        raise ValueError("directions must not contain duplicates")

    result = {str(key): np.array(value, copy=True) for key, value in state.items()}
    expected = 3 * hidden_size
    for suffix in directions:
        key = f"{module_prefix + '.' if module_prefix else ''}bias_ih_l0{suffix}"
        if key not in result:
            raise ValueError(f"missing GRU bias tensor: {key}")
        bias = result[key]
        if bias.shape != (expected,):
            raise ValueError(f"{key} must have shape ({expected},), got {bias.shape}")
        if not np.issubdtype(bias.dtype, np.floating) or not np.isfinite(bias).all():
            raise ValueError(f"{key} must contain finite floating-point values")
        bias[hidden_size:2 * hidden_size] += np.asarray(delta, dtype=bias.dtype)
        if not np.isfinite(bias).all():
            raise ValueError(f"{key} became non-finite after applying delta")
    return result


def hidden_failure_direction(
    error_hidden: Any,
    control_hidden: Any,
    *,
    error_match_ids: Any | None = None,
    control_match_ids: Any | None = None,
) -> np.ndarray:
    """Return a unit mean-difference direction from training-only activations.

    The caller defines matched failure/control rows using eligible training
    records. Rows must be ``[samples, hidden]`` with the same width. Optional
    match IDs make the direction an equal-match mean contrast: each match with
    both event types contributes its own group means once, regardless of its
    frame count. The mean difference is normalized so its magnitude does not
    silently set edit strength. Never populate this function from held-out
    labels.
    """
    errors = np.asarray(error_hidden, dtype=np.float64)
    controls = np.asarray(control_hidden, dtype=np.float64)
    if errors.ndim != 2 or controls.ndim != 2 or errors.shape[1:] != controls.shape[1:]:
        raise ValueError("error_hidden and control_hidden must be 2-D with equal hidden width")
    if not len(errors) or not len(controls):
        raise ValueError("both activation groups must be non-empty")
    if not np.isfinite(errors).all() or not np.isfinite(controls).all():
        raise ValueError("activations must be finite")
    if error_match_ids is None and control_match_ids is None:
        error_mean, control_mean = errors.mean(axis=0), controls.mean(axis=0)
    else:
        error_ids = np.asarray(error_match_ids)
        control_ids = np.asarray(control_match_ids)
        if error_ids.shape != (len(errors),) or control_ids.shape != (len(controls),):
            raise ValueError("match ID arrays must align with activation rows")
        common = sorted(set(error_ids.tolist()) & set(control_ids.tolist()), key=str)
        if not common:
            raise ValueError("no training match contains both delayed and control transitions")
        error_mean = np.mean([errors[error_ids == m].mean(axis=0) for m in common], axis=0)
        control_mean = np.mean([controls[control_ids == m].mean(axis=0) for m in common], axis=0)
    direction = error_mean - control_mean
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("failure and control activations do not define a direction")
    return (direction / norm).astype(np.float32)


def delayed_transition_activations(
    hidden: Any,
    target: Any,
    predicted: Any,
    sequence_ids: Any,
    *,
    require_nonempty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect late and correctly timed H/A boundary activations.

    A boundary at ``t`` is eligible only when the target changes at ``t``,
    remains stable at ``t+1``, and all three rows are known H/A classes in the
    same sequence. It is called *late* when the model still predicts the
    previous class at ``t`` but predicts the new class at ``t+1``. It is a
    control when the model predicts the new class at ``t``. Unknown/D rows,
    early predictions, multi-frame delays, and sequence edges are excluded. Call only on
    eligible training records; the returned groups define the rank-one edit
    direction without using evaluation labels.
    """
    activations = np.asarray(hidden, dtype=np.float64)
    truth = np.asarray(target)
    prediction = np.asarray(predicted)
    groups = np.asarray(sequence_ids)
    if activations.ndim != 2 or truth.ndim != 1 or prediction.shape != truth.shape or groups.shape != truth.shape or len(truth) != len(activations):
        raise ValueError("hidden must be [N,H] and target, predicted, sequence_ids must be [N]")
    if not np.isfinite(activations).all():
        raise ValueError("hidden activations must be finite")
    valid = (truth == 0) | (truth == 1)
    late_idx, control_idx = [], []
    for t in range(1, len(truth) - 1):
        if (groups[t - 1] != groups[t] or groups[t] != groups[t + 1]
                or not (valid[t - 1] and valid[t] and valid[t + 1])
                or truth[t] == truth[t - 1] or truth[t + 1] != truth[t]):
            continue
        new_class, old_class = truth[t], truth[t - 1]
        if prediction[t - 1] != new_class and prediction[t] == old_class and prediction[t + 1] == new_class:
            late_idx.append(t)
        elif prediction[t - 1] != new_class and prediction[t] == new_class:
            control_idx.append(t)
    if require_nonempty and (not late_idx or not control_idx):
        raise ValueError("training data must contain both delayed and correctly timed H/A transitions")
    return activations[late_idx].astype(np.float32), activations[control_idx].astype(np.float32)


def extract_training_activations(record: Mapping[str, Any], model: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract pre-output GRU activations and deployed H/A predictions.

    The caller must supply an eligible TRAIN record. Targets are returned only
    for the train-only diagnostic; inference itself uses feature/schema/time
    metadata and the frozen model. Hidden states use the native orientation;
    predictions average native and team-swapped probabilities, matching the
    model's inference symmetrization. Every ``cap._record_chunks`` core gets a
    distinct sequence ID so transition diagnostics cannot cross breaks.
    """
    import torch
    from experiments import learned_temporal as lt, soccernet_capacity_model as cap
    from experiments import supervised

    required = ("x", "names", "period", "y")
    if any(key not in record for key in required):
        raise ValueError(f"training record is missing one of {required}")
    names = list(record["names"])
    raw = np.asarray(record["x"])
    target = np.asarray(record["y"])
    if raw.ndim != 2 or target.shape != (len(raw),):
        raise ValueError("training record x/y shapes do not align")
    feature = {key: record[key] for key in ("x", "names", "period")}
    for key in ("timestamps", "fps"):
        if key in record:
            feature[key] = record[key]
    chunks = cap._record_chunks(feature)
    if not chunks:
        raise ValueError("training record has no inference chunks")
    if names != list(model.get("feature_names", [])):
        raise ValueError("training record schema differs from checkpoint")
    torch.set_num_threads(1)
    net = cap.network(int(model["input_features"]), int(model["hidden"]),
                      float(model["training"]["input_dropout"]))
    net.load_state_dict({k: torch.as_tensor(v) for k, v in model["state"].items()})
    net.eval()
    hidden = np.empty((len(raw), 2 * int(model["hidden"])), dtype=np.float32)
    probability_views = [np.empty((len(raw), 2), dtype=np.float32) for _ in range(2)]
    normalized_views = []
    for swap in (False, True):
        view = supervised.swap_team_features(raw, names) if swap else raw
        normalized_views.append(lt.normalize(view, model["median"], model["scale"]))
    with torch.inference_mode():
        for offset in range(0, len(chunks), 64):
            batch = chunks[offset:offset + 64]
            for view_id, normalized in enumerate(normalized_views):
                x = torch.as_tensor(np.stack([normalized[full] for _, _, full in batch]), dtype=torch.float32)
                h = net.sequence(net.input_dropout(x))[0]
                logits = net.output(h)
                independent_logits = h @ net.output.weight.T + net.output.bias
                if not torch.allclose(logits, independent_logits, atol=1e-6, rtol=1e-6):
                    raise RuntimeError("pre-output activation projection does not reproduce network logits")
                probs = torch.softmax(logits[:, cap.CONTEXT:cap.CONTEXT + cap.CORE], dim=-1).cpu().numpy()
                if view_id == 0:
                    hcore = h[:, cap.CONTEXT:cap.CONTEXT + cap.CORE].cpu().numpy()
                for i, (start, size, _) in enumerate(batch):
                    probability_views[view_id][start:start + size] = probs[i, :size, ::-1] if view_id else probs[i, :size]
                    if view_id == 0:
                        hidden[start:start + size] = hcore[i, :size]
    mean_probability = (probability_views[0] + probability_views[1]) * .5
    predicted = mean_probability.argmax(axis=1).astype(np.int8)
    sequence_ids = np.full(len(raw), -1, dtype=np.int32)
    for sequence_id, (start, size, _) in enumerate(chunks):
        sequence_ids[start:start + size] = sequence_id
    if np.any(sequence_ids < 0):
        raise RuntimeError("activation extraction left rows outside the model's inference chunks")
    return hidden, target.astype(np.int8, copy=False), predicted, sequence_ids


def diagnose_model(model: Mapping[str, Any], teacher: Mapping[str, Any],
                   training_records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive a delayed-transition direction from eligible TRAIN records only.

    Each record is one clean or fixed first-v2 view for a single TRAIN match.
    Every view is evaluated in its native and team-swapped orientation. The
    caller is responsible for passing only the provider-excluded original
    TRAIN cohort. This function checks teacher/model schema and normalization,
    captures exact pre-output GRU activations, and returns event counts and an
    equal-match direction suitable for a frozen artifact. It reads no dev or
    GAR evaluation records.
    """
    import hashlib
    from experiments import supervised

    if not training_records:
        raise ValueError("training_records must be non-empty")
    if (list(model.get("feature_names", [])) != list(teacher.get("feature_names", []))
            or not np.array_equal(model.get("median"), teacher.get("median"))
            or not np.array_equal(model.get("scale"), teacher.get("scale"))):
        raise ValueError("model and teacher must share frozen feature schema/normalization")
    late_rows, control_rows, late_match_ids, control_match_ids = [], [], [], []
    details = []
    seen = set()
    for record in training_records:
        match = str(record.get("match", ""))
        view = str(record.get("view_name", record.get("view", "clean")))
        if not match:
            raise ValueError("each training record must identify its match")
        key = (match, view)
        if key in seen:
            raise ValueError(f"duplicate training match/view record: {key}")
        seen.add(key)
        names = list(record["names"])
        y = np.asarray(record["y"])
        x = np.asarray(record["x"])
        for orientation in ("native", "swapped"):
            current = dict(record)
            if orientation == "swapped":
                current["x"] = supervised.swap_team_features(x, names)
                swapped_y = y.copy()
                h_or_a = (y == 0) | (y == 1)
                swapped_y[h_or_a] = 1 - y[h_or_a]
                current["y"] = swapped_y
            hidden, target, predicted, sequence_ids = extract_training_activations(current, model)
            late, correct = delayed_transition_activations(
                hidden, target, predicted, sequence_ids, require_nonempty=False)
            # Count eligible single-step boundaries independently from outcome.
            eligible = 0
            for t in range(1, len(target) - 1):
                if (sequence_ids[t - 1] == sequence_ids[t] == sequence_ids[t + 1]
                        and target[t - 1] in (0, 1) and target[t] in (0, 1)
                        and target[t + 1] == target[t] and target[t] != target[t - 1]
                        and predicted[t - 1] != target[t]):
                    eligible += 1
            details.append({"match": match, "view": view, "orientation": orientation,
                            "eligible_transitions": int(eligible), "late_1_sample": len(late),
                            "correct_at_boundary": len(correct),
                            "late_rate": float(len(late) / eligible) if eligible else None})
            if len(late):
                late_rows.append(late)
                late_match_ids.extend([match] * len(late))
            if len(correct):
                control_rows.append(correct)
                control_match_ids.extend([match] * len(correct))
    if not late_rows or not control_rows:
        raise ValueError("eligible TRAIN cohort lacks a delayed or correctly timed transition group")
    direction = hidden_failure_direction(
        np.concatenate(late_rows), np.concatenate(control_rows),
        error_match_ids=np.asarray(late_match_ids),
        control_match_ids=np.asarray(control_match_ids))
    matched_matches = sorted(set(late_match_ids) & set(control_match_ids))
    digest = hashlib.sha256(np.ascontiguousarray(direction).tobytes()).hexdigest()
    return {
        "direction": direction.tolist(),
        "direction_sha256": digest,
        "direction_width": int(direction.size),
        "training_match_count": len({str(r["match"]) for r in training_records}),
        "direction_contributing_matches": matched_matches,
        "direction_contributing_match_count": len(matched_matches),
        "view_count": len(seen),
        "activation_diagnostic": details,
        "diagnostic_definition": "one-sample-late vs correctly-timed stable known H/A transition; equal match means; clean and first fixed v2 TRAIN view; native and swapped orientations; no GAR/dev labels",
        "evaluation_labels_used": False,
    }


def subtract_output_direction(
    weight: Any,
    direction: Any,
    *,
    alpha: float,
    preserve_common_mode: bool = True,
) -> np.ndarray:
    """Subtract a rank-one hidden-direction contribution from an output matrix.

    For unit ``v``, the direct edit is ``W' = W - alpha (Wv) v.T``.  For a
    two-class H/A head, ``preserve_common_mode=True`` first projects ``Wv`` onto
    the zero-sum H/A logit contrast.  Thus the edit changes only the H/A
    difference along ``v`` and leaves the two-logit common-mode response
    unchanged.  ``alpha=0`` is exact identity; ``alpha=1`` removes that
    directional response.  This is a direct tensor edit, not optimization.
    """
    matrix = np.asarray(weight)
    vector = np.asarray(direction, dtype=np.float64)
    if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.floating):
        raise ValueError("weight must be a floating-point matrix")
    if vector.ndim != 1 or vector.shape[0] != matrix.shape[1]:
        raise ValueError("direction width must match weight input dimension")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("weight and direction must be finite")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > 1e-5:
        raise ValueError("direction must be normalized")
    contribution = np.asarray(matrix, dtype=np.float64) @ vector
    if preserve_common_mode:
        if matrix.shape[0] != 2:
            raise ValueError("common-mode preservation is defined only for a two-logit H/A head")
        contribution = contribution - contribution.mean()
    edited = np.asarray(matrix, dtype=np.float64) - float(alpha) * np.outer(contribution, vector)
    result = edited.astype(matrix.dtype)
    if not np.isfinite(result).all():
        raise ValueError("edited output matrix is non-finite")
    return result


def edit_had_state_ha_projection(
    state: Mapping[str, Any],
    direction: Any,
    *,
    alpha: float,
    output_key: str = "output.weight",
) -> dict[str, np.ndarray]:
    """Apply the low-rank edit to H/A rows of a 3-class model state only.

    The row order is H, A, D. The D output row and every other state tensor
    are copied unchanged; separately stored ``d_state`` remains outside this
    function and is consequently unaffected.
    """
    if output_key not in state:
        raise ValueError(f"missing output projection: {output_key}")
    weight = np.asarray(state[output_key])
    if weight.ndim != 2 or weight.shape[0] != 3:
        raise ValueError("expected a three-row H/A/D output projection")
    edited = dict((str(key), np.array(value, copy=True)) for key, value in state.items())
    edited[output_key][:2] = subtract_output_direction(
        weight[:2], direction, alpha=alpha, preserve_common_mode=True)
    return edited


def edit_conditional_ha_state_projection(
    state: Mapping[str, Any],
    direction: Any,
    *,
    alpha: float,
    output_key: str = "output.weight",
) -> dict[str, np.ndarray]:
    """Edit a two-row conditional H/A projection, preserving all other state.

    The v2 dense member stores a two-row conditional H/A GRU head and keeps its
    D path in a separate ``d_state`` mapping. The caller passes only the dense
    member's ``state`` here; the D state cannot be edited through this API.
    """
    if output_key not in state:
        raise ValueError(f"missing output projection: {output_key}")
    weight = np.asarray(state[output_key])
    if weight.ndim != 2 or weight.shape[0] != 2:
        raise ValueError("expected a two-row conditional H/A output projection")
    edited = dict((str(key), np.array(value, copy=True)) for key, value in state.items())
    edited[output_key] = subtract_output_direction(
        weight, direction, alpha=alpha, preserve_common_mode=True)
    return edited
