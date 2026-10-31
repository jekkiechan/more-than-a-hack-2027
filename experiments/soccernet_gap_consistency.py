"""Training-only missing-ball consistency for sparse SoccerNet segments.

The augmentation is applied to raw tracking inputs, before feature extraction.
Its auxiliary loss uses only the clean view's detached conditional H/A
distribution; it does not create possession, D, or event labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CONSISTENCY_WEIGHT = 0.05
# At the verified ~30 Hz native rate: up to ten seconds, one start opportunity
# per twenty observed seconds on average. Draws are clipped to real segments.
MAX_GAP_FRAMES = 300
GAP_START_PROBABILITY = 1.0 / 600.0
TEACHER_CONFIDENCE = 0.80
SAMPLE_STRIDE = 6
RECIPE_VERSION = "soccernet_raw_gap_consistency_v1"


def _segments(inputs: dict, n: int) -> list[np.ndarray]:
    """Return contiguous valid runs, respecting period and timestamp breaks."""
    if "tracking_valid" not in inputs:
        raise ValueError("tracking_valid is required for raw input segmentation")
    valid = np.asarray(inputs["tracking_valid"], dtype=bool)
    period = np.asarray(inputs["period"])
    times = np.asarray(inputs["timestamps"], dtype=float)
    if valid.shape != (n,) or period.shape != (n,) or times.shape != (n,):
        raise ValueError("tracking_valid, period, and timestamps must match ball rows")
    fps = float(np.asarray(inputs["fps"]))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    boundaries = np.ones(n, dtype=bool)
    if n > 1:
        boundaries[1:] = (~valid[:-1] | ~valid[1:] | (period[1:] != period[:-1]) |
                          ~np.isfinite(times[1:]) | ~np.isfinite(times[:-1]) |
                          (times[1:] <= times[:-1]) |
                          (times[1:] - times[:-1] > 1.5 / fps))
    starts = np.flatnonzero(valid & boundaries)
    result = []
    for start in starts:
        end = start + 1
        while end < n and valid[end] and not boundaries[end]:
            end += 1
        result.append(np.arange(start, end))
    return result


def _validate_continuous_native(inputs: dict) -> None:
    n = len(np.asarray(inputs.get("ball", [])))
    if n == 0:
        raise ValueError("raw native segment is empty")
    if "tracking_valid" not in inputs:
        raise ValueError("tracking_valid is required for paired feature preparation")
    valid = np.asarray(inputs["tracking_valid"], dtype=bool)
    if valid.shape != (n,) or not valid.all():
        raise ValueError("raw input must be one continuous segment with every row valid")
    fps = float(np.asarray(inputs.get("fps", np.nan)))
    if not np.isfinite(fps) or min(abs(fps - 29.97), abs(fps - 30.0)) > 0.1:
        raise ValueError("paired features require native SoccerNet FPS near 29.97 or 30")
    if len(_segments(inputs, n)) != 1:
        raise ValueError("raw input contains a period or timestamp discontinuity; split it first")
    times = np.asarray(inputs["timestamps"], dtype=float)
    if n > 1 and not np.allclose(np.diff(times), 1.0 / fps, rtol=0.02, atol=1e-5):
        raise ValueError("raw timestamps are not one continuous native-rate segment")


def corrupt_ball_gaps(inputs: dict, *, rng: np.random.Generator,
                      max_gap_frames: int = MAX_GAP_FRAMES,
                      start_probability: float = GAP_START_PROBABILITY
                      ) -> tuple[dict, np.ndarray]:
    """Copy raw inputs and hide sampled contiguous ball gaps within valid runs.

    Existing missing/unknown ball observations remain missing and are not
    counted as newly corrupted. Player tracking and all metadata are preserved.
    The supplied Generator makes the view reproducible for a fixed seed.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if isinstance(max_gap_frames, bool) or int(max_gap_frames) != max_gap_frames or max_gap_frames < 1:
        raise ValueError("max_gap_frames must be a positive integer")
    if not 0.0 <= float(start_probability) <= 1.0:
        raise ValueError("start_probability must be in [0, 1]")
    if "ball" not in inputs or "ball_observed" not in inputs:
        raise ValueError("raw inputs require ball and ball_observed")
    if "tracking_valid" not in inputs:
        raise ValueError("tracking_valid is required for ball-gap corruption")
    ball = np.asarray(inputs["ball"])
    observed = np.asarray(inputs["ball_observed"], dtype=bool)
    if ball.ndim != 2 or ball.shape[1] != 2 or observed.shape != (len(ball),):
        raise ValueError("ball must have shape (time, 2), with matching ball_observed")
    view = dict(inputs)
    corrupted = np.zeros(len(ball), dtype=bool)
    new_ball = np.array(ball, copy=True)
    new_observed = np.array(observed, copy=True)
    for run in _segments(inputs, len(ball)):
        i = 0
        while i < len(run):
            row = int(run[i])
            if not observed[row] or rng.random() >= start_probability:
                i += 1
                continue
            length = int(rng.integers(1, int(max_gap_frames) + 1))
            stop = min(i + length, len(run))
            rows = run[i:stop]
            rows = rows[observed[rows]]
            new_ball[rows] = np.nan
            new_observed[rows] = False
            corrupted[rows] = True
            i += length
    view["ball"] = new_ball
    view["ball_observed"] = new_observed
    return view, corrupted


def reliable_teacher_mask(inputs: dict) -> np.ndarray:
    """Rows where clean teacher targets are grounded in valid observed tracking."""
    n = len(np.asarray(inputs.get("ball", [])))
    if "tracking_valid" not in inputs:
        raise ValueError("tracking_valid is required for consistency targets")
    valid = np.asarray(inputs["tracking_valid"], dtype=bool)
    observed = np.asarray(inputs.get("ball_observed", np.zeros(n, dtype=bool)), dtype=bool)
    ball = np.asarray(inputs.get("ball", np.empty((0, 2))))
    if valid.shape != (n,) or observed.shape != (n,) or ball.shape != (n, 2):
        raise ValueError("raw tracking masks and ball coordinates have inconsistent shapes")
    return valid & observed & np.isfinite(ball).all(axis=1)


def _array_hash(inputs: dict) -> str:
    digest = hashlib.sha256()
    keys = ["ball", "players", "teams", "period", "fps", "timestamps",
            "pitch", "ball_observed", "tracking_valid"]
    if "player_observed" in inputs:
        keys.append("player_observed")
    for key in keys:
        if key not in inputs:
            raise ValueError(f"raw input is missing required field {key}")
        value = np.ascontiguousarray(np.asarray(inputs[key]))
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def prepare_consistency_record(raw_inputs: dict, *, seed: int, names: list[str]) -> dict:
    """Build aligned clean/gap joint features from raw SoccerNet tracking.

    Raw tracking is required because a saved feature record cannot recover
    ball coordinates needed for feature recomputation. The source runner must
    restrict calls to SoccerNet TRAIN segments. This helper reads no labels and
    requires explicit ``tracking_valid`` to prevent invalid rows becoming
    teacher targets.
    """
    _validate_continuous_native(raw_inputs)
    reliable_native = reliable_teacher_mask(raw_inputs)
    raw_hash = _array_hash(raw_inputs)
    clean_inputs = {key: np.array(value, copy=True) for key, value in raw_inputs.items()}
    gap_inputs, newly_corrupted = corrupt_ball_gaps(
        clean_inputs, rng=np.random.default_rng(int(seed)))

    # The tracker validity field is provenance/masking metadata, not a model
    # feature. The existing feature APIs accept the same allowlisted raw view.
    feature_keys = ("ball", "players", "teams", "period", "fps", "timestamps",
                    "pitch", "ball_observed", "player_observed")
    clean_feature_inputs = {key: clean_inputs[key] for key in feature_keys if key in clean_inputs}
    gap_feature_inputs = {key: gap_inputs[key] for key in feature_keys if key in gap_inputs}
    from experiments import learned_temporal as lt, refine_features

    clean_team, clean_team_names = refine_features.features(clean_feature_inputs)
    clean_player, clean_player_names = lt.player_features(clean_feature_inputs)
    gap_team, gap_team_names = refine_features.features(gap_feature_inputs)
    gap_player, gap_player_names = lt.player_features(gap_feature_inputs)
    expected_names = list(clean_team_names) + ["D_" + name for name in clean_player_names]
    expected_gap_names = list(gap_team_names) + ["D_" + name for name in gap_player_names]
    if expected_names != expected_gap_names or expected_names != list(names):
        raise ValueError("paired raw feature schema differs from requested names")
    clean_native = np.column_stack((clean_team, clean_player)).astype(np.float32)
    gap_native = np.column_stack((gap_team, gap_player)).astype(np.float32)
    if clean_native.shape != gap_native.shape or clean_native.shape[0] != len(reliable_native):
        raise ValueError("paired feature arrays have inconsistent dimensions")
    indices = np.arange(0, len(clean_native), SAMPLE_STRIDE, dtype=np.int64)
    candidate_native = newly_corrupted & reliable_native
    source_hashes = {}
    for module in (refine_features, lt):
        module_path = Path(module.__file__)
        source_hashes[module_path.name] = hashlib.sha256(module_path.read_bytes()).hexdigest()
    return {
        "clean_x": clean_native[indices],
        "gap_x": gap_native[indices],
        "names": expected_names,
        "indices": indices,
        "consistency_mask": candidate_native[indices],
        "native_corruption_mask": newly_corrupted,
        "native_reliable_mask": reliable_native,
        "native_ball_observed": np.asarray(clean_inputs["ball_observed"], dtype=bool).copy(),
        "native_tracking_valid": np.asarray(clean_inputs["tracking_valid"], dtype=bool).copy(),
        "provenance": {
            "recipe": RECIPE_VERSION,
            "recipe_parameters": {
                "seed": int(seed), "max_gap_frames": MAX_GAP_FRAMES,
                "start_probability": GAP_START_PROBABILITY,
                "teacher_confidence": TEACHER_CONFIDENCE,
                "sample_stride": SAMPLE_STRIDE,
                "consistency_weight": CONSISTENCY_WEIGHT,
            },
            "raw_input_sha256": raw_hash,
            "feature_source_sha256": source_hashes,
        },
    }


def attach_consistency_view(record: dict, raw_inputs: dict, *, seed: int) -> dict:
    """Verify clean parity and attach the paired gap view to a SoccerNet record.

    The caller must source ``raw_inputs`` only from a validated SoccerNet TRAIN
    segment. The returned shallow copy preserves the original target and all
    native audit masks, while adding only the experimental view and provenance.
    """
    source = record.get("source")
    if isinstance(source, np.ndarray) and source.ndim == 0:
        source = source.item()
    if str(source) != "soccernet":
        raise ValueError("consistency views may attach only to SoccerNet records")
    paired = prepare_consistency_record(raw_inputs, seed=seed,
                                        names=list(record.get("names", [])))
    x = np.asarray(record.get("x"))
    if x.shape != paired["clean_x"].shape or not np.array_equal(x, paired["clean_x"], equal_nan=True):
        raise ValueError("clean raw feature view does not exactly match the prepared SoccerNet record")
    indices = np.asarray(record.get("indices"), dtype=np.int64)
    if not np.array_equal(indices, paired["indices"]):
        raise ValueError("record sampled indices differ from paired raw feature indices")
    if int(np.asarray(record.get("native_n", -1))) != len(raw_inputs["ball"]):
        raise ValueError("record native length differs from raw input")
    native_fps = float(np.asarray(record.get("native_fps", record.get("fps", np.nan))))
    raw_fps = float(np.asarray(raw_inputs["fps"]))
    if abs(native_fps - raw_fps) > 0.1:
        raise ValueError("record native FPS differs from raw input")
    timestamps = np.asarray(raw_inputs["timestamps"], dtype=float)
    stored_times = record.get("full_timestamps", record.get("native_timestamps"))
    if stored_times is None or not np.array_equal(np.asarray(stored_times), timestamps):
        raise ValueError("record native timestamps differ from raw input")
    stored_period = record.get("full_period")
    if stored_period is None:
        raise ValueError("record is missing native period audit data")
    if not np.array_equal(np.diff(np.asarray(stored_period)) != 0,
                          np.diff(np.asarray(raw_inputs["period"])) != 0):
        raise ValueError("record native period boundaries differ from raw input")
    attached = dict(record)
    attached["gap_x"] = paired["gap_x"]
    attached["consistency_mask"] = paired["consistency_mask"]
    attached["gap_provenance"] = paired["provenance"]
    return attached


def consistency_loss(clean_logits, corrupted_logits, reliable_mask):
    """Fixed-weight detached-teacher KL for conditional home/away logits.

    Inputs are torch logits shaped ``(..., time, 2)`` and a boolean mask that
    broadcasts to their leading/time dimensions. Returns a scalar torch loss.
    The fixed 0.05 multiplier scales this auxiliary term; callers should add it
    to their existing SoccerNet H/A objective. No D logits are read.
    """
    import torch
    from torch.nn import functional as F

    if clean_logits.shape != corrupted_logits.shape or clean_logits.shape[-1] != 2:
        raise ValueError("clean and corrupted conditional logits must match (..., time, 2)")
    mask = torch.as_tensor(reliable_mask, dtype=torch.bool, device=clean_logits.device)
    try:
        mask = torch.broadcast_to(mask, clean_logits.shape[:-1])
    except RuntimeError as exc:
        raise ValueError("reliable_mask does not broadcast to the logit rows") from exc
    # Empty reliable populations should contribute a differentiable zero.
    confidence = torch.softmax(clean_logits.detach(), dim=-1).amax(dim=-1)
    mask = mask & (confidence >= TEACHER_CONFIDENCE)
    if not torch.any(mask):
        return corrupted_logits.sum() * 0.0
    teacher = F.softmax(clean_logits.detach(), dim=-1)
    row_kl = F.kl_div(F.log_softmax(corrupted_logits, dim=-1), teacher,
                      reduction="none").sum(dim=-1)
    return CONSISTENCY_WEIGHT * row_kl[mask].mean()
