"""Label-free aggregate audit of Smart Tagging against tracking and frozen probabilities.

This tool never reads a target label or official reference. It does no fitting or
scoring. It writes aggregate JSON only; source arrays and frame-level comparisons
remain in memory and are never printed or saved.

Declaration JSON schema::

    {
      "version": 1, "study_id": "smart_tagging_validation_v1",
      "input_sha256": "...", "xml_sha256": "...",
      "probabilities_sha256": null,
      "source_sha256": {"experiments/smart_tagging_audit.py": "...",
                         "experiments/smart_tagging_prior.py": "...",
                         "starter/load_smart_tagging.py": "..."},
      "team_to_side": {"<XML home club>": "H", "<XML away club>": "A"},
      "missing_ball_only": true,
      "anchor": {"own_max_m": 1.5, "opponent_min_m": 3.0},
      "claim": "label-free feasibility; no accuracy or adoption"
    }

All hashes are SHA-256. With no probability file, its path and expected hash
must both be null. Freeze the declaration before running against genuine assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import games
from experiments import smart_tagging_prior
from starter import load_smart_tagging


FPS = float(games.GAMES[games.GAME]["fps"])
INPUT_KEYS = ("ball", "players", "teams", "period", "fps", "timestamps", "pitch")
DECLARATION_KEYS = {"version", "study_id", "input_sha256", "xml_sha256",
                    "probabilities_sha256", "source_sha256", "team_to_side",
                    "missing_ball_only", "anchor", "claim"}
SOURCE_HASH_KEYS = {"experiments/smart_tagging_audit.py",
                    "experiments/smart_tagging_prior.py",
                    "starter/load_smart_tagging.py"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz_selected(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read only named numeric arrays; never access a labels member if present."""
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(keys) - set(archive.files))
        if missing:
            raise ValueError("NPZ is missing required non-label input arrays")
        return {key: np.array(archive[key], copy=True) for key in keys}


def load_inputs(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        optional = tuple(key for key in ("ball_observed", "player_observed") if key in archive.files)
        keys = INPUT_KEYS + optional
        data = {key: np.array(archive[key], copy=True) for key in keys if key in archive.files}
    missing = sorted(set(INPUT_KEYS) - set(data))
    if missing:
        raise ValueError("Input NPZ is missing required non-label input arrays")
    n = games.GAMES[games.GAME]["n_video_frames"]
    if data["ball"].shape != (n, 2) or data["players"].ndim != 3 or data["players"].shape[0] != n or data["players"].shape[2] != 2:
        raise ValueError("Input tracking shapes do not match challenge frame/coordinate contract")
    if data["teams"].shape != (data["players"].shape[1],):
        raise ValueError("Player team vector does not match tracking slots")
    if data["period"].shape != (n,) or data["timestamps"].shape != (n,):
        raise ValueError("Input period/timestamp arrays do not match challenge frame count")
    if np.asarray(data["fps"]).size != 1 or not np.isclose(float(data["fps"]), FPS):
        raise ValueError("Input frame rate does not match game metadata")
    if not np.isfinite(data["timestamps"]).all():
        raise ValueError("Input timestamps must be finite")
    return data


def load_probabilities(path: Path, n: int) -> np.ndarray:
    # Accept only the known native-probability member names; any labels member is
    # ignored without access. No decoded reference or prediction labels are read.
    with np.load(path, allow_pickle=False) as archive:
        key = "probabilities" if "probabilities" in archive.files else "probability"
        if key not in archive.files:
            raise ValueError("Probability NPZ needs a probabilities member")
        values = np.array(archive[key], dtype=np.float64, copy=True)
    if values.shape != (n, 3) or not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Baseline probabilities must be finite normalized native H/A/D rows")
    return values


def graded_halves(n: int, period: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    mask = np.zeros(n, dtype=bool)
    masks: dict[int, np.ndarray] = {}
    for half, first, last in games.half_video_blocks(games.GAME):
        block = np.zeros(n, dtype=bool)
        block[first:last + 1] = True
        if not np.all(np.asarray(period)[block] == half):
            raise ValueError("Input period IDs disagree with game half metadata")
        mask |= block
        masks[half] = block
    return mask, masks


def ball_gap_ge_10s(ball_observed: np.ndarray, period: np.ndarray,
                    timestamps: np.ndarray, fps: float) -> np.ndarray:
    """Mark missing rows inside bounded observation gaps lasting at least 10s."""
    observed = np.asarray(ball_observed, dtype=bool)
    periods = np.asarray(period)
    times = np.asarray(timestamps, dtype=np.float64)
    marked = np.zeros(len(observed), dtype=bool)
    boundaries = np.r_[0, np.flatnonzero((np.diff(periods) != 0) |
                                         (np.diff(times) <= 0) |
                                         (np.diff(times) > 1.5 / fps)) + 1,
                       len(observed)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        valid = np.flatnonzero(observed[start:stop]) + start
        for left, right in zip(valid[:-1], valid[1:]):
            elapsed = times[right] - times[left]
            if elapsed >= 10.0 and right > left + 1:
                marked[left + 1:right] = True
    return marked


def tracking_anchors(inputs: dict[str, np.ndarray], ball_observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descriptive same-frame anchors: ball <1.5m from one side, other >=3m."""
    ball = np.asarray(inputs["ball"], dtype=np.float64)
    players = np.asarray(inputs["players"], dtype=np.float64)
    teams = np.asarray(inputs["teams"])
    if not np.isin(teams, (0, 1)).all():
        raise ValueError("Tracking team vector must encode home/away as 0/1")
    observed_players = np.isfinite(players).all(axis=2)
    if "player_observed" in inputs:
        observed_players &= np.asarray(inputs["player_observed"], dtype=bool)
    distances = np.linalg.norm(players - ball[:, None, :], axis=2)
    distances[~observed_players] = np.inf
    distances[~np.asarray(ball_observed, dtype=bool)] = np.inf
    nearest = np.full((len(ball), 2), np.inf, dtype=np.float64)
    team_visible = np.zeros((len(ball), 2), dtype=bool)
    for side in (0, 1):
        slots = teams == side
        if slots.any():
            nearest[:, side] = np.min(distances[:, slots], axis=1)
            team_visible[:, side] = np.isfinite(distances[:, slots]).any(axis=1)
    anchors = np.full(len(ball), -1, dtype=np.int8)
    both_visible = team_visible.all(axis=1)
    anchors[both_visible & (nearest[:, 0] < 1.5) & (nearest[:, 1] >= 3.0)] = 0
    anchors[both_visible & (nearest[:, 1] < 1.5) & (nearest[:, 0] >= 3.0)] = 1
    support = anchors >= 0
    return anchors, support


def summarize(events: list[dict], mapping: dict[str, str], inputs: dict[str, np.ndarray],
              probabilities: np.ndarray | None, *, missing_ball_only: bool) -> dict:
    n = len(inputs["period"])
    period = np.asarray(inputs["period"])
    ball_observed = np.isfinite(inputs["ball"]).all(axis=1)
    if "ball_observed" in inputs:
        ball_observed &= np.asarray(inputs["ball_observed"], dtype=bool)
    graded, half_masks = graded_halves(n, period)
    gap10 = ball_gap_ge_10s(ball_observed, period, inputs["timestamps"], float(inputs["fps"]))
    anchor_side, anchor_support = tracking_anchors(inputs, ball_observed)

    # Run the production adapter on neutral synthetic probabilities to recover
    # only its tag-supported direction. This is not a model fit or a score.
    neutral = np.tile(np.array([0.5, 0.5, 0.0], dtype=np.float64), (n, 1))
    tag_prior = smart_tagging_prior.apply_smart_tagging_prior(
        neutral, events, mapping, float(inputs["fps"]), period, ball_observed,
        missing_ball_only=missing_ball_only,
        timestamps=inputs["timestamps"],
    )
    active = np.abs(tag_prior[:, 0] - neutral[:, 0]) > 1e-14
    tag_side = np.full(n, -1, dtype=np.int8)
    tag_side[active] = (tag_prior[active, 1] > tag_prior[active, 0]).astype(np.int8)
    active &= graded
    # Shadow direction uses the same frozen tag taxonomy, taper and H/A rules,
    # but ignores the candidate's ball-missing gate. It exists only to compare
    # broad phase labels with local tracking anchors on observed-ball frames.
    shadow_prior = smart_tagging_prior.apply_smart_tagging_prior(
        neutral, events, mapping, float(inputs["fps"]), period, ball_observed,
        missing_ball_only=False, timestamps=inputs["timestamps"],
    )
    shadow_active = np.abs(shadow_prior[:, 0] - neutral[:, 0]) > 1e-14
    shadow_side = np.full(n, -1, dtype=np.int8)
    shadow_side[shadow_active] = (shadow_prior[shadow_active, 1] > shadow_prior[shadow_active, 0]).astype(np.int8)
    shadow_active &= graded

    tag_frame_conflict = np.zeros(n, dtype=bool)
    # Keep unmasked interval overlaps separate from candidate-eligible overlaps.
    event_stats = {"total": len(events), "mapped_usable": 0,
                   "mapped_attacking": 0, "mapped_defending": 0,
                   "ignored_unknown_or_neutral": 0, "unmapped_team": 0,
                   "invalid_or_empty_interval": 0}
    from experiments.smart_tagging_prior import classify_smart_tagging_event
    direction_raw = np.zeros(n, dtype=np.int32)
    coverage_raw = np.zeros(n, dtype=np.int32)
    direction_effective = np.zeros(n, dtype=np.int32)
    coverage_effective = np.zeros(n, dtype=np.int32)
    for event in events:
        category, side = classify_smart_tagging_event(event, mapping)
        sign = 1 if category == "attacking" else (-1 if category == "defending" else 0)
        start, end = event.get("start_frame"), event.get("end_frame")
        if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)) or end <= start:
            event_stats["invalid_or_empty_interval"] += 1
            continue
        raw_team = str((event.get("tags") or {}).get("Team") or "").strip().lower()
        mapped_names = {str(name).strip().lower() for name in mapping}
        if side not in {"H", "A"} and raw_team not in mapped_names:
            event_stats["unmapped_team"] += 1
            continue
        if sign == 0:
            event_stats["ignored_unknown_or_neutral"] += 1
            continue
        half_tag = str((event.get("tags") or {}).get("Half") or "").strip().lower()
        event_half = 1 if half_tag in {"1st half", "first half", "1", "period 1"} else (
            2 if half_tag in {"2nd half", "second half", "2", "period 2"} else None)
        if event_half is None:
            positive_periods = np.unique(period[period > 0])
            event_half = int(positive_periods[0]) if len(positive_periods) == 1 else None
        if event_half is None:
            event_stats["ignored_unknown_or_neutral"] += 1
            continue
        event_stats["mapped_usable"] += 1
        event_stats["mapped_attacking" if sign > 0 else "mapped_defending"] += 1
        implied = side if sign > 0 else ("A" if side == "H" else "H")
        lo, hi = max(0, start), min(n, end)
        if hi <= lo:
            continue
        valid = (period[lo:hi] > 0) & (period[lo:hi] == event_half) if event_half else (period[lo:hi] > 0)
        indices_raw = np.arange(lo, hi)[valid]
        coverage_raw[indices_raw] += 1
        direction_raw[indices_raw] += 1 if implied == "H" else -1
        effective = valid & ((~ball_observed[lo:hi]) if missing_ball_only else True)
        indices_effective = np.arange(lo, hi)[effective]
        coverage_effective[indices_effective] += 1
        direction_effective[indices_effective] += 1 if implied == "H" else -1
    raw_conflict = (coverage_raw > 1) & (np.abs(direction_raw) < coverage_raw)
    effective_conflict = (coverage_effective > 1) & (np.abs(direction_effective) < coverage_effective)
    adjusted = None
    baseline_argmax = None
    adjusted_argmax = None
    if probabilities is not None:
        baseline_argmax = np.argmax(probabilities, axis=1)
        adjusted = smart_tagging_prior.apply_smart_tagging_prior(
            probabilities, events, mapping, float(inputs["fps"]), period,
            ball_observed, missing_ball_only=missing_ball_only,
            timestamps=inputs["timestamps"],
        )
        adjusted_argmax = np.argmax(adjusted, axis=1)

    def metrics(mask: np.ndarray) -> dict:
        scope = np.asarray(mask, dtype=bool) & graded
        tagged = scope & active
        raw_tagged = scope & (coverage_raw > 0)
        candidate_conflict = scope & effective_conflict
        raw_conflict_frames = int(np.sum(scope & raw_conflict))
        supported_anchor = tagged & anchor_support
        agreements = int(np.sum(tag_side[supported_anchor] == anchor_side[supported_anchor]))
        contradictions = int(np.sum(tag_side[supported_anchor] != anchor_side[supported_anchor]))
        row = {
            "frames": int(scope.sum()),
            "raw_mapped_tag_interval_frames": int(raw_tagged.sum()),
            "raw_mapped_tag_interval_fraction": float(raw_tagged.sum() / scope.sum()) if scope.any() else None,
            "tag_supported_frames": int(tagged.sum()),
            "tag_support_fraction": float(tagged.sum() / scope.sum()) if scope.any() else None,
            "raw_overlapping_opposing_tag_frames": raw_conflict_frames,
            "candidate_effective_conflict_frames": int(candidate_conflict.sum()),
            "tracking_anchor_support_frames": int(np.sum(scope & anchor_support)),
            "tag_and_anchor_overlap_frames": int(supported_anchor.sum()),
            "tag_anchor_agreement_frames": agreements,
            "tag_anchor_contradiction_frames": contradictions,
            "tag_anchor_agreement_fraction": float(agreements / supported_anchor.sum()) if supported_anchor.any() else None,
        }
        shadow_overlap = scope & shadow_active & anchor_support
        shadow_match = shadow_overlap & (shadow_side == anchor_side)
        row.update({
            "raw_phase_shadow_anchor_overlap_frames": int(shadow_overlap.sum()),
            "raw_phase_shadow_anchor_agreement_frames": int(shadow_match.sum()),
            "raw_phase_shadow_anchor_contradiction_frames": int(np.sum(shadow_overlap & (shadow_side != anchor_side))),
            "raw_phase_shadow_anchor_agreement_fraction": float(shadow_match.sum() / shadow_overlap.sum()) if shadow_overlap.any() else None,
        })
        if probabilities is not None:
            tagged_side_idx = (tag_side == 1).astype(np.int8)
            compare = tagged & (baseline_argmax < 2)
            agree = compare & (baseline_argmax == tagged_side_idx)
            changed = scope & (adjusted_argmax != baseline_argmax)
            row.update({
                "baseline_vs_tag_adjusted_argmax_comparison_frames": int(scope.sum()),
                "baseline_tag_adjusted_argmax_agreement_frames": int(np.sum(scope & (baseline_argmax == adjusted_argmax))),
                "baseline_tag_adjusted_argmax_agreement_fraction": float(np.sum(scope & (baseline_argmax == adjusted_argmax)) / scope.sum()) if scope.any() else None,
                "baseline_vs_tag_direction_comparison_frames": int(compare.sum()),
                "baseline_tag_direction_agreement_frames": int(agree.sum()),
                "baseline_tag_direction_agreement_fraction": float(agree.sum() / compare.sum()) if compare.any() else None,
                "tag_prior_changed_baseline_argmax_frames": int(changed.sum()),
                "agreement_interpretation": "model/tag AGREEMENT ONLY; not accuracy",
                "model_comparison_interpretation": "baseline/tag-adjusted model AGREEMENT ONLY; not accuracy",
            })
            anchor_compare = scope & anchor_support
            anchor_agree = anchor_compare & (baseline_argmax == anchor_side)
            row.update({
                "baseline_vs_tracking_anchor_comparison_frames": int(anchor_compare.sum()),
                "baseline_tracking_anchor_agreement_frames": int(anchor_agree.sum()),
                "baseline_tracking_anchor_agreement_fraction": float(anchor_agree.sum() / anchor_compare.sum()) if anchor_compare.any() else None,
                "anchor_agreement_interpretation": "tracking/model AGREEMENT ONLY; not accuracy",
            })
        return row

    all_mask = np.ones(n, dtype=bool)
    strata = {
        "graded_all": metrics(all_mask),
        "ball_observed": metrics(ball_observed),
        "ball_missing": metrics(~ball_observed),
        "missing_ball_gap_at_least_10s": metrics(gap10),
    }
    for half, mask in half_masks.items():
        strata[f"half_{half}"] = metrics(mask)
        strata[f"half_{half}_ball_observed"] = metrics(mask & ball_observed)
        strata[f"half_{half}_ball_missing"] = metrics(mask & ~ball_observed)
        strata[f"half_{half}_missing_ball_gap_at_least_10s"] = metrics(mask & gap10)
    shadow_scope = graded & shadow_active & anchor_support
    shadow_agree = shadow_scope & (shadow_side == anchor_side)
    shadow = {
        "scope": "raw-phase shadow only; same tag prior without ball-missing gate; descriptive, not accuracy",
        "tag_and_tracking_anchor_overlap_frames": int(shadow_scope.sum()),
        "tag_tracking_anchor_agreement_frames": int(shadow_agree.sum()),
        "tag_tracking_anchor_contradiction_frames": int(np.sum(shadow_scope & (shadow_side != anchor_side))),
        "agreement_fraction": float(shadow_agree.sum() / shadow_scope.sum()) if shadow_scope.any() else None,
        "agreement_interpretation": "same-source phase/tracking ANCHOR AGREEMENT ONLY; not accuracy",
    }
    return {
        "evidence_label": "label-free descriptive audit; no accuracy or efficacy estimate",
        "event_counts": event_stats,
        "raw_tag_overlap_conflict_definition": "overlapping recognized tag intervals imply opposing sides; ball mask ignored",
        "effective_tag_overlap_conflict_definition": "same overlap after candidate ball-missing eligibility gate",
        "strata": strata,
        "raw_phase_shadow_anchor_alignment": shadow,
    }


def _read_declaration(path: Path) -> dict:
    declaration = json.loads(path.read_text(encoding="utf-8"))
    if set(declaration) != DECLARATION_KEYS:
        raise ValueError("Declaration has an unexpected schema")
    if declaration["version"] != 1 or declaration["study_id"] != "smart_tagging_validation_v1":
        raise ValueError("Unexpected study declaration identity")
    if set(declaration["source_sha256"]) != SOURCE_HASH_KEYS:
        raise ValueError("Declaration hash inventory has an unexpected schema")
    if not isinstance(declaration["missing_ball_only"], bool):
        raise ValueError("missing_ball_only must be a boolean")
    if declaration["anchor"] != {"own_max_m": 1.5, "opponent_min_m": 3.0}:
        raise ValueError("Anchor thresholds differ from the frozen protocol")
    if declaration["claim"] != "label-free feasibility; no accuracy or adoption":
        raise ValueError("Declaration claim differs from the frozen protocol")
    mapping = declaration["team_to_side"]
    if not isinstance(mapping, dict) or len(mapping) != 2 or set(mapping.values()) != {"H", "A"}:
        raise ValueError("Declaration must map exactly two XML team names to H/A")
    return declaration


def run(inputs_path: Path, xml_path: Path, declaration_path: Path,
        output_path: Path, probability_path: Path | None = None) -> dict:
    declaration = _read_declaration(declaration_path)
    if (probability_path is None) != (declaration["probabilities_sha256"] is None):
        raise ValueError("Probability path and declaration hash must both be present or absent")
    source_paths = {
        "experiments/smart_tagging_audit.py": Path(__file__).resolve(),
        "experiments/smart_tagging_prior.py": Path(smart_tagging_prior.__file__).resolve(),
        "starter/load_smart_tagging.py": Path(load_smart_tagging.__file__).resolve(),
    }
    if sha256(inputs_path) != declaration["input_sha256"]:
        raise ValueError("Declared input hash mismatch")
    if sha256(xml_path) != declaration["xml_sha256"]:
        raise ValueError("Declared XML hash mismatch")
    if probability_path and sha256(probability_path) != declaration["probabilities_sha256"]:
        raise ValueError("Declared probability NPZ hash mismatch")
    for key, path in source_paths.items():
        if sha256(path) != declaration["source_sha256"][key]:
            raise ValueError(f"Declared source hash mismatch: {key}")
    data = load_inputs(inputs_path)
    n = len(data["period"])
    probabilities = load_probabilities(probability_path, n) if probability_path else None
    events = load_smart_tagging.load(data_dir=str(xml_path.resolve().parents[1]))
    # loader resolves canonical game path from data_dir. Require the supplied XML
    # to be that exact tactical asset to prevent auditing a silently different file.
    if Path(load_smart_tagging.pattern_path(data_dir=str(xml_path.resolve().parents[1]))).resolve() != xml_path.resolve():
        raise ValueError("XML path must be the canonical tactical Smart Tagging asset")
    report = {
        "protocol": "smart_tagging_label_free_audit_v1",
        "provenance": {
            "inputs_npz_sha256": sha256(inputs_path),
            "xml_sha256": sha256(xml_path),
            "probabilities_npz_sha256": sha256(probability_path) if probability_path else None,
            "audit_script_sha256": sha256(source_paths["experiments/smart_tagging_audit.py"]),
            "prior_adapter_sha256": sha256(source_paths["experiments/smart_tagging_prior.py"]),
            "xml_loader_sha256": sha256(source_paths["starter/load_smart_tagging.py"]),
            "declaration_sha256": sha256(declaration_path),
        },
        "config": {"missing_ball_only": declaration["missing_ball_only"],
                   "fps": FPS, "anchor_near_m": 1.5, "anchor_other_team_min_m": 3.0,
                   "long_ball_gap_seconds": 10.0},
        "audit": summarize(events, declaration["team_to_side"], data, probabilities,
                           missing_ball_only=declaration["missing_ball_only"]),
    }
    try:
        output_path.resolve().relative_to((ROOT / "data").resolve())
    except ValueError as exc:
        raise ValueError("Aggregate output must be stored under the ignored private data/ directory") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.inputs, args.xml, args.declaration, args.output, args.probabilities)
    print("Wrote aggregate label-free Smart Tagging audit JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
