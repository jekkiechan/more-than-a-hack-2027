"""Corrected-origin, nested whole-match Sportec event-to-possession calibration."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import product
import hashlib
import json
import math
import os
from numbers import Real
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from experiments import metrics

ROOT = Path(__file__).resolve().parents[1]


def candidate_grid() -> list[dict]:
    """Return the predeclared 24-arm sensitivity grid for GAR touch runs.

    CH/RE end any run in v1; CL ends a run only when it is a single-frame run.
    The second candidate expands CL to any run; the third treats no end code
    as evidence of an uncontrolled run.
    """
    end_rules = {
        "v1_ch_re_any_cl_single": {"CH": "any", "RE": "any", "CL": "single_frame"},
        "cl_any_run_sensitivity": {"CH": "any", "RE": "any", "CL": "any"},
        "no_uncontrolled_end_codes": {},
    }
    # Keep the calibration bounded: two realistic boundary policies per axis.
    return [
        {
            "uncontrolled_end_rules": dict(end_rules[end_name]),
            "uncontrolled_rule": end_name,
            "uncontrolled_opponent": opponent,
            "censored_run_controlled": censored,
            "out_tail_grace_seconds": grace,
        }
        for end_name, opponent, censored, grace in product(
            end_rules, ("hold", "literal"), (True, False), (0.0, 3.0)
        )
    ]


def select_by_provider_mean(scores_by_provider: dict[str, dict[str, float]],
                            excluded_provider: str) -> dict:
    """Select a candidate using only whole-provider training scores.

    Input maps provider -> candidate-id -> scalar loss (lower is better).
    The excluded provider must be absent, and every included provider must
    provide the same candidate set. One eligible training provider is enough
    (for example, when only Metrica has usable event mapping in a fold). This
    does not itself compute a score or prove match-level separation: the caller
    must ensure each score uses training matches only, and must document the
    match IDs and any inner whole-match split used to produce it.
    """
    if excluded_provider in scores_by_provider:
        raise ValueError("excluded provider appears in calibration data")
    if not scores_by_provider:
        raise ValueError("need at least one eligible training provider")
    providers = sorted(scores_by_provider)
    candidate_ids = set(scores_by_provider[providers[0]])
    if not candidate_ids:
        raise ValueError("empty candidate score set")
    for provider in providers:
        scores = scores_by_provider[provider]
        if set(scores) != candidate_ids:
            raise ValueError("candidate sets differ across providers")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(float(v))
               for v in scores.values()):
            raise ValueError("candidate losses must be finite real numbers")
    means = {
        candidate: sum(float(scores_by_provider[p][candidate]) for p in providers) / len(providers)
        for candidate in sorted(candidate_ids)
    }
    chosen = min(means, key=lambda key: (means[key], key))
    return {"candidate": chosen, "provider_mean_loss": means[chosen],
            "provider_losses": {p: float(scores_by_provider[p][chosen]) for p in providers},
            "training_providers": providers, "excluded_provider": excluded_provider}


# A source-specific adapter to Sportec's documented event vocabulary. It maps
# explicit TacklingGame.PossessionChange, event teams and stoppage/restart times
# to a sparse event state stream; it never reads target labels to construct ops.
SPORTEC_MATCHES = ("J03WMX", "J03WN1", "J03WOH")  # TRAIN only
SPORTEC_EVENT_FILE_IDS = {"J03WMX": 51643493, "J03WN1": 51643496, "J03WOH": 51643499}
SPORTEC_INFO_FILE_IDS = {"J03WMX": 51643475, "J03WN1": 51643472, "J03WOH": 51643478}
SPORTEC_EVENT_MD5 = {
    "J03WMX": "2d88e180a1f63c7e4157feb980fc7255",
    "J03WN1": "83e8ed7ea8e8ebbbdc61f61ac3534008",
    "J03WOH": "ab66e51512f39d06f49f27dd3b4c2f40",
}
SPORTEC_INFO_MD5 = {
    "J03WMX": "bc5d5b21dbcf36aa30f4cd0428309f6b",
    "J03WN1": "37b0f85588c0acfb995fe633c7ac2e0d",
    "J03WOH": "b9710781d400dd812f62b0a6e3d7bce8",
}
SPORTEC_TRACKING_ORIGIN = {
    # First observed frame T from exact bytes 0-65535 of the pinned Figshare
    # positions XML. These timestamps, not planned/actual kickoff, are origin 0.
    "J03WMX": {"file_id": 51643514, "raw_md5": "e4273a2adf5dbd9d4ec42def2f01905c",
               "frame": 10000, "timestamp": "2023-05-27T13:30:12.600+00:00",
               "range_bytes": [0, 65535],
               "prefix_sha256": "1e05e0bbe89a5829baabd63f1a354609741454724a0974b82073ac6805cda822"},
    "J03WN1": {"file_id": 51643517, "raw_md5": "bacb2f609c1d1b0bfc906f6b9d4109f2",
               "frame": 10000, "timestamp": "2023-05-27T13:31:06.400+00:00",
               "range_bytes": [0, 65535],
               "prefix_sha256": "8f965a6b08f9a72a782565cd9968a19d87409bb6c5df5166713cf152662c15f7"},
    "J03WOH": {"file_id": 51643520, "raw_md5": "3efff25a86a61768e7d253b4bba137a3",
               "frame": 10000, "timestamp": "2022-08-26T16:32:09.720+00:00",
               "range_bytes": [0, 65535],
               "prefix_sha256": "77dc05846daa6156bf54a0aaa3c53102ffaa3f887d2982a2c493a4cc4e821ee0"},
}
RESTART_TYPES = frozenset(("ThrowIn", "FreeKick", "GoalKick", "CornerKick", "KickOff", "Penalty"))
STOPPAGE_TYPES = frozenset(("Foul", "Offside", "FinalWhistle", "GoalDisallowed",
                            "ShotWide", "SuccessfulShot", "Goal"))
ACTION_TYPES = frozenset(("Play", "OtherBallAction", "BallClaiming"))


def sportec_candidate_grid() -> list[dict]:
    """Four frozen sensitivity arms supported by Sportec's native events."""
    return [
        {"id": f"contest_{contest}_tail_{int(grace)}s",
         "nonchanging_duel": contest, "out_tail_grace_seconds": grace}
        for contest, grace in product(("hold", "literal"), (0.0, 3.0))
    ]


def _parse_time(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _team_code(team_id: str | None, home_id: str, away_id: str) -> str | None:
    if team_id == home_id:
        return "H"
    if team_id == away_id:
        return "A"
    return None


def _frame_at(timestamps: np.ndarray, seconds: float) -> int | None:
    """Nearest actual frame; return None when the event lies outside coverage."""
    k = int(np.searchsorted(timestamps, seconds, side="left"))
    options = [i for i in (k - 1, k) if 0 <= i < len(timestamps)]
    if not options:
        return None
    nearest = min(options, key=lambda i: (abs(float(timestamps[i]) - seconds), i))
    if abs(float(timestamps[nearest]) - seconds) > 0.5:
        return None
    return nearest


def load_sportec_operations(event_path: Path, info_path: Path,
                            timestamps: np.ndarray,
                            timestamp_origin: str | None = None) -> tuple[list[dict], dict]:
    """Parse source event anchors and stoppages, without accessing target arrays."""
    info = ET.parse(info_path).getroot().find(".//General").attrib
    home_id, away_id = info["HomeTeamId"], info["GuestTeamId"]
    kickoff = _parse_time(info["KickoffTime"])
    origin = _parse_time(timestamp_origin) if timestamp_origin else kickoff
    root = ET.parse(event_path).getroot()
    operations, counts = [], Counter()

    def add(absolute_time: float, kind: str, team: str | None = None,
            challenger: str | None = None, source_type: str = "") -> None:
        frame = _frame_at(timestamps, absolute_time - origin)
        counts[f"ops_{kind}"] += 1
        if frame is None:
            counts["events_outside_tracking_coverage"] += 1
            return
        operations.append({"frame": frame, "time": absolute_time - origin,
                           "kind": kind, "team": team,
                           "challenger": challenger, "source_type": source_type})

    for event in root.findall("Event"):
        event_time_text = event.get("EventTime")
        if not event_time_text:
            counts["events_without_time"] += 1
            continue
        event_time = _parse_time(event_time_text)
        children = list(event)
        outer_tags = [child.tag for child in children]
        counts["event_rows"] += 1
        counts["top_level_types" + ":" + (outer_tags[0] if outer_tags else "none")] += 1

        # A restart's DecisionTimestamp is the annotated stoppage onset; its
        # EventTime is the restart action. Keeping both preserves dead-ball time.
        for child in children:
            if child.tag in RESTART_TYPES:
                decision = child.get("DecisionTimestamp")
                if decision:
                    add(_parse_time(decision), "dead_start", source_type=child.tag)
                play = next(child.iter("Play"), None)
                team = _team_code(play.get("Team") if play is not None else child.get("Team"),
                                  home_id, away_id)
                add(event_time, "restart", team=team, source_type=child.tag)
            elif child.tag in STOPPAGE_TYPES:
                add(event_time, "dead_start", source_type=child.tag)
            elif child.tag == "TacklingGame":
                possession_change = child.get("PossessionChange") == "true"
                winner_team = _team_code(child.get("WinnerTeam"), home_id, away_id)
                challenger_team = None
                if child.get("WinnerRole") == "withoutBallControl":
                    challenger_team = winner_team
                elif child.get("LoserRole") == "withoutBallControl":
                    challenger_team = _team_code(child.get("LoserTeam"), home_id, away_id)
                if possession_change:
                    add(event_time, "control", team=winner_team, source_type="TacklingGame_RE")
                else:
                    add(event_time, "challenge", challenger=challenger_team,
                        source_type="TacklingGame_CH")
            elif child.tag in ACTION_TYPES:
                team = _team_code(child.get("Team"), home_id, away_id)
                if team is not None:
                    add(event_time, "control", team=team, source_type=child.tag)

    priorities = {"dead_start": 0, "challenge": 1, "control": 2, "restart": 3}
    operations.sort(key=lambda row: (row["time"], priorities[row["kind"]], row["source_type"]))
    counts["frame_origin_minus_kickoff_seconds"] = origin - kickoff
    counts["event_rows_without_control_operation"] = counts["event_rows"] - sum(
        counts[f"ops_{kind}"] for kind in ("control", "challenge", "restart", "dead_start"))
    return operations, dict(counts)


def predict_sportec(operations: list[dict], period: np.ndarray, n_frames: int,
                    config: dict, fps: float = 25.0) -> np.ndarray:
    """Create H/A/D/X from event ops only, preserving period boundaries."""
    pred = np.full(n_frames, "X", dtype="<U1")
    grace = int(round(float(config["out_tail_grace_seconds"]) * fps))
    challenge_policy = config["nonchanging_duel"]
    ops_by_period: dict[int, list[dict]] = {}
    for op in operations:
        frame = int(op["frame"])
        p = int(period[frame])
        if p > 0:
            ops_by_period.setdefault(p, []).append(op)

    for p, ops in ops_by_period.items():
        indices = np.flatnonzero(period == p)
        if len(indices) == 0:
            continue
        lo, hi = int(indices[0]), int(indices[-1]) + 1
        current = None
        dead = False
        cursor = lo
        last_touch = None
        for op in ops:
            frame = min(max(int(op["frame"]), lo), hi - 1)
            if op["kind"] == "dead_start":
                if dead:
                    continue
                if current is not None and last_touch is not None and frame - last_touch <= grace:
                    pred[cursor:frame] = current
                else:
                    pred[cursor:frame] = "X"
                cursor, dead, current = frame, True, None
            elif op["kind"] == "restart":
                if dead:
                    pred[cursor:frame] = "D"
                    dead = False
                else:
                    pred[cursor:frame] = current if current is not None else "X"
                cursor = frame
                current = op["team"]
                if current is not None:
                    pred[frame] = current
                    last_touch = frame
            elif op["kind"] == "control":
                if dead:
                    pred[cursor:frame] = "D"
                    dead = False
                else:
                    pred[cursor:frame] = current if current is not None else "X"
                cursor = frame
                current = op["team"]
                if current is not None:
                    pred[frame] = current
                    last_touch = frame
            elif op["kind"] == "challenge":
                if dead:
                    continue
                pred[cursor:frame] = current if current is not None else "X"
                if challenge_policy == "literal" and op["challenger"] is not None:
                    pred[frame] = op["challenger"]
                    cursor = frame + 1
                else:
                    if current is not None:
                        pred[frame] = current
                    cursor = frame
                last_touch = frame
        if dead:
            pred[cursor:hi] = "D"
        # Deliberately leave the post-final-touch tail unknown, matching the
        # native dense-label rule's right-censoring behavior.
    return pred


def nested_whole_match_cv(event_root: Path, target_root: Path,
                          info_root: Path | None = None,
                          timestamp_origins: dict[str, str] | None = None) -> dict:
    """Nested leave-one-whole-match-out calibration on Sportec TRAIN only."""
    candidates = sportec_candidate_grid()
    info_root = info_root or (event_root.parent.parent / "raw")
    if timestamp_origins is None or set(timestamp_origins) != set(SPORTEC_MATCHES):
        raise ValueError("one verified first-frame timestamp origin is required per TRAIN match")
    matches = []
    for match in SPORTEC_MATCHES:
        target_path = target_root / "external_audit/sportec/processed" / f"{match}.npz"
        event_path = event_root / f"{match}_events.xml"
        info_path = info_root / f"DFL_02_01_matchinformation_DFL-COM-{('000002' if match == 'J03WOH' else '000001')}_DFL-MAT-{match}.xml"
        with np.load(target_path, allow_pickle=False) as data:
            labels, period, timestamps = data["labels"], data["period"], data["timestamps"]
            fps = float(data["fps"])
        operations, audit = load_sportec_operations(event_path, info_path, timestamps,
                                                    timestamp_origins[match])
        matches.append({"match": match, "labels": labels, "period": period,
                        "timestamps": timestamps, "fps": fps,
                        "operations": operations, "event_audit": audit})

    rows = []
    for outer in matches:
        inner = [m for m in matches if m["match"] != outer["match"]]
        inner_scores = []
        for config in candidates:
            per_match = []
            for validation in inner:
                pred = predict_sportec(validation["operations"], validation["period"],
                                       len(validation["labels"]), config, validation["fps"])
                score = metrics.evaluate_tracking(validation["labels"], pred,
                    validation["period"], validation["fps"], validation["timestamps"])
                per_match.append({"match": validation["match"], "spell_f1_mean": score["spell_f1_mean"],
                                  "macro_f1": score["macro_f1"],
                                  "reference_coverage": score["label_coverage"],
                                  "prediction_known_rate": float(np.mean(pred != "X")),
                                  "prediction_x_count": int(np.sum(pred == "X"))})
            inner_scores.append({"config": config, "matches": per_match,
                                 "mean_spell_f1": float(np.mean([r["spell_f1_mean"] for r in per_match])),
                                 "mean_macro_f1": float(np.mean([r["macro_f1"] for r in per_match]))})
        selected = max(inner_scores, key=lambda row: (
            row["mean_spell_f1"], row["mean_macro_f1"],
            row["config"]["nonchanging_duel"] == "hold",
            row["config"]["out_tail_grace_seconds"] == 3.0))
        pred = predict_sportec(outer["operations"], outer["period"], len(outer["labels"]),
                               selected["config"], outer["fps"])
        score = metrics.evaluate_tracking(outer["labels"], pred, outer["period"],
                                          outer["fps"], outer["timestamps"])
        score["prediction_known_rate"] = float(np.mean(pred != "X"))
        score["prediction_x_count"] = int(np.sum(pred == "X"))
        score["prediction_known_count"] = int(np.sum(pred != "X"))
        score["reference_frame_counts"] = {
            label: int(np.sum(outer["labels"] == label)) for label in ("H", "A", "D", "X")}
        rows.append({"outer_match": outer["match"], "inner_training_matches": [m["match"] for m in inner],
                     "selected_config": selected["config"], "inner_candidate_scores": inner_scores,
                     "outer_metrics": score, "event_audit": outer["event_audit"]})
    return {"study_id": "provider_touch_sportec_train_nested_v2",
            "selection_metric": "equal-match mean official spell_f1_mean; macro_f1 tie-break",
            "candidate_count": len(candidates), "outer_folds": rows,
            "prediction_coverage": "Reported as prediction_known_rate and prediction_x_count; official spell scoring fills X within reference spells, so spell gains are not interpreted without coverage and frame-macro diagnostics.",
            "interpretation": "Nested whole-match development calibration on previously exposed Sportec TRAIN only with first-observed-frame clock origins; not fresh confirmation, not full-provider transfer."}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_declaration(data_root: Path, source_root: Path) -> dict:
    """Seal exact TRAIN inputs and first-frame origins before scoring."""
    data_root, source_root = Path(data_root), Path(source_root)
    manifest_path = ROOT / "experiments/eval_set/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    target_hashes = {row["match"]: row["sha256"]
                     for row in manifest["sources"]["sportec"]["train"]}
    inputs, origin_evidence, origin_times = [], {}, {}
    for match in SPORTEC_MATCHES:
        event_path = data_root / f"external_audit/sportec/raw/events/{match}_events.xml"
        info_path = data_root / "external_audit/sportec/raw" / (
            f"DFL_02_01_matchinformation_DFL-COM-{('000002' if match == 'J03WOH' else '000001')}_DFL-MAT-{match}.xml")
        target_path = source_root / "external_audit/sportec/processed" / f"{match}.npz"
        sidecar_path = source_root / "external_audit/sportec/processed" / f"{match}.json"
        checks = ((event_path, "provider_event_xml", SPORTEC_EVENT_MD5[match]),
                  (info_path, "match_info_xml", SPORTEC_INFO_MD5[match]),
                  (target_path, "provider_reference_npz", target_hashes[match]),
                  (sidecar_path, "provider_source_receipt", None))
        for path, kind, expected in checks:
            if not path.is_file():
                raise FileNotFoundError(path)
            sha = _sha256(path)
            if kind in ("provider_event_xml", "match_info_xml") and hashlib.md5(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Figshare MD5 mismatch for {path}")
            if kind == "provider_reference_npz" and sha != expected:
                raise ValueError(f"Provider reference NPZ SHA-256 mismatch for {path}")
            inputs.append({"match": match, "kind": kind, "path": str(path.resolve()),
                           "bytes": path.stat().st_size, "sha256": sha})
        receipt = json.loads(sidecar_path.read_text())
        source = next(item for item in receipt["sources"] if "positions_raw" in item["name"])
        origin = SPORTEC_TRACKING_ORIGIN[match]
        if source["file_id"] != origin["file_id"] or source["md5"] != origin["raw_md5"]:
            raise ValueError(f"First-frame probe does not match receipt-listed raw file for {match}")
        origin_times[match] = origin["timestamp"]
        kickoff = receipt["kickoff"]
        origin_evidence[match] = {**origin, "kickoff_utc": kickoff,
            "origin_minus_kickoff_seconds": _parse_time(origin["timestamp"]) - _parse_time(kickoff),
            "source_url": f"https://ndownloader.figshare.com/files/{origin['file_id']}",
            "sidecar_sha256": _sha256(sidecar_path)}
    code_paths = ("provider_touch_calibration_v2.py", "provider_touch_calibration.py",
                  "metrics.py", "soccernet_dense_labels.py", "audit_dataset.py")
    code_hashes = {name: _sha256(ROOT / "experiments" / name) for name in code_paths}
    return {
        "study_id": "provider_touch_sportec_train_nested_v2",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does an event-derived owner timeline that holds ownership through explicitly non-changing Sportec duels outperform literal challenger assignment on Sportec's own BallPossession reference?",
        "scope": "Source-specific Sportec event-rule calibration on TRAIN only. Metrica targets are event-derived and SkillCorner interruption spans use a different target definition, so neither is used to select these parameters.",
        "cohort": {"provider": "sportec", "matches": list(SPORTEC_MATCHES),
                   "status": "previously exposed development TRAIN matches; no fresh confirmation"},
        "protocol": {
            "outer": "Leave one whole TRAIN match out; score the selected candidate once on the outer match.",
            "inner": "On each outer fold, score each fixed candidate separately on each of the other two whole TRAIN matches; select by equal-match mean spell_f1_mean, then mean macro_f1.",
            "selection_tie_break": "Prefer hold over literal, then 3-second over 0-second tail grace (the current-rule setting).",
            "target_use": "BallStatus/BallPossession labels are only read by the official scorer; they do not create event anchors, classify event types, or adjust time alignment.",
            "coverage": "Every outer score reports fixed reference support and prediction known/X coverage; official spell F1 can fill X and is read with the coverage diagnostics.",
            "fitting": "No learned model or normalization is fit; candidates are fixed rule configurations."},
        "mapping": {
            "controlled_events": "Play, OtherBallAction, BallClaiming, and TacklingGame with PossessionChange=true (new owner WinnerTeam).",
            "uncontrolled_opponent_event": "TacklingGame with PossessionChange=false; challenger is the team whose role is withoutBallControl.",
            "dead_start": "Restart DecisionTimestamp; otherwise Foul, Offside, FinalWhistle, GoalDisallowed, ShotWide, SuccessfulShot, or Goal event time.",
            "dead_end": "EventTime for ThrowIn, FreeKick, GoalKick, CornerKick, KickOff, or Penalty; team only from its own nested action annotation.",
            "time_alignment": "Event times are relative to the first observed tracking-frame T from the pinned Figshare positions XML, not planned or actual KickoffTime; map to nearest NPZ timestamp with a fixed maximum 0.5-second alignment distance.",
            "out_tail_grace": "Propagate the current owner to dead_start only when the last touch anchor is within the candidate grace; otherwise keep the gap X.",
            "unsupported_v1_dimensions": "CL run-length behavior and censored-touch-run control cannot be mapped because Sportec event rows do not form GAR contiguous touch runs."},
        "timestamp_origin_evidence": origin_evidence,
        "timestamp_origins_utc": origin_times,
        "candidates": sportec_candidate_grid(),
        "selection_metric": "experiments.metrics.evaluate_tracking spell_f1_mean; macro_f1 tie-break; report prediction coverage",
        "inputs": {"manifest_path": str(manifest_path.resolve()),
                   "manifest_sha256": _sha256(manifest_path), "files": inputs,
                   "code_sha256": code_hashes},
        "excluded_data": ["all Sportec selection/test matches", "all Metrica labels", "all SkillCorner labels", "SoccerNet valid/test", "organizer labels"],
        "artifacts": {"report": "experiments/results/provider_touch_sportec_train_nested_v2.json"},
    }


def verify_declaration(declaration: dict) -> None:
    for item in declaration["inputs"]["files"]:
        path = Path(item["path"])
        if (not path.is_file() or path.stat().st_size != item["bytes"] or
                _sha256(path) != item["sha256"]):
            raise ValueError(f"Frozen input changed or unavailable: {item['path']}")
    for name, expected in declaration["inputs"]["code_sha256"].items():
        if _sha256(ROOT / "experiments" / name) != expected:
            raise ValueError(f"Frozen code changed after declaration: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--source-root", type=Path,
                        default=Path(os.environ.get("MTAH_SOURCE_CACHE", "/workspace/mtah-data")))
    parser.add_argument("--declaration", type=Path, default=ROOT / "experiments/results/provider_touch_sportec_train_nested_v2_declaration.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/provider_touch_sportec_train_nested_v2.json")
    args = parser.parse_args()
    if args.command == "freeze":
        if args.declaration.exists():
            raise FileExistsError(f"Declaration already exists: {args.declaration}")
        declaration = build_declaration(args.data_dir, args.source_root)
        args.declaration.parent.mkdir(parents=True, exist_ok=True)
        args.declaration.write_text(json.dumps(declaration, indent=2) + "\n")
        print(json.dumps({"study_id": declaration["study_id"],
                          "frozen_at_utc": declaration["frozen_at_utc"],
                          "input_count": len(declaration["inputs"]["files"]),
                          "candidate_count": len(declaration["candidates"])}, indent=2))
        return
    declaration = json.loads(args.declaration.read_text())
    verify_declaration(declaration)
    report = nested_whole_match_cv(
        args.data_dir / "external_audit/sportec/raw/events", args.source_root,
        args.data_dir / "external_audit/sportec/raw",
        declaration["timestamp_origins_utc"])
    report["declaration_sha256"] = _sha256(args.declaration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"study_id": report["study_id"], "output": str(args.output),
                      "outer_folds": len(report["outer_folds"])}, indent=2))


if __name__ == "__main__":
    main()
