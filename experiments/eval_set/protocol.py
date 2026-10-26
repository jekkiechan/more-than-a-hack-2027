"""Whole-match/source protocol using the unchanged official possession scorer.

This creates an honest development benchmark, not a new independent test set:
every available external match has already been used for training or evaluation.
No organizer assets or public-window labels are read here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import dead_transfer, supervised  # noqa: E402
from experiments.metrics import evaluate_tracking  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest():
    provenance = {
        "sportec": {"labels": "provider BallStatus/BallPossession", "quality": "provider_reference",
                    "shortcut": "Ball coordinates frozen during dead play"},
        "metrica": {"labels": "derived H/A/D from human event stream; inferred dead intervals",
                    "quality": "event_derived_silver", "shortcut": "Ball mostly blanked only during dead play"},
        "skillcorner": {"labels": "provider team possession + interruption-derived D cores; 1 s X margins",
                        "quality": "event_derived_silver", "shortcut": "Broadcast cuts reduce player/ball visibility"},
    }
    sources = {}
    for source, split in dead_transfer.SPLITS.items():
        def describe(path, role):
            if not path.exists():
                raise ValueError(f"Missing prepared external match: {path}")
            return {"match": path.stem, "group_id": path.stem,
                    "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                    "prior_exposure": "previously_used_for_training" if role == "train"
                    else "previously_evaluated_development_match"}
        sources[source] = {**provenance[source],
                           "train": [describe(p, "train") for p in split["train"]],
                           "selection": [describe(p, "selection") for p in split["eval"]]}
    folds = [{"name": "pooled", "train_sources": list(sources), "selection_sources": list(sources),
              "interpretation": "Within-provider, whole-match development comparison"}]
    folds += [{"name": f"leave_out_{source}", "train_sources": [s for s in sources if s != source],
               "selection_sources": [source],
               "interpretation": "Unseen-source transfer on historically evaluated development matches"}
              for source in sources]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Predeclare temporal D and LightGBM learning-rate development comparisons",
        "sources": sources, "folds": folds,
        "confirmation": {"matches": [], "status": "unavailable_no_unseen_external_match",
                         "reason": "Every available source/match has already influenced earlier training or evaluation; reshuffling cannot create a fresh test.",
                         "required_next_data": "Independent externally labelled tactical/broadcast match with real live-play ball occlusions and representative stoppages"},
        "historical_tests_excluded": ["J03WPY", "J03WR9", "J03WQQ"],
        "organizer_game": {"training": False, "selection": False, "manual_labels": False,
                           "public_5min_use": "One pipeline check after external selection and configuration freeze; no tuning or winner selection"},
        "metric": {"entrypoint": "experiments.metrics.evaluate_tracking",
                   "official_sha256": sha256(ROOT / "evaluation/possession_metrics.py"),
                   "adapter_sha256": sha256(ROOT / "experiments/metrics.py"),
                   "primary": "spell_f1_mean: official greedy one-to-one H/A matching averaged over IoU .3/.4/.5/.6/.7",
                   "secondary": ["macro_f1", "D precision/recall/F1", "D boundary precision/recall and latency at .5 s and 1 s"],
                   "aggregation": "Mean of match scores within source, then equal mean across sources; retain all per-match/source results",
                   "resampling": "Paired whole-match bootstrap stratified by source; never bootstrap correlated frames"},
        "controls": {"D_features": "player-only geometry/motion with observation masks; no ball-derived D cue",
                     "team_attribution": "Fixed pooled multisource HGB conditional H/A; never refit during D-head comparison",
                     "decoder_switch_cost": 3,
                     "skillcorner_min_visible": dead_transfer.MIN_VISIBLE,
                     "skillcorner_coverage": "Mask positions flagged unobserved; X for <12 visible players in both training and evaluation; report retained D/live coverage separately",
                     "temporal_context": "Centered offline windows; never cross match, half or timestamp discontinuity; fit normalization only to training folds",
                     "candidate_policy": "Freeze finite candidate set, epoch counts, seed and learning-rate budgets before selection; add no candidates after public check"},
        "limitations": ["Development comparisons are not fresh confirmation.",
                        "Only one Sportec and one Metrica selection match; uncertainty across providers cannot be estimated reliably.",
                        "Metrica/SkillCorner labels are event-derived silver, not adjudicated frame gold.",
                        "Visible-player filtering changes the SkillCorner population and can remove much of D; report coverage.",
                        "Synthetic missingness probes sensitivity and does not reproduce real broadcast occlusion."]}
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    sources = manifest["sources"]
    train, selection = [], []
    for source, values in sources.items():
        train += [(source, m["group_id"]) for m in values["train"]]
        selection += [(source, m["group_id"]) for m in values["selection"]]
    train_ids, select_ids = [m for _, m in train], [m for _, m in selection]
    if len(set(train_ids)) != len(train_ids) or len(set(select_ids)) != len(select_ids):
        raise ValueError("Duplicate match/group IDs across sources")
    if set(train_ids) & set(select_ids):
        raise ValueError("A match/group appears in both training and selection")
    for fold in manifest["folds"]:
        training, targets = set(fold["train_sources"]), set(fold["selection_sources"])
        if not training or not targets or not (training | targets) <= sources.keys():
            raise ValueError("Unknown or empty source fold")
        if fold["name"].startswith("leave_out_") and training & targets:
            raise ValueError("Unseen-source fold includes target source in training")
    if manifest["confirmation"]["matches"]:
        raise ValueError("Fresh confirmation requires a separate provenance-reviewed protocol")
    return True


def summarize_d_boundaries(labels, pred, period, fps, timestamps=None):
    """Official transition matching after collapsing H/A to live.

    Returns JSON-safe diagnostics; excludes X/half/time-gap boundaries, matches
    entering/leaving D separately, and expresses timing in seconds. Missing D
    boundaries stay None rather than inventing a timing error.
    """
    labels, pred = np.asarray(labels).astype(str), np.asarray(pred).astype(str)
    if not np.isin(labels, ["H", "A", "D", "X"]).all() or not np.isin(pred, ["H", "A", "D", "X"]).all():
        raise ValueError("Labels must be H/A/D/X")
    live_truth = np.where(np.isin(labels, ["H", "A"]), "H", labels)
    live_pred = np.where(np.isin(pred, ["H", "A"]), "H", pred)
    result = evaluate_tracking(live_truth, live_pred, period=period, fps=fps, timestamps=timestamps)
    summaries = {}
    for tolerance, values in result["transitions"].items():
        item = dict(values)
        for name in ("pairs", "missed", "spurious"):
            item.pop(name, None)
        for name in ("mean_latency", "median_latency", "mean_abs_latency", "p90_abs_latency"):
            value = item.pop(name)
            item[f"{name}_seconds"] = value / fps if value is not None else None
        item["tolerance_seconds"] = int(tolerance) / fps
        summaries[str(round(int(tolerance) / fps, 6))] = item
    return summaries


def audit_match(path, source):
    data = supervised.load(path)
    labels = data["labels"].astype(str)
    graded = np.isin(labels, ["H", "A", "D"])
    ball_seen = np.asarray(data.get("ball_observed", np.isfinite(data["ball"]).all(axis=1)), dtype=bool)
    player_seen = np.asarray(data.get("player_observed", np.isfinite(data["players"]).all(axis=2)), dtype=bool)
    counts = player_seen.sum(axis=1)
    keep = counts >= dead_transfer.MIN_VISIBLE if source == "skillcorner" else np.ones(len(labels), dtype=bool)
    dt = np.diff(data["timestamps"], prepend=np.nan)
    distance = np.linalg.norm(np.diff(data["ball"], axis=0, prepend=np.full_like(data["ball"][:1], np.nan)), axis=1)
    valid_motion = (np.r_[False, data["period"][1:] == data["period"][:-1]] & (dt > 0)
                    & (dt <= 1.5 / float(data["fps"])) & ball_seen & np.r_[False, ball_seen[:-1]])
    stationary = valid_motion & (distance <= .05 * dt)
    def conditional(mask):
        n = int(mask.sum())
        retained = mask & keep
        return {"frames": n, "retained_frames": int(retained.sum()),
                "retained_fraction": float(keep[mask].mean()) if n else None,
                "ball_missing_fraction": float((~ball_seen[mask]).mean()) if n else None,
                "mean_visible_players": float(counts[mask].mean()) if n else None,
                "stationary_fraction_given_observed_valid_motion": float(stationary[mask & valid_motion].mean())
                if np.any(mask & valid_motion) else None}
    return {"source": source, "match": path.stem, "frames": len(labels), "fps": float(data["fps"]),
            "label_quality": "provider_reference" if source == "sportec" else "event_derived_silver",
            "graded_coverage": float(graded.mean()),
            "total_retained_graded_fraction": float((graded & keep).sum() / max(graded.sum(), 1)),
            "live": conditional(np.isin(labels, ["H", "A"])), "D": conditional(labels == "D"),
            "stationary_threshold_metres_per_second": .05}


def paired_source_bootstrap(baseline, candidate, metric="spell_f1", repeats=2000, seed=42):
    """Paired source-balanced whole-match deltas, conditional on these sources.

    Rows require `match`, `eval_source`, and the metric. Call separately for
    pooled/each LOSO fold and for different corruptions. No frame-level CI.
    """
    def indexed(rows):
        indexed_rows = {(r["eval_source"], r["match"]): r for r in rows}
        if len(indexed_rows) != len(rows):
            raise ValueError("Duplicate source/match: filter candidates/folds first")
        return indexed_rows
    old, new = indexed(baseline), indexed(candidate)
    if not old or old.keys() != new.keys():
        raise ValueError("Paired reports must contain identical nonempty source/match sets")
    groups = {}
    for key in old:
        delta = float(new[key][metric]) - float(old[key][metric])
        if not np.isfinite(delta):
            raise ValueError("Nonfinite score delta")
        groups.setdefault(key[0], []).append(delta)
    if repeats < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    rng = np.random.default_rng(seed)
    samples = np.mean([np.mean(rng.choice(values, size=(repeats, len(values)), replace=True), axis=1)
                       for values in groups.values()], axis=0)
    return {"metric": metric, "candidate_minus_baseline": float(np.mean([np.mean(v) for v in groups.values()])),
            "paired_match_bootstrap_95_interval": np.quantile(samples, [.025, .975]).tolist(),
            "per_source_delta": {s: float(np.mean(v)) for s, v in groups.items()},
            "matches_per_source": {s: len(v) for s, v in groups.items()}, "repeats": repeats, "seed": seed,
            "interpretation": "Conditional on these historical development matches/sources; singleton sources cannot contribute sampling uncertainty and this is not unseen-provider uncertainty."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "audit"])
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        result = build_manifest()
        output = args.output or args.manifest
    else:
        manifest = json.loads(args.manifest.read_text())
        validate_manifest(manifest)
        rows = []
        for source, values in manifest["sources"].items():
            for match in values["selection"]:
                path = ROOT / match["path"]
                if sha256(path) != match["sha256"]:
                    raise ValueError(f"Dataset changed: {path}")
                rows.append(audit_match(path, source))
        result = {"manifest_sha256": sha256(args.manifest), "scope": "Historical selection matches only", "matches": rows}
        output = args.output or Path(__file__).with_name("shortcut_audit.json")
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
