"""Train-only directed H/A turnover event head with bounded Viterbi costs.

The head predicts an exact directed H/A change over adjacent eligible 5 Hz
samples from the frozen ensemble trajectory and local tracking context. Its
natural-prevalence log-likelihood ratio changes only the H<->A edge penalty;
the frozen H/A/D emission probabilities and all D/live transition penalties
remain unchanged.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import (  # noqa: E402
    factorized_had,
    learned_temporal as lt,
    metrics,
    objective_ensemble as oe,
    preparation as prep,
    spell_diagnostics,
    supervised,
)


STUDY_ID = "turnover_decoder_v1"
OUT_RELATIVE = Path("experiments") / STUDY_ID
DECLARATION = ROOT / "experiments/results/turnover_decoder_declaration.json"
FIT_RECEIPT = ROOT / "experiments/results/turnover_decoder_fit_receipt.json"
SCORE_RECEIPT = ROOT / "experiments/results/turnover_decoder_score_receipt.json"
ARTIFACT_RECEIPT = ROOT / "experiments/results/turnover_decoder_artifacts.json"
PROVIDERS = ("sportec", "metrica", "skillcorner")
SEEDS = (2026, 2027)
VIEWS = (("clean", "clean"), ("feed_like", "corrupt_1"), ("feed_like", "corrupt_2"))
CLASS_SOURCE, CLASS_DEST, CLASS_DEAD = 0, 1, 2
BASE_COST = 3.0
COST_MIN, COST_MAX = 1.5, 4.5
LOGISTIC_C = 0.1
MAX_ITER = 300
PROBABILITY_EPS = 1e-6
IOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
TRAIN_CACHE_ROOT = Path("experiments/objective_alignment_v1")
EVAL_CACHE_ROOT = Path("experiments/seed_gap_context_v1")
MEMBER_ROOT = Path("experiments/objective_alignment_v1")
ENSEMBLE_ROOT = Path("experiments/objective_ensemble_v1")
SOURCE_FILES = (
    "experiments/turnover_decoder.py",
    "experiments/factorized_had.py",
    "experiments/learned_temporal.py",
    "experiments/metrics.py",
    "experiments/objective_ensemble.py",
    "experiments/preparation.py",
    "experiments/spell_diagnostics.py",
    "experiments/supervised.py",
    "experiments/eval_set/protocol.py",
    "experiments/eval_set/manifest.json",
    "experiments/baseline_manifest.json",
    "experiments/model_decision_policy.json",
    "experiments/tests/test_turnover_decoder.py",
)

FEATURE_NAMES = (
    "p_source_prev", "p_destination_prev", "p_dead_prev",
    "p_source_now", "p_destination_now", "p_dead_now",
    "edge_logodds_prev", "edge_logodds_now", "edge_logodds_change",
    "control_advantage_now", "control_advantage_change",
    "contact_impulse_advantage_now", "contact_impulse_advantage_change",
    "nearest_distance_advantage_now", "nearest_distance_advantage_change",
    "within5m_advantage_now", "visible_player_advantage_now",
    "ball_speed_now", "ball_available_now",
)
TRACK_FEATURES = (
    "physical_control_evidence",
    "physical_contact_impulse",
    "distance1",
    "within5m",
    "visible_players",
)
SCORE_METRICS = (
    "spell_f1", "macro_f1", "weighted_70_30", "d_f1", "d_precision",
    "d_recall", "short_spell_recall",
)


class StudyError(RuntimeError):
    """Raised when a frozen input, model, or score contract fails."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def data_root(path: Path | None = None) -> Path:
    chosen = path or (Path(os.environ["MTAH_DATA_DIR"])
                      if os.environ.get("MTAH_DATA_DIR") else ROOT / "data")
    return chosen.expanduser().resolve()


def cells() -> list[dict[str, str]]:
    rows = []
    manifest = json.loads((ROOT / "experiments/eval_set/manifest.json").read_text())
    expected = {
        "sportec": {"J03WOY"},
        "metrica": {"metrica_3"},
        "skillcorner": {"skillcorner_had_1996435", "skillcorner_had_1996436",
                        "skillcorner_had_2006229"},
    }
    seen = {provider: set() for provider in expected}
    for provider in PROVIDERS:
        for item in manifest["sources"][provider]["selection"]:
            if item["match"] in expected[provider]:
                seen[provider].add(item["match"])
                for view, replicate in VIEWS:
                    rows.append(dict(source=provider, match=item["match"],
                                     view=view, replicate=replicate))
    if seen != expected or len(rows) != 15:
        raise StudyError(f"Evaluation cohort changed: {seen}")
    return rows


def split_for(provider: str) -> dict[str, Any]:
    return prep.split_plan(fold=f"leave_out_{provider}")


def cache_path(root: Path, relative: Path) -> Path:
    return root.joinpath(*relative.parts)


def _source_file_facts() -> list[dict[str, str]]:
    return [{"path": name, "sha256": sha256_file(ROOT / name)} for name in SOURCE_FILES]


def _match_train_cache_path(match: str, root: Path) -> Path:
    return root / TRAIN_CACHE_ROOT / f"train_{match}.joblib"


def _record_paths_for_cell(cell: dict[str, str], root: Path) -> dict[str, Path]:
    stem = f"{cell['match']}_{cell['view']}_{cell['replicate']}"
    return {
        "inputs_labels_record": root / EVAL_CACHE_ROOT / f"{stem}.joblib",
        "candidate_ensemble": root / ENSEMBLE_ROOT / f"candidate_{stem}.npz",
    }


def _member_path(root: Path, cell: dict[str, str], seed: int) -> Path:
    return root / MEMBER_ROOT / (
        f"{cell['source']}_{cell['match']}_{cell['view']}_{cell['replicate']}_combined_{seed}.npz")


def _load_member_probabilities(root: Path, cell: dict[str, str]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    members: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        path = _member_path(root, cell, seed)
        if not path.is_file():
            raise StudyError(f"Missing frozen member probabilities: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if not {"probability", "prediction"}.issubset(archive.files):
                raise StudyError(f"Malformed member probability archive: {path}")
            members[seed] = np.asarray(archive["probability"])
    ensemble = oe.blend_probabilities(members[2026], members[2027])
    return ensemble, members


def _verify_eval_cell_snapshot(root: Path, cell: dict[str, str]) -> dict[str, Any]:
    paths = _record_paths_for_cell(cell, root)
    for path in paths.values():
        if not path.is_file():
            raise StudyError(f"Missing sealed evaluation artifact: {path}")
    candidate = paths["candidate_ensemble"]
    member_ensemble, members = _load_member_probabilities(root, cell)
    with np.load(candidate, allow_pickle=False) as stored:
        if not {"probability", "prediction"}.issubset(stored.files):
            raise StudyError(f"Malformed saved ensemble candidate: {candidate}")
        saved_probability = stored["probability"]
        saved_prediction = stored["prediction"]
    if not np.array_equal(member_ensemble, saved_probability):
        raise StudyError(f"Member blend does not reproduce candidate archive: {cell}")
    job_path = paths["inputs_labels_record"]
    cache = joblib.load(job_path)
    record = cache.get("record")
    if not isinstance(record, dict) or record.get("source") != cell["source"] or record.get("match") != cell["match"]:
        raise StudyError(f"Evaluation record identity mismatch: {cell}")
    if len(cache.get("labels", ())) != len(saved_probability) or len(cache["inputs"]["period"]) != len(saved_probability):
        raise StudyError(f"Evaluation inputs, labels and probabilities do not align: {cell}")
    decoded = supervised.decode(saved_probability, supervised.model_periods(cache["inputs"]),
                                float(cache["inputs"]["fps"]), BASE_COST)
    if not np.array_equal(decoded.astype("U1"), saved_prediction):
        raise StudyError(f"Saved cost-3 decode does not match archived ensemble probabilities: {cell}")
    return {
        "cell": cell,
        "inputs_labels_record": str(paths["inputs_labels_record"].relative_to(root)),
        "inputs_labels_record_sha256": sha256_file(job_path),
        "candidate_ensemble": str(candidate.relative_to(root)),
        "candidate_ensemble_sha256": sha256_file(candidate),
        "candidate_probability_sha256": sha256_array(saved_probability),
        "candidate_prediction_sha256": sha256_array(saved_prediction),
        "member_probabilities": {
            str(seed): {"path": str(_member_path(root, cell, seed).relative_to(root)),
                        "sha256": sha256_file(_member_path(root, cell, seed)),
                        "probability_sha256": sha256_array(members[seed])}
            for seed in SEEDS
        },
        "frames": int(len(saved_probability)),
    }


def _fold_inputs(root: Path, provider: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    plan = split_for(provider)
    training_by_match = {item["match"]: item for item in plan["train"]}
    if not training_by_match:
        raise StudyError(f"Empty train fold for {provider}")
    records = []
    training_facts: dict[str, str] = {}
    for match, item in sorted(training_by_match.items()):
        path = _match_train_cache_path(match, root)
        if not path.is_file():
            raise StudyError(f"Missing verified training cache: {path}")
        if sha256_file(path) == "":
            raise StudyError(f"Could not hash training cache: {path}")
        outer = joblib.load(path)
        if outer.get("match") != match or outer.get("source") != item["source"]:
            raise StudyError(f"Training cache identity differs from split plan: {match}")
        if outer.get("input_sha256") != item["sha256"]:
            raise StudyError(f"Training cache raw input seal differs from split plan: {match}")
        if not {"full_y", "full_period", "records", "fps"}.issubset(outer):
            raise StudyError(f"Training cache lacks native target or feature records: {match}")
        record = outer["records"].get("clean")
        if not isinstance(record, dict) or record.get("match") != match or record.get("source") != item["source"]:
            raise StudyError(f"Clean training record identity mismatch: {match}")
        model_period = np.asarray(outer["full_period"])
        # The cache stores model_periods(inputs), built from raw time metadata.
        # Do not infer blocks from X; target masks affect edge eligibility only.
        if model_period.shape != np.asarray(outer["full_y"]).shape:
            raise StudyError(f"Native target/time-block shapes differ: {match}")
        if not np.array_equal(record["period"], model_period[record["indices"]]):
            raise StudyError(f"Sampled record blocks disagree with native time blocks: {match}")
        training_facts[match] = sha256_file(path)
        records.append({
            "source": item["source"], "match": match, "record": record,
            "full_y": np.asarray(outer["full_y"], dtype=np.int8),
            "full_period": model_period,
            "fps": float(outer["fps"]), "input_sha256": item["sha256"],
            "cache_sha256": training_facts[match],
        })
    if {r["source"] for r in records} != {item["source"] for item in plan["train"]}:
        raise StudyError(f"Training provider groups differ from split: {provider}")
    return plan, records, training_facts


def _checkpoint_facts(root: Path, provider: str) -> tuple[dict[int, dict], dict[str, Any]]:
    model_root = root / TRAIN_CACHE_ROOT
    manifest_path = model_root / "checkpoints_frozen.json"
    predeclared_path = model_root / "predeclared.json"
    if not manifest_path.is_file() or not predeclared_path.is_file():
        raise StudyError("Missing objective-alignment checkpoint receipts")
    frozen = json.loads(manifest_path.read_text())
    previous = json.loads(predeclared_path.read_text())
    rows = {row["file"]: row for row in frozen.get("new_checkpoints", [])}
    models = {}
    facts = {}
    expected_sources = previous["training_sources"][provider]
    expected_matches = previous["training_matches"][provider]
    plan = split_for(provider)
    actual_matches = sorted(item["match"] for item in plan["train"])
    plan_sources = sorted({item["source"] for item in plan["train"]})
    if sorted(expected_sources) != plan_sources or sorted(expected_matches) != actual_matches:
        raise StudyError(f"Frozen backbone fold differs from split plan: {provider}")
    for seed in SEEDS:
        filename = f"without_{provider}_combined_{seed}.joblib"
        entry = rows.get(filename)
        model_path = model_root / filename
        receipt_path = model_root / f"{filename.removesuffix('.joblib')}.json"
        if entry is None or not model_path.is_file() or not receipt_path.is_file():
            raise StudyError(f"Missing frozen fold checkpoint or receipt: {filename}")
        actual_sha = sha256_file(model_path)
        receipt_sha = sha256_file(receipt_path)
        if actual_sha != entry.get("sha256") or receipt_sha != entry.get("receipt_sha256"):
            raise StudyError(f"Frozen fold checkpoint checksum mismatch: {filename}")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("heldout") != provider or receipt.get("seed") != seed or
                receipt.get("objective") != "combined" or receipt.get("arm") != "combined" or
                sorted(receipt.get("sources", [])) != sorted(expected_sources) or
                sorted(receipt.get("training_matches", [])) != actual_matches):
            raise StudyError(f"Frozen fold fit receipt mismatch: {filename}")
        model = joblib.load(model_path)
        if model.get("seed") != seed or model.get("feature_names") is None:
            raise StudyError(f"Frozen model metadata mismatch: {filename}")
        models[seed] = model
        facts[str(seed)] = {"model": str(model_path.relative_to(root)), "sha256": actual_sha,
                            "fit_receipt": str(receipt_path.relative_to(root)),
                            "fit_receipt_sha256": receipt_sha}
    return models, {"checkpoint_manifest_sha256": sha256_file(manifest_path),
                    "backbone_declaration_sha256": sha256_file(predeclared_path),
                    "models": facts}


def _edge_pairs(record: dict[str, Any]) -> np.ndarray:
    indices = np.asarray(record["indices"], dtype=np.int64)
    period = np.asarray(record["period"])
    fps = float(record["fps"])
    stride = int(round(fps / 5))
    if stride < 1 or not np.isclose(stride * 5, fps):
        raise ValueError("Expected 5 Hz samples on an integer native-frame stride")
    if len(indices) != len(period):
        raise ValueError("Record indices and periods must align")
    if len(indices) < 2:
        return np.empty((0, 2), dtype=np.int64)
    valid = ((np.diff(period) == 0) & (np.diff(indices) == stride))
    right = np.flatnonzero(valid) + 1
    return np.column_stack((right - 1, right)).astype(np.int64)


def _feature_index(record: dict[str, Any]) -> dict[str, int]:
    names = list(record["names"])
    index = {name: i for i, name in enumerate(names)}
    required = ["ball_speed", "ball_available"]
    for team in ("H", "A"):
        required.extend(f"{team}_{name}" for name in TRACK_FEATURES[:2])
        required.extend((f"{team}_distance1", f"{team}_within5m", f"{team}_visible_players"))
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"Training/evaluation feature schema is missing: {missing}")
    return index


def edge_features(record: dict[str, Any], probabilities_5hz: np.ndarray,
                  pairs: np.ndarray | None = None) -> np.ndarray:
    """Build oriented H/A edge features; no labels are read here."""
    x = np.asarray(record["x"], dtype=np.float64)
    p = np.asarray(probabilities_5hz, dtype=np.float64)
    if x.ndim != 2 or p.shape != (len(x), 3):
        raise ValueError("Expected aligned (n, feature) x and (n, 3) probabilities")
    if not np.isfinite(p).all() or (p < 0).any() or np.any(p.sum(axis=1) <= 0):
        raise ValueError("Probabilities must be finite, nonnegative and nonempty")
    pairs = _edge_pairs(record) if pairs is None else np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or (pairs.size and (pairs.min() < 0 or pairs.max() >= len(x))):
        raise ValueError("Malformed sample edge pairs")
    if len(pairs) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    at, nxt = pairs[:, 0], pairs[:, 1]
    index = _feature_index(record)
    eps = PROBABILITY_EPS

    def clip_prob(values: np.ndarray) -> np.ndarray:
        return np.clip(values, eps, 1.0)

    def oriented(source: int, destination: int) -> np.ndarray:
        ps0, pd0, dead0 = (clip_prob(p[at, source]), clip_prob(p[at, destination]),
                           clip_prob(p[at, CLASS_DEAD]))
        ps1, pd1, dead1 = (clip_prob(p[nxt, source]), clip_prob(p[nxt, destination]),
                           clip_prob(p[nxt, CLASS_DEAD]))
        odds0 = np.log(pd0) - np.log(ps0)
        odds1 = np.log(pd1) - np.log(ps1)

        def team_value(team: int, feature: str, rows: np.ndarray) -> np.ndarray:
            return x[rows, index[f"{'H' if team == 0 else 'A'}_{feature}"]]

        def adv(feature: str, now_rows: np.ndarray = nxt,
                before_rows: np.ndarray = at, reverse_distance: bool = False) -> tuple[np.ndarray, np.ndarray]:
            src_now = team_value(source, feature, now_rows)
            dst_now = team_value(destination, feature, now_rows)
            src_before = team_value(source, feature, before_rows)
            dst_before = team_value(destination, feature, before_rows)
            if reverse_distance:
                now = src_now - dst_now
                before = src_before - dst_before
            else:
                now = dst_now - src_now
                before = dst_before - src_before
            return now, now - before

        control, control_delta = adv("physical_control_evidence")
        impulse, impulse_delta = adv("physical_contact_impulse")
        distance, distance_delta = adv("distance1", reverse_distance=True)
        within5, _ = adv("within5m")
        visible, _ = adv("visible_players")
        ball_speed = x[nxt, index["ball_speed"]]
        ball_available = x[nxt, index["ball_available"]]
        return np.column_stack((
            ps0, pd0, dead0, ps1, pd1, dead1,
            odds0, odds1, odds1 - odds0,
            control, control_delta, impulse, impulse_delta,
            distance, distance_delta, within5, visible, ball_speed, ball_available,
        ))

    result = np.vstack((oriented(CLASS_SOURCE, CLASS_DEST),
                        oriented(CLASS_DEST, CLASS_SOURCE)))
    if result.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("Feature schema width changed")
    result[~np.isfinite(result)] = np.nan
    return result


def eligible_training_events(record: dict[str, Any], full_y: np.ndarray,
                             full_period: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return paired sample rows and directed labels using native eligibility.

    Time blocks come only from tracking period/timestamps. Native target masks
    exclude an interval if any row is D or X; X never creates a new block.
    Both possible orientations are represented for each valid interval, with
    natural event prevalence and no event-class balancing or undersampling.
    """
    y = np.asarray(record["y"], dtype=np.int8)
    indices = np.asarray(record["indices"], dtype=np.int64)
    sampled_period = np.asarray(record["period"])
    full_y = np.asarray(full_y, dtype=np.int8)
    full_period = np.asarray(full_period)
    if full_y.shape != full_period.shape or (indices.size and indices.max() >= len(full_y)):
        raise ValueError("Native training labels/time blocks do not align")
    pairs = _edge_pairs(record)
    eligible_pairs, target_ha, target_ah = [], [], []
    for left, right in pairs:
        start, stop = int(indices[left]), int(indices[right])
        if stop <= start or not np.all(full_period[start:stop + 1] == full_period[start]):
            continue
        # Do not use X as a segmentation rule. An interval touching X or D is
        # omitted from the event target, while the next tracking-contiguous
        # interval remains eligible independently.
        native = full_y[start:stop + 1]
        if native.size != stop - start + 1 or not np.isin(native, (0, 1)).all():
            continue
        if int(y[left]) not in (0, 1) or int(y[right]) not in (0, 1):
            raise ValueError("Sampled and native live target masks disagree")
        if int(full_y[start]) != int(y[left]) or int(full_y[stop]) != int(y[right]):
            raise ValueError("Sampled and native team targets disagree")
        eligible_pairs.append((left, right))
        source_to_dest = int(y[left]) == 0 and int(y[right]) == 1
        dest_to_source = int(y[left]) == 1 and int(y[right]) == 0
        target_ha.append(int(source_to_dest))
        target_ah.append(int(dest_to_source))
    return (np.asarray(eligible_pairs, dtype=np.int64).reshape(-1, 2),
            np.asarray(target_ha + target_ah, dtype=np.int8))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column]) & np.isfinite(weights) & (weights > 0)
        if not finite.any():
            continue
        x, w = values[finite, column], weights[finite]
        order = np.argsort(x, kind="mergesort")
        x, w = x[order], w[order]
        result[column] = x[min(int(np.searchsorted(np.cumsum(w), w.sum() * .5, side="left")), len(x)-1)]
    return result


def _balanced_source_match_weights(blocks: list[dict[str, Any]]) -> np.ndarray:
    sources = sorted({row["source"] for row in blocks})
    matches_by_source = {source: sum(row["source"] == source for row in blocks)
                         for source in sources}
    weights = []
    for row in blocks:
        count = len(row["target"])
        if count == 0:
            weights.append(np.empty(0, dtype=np.float64))
            continue
        mass = 1.0 / (len(sources) * matches_by_source[row["source"]])
        weights.append(np.full(count, mass / count, dtype=np.float64))
    combined = np.concatenate(weights) if weights else np.empty(0)
    if combined.size:
        combined /= combined.sum()
    return combined


def build_training_blocks(root: Path, provider: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create OOF-by-provider-fold target/feature blocks; no eval labels read."""
    plan, records, cache_facts = _fold_inputs(root, provider)
    models, checkpoint_facts = _checkpoint_facts(root, provider)
    blocks = []
    prediction_facts = {}
    for row in records:
        record = row["record"]
        if not np.array_equal(record["names"], models[2026]["feature_names"]):
            raise StudyError(f"Feature schema mismatch in {row['match']}")
        if models[2026]["feature_names"] != models[2027]["feature_names"]:
            raise StudyError(f"Seed feature schemas differ in fold {provider}")
        p0 = factorized_had.predict(record, models[2026])
        p1 = factorized_had.predict(record, models[2027])
        p = (p0.astype(np.float64) + p1.astype(np.float64)) / 2.0
        pairs, target = eligible_training_events(record, row["full_y"], row["full_period"])
        design = edge_features(record, p, pairs)
        if len(design) != len(target):
            raise StudyError(f"Training event feature/target rows differ: {row['match']}")
        blocks.append({"source": row["source"], "match": row["match"],
                       "features": design, "target": target,
                       "eligible_intervals": int(len(pairs)),
                       "directed_edges": int(len(target)),
                       "directed_events": int(target.sum()),
                       "full_y_sha256": sha256_array(row["full_y"]),
                       "full_period_sha256": sha256_array(row["full_period"]),
                       "record_x_sha256": sha256_array(record["x"]),
                       "ensemble_train_probability_sha256": sha256_array(p)})
        prediction_facts[row["match"]] = {
            "seed_probability_sha256": {"2026": sha256_array(p0), "2027": sha256_array(p1)},
            "ensemble_probability_sha256": sha256_array(p),
            "training_is_in_sample_for_backbone": True,
            "backbone_fold": provider,
        }
    useful = [row for row in blocks if len(row["target"]) > 0]
    if len(useful) != len(blocks):
        raise StudyError(f"At least one training match has no eligible edge targets for {provider}")
    return blocks, {"fold": plan, "training_cache_sha256": cache_facts,
                    "checkpoint_facts": checkpoint_facts,
                    "train_probability_facts": prediction_facts}


def _fit_transform(features: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    median = _weighted_median(features, weights)
    imputed = np.where(np.isfinite(features), features, median[None, :])
    mean = np.average(imputed, axis=0, weights=weights)
    variance = np.average((imputed - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    transformed = ((imputed - mean) / scale).astype(np.float64, copy=False)
    return transformed, {"median": median, "mean": mean, "scale": scale}


def fit_head(blocks: list[dict[str, Any]], provider: str) -> dict[str, Any]:
    """Fit the one frozen regularized event head for a held-out provider."""
    blocks = [row for row in blocks if len(row["target"]) > 0]
    if not blocks:
        raise StudyError(f"No training event examples for {provider}")
    features = np.concatenate([row["features"] for row in blocks], axis=0)
    target = np.concatenate([row["target"] for row in blocks], axis=0).astype(np.int8)
    weights = _balanced_source_match_weights(blocks)
    if len(weights) != len(target) or not set(np.unique(target)).issubset({0, 1}) or len(np.unique(target)) != 2:
        raise StudyError(f"Expected natural-prevalence binary H/A turnover examples for {provider}")
    transformed, transform = _fit_transform(features, weights)
    prior = float(np.sum(weights * target) / np.sum(weights))
    if not 0 < prior < 1:
        raise StudyError("Natural training event prevalence must be strictly between zero and one")
    estimator = LogisticRegression(C=LOGISTIC_C, solver="lbfgs",
                                   max_iter=MAX_ITER, random_state=0, fit_intercept=True)
    estimator.fit(transformed, target, sample_weight=weights)
    return {
        "study_id": STUDY_ID, "provider_excluded": provider,
        "feature_names": list(FEATURE_NAMES), "track_feature_names": list(TRACK_FEATURES),
        "transform": transform, "estimator": estimator,
        "natural_event_prevalence": prior,
        "training": {
            "sources": sorted({row["source"] for row in blocks}),
            "matches": [row["match"] for row in blocks],
            "directed_edges": int(len(target)), "directed_events": int(target.sum()),
            "source_match_balanced_weights": True,
            "weighted_natural_prevalence": prior,
            "class_balancing": False, "event_undersampling": False,
            "target": "exact directed H/A change across adjacent eligible native-checked 5 Hz rows",
            "training_is_in_sample_for_backbone": True,
            "training_backbone_provider_exclusion": provider,
            "n_iter": int(estimator.n_iter_[0]),
            "converged": bool(estimator.n_iter_[0] < MAX_ITER),
        },
        "mapping": {"base_cost_10hz": BASE_COST, "lower_cost_10hz": COST_MIN,
                    "upper_cost_10hz": COST_MAX,
                    "rule": "clip(3 - (logit(q_edge) - logit(weighted_training_event_prevalence)), 1.5, 4.5)"},
    }


def _transform_features(head: dict[str, Any], features: np.ndarray) -> np.ndarray:
    transform = head["transform"]
    imputed = np.where(np.isfinite(features), features, transform["median"][None, :])
    return (imputed - transform["mean"]) / transform["scale"]


def _logit(value: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(value, dtype=np.float64), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    return np.log(p) - np.log1p(-p)


def cost_from_llr(llr: np.ndarray | float) -> np.ndarray:
    """The single fixed event-evidence to decoder-cost mapping."""
    return np.clip(BASE_COST - np.asarray(llr, dtype=np.float64), COST_MIN, COST_MAX)


def predict_edge_events(head: dict[str, Any], record: dict[str, Any],
                        probabilities_5hz: np.ndarray) -> dict[str, np.ndarray]:
    pairs = _edge_pairs(record)
    design = edge_features(record, probabilities_5hz, pairs)
    count = len(pairs)
    if not count:
        empty = np.empty(0, dtype=np.float64)
        return {"pairs": pairs, "q_home_away": empty, "q_away_home": empty,
                "llr_home_away": empty, "llr_away_home": empty,
                "cost_home_away": empty, "cost_away_home": empty}
    q = head["estimator"].predict_proba(_transform_features(head, design))[:, 1]
    q_ha, q_ah = q[:count], q[count:]
    baseline_odds = float(_logit(head["natural_event_prevalence"]))
    llr_ha, llr_ah = _logit(q_ha) - baseline_odds, _logit(q_ah) - baseline_odds
    cost_ha = cost_from_llr(llr_ha)
    cost_ah = cost_from_llr(llr_ah)
    return {"pairs": pairs, "q_home_away": q_ha, "q_away_home": q_ah,
            "llr_home_away": llr_ha, "llr_away_home": llr_ah,
            "cost_home_away": cost_ha, "cost_away_home": cost_ah}


def native_edge_costs(record: dict[str, Any], native_n: int, event: dict[str, np.ndarray]) -> np.ndarray:
    """Project each 200 ms sampled edge likelihood over its native interval."""
    indices = np.asarray(record["indices"], dtype=np.int64)
    costs = np.full((native_n, 2), BASE_COST, dtype=np.float64)
    pairs = event["pairs"]
    if len(pairs) != len(event["cost_home_away"]) or len(pairs) != len(event["cost_away_home"]):
        raise ValueError("Event outputs and sampled edge pairs do not align")
    for index, (left, right) in enumerate(pairs):
        start, stop = int(indices[left]), int(indices[right])
        if start < 0 or stop >= native_n or stop <= start:
            raise ValueError("Sample pair falls outside native grid")
        # Each 5 Hz probability row summarizes one adjacent 0.2 s event window.
        # Hold that event LLR over the possible native boundaries in the same
        # interval; the Viterbi penalty later scales by fps/10 in cost units.
        costs[start + 1:stop + 1, 0] = event["cost_home_away"][index]
        costs[start + 1:stop + 1, 1] = event["cost_away_home"][index]
    return costs


def transition_penalties(probabilities: np.ndarray, edge_costs: np.ndarray,
                         fps: float) -> np.ndarray:
    p = np.asarray(probabilities)
    costs = np.asarray(edge_costs, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or costs.shape != (len(p), 2):
        raise ValueError("Expected (n,3) probabilities and (n,2) directed costs")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(costs).all():
        raise ValueError("fps and edge costs must be finite")
    penalties = np.full((len(p), 3, 3), BASE_COST * float(fps) / 10.0, dtype=np.float64)
    for label in range(3):
        penalties[:, label, label] = 0.0
    penalties[:, CLASS_SOURCE, CLASS_DEST] = costs[:, 0] * float(fps) / 10.0
    penalties[:, CLASS_DEST, CLASS_SOURCE] = costs[:, 1] * float(fps) / 10.0
    return penalties


def decode_dynamic(probabilities: np.ndarray, period: np.ndarray, fps: float,
                   edge_costs: np.ndarray) -> np.ndarray:
    """Match supervised.decode with time-varying directed H/A penalties."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    period = np.asarray(period)
    edge_costs = np.asarray(edge_costs, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != (len(period), 3):
        raise ValueError("Expected probabilities shaped (n,3) and aligned periods")
    if edge_costs.shape != (len(period), 2):
        raise ValueError("Directed edge costs must be (n,2)")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.all(probabilities.sum(axis=1) > 0):
        raise ValueError("Probability rows must have positive mass")
    valid_classes = np.flatnonzero(probabilities.sum(axis=0) > 0)
    logp = np.log(np.maximum(probabilities[:, valid_classes], 1e-7))
    full_penalty = transition_penalties(probabilities, edge_costs, fps)
    penalty = full_penalty[:, valid_classes][:, :, valid_classes]
    result = np.empty(len(period), dtype=np.int8)
    for start, stop in supervised.blocks(period):
        back = np.zeros((stop - start, len(valid_classes)), dtype=np.int8)
        value = logp[start].copy()
        for row in range(start + 1, stop):
            transition = value[:, None] - penalty[row]
            np.fill_diagonal(transition, value)
            previous = np.argmax(transition, axis=0)
            back[row-start] = previous
            value = transition[previous, np.arange(len(valid_classes))] + logp[row]
        state = int(np.argmax(value))
        for row in range(stop - 1, start - 1, -1):
            result[row] = valid_classes[state]
            state = int(back[row-start, state])
    return supervised.LABELS[result]


def event_costs_for_native(head: dict[str, Any], record: dict[str, Any],
                           probability_native: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    probability_native = np.asarray(probability_native, dtype=np.float64)
    indices = np.asarray(record["indices"], dtype=np.int64)
    if probability_native.ndim != 2 or probability_native.shape[1] != 3:
        raise ValueError("Expected native three-class probabilities")
    if indices.size and indices.max() >= len(probability_native):
        raise ValueError("Sample indices exceed native probability array")
    event = predict_edge_events(head, record, probability_native[indices])
    costs = native_edge_costs(record, len(probability_native), event)
    diag = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in event.items()}
    diag["cost_mean_home_away"] = float(np.mean(event["cost_home_away"])) if len(event["pairs"]) else BASE_COST
    diag["cost_mean_away_home"] = float(np.mean(event["cost_away_home"])) if len(event["pairs"]) else BASE_COST
    diag["cost_clipped_fraction"] = float(np.mean(
        (event["cost_home_away"] <= COST_MIN) | (event["cost_home_away"] >= COST_MAX) |
        (event["cost_away_home"] <= COST_MIN) | (event["cost_away_home"] >= COST_MAX))) if len(event["pairs"]) else 0.0
    return costs, diag


def score_prediction(labels: np.ndarray, prediction: np.ndarray, inputs: dict,
                     cell: dict[str, str], probability: np.ndarray,
                     probability_file_sha256: str | None = None) -> dict[str, Any]:
    truth = np.asarray(labels).astype(str)
    guess = np.asarray(prediction).astype(str)
    result = metrics.evaluate_tracking(truth, guess, period=inputs["period"],
                                      fps=float(inputs["fps"]), timestamps=inputs["timestamps"])
    diag = spell_diagnostics.spell_diagnostics(truth, guess, period=inputs["period"],
                                               fps=float(inputs["fps"]), timestamps=inputs["timestamps"])
    d = result["per_class"]["D"]
    short = diag["per_iou"]["0.5"]["reference_recall_groups"]["under_3_seconds"]
    return {
        **cell, "spell_f1": float(result["spell_f1_mean"]),
        "macro_f1": float(result["macro_f1"]),
        "weighted_70_30": float(.7 * result["spell_f1_mean"] + .3 * result["macro_f1"]),
        "d_f1": float(d["f1"]), "d_precision": float(d["precision"]),
        "d_recall": float(d["recall"]), "short_spell_recall": short["recall"],
        "short_reference_spells": int(short["n_gt"]),
        "short_matched_spells": int(short["n_matched"]),
        "spell_by_iou": {str(t): float(result["spell_f1"][t]["f1"]) for t in IOU_THRESHOLDS},
        "n_gt_spells": int(result["n_gt_spells"]),
        "n_pred_spells": int(result["n_pred_spells"]),
        "false_extra_spells": int(diag["per_iou"]["0.5"]["n_spurious"]),
        "spell_count_error": int(diag["spell_count_error"]),
        "fragmented_reference_spells": int(diag["overlap_geometry"]["reference_overlapped_by_multiple_same_team_predictions"]),
        "merged_reference_spells": int(diag["overlap_geometry"]["predictions_overlapping_multiple_same_team_references"]),
        "reference_spells_touching_graded_run_edge": int(diag["reference_spells_touching_graded_run_edge"]),
        "complete_spell_boundary_timing_iou_0_5": diag["matched_complete_boundary_errors_seconds_at_iou_0_5"],
        "graded_frames": int(result["n_graded"]),
        "probability_sha256": sha256_array(probability),
        "prediction_sha256": sha256_array(guess),
        "probability_file_sha256": probability_file_sha256,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for view in ("clean", "feed_like"):
        providers = {}
        for provider in PROVIDERS:
            cells_for_provider = [row for row in rows if row["view"] == view and row["source"] == provider]
            if not cells_for_provider:
                raise StudyError(f"No rows for {provider}/{view}")
            metrics = (set(SCORE_METRICS) | {f"iou_{t:.1f}" for t in IOU_THRESHOLDS} |
                       {"false_extra_spells", "fragmented_reference_spells",
                        "merged_reference_spells", "n_gt_spells", "n_pred_spells",
                        "short_reference_spells", "short_matched_spells"})
            providers[provider] = {
                name: float(np.mean([
                    row["spell_by_iou"][f"{float(name[4:]):.1f}"] if name.startswith("iou_") else row[name]
                    for row in cells_for_provider]))
                for name in metrics
            }
        equal = {key: float(np.mean([v[key] for v in providers.values()]))
                 for key in next(iter(providers.values()))}
        result[view] = {"equal_provider_mean": equal, "per_provider": providers}
    return result


def _baseline_score_rows(root: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    report = json.loads((root / ENSEMBLE_ROOT / "runs.json").read_text())
    rows = [row for row in report["rows"] if row.get("recipe") == oe.RECIPE_CANDIDATE]
    indexed = {(row["source"], row["match"], row["view"], row["replicate"]): row for row in rows}
    if len(indexed) != 15:
        raise StudyError("Expected exactly 15 archived combined-objective baseline rows")
    return indexed


def _verify_baseline_row(actual: dict[str, Any], archived: dict[str, Any], key: tuple[str, ...]) -> None:
    for name in ("spell_f1", "macro_f1", "weighted_70_30", "d_f1", "d_precision", "d_recall"):
        if not math.isclose(float(actual[name]), float(archived[name]), rel_tol=0, abs_tol=1e-12):
            raise StudyError(f"Baseline metric parity failed for {key}: {name}")
    for threshold, score in actual["spell_by_iou"].items():
        if not math.isclose(float(score), float(archived["spell_by_iou"][threshold]), rel_tol=0, abs_tol=1e-12):
            raise StudyError(f"Baseline spell IoU parity failed for {key}: {threshold}")
    if actual["n_gt_spells"] != archived["n_gt_spells"] or actual["n_pred_spells"] != archived["n_pred_spells"]:
        raise StudyError(f"Baseline spell count parity failed for {key}")
    if actual["probability_sha256"] != archived["probability_sha256"] or actual["prediction_sha256"] != archived["prediction_sha256"]:
        raise StudyError(f"Baseline archived array parity failed for {key}")


def freeze(data: Path | None = None) -> dict[str, Any]:
    root = data_root(data)
    if DECLARATION.exists():
        raise StudyError(f"Declaration is immutable once written: {DECLARATION}")
    source_facts = _source_file_facts()
    current_manifest = json.loads((ROOT / "experiments/baseline_manifest.json").read_text())
    policy = json.loads((ROOT / "experiments/model_decision_policy.json").read_text())
    if (current_manifest.get("model") != "joint_gru_combined_ensemble" or
            current_manifest.get("package_version") != "joint_gru_combined_ensemble_v1"):
        raise StudyError("The active baseline changed before this study was frozen")
    if policy.get("working_candidate", {}).get("model") != "joint_gru_combined_ensemble":
        raise StudyError("The active policy no longer names the combined ensemble")
    fold_facts = {}
    for provider in PROVIDERS:
        plan, training_rows, train_cache_facts = _fold_inputs(root, provider)
        _, checkpoints = _checkpoint_facts(root, provider)
        fold_facts[provider] = {
            "fold": plan, "train_cache_sha256": train_cache_facts,
            "training_sources": sorted({item["source"] for item in plan["train"]}),
            "training_matches": sorted(r["match"] for r in training_rows),
            "heldout_provider_excluded": provider,
            "base_prediction_policy": "Apply the exact two frozen without_<heldout provider> combined checkpoints to that fold's TRAIN records, then arithmetic-average their full H/A/D probabilities. Train-record base predictions are in-sample for the backbone; all learned components exclude the held-out evaluation provider.",
            "checkpoint_facts": checkpoints,
        }
    cell_facts = [_verify_eval_cell_snapshot(root, cell) for cell in cells()]
    declaration = {
        "study_id": STUDY_ID, "status": "frozen_before_fitting",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "possession/short-spell-recovery",
        "hypothesis": "A compact natural-prevalence event head over the current frozen GRU ensemble's 5 Hz probability trajectory and local tracking context can identify an exact directed H/A turnover interval, allowing a bounded edge-specific H/A Viterbi cost to retain short spells suppressed by uniform cost 3.",
        "falsification": "The candidate fails the predeclared clean +.01 spell screen, feed_like -0.01 floor, or any held-out provider clean macro/D-F1 -0.02 guard; report all outcomes without further arms.",
        "prior_distinction": "PR #13 directional head targeted dead/live stoppage/restart events and altered only D/live costs; earlier H/A boundary decoder targeted a class-balanced +/-0.16 s neighborhood and used non-calibrated evidence. This head targets exact directed H/A changes between native-eligible adjacent 5 Hz samples, uses natural prevalence and LLR costs, and adapts only H/A edges under the current full-provider-held-out GRU ensemble.",
        "baseline": {"manifest": "experiments/baseline_manifest.json",
                     "manifest_sha256": sha256_file(ROOT / "experiments/baseline_manifest.json"),
                     "policy": "experiments/model_decision_policy.json",
                     "policy_sha256": sha256_file(ROOT / "experiments/model_decision_policy.json"),
                     "ensemble": "joint_gru_combined_ensemble_v1",
                     "decoder": "Viterbi switch cost 3, native fps scaled by fps/10"},
        "question_and_mechanism": {
            "mechanism": "One L2 logistic edge-event head per held-out provider, fit only on that fold's existing TRAIN matches. Head predicts two oriented edge rows per valid 5 Hz interval. Features combine source/destination H/A/D ensemble probabilities at both endpoints, oriented H/A log-odds/change, and destination-versus-source control/contact impulse, nearest distance, nearby-player support, plus global ball speed/availability.",
            "target": "For each valid pair, H->A positive only for an exact H then A target pair; A->H positive only for A then H. Both orientations are present for every eligible pair. No class balancing or event undersampling.",
            "native_eligibility": "Training pair requires consecutive 5 Hz indices in the same tracking-derived model_period and every native target in the sampled interval known H/A. D/X-containing intervals are excluded. Unknown X never creates a block reset; time blocks are built from period/timestamps only.",
            "sampling": "5 Hz uniform block-local samples from frozen caches. No forced minimum duration; one-native-frame spells remain representable by the decoder.",
            "training_weighting": "Equal provider mass, then equal match mass within provider, then equal edge-row mass within match; natural weighted event prevalence is retained.",
            "training_prediction_provenance": "For leave_out_X, the exact same without_X two-seed frozen combined models generate ensemble probabilities on every TRAIN record. These backbone predictions are in-sample for training matches, disclosed; evaluation provider X is excluded from every learned component.",
            "features": list(FEATURE_NAMES),
            "transform": "Weighted training-only feature median imputation, weighted mean and standard deviation; no evaluation-derived statistic.",
            "head": {"class": "sklearn.linear_model.LogisticRegression", "penalty": "l2",
                     "C": LOGISTIC_C, "solver": "lbfgs", "max_iter": MAX_ITER,
                     "head_fits": 3, "random_state": 0},
            "fixed_cost_mapping": {
                "q": "predicted natural-prevalence probability for one directed edge over the adjacent 0.2 s event window",
                "llr": "logit(q) - logit(weighted training event prevalence)",
                "cost": "clip(3 - llr, 1.5, 4.5), in supervised.decode 10 Hz log-likelihood units",
                "native_time_grid": "Hold each sampled 0.2 s interval's two directed costs across native rows (left_sample+1 ... right_sample inclusive); the decoder multiplies by fps/10. At block starts/discontinuities and uncovered rows, cost remains 3.",
                "D_live": "all H/A<->D transition penalties remain exactly 3*fps/10",
            },
            "emissions": "Use archived current equal-probability seed ensemble unchanged; no P(H/A/D) modification or renormalization.",
            "seeds": "One deterministic head per held-out provider trained on the actual 50/50 two-seed ensemble trajectory; no separate head-seed fits.",
        },
        "cohort": {"development_matches": [cell["match"] for cell in cells()],
                   "cells": cell_facts, "provider_folds": fold_facts,
                   "historical_exposure": "All five evaluation matches are previously exposed development data; Metrica/SkillCorner labels are event-derived silver; feed_like is synthetic."},
        "evaluation": {
            "cells": 15, "views": ["clean", "feed_like"], "stress_replicates": ["corrupt_1", "corrupt_2"],
            "primary": "official greedy H/A spell F1, mean IoU 0.3/0.4/0.5/0.6/0.7",
            "tie_break": "macro F1 on exact spell ties", "auxiliary": "0.70 spell + 0.30 macro; not a ranking metric",
            "aggregation": "average feed_like replicates within match; matches within provider; providers equally",
            "guardrails": {"clean_spell": "candidate >= exact cost-3 baseline + 0.01",
                           "feed_like_spell": "candidate >= exact cost-3 baseline - 0.01",
                           "each_provider_clean_macro": "candidate delta >= -0.02",
                           "each_provider_clean_d_f1": "candidate delta >= -0.02"},
            "diagnostics": ["both frozen component seeds and actual probability ensemble",
                            "all five spell IoUs", "under-3s recall", "false extra spells",
                            "fragmented references", "merged references", "D precision/recall",
                            "graded-edge censoring", "complete-boundary timing", "edge-cost distribution"],
            "public_labels": "Never read, score, select or tune on organizer public labels; no organizer scoring gate is opened.",
            "automatic_promotion": False,
        },
        "budgets": {"fits": 3, "fit_method": "CPU L2 logistic only; no GRU refits",
                    "evaluation_cells": 15, "extra_arms": 0, "sweeps": 0},
        "source_sha256": {row["path"]: row["sha256"] for row in source_facts},
        "dependency_versions": {name: __import__("importlib.metadata", fromlist=["version"]).version(name)
                                for name in ("numpy", "scikit-learn", "joblib", "torch")},
        "data_root": str(root), "output_directory": str((root / OUT_RELATIVE).relative_to(root)),
    }
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    DECLARATION.write_text(json.dumps(declaration, indent=2, allow_nan=False) + "\n")
    return {"declaration": str(DECLARATION.relative_to(ROOT)),
            "declaration_sha256": sha256_file(DECLARATION),
            "source_sha256": declaration["source_sha256"], "folds": list(fold_facts),
            "cells": len(cell_facts)}


def _load_frozen_declaration(data: Path) -> tuple[dict[str, Any], str]:
    if not DECLARATION.is_file():
        raise StudyError("Run freeze before fitting")
    declaration = json.loads(DECLARATION.read_text())
    for name, expected in declaration["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise StudyError(f"Frozen source changed after declaration: {name}")
    if Path(declaration["data_root"]).resolve() != data.resolve():
        raise StudyError("Frozen data root differs from the current invocation")
    return declaration, sha256_file(DECLARATION)


def fit_all(data: Path | None = None) -> dict[str, Any]:
    root = data_root(data)
    declaration, declaration_sha = _load_frozen_declaration(root)
    output = root / OUT_RELATIVE
    output.mkdir(parents=True, exist_ok=True)
    if FIT_RECEIPT.exists() or any((output / f"head_{p}.joblib").exists() for p in PROVIDERS):
        raise StudyError("Fitted artifacts are immutable once written")
    receipts, summaries = [], {}
    for provider in PROVIDERS:
        expected = declaration["cohort"]["provider_folds"][provider]
        blocks, provenance = build_training_blocks(root, provider)
        if provenance["fold"]["train"] != expected["fold"]["train"]:
            raise StudyError(f"Training fold changed after freeze: {provider}")
        if provenance["training_cache_sha256"] != expected["train_cache_sha256"]:
            raise StudyError(f"Training cache changed after freeze: {provider}")
        if provenance["checkpoint_facts"] != expected["checkpoint_facts"]:
            raise StudyError(f"Frozen backbone changed after declaration: {provider}")
        head = fit_head(blocks, provider)
        path = output / f"head_{provider}.joblib"
        joblib.dump(head, path, compress=3)
        model_fact = {"provider_excluded": provider, "path": str(path.relative_to(root)),
                      "sha256": sha256_file(path), "fit": head["training"],
                      "training_cache_sha256": provenance["training_cache_sha256"],
                      "checkpoint_facts": provenance["checkpoint_facts"],
                      "train_probability_facts": provenance["train_probability_facts"],
                      "training_blocks": [{k: v for k, v in row.items() if k not in ("features", "target")}
                                          for row in blocks]}
        receipts.append(model_fact)
        summaries[provider] = {"training": head["training"], "mapping": head["mapping"],
                               "model_sha256": model_fact["sha256"]}
    receipt = {"study_id": STUDY_ID, "declaration_sha256": declaration_sha,
               "source_sha256": declaration["source_sha256"],
               "fitted_utc": datetime.now(timezone.utc).isoformat(),
               "fit_count": len(receipts), "heads": receipts,
               "scores_read_before_fit_seal": False}
    FIT_RECEIPT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return {"fit_receipt": str(FIT_RECEIPT.relative_to(ROOT)),
            "fit_receipt_sha256": sha256_file(FIT_RECEIPT), "heads": summaries}


def seal(data: Path | None = None) -> dict[str, Any]:
    root = data_root(data)
    declaration, declaration_sha = _load_frozen_declaration(root)
    if not FIT_RECEIPT.is_file():
        raise StudyError("All three heads must be fitted before scoring")
    receipt = json.loads(FIT_RECEIPT.read_text())
    if (receipt.get("declaration_sha256") != declaration_sha or receipt.get("fit_count") != 3 or
            receipt.get("source_sha256") != declaration["source_sha256"]):
        raise StudyError("Fit receipt does not match the frozen 3-head declaration")
    output = root / OUT_RELATIVE
    for provider, head_fact in zip(PROVIDERS, receipt["heads"]):
        path = output / f"head_{provider}.joblib"
        if head_fact["provider_excluded"] != provider or sha256_file(path) != head_fact["sha256"]:
            raise StudyError(f"Head model failed its frozen seal: {provider}")
    seal_fact = {"study_id": STUDY_ID, "declaration_sha256": declaration_sha,
                 "fit_receipt_sha256": sha256_file(FIT_RECEIPT),
                 "head_sha256": {provider: sha256_file(output / f"head_{provider}.joblib")
                                 for provider in PROVIDERS},
                 "sealed_utc": datetime.now(timezone.utc).isoformat(),
                 "before_evaluation_score_read": True}
    path = output / "model_seal.json"
    if path.exists():
        raise StudyError("Model seal is immutable once written")
    path.write_text(json.dumps(seal_fact, indent=2) + "\n")
    return {"model_seal": str(path.relative_to(root)), "model_seal_sha256": sha256_file(path),
            "heads": seal_fact["head_sha256"]}


def _verify_model_seal(root: Path, declaration_sha: str) -> tuple[dict, dict[str, dict]]:
    output = root / OUT_RELATIVE
    seal_path = output / "model_seal.json"
    if not seal_path.is_file() or not FIT_RECEIPT.is_file():
        raise StudyError("Scoring requires a complete frozen three-head model seal")
    seal_fact = json.loads(seal_path.read_text())
    fit_receipt = json.loads(FIT_RECEIPT.read_text())
    declaration = json.loads(DECLARATION.read_text())
    if (seal_fact.get("declaration_sha256") != declaration_sha or
            fit_receipt.get("declaration_sha256") != declaration_sha or
            fit_receipt.get("source_sha256") != declaration["source_sha256"]):
        raise StudyError("Model seal/declaration mismatch")
    if seal_fact.get("fit_receipt_sha256") != sha256_file(FIT_RECEIPT):
        raise StudyError("Fit receipt changed after model sealing")
    heads = {}
    for provider in PROVIDERS:
        path = output / f"head_{provider}.joblib"
        if sha256_file(path) != seal_fact["head_sha256"].get(provider):
            raise StudyError(f"Frozen head changed after seal: {provider}")
        heads[provider] = joblib.load(path)
    return seal_fact, heads


def score_all(data: Path | None = None) -> dict[str, Any]:
    root = data_root(data)
    declaration, declaration_sha = _load_frozen_declaration(root)
    seal_fact, heads = _verify_model_seal(root, declaration_sha)
    if SCORE_RECEIPT.exists():
        raise StudyError("Score receipt is immutable once written")
    archived_baseline_rows = _baseline_score_rows(root)
    outputs = root / OUT_RELATIVE
    rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    frozen_cells = {(row["cell"]["source"], row["cell"]["match"],
                     row["cell"]["view"], row["cell"]["replicate"]): row
                    for row in declaration["cohort"]["cells"]}
    for cell in cells():
        paths = _record_paths_for_cell(cell, root)
        for path in paths.values():
            if not path.is_file():
                raise StudyError(f"Missing frozen evaluation artifact: {path}")
        frozen = frozen_cells[(cell["source"], cell["match"], cell["view"], cell["replicate"])]
        current_facts = _verify_eval_cell_snapshot(root, cell)
        if current_facts != frozen:
            raise StudyError(f"Evaluation cell facts differ from the complete frozen receipt: {cell}")
        cache = joblib.load(paths["inputs_labels_record"])
        record = cache["record"]
        labels, inputs = np.asarray(cache["labels"]).astype(str), cache["inputs"]
        member_ensemble, members = _load_member_probabilities(root, cell)
        with np.load(paths["candidate_ensemble"], allow_pickle=False) as archive:
            candidate_prob = archive["probability"]
            candidate_saved_pred = archive["prediction"].astype("U1")
        if not np.array_equal(member_ensemble, candidate_prob):
            raise StudyError(f"Frozen member blend changed before scoring: {cell}")
        time_blocks = supervised.model_periods(inputs)
        fps = float(inputs["fps"])
        if labels.shape != time_blocks.shape or candidate_prob.shape != (len(labels), 3):
            raise StudyError(f"Scoring arrays have inconsistent shapes: {cell}")
        baseline_prediction = supervised.decode(candidate_prob, time_blocks, fps, BASE_COST)
        if not np.array_equal(baseline_prediction, candidate_saved_pred):
            raise StudyError(f"Archived cost-3 baseline prediction mismatch: {cell}")
        key = (cell["source"], cell["match"], cell["view"], cell["replicate"])
        baseline_row = score_prediction(labels, baseline_prediction, inputs, cell,
                                         candidate_prob, sha256_file(paths["candidate_ensemble"]))
        _verify_baseline_row(baseline_row, archived_baseline_rows[key], key)
        model = heads[cell["source"]]
        edge_cost, edge_diag = event_costs_for_native(model, record, candidate_prob)
        candidate_prediction = decode_dynamic(candidate_prob, time_blocks, fps, edge_cost)
        candidate_row = score_prediction(labels, candidate_prediction, inputs, cell,
                                         candidate_prob, sha256_file(paths["candidate_ensemble"]))
        candidate_row.update({"recipe": "turnover_edge_cost", "edge_cost_diagnostics": {
            "q_home_away_mean": float(np.mean(edge_diag["q_home_away"])) if edge_diag["pairs"] else None,
            "q_away_home_mean": float(np.mean(edge_diag["q_away_home"])) if edge_diag["pairs"] else None,
            "llr_home_away_mean": float(np.mean(edge_diag["llr_home_away"])) if edge_diag["pairs"] else None,
            "llr_away_home_mean": float(np.mean(edge_diag["llr_away_home"])) if edge_diag["pairs"] else None,
            "cost_mean_home_away": edge_diag["cost_mean_home_away"],
            "cost_mean_away_home": edge_diag["cost_mean_away_home"],
            "cost_clipped_fraction": edge_diag["cost_clipped_fraction"],
            "eligible_sampled_edge_intervals": len(edge_diag["pairs"]),
            "native_costs_sha256": sha256_array(edge_cost),
        }})
        control_file = outputs / f"baseline_{cell['match']}_{cell['view']}_{cell['replicate']}.npz"
        candidate_file = outputs / f"candidate_{cell['match']}_{cell['view']}_{cell['replicate']}.npz"
        np.savez_compressed(control_file, probability=candidate_prob,
                            prediction=baseline_prediction.astype("U1"))
        np.savez_compressed(candidate_file, probability=candidate_prob,
                            prediction=candidate_prediction.astype("U1"),
                            edge_costs=edge_cost)
        baseline_row["probability_file"] = control_file.name
        baseline_row["probability_file_sha256"] = sha256_file(control_file)
        candidate_row["probability_file"] = candidate_file.name
        candidate_row["probability_file_sha256"] = sha256_file(candidate_file)
        rows.extend((baseline_row, candidate_row))
        for seed in SEEDS:
            seed_base = supervised.decode(members[seed], time_blocks, fps, BASE_COST)
            seed_candidate = decode_dynamic(members[seed], time_blocks, fps, edge_cost)
            fit_rows.append({**score_prediction(labels, seed_base, inputs, cell, members[seed]),
                             "component_seed": seed, "recipe": f"cost3_seed_{seed}"})
            fit_rows.append({**score_prediction(labels, seed_candidate, inputs, cell, members[seed]),
                             "component_seed": seed, "recipe": f"shared_head_seed_{seed}"})
    baseline = [row for row in rows if row.get("recipe") != "turnover_edge_cost"]
    candidate = [row for row in rows if row.get("recipe") == "turnover_edge_cost"]
    if len(baseline) != 15 or len(candidate) != 15 or len(fit_rows) != 60:
        raise StudyError("Completed cell count differs from frozen budget")
    baseline_aggregate = aggregate_rows(baseline)
    candidate_aggregate = aggregate_rows(candidate)
    component_seed_aggregate = {
        str(seed): {
            recipe: aggregate_rows([row for row in fit_rows if row["component_seed"] == seed and row["recipe"] == recipe])
            for recipe in (f"cost3_seed_{seed}", f"shared_head_seed_{seed}")
        } for seed in SEEDS
    }
    deltas = {}
    for view in ("clean", "feed_like"):
        deltas[view] = {
            name: candidate_aggregate[view]["equal_provider_mean"][name] -
                  baseline_aggregate[view]["equal_provider_mean"][name]
            for name in candidate_aggregate[view]["equal_provider_mean"]
        }
        deltas[view]["per_provider"] = {
            provider: {
                name: candidate_aggregate[view]["per_provider"][provider][name] -
                      baseline_aggregate[view]["per_provider"][provider][name]
                for name in candidate_aggregate[view]["per_provider"][provider]
            } for provider in PROVIDERS
        }
    checks = {
        "clean_spell_plus_0_01": deltas["clean"]["spell_f1"] >= .01,
        "feed_like_spell_no_worse_than_minus_0_01": deltas["feed_like"]["spell_f1"] >= -.01,
        "each_provider_clean_macro_within_0_02": all(deltas["clean"]["per_provider"][p]["macro_f1"] >= -.02 for p in PROVIDERS),
        "each_provider_clean_d_f1_within_0_02": all(deltas["clean"]["per_provider"][p]["d_f1"] >= -.02 for p in PROVIDERS),
    }
    aggregate = {"baseline": baseline_aggregate, "candidate": candidate_aggregate,
                "component_seed_diagnostics": component_seed_aggregate,
                "candidate_minus_baseline": deltas,
                "predeclared_guardrails": {"checks": checks, "passes": all(checks.values()),
                                           "automatic_promotion": False}}
    run_payload = {"study_id": STUDY_ID, "declaration_sha256": declaration_sha,
                   "model_seal_sha256": sha256_file(outputs / "model_seal.json"),
                   "scored_utc": datetime.now(timezone.utc).isoformat(),
                   "rows": rows, "component_seed_rows": fit_rows, "aggregate": aggregate}
    run_path = outputs / "runs.json"
    if run_path.exists():
        raise StudyError("Run output already exists")
    run_path.write_text(json.dumps(run_payload, indent=2, allow_nan=False) + "\n")
    receipt = {"study_id": STUDY_ID, "declaration_sha256": declaration_sha,
               "model_seal_sha256": sha256_file(outputs / "model_seal.json"),
               "run_path": str(run_path.relative_to(root)), "run_sha256": sha256_file(run_path),
               "scored_cells": 15, "baseline_rows": 15, "candidate_rows": 15,
               "component_seed_rows": 60,
               "source_sha256": declaration["source_sha256"],
               "no_organizer_scoring": True, "automatic_promotion": False,
               "scores_read_after_all_heads_sealed": True,
               "aggregate": aggregate}
    SCORE_RECEIPT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def artifact_receipt(data: Path | None = None) -> dict[str, Any]:
    root = data_root(data)
    result = {}
    for name in (DECLARATION, FIT_RECEIPT, SCORE_RECEIPT):
        if name.is_file():
            result[name.name] = {"path": str(name.relative_to(ROOT)), "sha256": sha256_file(name)}
    output = root / OUT_RELATIVE
    if output.is_dir():
        result["private_outputs"] = {
            path.name: {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)}
            for path in sorted(output.iterdir()) if path.is_file()
        }
    payload = {"study_id": STUDY_ID, "created_utc": datetime.now(timezone.utc).isoformat(),
               "artifacts": result, "private_artifacts_are_local_only": True}
    ARTIFACT_RECEIPT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "fit", "seal", "score", "receipt"))
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    handlers = {"freeze": freeze, "fit": fit_all, "seal": seal,
                "score": score_all, "receipt": artifact_receipt}
    result = handlers[args.command](args.data_dir)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
