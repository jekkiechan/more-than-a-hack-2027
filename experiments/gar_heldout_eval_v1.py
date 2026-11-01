"""Held-out evaluation of the sealed possession arms on SoccerNet-GAR VALID and TEST (silver labels).

Study `gar_heldout_eval_v1`. No fitting, no promotion, no organizer labels. The 19 GAR VALID (9 games)
and TEST (10 games) matches (PFF World Cup 2022 tracking) were never used for training or selection by
any sealed arm; the user authorised scoring them here.

Arms (all sealed leave-one-provider-out fold checkpoints; each arm uses the equal-probability average of
all 6 folds = providers Metrica, SkillCorner, Sportec x seeds 2026, 2027, each with its matching teacher):

* `adop`     adopted calibrated dense D2 heads
* `cand`     `original_full` candidate
* `raw`      raw-feed augmented (v1 views)
* `v2train`  raw-feed recipe with feed_like_v2 training views

Views: `clean` (native GAR tracking) and `feed_v2_1`, `feed_v2_2` (challenge-calibrated feed_like_v2
corruption of the clean tracking, 2 realizations). Modes: `phase0` (the 5 Hz grid of the dense records,
bit-identical to the stored features) and `phase_tta` (probability average over the 6 sampling phases of
the native 29.97 fps tracking). Decoder: the frozen f062 decoder; scoring: the official spell / macro F1.

Aggregation: the dense records of one game (contiguous tracking segments, each with at least one labelled
frame; unlabelled frames are `X`) are concatenated into one game-level input with a distinct period id per
record, so no spell or smoothing crosses a record boundary (breaks preserved). Metrics are computed per
game; split and overall numbers are unweighted means over games. Pooled-over-split scores (records of all
games of a split concatenated, as in `sn_dense_train.make_gar_validator`) are reported as a secondary view.

Steps: `sanity`, `games` (resumable, per-game worker processes), `report`.
Per-game raw outputs (decoded labels, ignored) are under data/experiments/gar_heldout_eval_v1/games;
experiments/results/gar_heldout_eval_v1.json holds aggregates only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MTAH_DATA_DIR", str(ROOT.parent / "mtah-data"))
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import joblib
import numpy as np

from experiments import feed_benchmark_audit_v1 as audit
from experiments import feed_like_v2 as v2
from experiments import feed_like_v2_benchmark as v2bench  # registers arm `v2train` in audit.model_path
from experiments import soccernet_dense_labels as dl
from experiments import soccernet_extension as old
from experiments import soccernet_segments as seg
from experiments import sn_dense_train as sn
from experiments import supervised

STUDY = "gar_heldout_eval_v1"
OUT = ROOT / "data/experiments" / STUDY
RECORDS = OUT / "records"
GAMES = OUT / "games"
RAW = dl.DEFAULT_RAW
RESULT = ROOT / "experiments/results" / f"{STUDY}.json"
RECEIPT = ROOT / "experiments/results" / f"{STUDY}_download_receipt.json"
CALIBRATION = ROOT / "experiments/results/feed_like_v2_calibration.json"
PROVIDERS = ("Metrica", "SkillCorner", "Sportec")
SEEDS = (2026, 2027)
ARMS = ("adop", "cand", "raw", "v2train")
SPLITS = ("valid", "test")
EXPECTED = {"valid": (9, 2191), "test": (10, 2439)}
VIEWS = ("clean", "feed_v2_1", "feed_v2_2")
MODES = ("phase0", "phase_tta")
KEYS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall", "predicted_spell_count",
        "reference_spell_count", "short_spell_recall_iou_0_5_lt3s", "graded_frames")
WORKERS = int(os.environ.get("AUDIT_WORKERS", "2"))
LABEL_TEXT = np.array(["H", "A", "D", "X"])


# ----------------------------------------------------------------------------- inference

def ensemble_probability(record, native, arm, providers=PROVIDERS, seeds=SEEDS):
    """Equal-probability average over provider folds x seeds (each with its matching teacher).

    Sampled probabilities are averaged, then interpolated once to the native grid; interpolation is
    linear in each continuous block, so this equals averaging the per-model native probabilities.
    """
    rec = old.inference_record(record)
    rec["names"] = list(rec["names"])
    sampled = [sn.predict(rec, audit._load(v2bench.model_path(arm, p, s)), audit._load(audit.teacher_path(p, s)))
               for p in providers for s in seeds]
    return old.upsample(np.mean(sampled, axis=0), rec, native)


def score_game(labels, period, timestamps, fps, native_probability):
    model_period = supervised.model_periods(dict(period=period, timestamps=timestamps, fps=fps))
    prediction = audit.decode(native_probability, model_period, fps)
    cell = dict(labels=labels, period=period, timestamps=timestamps, fps=fps)
    row = audit.score(cell, prediction)
    return {k: row[k] for k in KEYS if k in row}, prediction


# ----------------------------------------------------------------------------- game inputs

def manifest(split, records_root=RECORDS):
    return json.loads((Path(records_root) / split / "manifest.json").read_text())


def load_game_records(split, game, records_root=RECORDS):
    """Sealed per-game dense records (hash checked against the manifest)."""
    man = manifest(split, records_root)
    name = f"{game}.joblib"
    path = Path(records_root) / split / "games" / name
    if seg._sha256(path) != man["game_files"][name]:
        raise ValueError(f"seal mismatch: {path}")
    records = joblib.load(path)["records"]
    records.sort(key=lambda r: (int(r["game_period"]), int(r["full_frame_numbers"][0])))
    return records


def capture_inputs(split, game, tracking=None):
    """Re-run the frozen (dense-patched) assembler and capture each segment's challenge-format inputs.

    The assembler builds ball / players / teams / period / fps / timestamps / pitch / ball_observed /
    player_observed for every contiguous segment and passes them to `refine_features.features`; that call
    is wrapped to keep them. Nothing is changed in the frozen modules.
    """
    import io
    import pyarrow.parquet as pq
    from experiments import refine_features
    tracking = dl.tracking_root(RAW) if tracking is None else tracking
    samples = [s for s in dl.read_split_samples(tracking, split) if s["game_id"] == game]
    samples.sort(key=lambda s: (s["start_frame"], s["sample_id"]))
    shards = sorted({s["shard_name"] for s in samples})
    archives = {n: __import__("tarfile").open(Path(tracking) / split / "shards" / n, mode="r:") for n in shards}
    columns = list(seg.RAW_COLUMNS) + list(dl.EXTRA_COLUMNS)

    def read_clip(sample):
        archive = archives[sample["shard_name"]]
        handle = archive.extractfile(archive.getmember(sample["wds_member"]))
        table = pq.read_table(io.BytesIO(handle.read()), columns=columns)
        rows = table.to_pylist()
        if len(rows) != 144:
            raise ValueError(f'{sample["sample_id"]} has {len(rows)} rows')
        return rows

    captured, original = {}, refine_features.features

    def capture(inputs):
        key = (int(inputs["period"][0]), float(inputs["timestamps"][0]), int(len(inputs["timestamps"])))
        captured[key] = inputs
        return original(inputs)

    refine_features.features = capture
    try:
        dl._patched_assemble()(game, samples, read_clip)
    finally:
        refine_features.features = original
        for a in archives.values():
            a.close()
    return captured


def build_game(records, captured):
    """Concatenate record-level inputs into one game-level input (period = distinct positive id per record)."""
    parts, labels, base_x = [], [], []
    for i, r in enumerate(records):
        key = (int(r["game_period"]), float(r["full_timestamps"][0]), int(r["native_n"]))
        inp = captured[key]
        if not np.array_equal(inp["timestamps"], r["full_timestamps"]):
            raise ValueError("captured inputs do not match the dense record")
        parts.append(inp)
        labels.append(LABEL_TEXT[np.where(np.asarray(r["full_y"]) < 0, 3, np.asarray(r["full_y"]))])
        base_x.append(r["x"])
    slots = {p["players"].shape[1] for p in parts}
    if len(slots) != 1 or len({tuple(p["teams"]) for p in parts}) != 1:
        raise ValueError("Player slots differ between segments of one game")
    fps = float(parts[0]["fps"])
    game = dict(
        ball=np.concatenate([p["ball"] for p in parts]), players=np.concatenate([p["players"] for p in parts]),
        teams=np.asarray(parts[0]["teams"]),
        period=np.concatenate([np.full(len(p["ball"]), i + 1, np.int32) for i, p in enumerate(parts)]),
        fps=np.asarray(fps), timestamps=np.concatenate([p["timestamps"] for p in parts]),
        pitch=np.asarray(parts[0]["pitch"]),
        ball_observed=np.concatenate([p["ball_observed"] for p in parts]),
        player_observed=np.concatenate([p["player_observed"] for p in parts]))
    return game, np.concatenate(labels), np.concatenate(base_x), [r["names"] for r in records][0]


def phase_set(inputs, names, sampled_fps):
    """The six 5 Hz phase records of a (possibly corrupted) game-level input."""
    base = dict(names=list(names), native_n=len(inputs["ball"]), fps=np.asarray(sampled_fps))
    return audit.phase_records(inputs, base)


def evaluate_view(view_inputs, labels, names, sampled_fps, arms=ARMS, probabilities=None):
    """Score every arm in phase0 and phase_tta modes on one view; also return decoded predictions."""
    records = phase_set(view_inputs, names, sampled_fps)
    native = audit.tc.inputs_only(view_inputs)
    fps = float(view_inputs["fps"])
    scores, predictions = {}, {}
    for arm in arms:
        phases = [ensemble_probability(r, native, arm) for r in records]
        for mode, p in (("phase0", phases[0]), ("phase_tta", np.mean(phases, axis=0))):
            scores.setdefault(arm, {})[mode], pred = score_game(
                labels, view_inputs["period"], view_inputs["timestamps"], fps, p)
            predictions[f"{arm}__{mode}"] = np.frombuffer(pred.astype("S1").tobytes(), np.uint8)
    return scores, predictions, records[0]


def feed_view(game, labels, calibration, game_id, rep):
    seed = int(hashlib.sha256(f"{STUDY}/{rep}/{game_id}".encode()).hexdigest()[:8], 16)
    return v2.corrupt(game, labels, calibration, seed), seed


def run_game(args):
    split, game_id, records_root, dest_dir, split_dir, arms = args
    dest = Path(dest_dir) / f"{split}__{game_id}.json"
    if dest.exists():
        return str(dest)
    started = time.time()
    records = load_game_records(split_dir or split, game_id, records_root)
    captured = capture_inputs(split, game_id)
    game, labels, saved_x, names = build_game(records, captured)
    sampled_fps = float(records[0]["fps"])
    calibration = json.loads(CALIBRATION.read_text())["calibration"]
    out = dict(split=split, game=game_id, records=len(records), native_frames=int(len(labels)),
               graded_frames=int((labels != "X").sum()), hours_graded=float((labels != "X").sum() / game["fps"] / 3600),
               native_fps=float(game["fps"]), arms=list(arms), views={})
    preds = {}
    for view in VIEWS:
        if view == "clean":
            inputs = game
        else:
            inputs, seed = feed_view(game, labels, calibration, game_id, int(view[-1]))
        scores, predictions, first = evaluate_view(inputs, labels, names, sampled_fps, arms)
        if view == "clean":
            # Phase 0 must replay the stored dense-record features exactly.
            if first["x"].shape != saved_x.shape or not np.array_equal(first["x"], saved_x, equal_nan=True):
                raise ValueError(f"phase-0 features differ from the stored dense records: {game_id}")
            out["phase0_features_equal_stored"] = True
        out["views"][view] = dict(
            scores=scores, ball_missing=float((~inputs["ball_observed"][labels != "X"]).mean()),
            **({} if view == "clean" else dict(corrupt_seed=seed)))
        preds.update({f"{view}__{k}": v for k, v in predictions.items()})
    np.savez_compressed(Path(dest_dir) / f"{split}__{game_id}__pred.npz", labels=np.frombuffer(
        labels.astype("S1").tobytes(), np.uint8), period=game["period"], timestamps=game["timestamps"], **preds)
    out["seconds"] = time.time() - started
    dest.parent.mkdir(parents=True, exist_ok=True)
    audit.write(dest, out)
    return str(dest)


def game_ids(split, records_root=RECORDS):
    return list(manifest(split, records_root)["games"])


def run_games(workers=WORKERS, only=None):
    GAMES.mkdir(parents=True, exist_ok=True)
    tasks = [(s, g, str(RECORDS), str(GAMES), None, ARMS) for s in SPLITS for g in game_ids(s)
             if only is None or f"{s}:{g}" in only]
    # Largest games first keeps the two workers balanced.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        for path in ex.map(run_game, tasks):
            print("done", path, flush=True)


# ----------------------------------------------------------------------------- sanity

def run_sanity():
    result = {}
    # 1. Replay a development match: the multi-fold path restricted to its own provider must reproduce the
    #    archived two-seed probabilities of the audit (same checkpoints, same teachers).
    cached = joblib.load(audit.EVAL_INPUTS / "J03WOY_clean_clean.joblib")
    inputs, record = cached["inputs"], cached["record"]
    native = audit.tc.inputs_only(inputs)
    archived = audit.load_cells(audit.CANDIDATE)[("Sportec", "J03WOY", "clean")]["probabilities"]
    dev = {}
    for arm in ARMS:
        ref = audit.predict(inputs, record, arm, "Sportec")
        mine = ensemble_probability(record, native, arm, providers=("Sportec",))
        dev[arm] = float(np.abs(ref - mine).max())
    mine = ensemble_probability(record, native, "cand", providers=("Sportec",))
    dev["cand_vs_archived_npz"] = float(np.abs(mine - archived).max())
    result["dev_replay_max_abs_diff"] = dev
    assert max(dev.values()) < 1e-6, dev
    # 2. A GAR-TRAIN-format dense record: valid probabilities, deterministic six-fold ensemble.
    train_root = ROOT / "data/experiments/dense_label_weight_v1/records"
    train_game = sorted(manifest("train", train_root)["games"])[0]
    rec = load_game_records("train", train_game, train_root)
    rec = max(rec[:50], key=lambda r: len(r["x"]))
    native_rec = dict(period=rec["full_period"], timestamps=rec["full_timestamps"], fps=float(rec["native_fps"]))
    det = {}
    for arm in ARMS:
        a = ensemble_probability(rec, native_rec, arm)
        b = ensemble_probability(rec, native_rec, arm)
        det[arm] = dict(finite=bool(np.isfinite(a).all()), min=float(a.min()), max=float(a.max()),
                        row_sum_max_abs_dev=float(np.abs(a.sum(1) - 1).max()), identical_rerun=bool(np.array_equal(a, b)),
                        rows=int(len(a)))
        assert det[arm]["finite"] and det[arm]["identical_rerun"] and det[arm]["row_sum_max_abs_dev"] < 1e-4, det[arm]
        assert det[arm]["min"] >= 0 and det[arm]["max"] <= 1 + 1e-6
    result["train_format_record"] = dict(game=train_game, segment=str(rec["segment_id"]), arms=det)
    audit.write(OUT / "sanity.json", result)
    print(json.dumps(result, indent=1))
    return result


# ----------------------------------------------------------------------------- report

def pooled(split, mode_key, view):
    """Score the concatenation of all games of a split (records keep distinct period ids)."""
    labels, period, stamps, preds, offset = [], [], [], [], 0
    fps = None
    for g in game_ids(split):
        with np.load(GAMES / f"{split}__{g}__pred.npz") as z:
            labels.append(z["labels"]); preds.append(z[f"{view}__{mode_key}"])
            period.append(z["period"] + offset); stamps.append(z["timestamps"])
        offset = int(period[-1].max())
        fps = json.loads((GAMES / f"{split}__{g}.json").read_text())["native_fps"]
    lab = np.frombuffer(np.concatenate(labels).tobytes(), "S1").astype(str)
    prd = np.frombuffer(np.concatenate(preds).tobytes(), "S1").astype(str)
    cell = dict(labels=lab, period=np.concatenate(period), timestamps=np.concatenate(stamps), fps=fps)
    row = audit.score(cell, prd)
    return {k: row[k] for k in ("spell_f1", "macro_f1", "d_f1")}


def bootstrap(table, a, b, group, draws=2000, seed=20261101, metric="spell_f1"):
    """Paired bootstrap over games of mean(a - b); table[(arm_mode, view)] -> {game: {metric: v}}."""
    games = sorted(group)
    diff = np.array([table[a][g][metric] - table[b][g][metric] for g in games])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(games), (draws, len(games)))
    means = diff[idx].mean(axis=1)
    return dict(delta=float(diff.mean()), ci95=[float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                ci90=[float(np.percentile(means, 5)), float(np.percentile(means, 95))],
                p_positive=float((means > 0).mean()), games_positive=int((diff > 0).sum()), games=len(games))


def run_report():
    files = {(s, g): json.loads((GAMES / f"{s}__{g}.json").read_text()) for s in SPLITS for g in game_ids(s)}
    counts = {s: dict(games=len(game_ids(s)), records=sum(f["records"] for (sp, _), f in files.items() if sp == s),
                      hours_graded=sum(f["hours_graded"] for (sp, _), f in files.items() if sp == s),
                      graded_frames=sum(f["graded_frames"] for (sp, _), f in files.items() if sp == s))
              for s in SPLITS}
    for s in SPLITS:
        assert (counts[s]["games"], counts[s]["records"]) == EXPECTED[s], (s, counts[s])
    groups = {"valid": [g for (s, g) in files if s == "valid"], "test": [g for (s, g) in files if s == "test"]}
    # table[(arm, mode, view)][game id with split prefix] -> metrics; feed view = mean of 2 realizations.
    names = [f"{s}:{g}" for (s, g) in files]

    def cell(arm, mode, view, metric):
        out = {}
        for (s, g), f in files.items():
            if view == "feed_v2":
                v = np.mean([f["views"][f"feed_v2_{r}"]["scores"][arm][mode][metric] for r in (1, 2)])
            else:
                v = f["views"][view]["scores"][arm][mode][metric]
            out[f"{s}:{g}"] = float(v)
        return out

    members = {"valid": [f"valid:{g}" for g in groups["valid"]], "test": [f"test:{g}" for g in groups["test"]],
               "all": names}
    metrics = ("spell_f1", "macro_f1", "d_f1")
    views = ("clean", "feed_v2")
    result = dict(study=STUDY, organizer_reference_access=False, public_scoring=False, fitting=False,
                  promotion=False, frame_arrays_published=False, counts=counts,
                  aggregation="per-game metrics on the concatenation of the game's dense records (distinct period id "
                              "per record, breaks preserved); split/overall = unweighted mean over games; feed_v2 = "
                              "mean of 2 realizations per game", arms=list(ARMS), providers=list(PROVIDERS),
                  seeds=list(SEEDS), decoder="f062", means={}, per_game={}, pooled_split={}, bootstrap={})
    tables = {}
    for view in views:
        for arm in ARMS:
            for mode in MODES:
                for metric in metrics:
                    t = cell(arm, mode, view, metric)
                    tables[(arm, mode, view, metric)] = t
                    for grp, mem in members.items():
                        vals = [t[m] for m in mem]
                        result["means"].setdefault(view, {}).setdefault(arm, {}).setdefault(mode, {}).setdefault(
                            metric, {})[grp] = dict(mean=float(np.mean(vals)), sd_over_games=float(np.std(vals, ddof=1)),
                                                    min=float(np.min(vals)), max=float(np.max(vals)), n=len(vals))
    result["per_game"] = {m: {"hours_graded": float(files[tuple(m.split(":"))]["hours_graded"]),
                              **{f"{view}/{arm}/{mode}": {k: tables[(arm, mode, view, k)][m] for k in metrics}
                                 for view in views for arm in ARMS for mode in MODES}} for m in names}
    for view in ("clean", "feed_v2_1", "feed_v2_2"):
        for arm in ARMS:
            for mode in MODES:
                result["pooled_split"].setdefault(view, {}).setdefault(arm, {})[mode] = {
                    s: pooled(s, f"{arm}__{mode}", view) for s in SPLITS}
    comparisons = []
    for view in views:
        for metric in ("spell_f1", "macro_f1"):
            for mode in MODES:
                comparisons += [(view, metric, f"cand-adop/{mode}", ("cand", mode), ("adop", mode)),
                                (view, metric, f"raw-adop/{mode}", ("raw", mode), ("adop", mode)),
                                (view, metric, f"v2train-adop/{mode}", ("v2train", mode), ("adop", mode)),
                                (view, metric, f"v2train-cand/{mode}", ("v2train", mode), ("cand", mode)),
                                (view, metric, f"v2train-raw/{mode}", ("v2train", mode), ("raw", mode))]
            for arm in ARMS:
                comparisons.append((view, metric, f"tta-minus-phase0/{arm}", (arm, "phase_tta"), (arm, "phase0")))
    for view, metric, name, (arm_a, mode_a), (arm_b, mode_b) in comparisons:
        ta, tb = tables[(arm_a, mode_a, view, metric)], tables[(arm_b, mode_b, view, metric)]
        merged = {k: {metric: ta[k] - tb[k]} for k in ta}
        zero = {k: {metric: 0.0} for k in ta}
        for grp, mem in members.items():
            result["bootstrap"].setdefault(view, {}).setdefault(metric, {}).setdefault(name, {})[grp] = bootstrap(
                {"a": merged, "b": zero}, "a", "b", mem, metric=metric)
    result["bootstrap_method"] = ("paired percentile bootstrap over games, 2,000 draws, seed 20261101; games resampled "
                                  "with replacement within the group (valid 9, test 10, all 19); statistic = mean over "
                                  "games of the per-game difference")
    audit.write(RESULT, result)
    print("wrote", RESULT)
    return result


STEPS = dict(sanity=run_sanity, games=run_games, report=run_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=tuple(STEPS))
    parser.add_argument("--only", nargs="*", default=None, help="split:game ids")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.step == "games":
        run_games(min(args.workers, 2), args.only)
    else:
        STEPS[args.step]()
