"""Raw-feed training with challenge-calibrated `feed_like_v2` views.

Identical to `feed_robustness_v1` (same teachers, records, GAR dense labels, width96,
batch32, half original/half GAR, lr 3e-4 / D lr 1e-4, 2,048 updates, seeds 2026/2027,
three full provider exclusions) except that the original-provider TRAIN feature views
come from `feed_like_v2.corrupt` instead of the state-blind raw-feed corruption.

Steps:
* `prepare`  build one v2 TRAIN view per original TRAIN match (TRAIN labels only place
             simulated defects; features are tracking-only).
* `parity`   refit one archived raw-feed model from its archived v1 views to verify that
             this training path reproduces the archive before any v2 fit.
* `fit`      fit one fold/seed with v2 views.
No development or organizer labels are read here. Evaluation lives in
feed_like_v2_benchmark (arm `v2train`) and feed_benchmark_audit_v1-style replays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from experiments import feed_robustness_v1 as rf, feed_like_v2 as v2
from experiments import preparation as prep, multisource_eval as ms

STUDY = "feed_like_v2_train"
ROOT = rf.ROOT
OUT = ROOT / "data/experiments" / STUDY
CALIBRATION = ROOT / "experiments/results/feed_like_v2_calibration.json"
ARCHIVE = Path(os.environ["MTAH_DATA_DIR"]) / "experiments/feed_robustness_v1/artifacts"
TRAIN_SEED = 20261102


def view_path(match, source):
    return (ARCHIVE / "views" if source == "v1" else OUT / "views") / f"{match}.joblib"


def prepare():
    calibration = json.loads(CALIBRATION.read_text())["calibration"]
    receipts = {}
    for item in prep.split_plan()["train"]:
        path = prep.data_path(item, rf.old.DATA)
        if rf.sha(path) != item["sha256"]:
            raise ValueError("Raw TRAIN checksum mismatch")
        dest = view_path(item["match"], "v2")
        if dest.exists():
            continue
        inputs, labels = ms.view(path)
        seed = int(hashlib.sha256(f"{STUDY}/{TRAIN_SEED}/{item['match']}".encode()).hexdigest()[:8], 16)
        viewed = v2.corrupt(inputs, labels, calibration, seed)
        record = prep.build_record(viewed, labels, item["source"], item["match"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(record, dest)
        receipts[item["match"]] = dict(seed=seed, sha256=rf.sha(dest), raw_sha256=item["sha256"],
                                       profile_truth_state=v2.profile(viewed, np.where(labels == "D", "D", "H")))
        print("PREPARED", item["match"], flush=True)
    if receipts:
        (OUT / "preparation.json").write_text(json.dumps(receipts, indent=1, default=float))


def training(fold, source):
    """feed_robustness_v1.training with the view directory as the only difference."""
    records = []
    spec = rf.old.old_spec()
    for item in prep.split_plan(fold=fold)["train"]:
        entry = spec["prepared_training_caches"][item["match"]]
        path = rf.old.DATA / Path(entry["cache"]).relative_to("data")
        if rf.sha(path) != entry["sha256"]:
            raise ValueError("Original training cache changed")
        cache = joblib.load(path)
        paired = rf.deployment.paired_records(cache, item["match"])
        view = joblib.load(view_path(item["match"], source))
        for key in ("indices", "period", "y", "names"):
            np.testing.assert_array_equal(view[key], cache["records"]["clean"][key])
        records.extend(rf.replace_view(paired, view["x"]))
    excluded = fold.removeprefix("leave_out_")
    if any(str(r["source"]) == excluded for r in records):
        raise ValueError("Held-out provider leaked into training")
    return records


def _fit(fold, seed, source, dest):
    teacher = joblib.load(rf.sc.initial_path(fold, seed))
    records = training(fold, source)
    gar = rf.dl.load_dense_records("train", output=rf.TRAIN_RECORDS)
    started = time.monotonic()
    result = rf.sn.train(teacher, records, gar, width=96, lr=3e-4, seed=seed, steps=2048,
                         fraction=.5, batch_size=32, ha_target="dense", train_d=True,
                         require_full_native=True, log_every=256, verbose=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    joblib.dump(result["model"], dest)
    Path(str(dest) + ".json").write_text(json.dumps(dict(
        fold=fold, seed=seed, source=source, sha256=rf.sha(dest), wall_seconds=time.monotonic()-started,
        old_matches=sorted({str(r["match"]) for r in records}), training_config=result["config"],
        history=result.get("history", [])), indent=1, default=float))
    return result["model"]


def parity(fold="leave_out_metrica", seed=2026):
    """Refit an archived raw-feed model from archived v1 views and compare parameters."""
    dest = OUT / "parity" / f"{fold}_{seed}.joblib"
    model = joblib.load(dest) if dest.exists() else _fit(fold, seed, "v1", dest)
    archived = joblib.load(ARCHIVE / "models" / f"{fold}_{seed}.joblib")
    diffs = {}

    def walk(a, b, key=""):
        if isinstance(a, dict):
            for k in a:
                walk(a[k], b[k], f"{key}/{k}")
        elif hasattr(a, "shape") or hasattr(a, "detach"):
            x = np.asarray(a.detach().cpu() if hasattr(a, "detach") else a, dtype=float)
            y = np.asarray(b.detach().cpu() if hasattr(b, "detach") else b, dtype=float)
            diffs[key] = float(np.abs(x-y).max()) if x.shape == y.shape else float("inf")
    walk(model, archived)
    result = dict(fold=fold, seed=seed, sha_equal=rf.sha(dest) == rf.sha(ARCHIVE / "models" / f"{fold}_{seed}.joblib"),
                  max_abs_parameter_difference=max(diffs.values()) if diffs else None, arrays=len(diffs))
    (OUT / "parity.json").write_text(json.dumps(result, indent=1))
    print(result, flush=True)
    return result


def fit(fold, seed):
    dest = OUT / "models" / f"{fold}_{seed}.joblib"
    if dest.exists():
        return
    _fit(fold, seed, "v2", dest)
    print("FIT COMPLETE", fold, seed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("prepare", "parity", "fit"))
    parser.add_argument("--fold", choices=rf.FOLDS)
    parser.add_argument("--seed", type=int, choices=rf.SEEDS)
    a = parser.parse_args()
    if a.step == "fit":
        fit(a.fold, a.seed)
    elif a.step == "parity":
        parity(*(x for x in (a.fold, a.seed) if x is not None))
    else:
        prepare()
