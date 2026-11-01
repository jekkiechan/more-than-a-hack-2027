"""Challenge-calibrated `feed_like_v2` benchmark on the five development matches.

Steps (no organizer labels, no public scoring, no promotion):

* `calibrate`  derive the v2 calibration and the organizer feed profile from organizer
               tracking inputs and the adopted model's sealed label-free CSV.
* `views`      build 4 v2 realizations per development match and score sealed arms.
* `fidelity`   compare label-free profiles: organizer feed vs original feed_like vs v2,
               each conditioned on the adopted model's own predicted states.
* `report`     aggregate JSON for experiments/results (no frame arrays).

Arms: `adop` (adopted calibrated dense D2 fold models), `cand` (`original_full`),
`raw` (raw-feed augmented, v1 views) and, once fitted, `v2train` (identical raw-feed
recipe with v2 training views; see feed_like_v2_train).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import glob
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np

from experiments import feed_benchmark_audit_v1 as audit, feed_like_v2 as v2
from experiments import preparation as prep, supervised

STUDY = "feed_like_v2_benchmark"
ROOT = audit.ROOT
OUT = ROOT / "data/experiments" / STUDY
RESULTS = ROOT / "experiments/results"
CALIBRATION = RESULTS / "feed_like_v2_calibration.json"
RESULT = RESULTS / f"{STUDY}.json"
REPLICATES = (1, 2, 3, 4)
ARMS = tuple(os.environ.get("V2_ARMS", "adop,cand,raw").split(","))
PHASE_ARMS = ("adop", "cand")
V2TRAIN = ROOT / "data/experiments/feed_like_v2_train/models"


_AUDIT_MODEL_PATH = audit.model_path


def model_path(arm, provider, seed):
    if arm == "v2train":
        return V2TRAIN / f"leave_out_{audit.FOLD[provider]}_{seed}.joblib"
    return _AUDIT_MODEL_PATH(arm, provider, seed)


audit.model_path = model_path  # audit.predict resolves checkpoints through this name


def challenge_inputs():
    with np.load(audit.DATA / "challenge/inputs.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files if k != "labels"}  # organizer labels never loaded


def adopted_prediction():
    return np.array([r[1] for r in csv.reader(audit.ADOPTED_CSV.open())][1:])


def run_calibrate():
    inputs, prediction = challenge_inputs(), adopted_prediction()
    calibration = v2.calibrate(inputs, prediction)
    audit.write(CALIBRATION, dict(calibration=calibration, organizer_profile=v2.profile(inputs, prediction),
                                  source="organizer tracking inputs + adopted label-free CSV; no labels",
                                  csv_sha256=hashlib.sha256(audit.ADOPTED_CSV.read_bytes()).hexdigest()))


def _view_job(arg):
    provider, match, rep = arg
    dest = OUT / "scores" / f"{match}__{rep}.json"
    have = audit.read(dest) if dest.exists() else {}
    todo = [a for a in ARMS if a not in have]
    if not todo:
        return str(dest)
    calibration = audit.read(CALIBRATION)["calibration"]
    cached = joblib.load(audit.EVAL_INPUTS / f"{audit.FILE[match]}_clean_clean.joblib")
    labels = np.asarray(cached["labels"]).astype(str)
    seed = int(hashlib.sha256(f"{STUDY}/{rep}/{match}".encode()).hexdigest()[:8], 16)
    view = v2.corrupt(cached["inputs"], labels, calibration, seed)
    record = prep.build_record(view, labels, provider, audit.FILE[match])
    cell = dict(labels=labels, period=view["period"], timestamps=view["timestamps"], fps=float(view["fps"]),
                model_period=supervised.model_periods(view))
    records = audit.phase_records(view, record) if any(a in PHASE_ARMS for a in todo) else [record]
    for arm in todo:
        phases = [audit.predict(view, r, arm, provider) for r in (records if arm in PHASE_ARMS else [record])]
        for mode, p in (("", phases[0]), ("_phase_tta", np.mean(phases, 0))):
            if mode and arm not in PHASE_ARMS:
                continue
            prediction = audit.decode(p, cell["model_period"], cell["fps"])
            r = audit.score(cell, prediction)
            have[arm + mode] = {k: r[k] for k in ("spell_f1", "macro_f1", "d_f1", "predicted_spell_count",
                                                  "reference_spell_count", "short_spell_recall_iou_0_5_lt3s")}
            if arm == "adop" and not mode:
                have["profile_predicted"] = v2.profile(view, prediction)
    have["arm_seed"] = seed
    dest.parent.mkdir(parents=True, exist_ok=True)
    audit.write(dest, have)
    return str(dest)


def run_views():
    jobs = [(p, m, r) for p, m, _ in audit.MATCHES for r in REPLICATES]
    with ProcessPoolExecutor(audit.WORKERS) as ex:
        list(ex.map(_view_job, jobs))


def _v1_profiles():
    adopted = audit.load_adopted()
    out = {}
    for k, prediction in adopted.items():
        if k[2] == "clean":
            continue
        inputs = joblib.load(audit.EVAL_INPUTS / f"{audit.FILE[k[1]]}_{k[2]}.joblib")["inputs"]
        out[k] = v2.profile(inputs, prediction)
    return out


def _provider_mean(rows, key):
    out = {}
    for (provider, _, _), r in rows.items():
        if r.get(key) is not None:
            out.setdefault(provider, []).append(r[key])
    return {p: float(np.mean(v)) for p, v in out.items()}


def run_fidelity():
    organizer = audit.read(CALIBRATION)["organizer_profile"]
    v1 = _v1_profiles()
    v2rows = {}
    for f in glob.glob(str(OUT / "scores" / "*.json")):
        match, rep = Path(f).name[:-5].split("__")
        v2rows[(audit.PROVIDER[match], match, rep)] = audit.read(f)["profile_predicted"]
    table = {}
    for key, target in organizer.items():
        p1, p2 = _provider_mean(v1, key), _provider_mean(v2rows, key)
        table[key] = dict(organizer=target, feed_like_v1=p1, feed_like_v2=p2)
    full_pitch = ("Sportec", "Metrica")

    def gap(kind):
        errs = []
        for key, row in table.items():
            scale = max(abs(row["organizer"]), .05)
            errs += [abs(row[kind][p]-row["organizer"])/scale for p in full_pitch if p in row[kind]]
        return float(np.median(errs))
    audit.write(OUT / "fidelity.json", dict(table=table, median_relative_error_full_pitch=dict(
        feed_like_v1=gap("feed_like_v1"), feed_like_v2=gap("feed_like_v2"))))


def summarize():
    rows = {}
    for f in glob.glob(str(OUT / "scores" / "*.json")):
        match, rep = Path(f).name[:-5].split("__")
        rows[(audit.PROVIDER[match], match, rep)] = audit.read(f)
    arms = sorted({k for r in rows.values() for k in r if isinstance(r[k], dict) and "spell_f1" in r[k]})
    out = {}
    for arm in arms:
        sub = {k: r[arm] for k, r in rows.items() if arm in r}
        if len(sub) != len(rows):
            continue
        spell, per = audit.equal_provider({k: r["spell_f1"] for k, r in sub.items()})
        out[arm] = dict(spell_f1=spell, providers=per,
                        macro_f1=audit.equal_provider({k: r["macro_f1"] for k, r in sub.items()})[0],
                        full_pitch_spell_f1=float(np.mean([per["Sportec"], per["Metrica"]])),
                        per_replicate=[audit.equal_provider({k: r["spell_f1"] for k, r in sub.items()
                                                             if k[2] == str(rep)})[0] for rep in REPLICATES])
    return out


def run_report():
    cal = audit.read(CALIBRATION)
    audit.write(RESULT, dict(study=STUDY, organizer_reference_access=False, public_scoring=False,
                             promotion=False, frame_arrays_published=False, replicates=list(REPLICATES),
                             organizer_profile=cal["organizer_profile"], fidelity=audit.read(OUT / "fidelity.json"),
                             arms=summarize()))


STEPS = dict(calibrate=run_calibrate, views=run_views, fidelity=run_fidelity, report=run_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=(*STEPS, "all"))
    step = parser.parse_args().step
    for name in (STEPS if step == "all" else (step,)):
        print("STEP", name, flush=True)
        STEPS[name]()
