"""Score sealed arms on every development benchmark: original clean/feed-like cells,
the audit's random/matched gap views and the feed_like_v2 views (via the benchmark).

No organizer labels, public scoring or promotion. Writes aggregate JSON only.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import glob
import os
from pathlib import Path

import joblib
import numpy as np

from experiments import feed_benchmark_audit_v1 as audit, feed_like_v2_benchmark as bench
from experiments import supervised

STUDY = "feed_like_v2_eval"
OUT = audit.ROOT / "data/experiments" / STUDY
RESULT = audit.ROOT / "experiments/results" / f"{STUDY}.json"
ARMS = tuple(os.environ.get("EVAL_ARMS", "v2train").split(","))
AUDIT_VIEWS = audit.OUT / "views"


def _original_job(arg):
    arm, name = arg
    dest = OUT / "original" / f"{arm}__{name}.json"
    if dest.exists():
        return str(dest)
    match = next(m for m in audit.PROVIDER if name.startswith(audit.FILE[m]))
    cached, cell = audit.tracking_cell(audit.EVAL_INPUTS / f"{name}.joblib")
    p = audit.predict(cached["inputs"], cached["record"], arm, audit.PROVIDER[match])
    r = audit.score(cell, audit.decode(p, cell["model_period"], cell["fps"]))
    dest.parent.mkdir(parents=True, exist_ok=True)
    audit.write(dest, {k: r[k] for k in ("spell_f1", "macro_f1", "d_f1", "short_spell_recall_iou_0_5_lt3s")})
    return str(dest)


def _audit_view_job(arg):
    arm, path = arg
    match, kind, rep = Path(path).name[:-7].split("__")
    dest = OUT / "audit_views" / f"{arm}__{match}__{kind}__{rep}.json"
    if dest.exists():
        return str(dest)
    cached, cell = audit.tracking_cell(path)
    p = audit.predict(cached["inputs"], cached["record"], arm, audit.PROVIDER[match])
    r = audit.score(cell, audit.decode(p, cell["model_period"], cell["fps"]))
    dest.parent.mkdir(parents=True, exist_ok=True)
    audit.write(dest, {k: r[k] for k in ("spell_f1", "macro_f1", "d_f1")})
    return str(dest)


def run():
    names = sorted({Path(f).name[:-7] for f in glob.glob(str(audit.EVAL_INPUTS / "*.joblib"))})
    views = sorted(glob.glob(str(AUDIT_VIEWS / "*__random__*.joblib")) + glob.glob(str(AUDIT_VIEWS / "*__matched__*.joblib")))
    with ProcessPoolExecutor(audit.WORKERS) as ex:
        list(ex.map(_original_job, [(a, n) for a in ARMS for n in names]))
        list(ex.map(_audit_view_job, [(a, v) for a in ARMS for v in views]))
    out = {}
    for arm in ARMS:
        rows = {}
        for f in glob.glob(str(OUT / "original" / f"{arm}__*.json")):
            name = Path(f).name[len(arm)+2:-5]
            match = next(m for m in audit.PROVIDER if name.startswith(audit.FILE[m]))
            view = "clean" if name.endswith("clean_clean") else "feed_like_" + name.rsplit("feed_like_", 1)[1]
            rows[(audit.PROVIDER[match], match, view)] = audit.read(f)
        entry = {"original": {"spell": audit.summary(rows), "macro": audit.summary(rows, "macro_f1")}}
        for kind in ("random", "matched"):
            sub = {}
            for f in glob.glob(str(OUT / "audit_views" / f"{arm}__*__{kind}__*.json")):
                _, match, _, rep = Path(f).name[:-5].split("__")
                sub[(audit.PROVIDER[match], match, rep)] = audit.read(f)["spell_f1"]
            entry[kind] = dict(zip(("spell_f1", "providers"), audit.equal_provider(sub))) if sub else None
        out[arm] = entry
    audit.write(RESULT, dict(study=STUDY, organizer_reference_access=False, public_scoring=False,
                             promotion=False, arms=out))
    return out





# ---------------------------------------------------------------- scorecard (all arms, all views)

TTA_ARMS = ("raw", "v2train")


def _tta_original_job(arg):
    arm, name = arg
    dest = OUT / "original_tta" / f"{arm}__{name}.json"
    if dest.exists():
        return str(dest)
    match = next(m for m in audit.PROVIDER if name.startswith(audit.FILE[m]))
    cached, cell = audit.tracking_cell(audit.EVAL_INPUTS / f"{name}.joblib")
    out = {}
    phases = [audit.predict(cached["inputs"], r, arm, audit.PROVIDER[match])
              for r in audit.phase_records(cached["inputs"], cached["record"])]
    for mode, p in (("phase0", phases[0]), ("phase_tta", np.mean(phases, 0))):
        r = audit.score(cell, audit.decode(p, cell["model_period"], cell["fps"]))
        out[mode] = {k: r[k] for k in ("spell_f1", "macro_f1", "d_f1")}
    dest.parent.mkdir(parents=True, exist_ok=True)
    audit.write(dest, out)
    return str(dest)


def _tta_v2_job(arg):
    """Regenerate the benchmark's deterministic v2 view and add phase-TTA arms."""
    arm, provider, match, rep = arg
    dest = bench.OUT / "scores" / f"{match}__{rep}.json"
    have = audit.read(dest)
    key = f"{arm}_phase_tta"
    if key in have:
        return str(dest)
    import hashlib
    from experiments import feed_like_v2 as v2, preparation as prep
    calibration = audit.read(bench.CALIBRATION)["calibration"]
    cached = joblib.load(audit.EVAL_INPUTS / f"{audit.FILE[match]}_clean_clean.joblib")
    labels = np.asarray(cached["labels"]).astype(str)
    seed = int(hashlib.sha256(f"{bench.STUDY}/{rep}/{match}".encode()).hexdigest()[:8], 16)
    if seed != have["arm_seed"]:
        raise ValueError("v2 view seed differs from the scored benchmark view")
    view = v2.corrupt(cached["inputs"], labels, calibration, seed)
    record = prep.build_record(view, labels, provider, audit.FILE[match])
    cell = dict(labels=labels, period=view["period"], timestamps=view["timestamps"], fps=float(view["fps"]),
                model_period=supervised.model_periods(view))
    phases = [audit.predict(view, r, arm, provider) for r in audit.phase_records(view, record)]
    if not np.isclose(audit.score(cell, audit.decode(phases[0], cell["model_period"], cell["fps"]))["spell_f1"],
                      have[arm]["spell_f1"]):
        raise ValueError("Regenerated v2 view does not replay the benchmark score")
    r = audit.score(cell, audit.decode(np.mean(phases, 0), cell["model_period"], cell["fps"]))
    have = audit.read(dest)  # re-read: another worker may have added a different arm
    have[key] = {k: r[k] for k in ("spell_f1", "macro_f1", "d_f1", "predicted_spell_count",
                                   "reference_spell_count", "short_spell_recall_iou_0_5_lt3s")}
    audit.write(dest, have)
    return str(dest)


def scorecard():
    names = sorted({Path(f).name[:-7] for f in glob.glob(str(audit.EVAL_INPUTS / "*.joblib"))})
    with ProcessPoolExecutor(audit.WORKERS) as ex:
        list(ex.map(_tta_original_job, [(a, n) for a in TTA_ARMS for n in names]))
    for arm in TTA_ARMS:  # one arm at a time: workers update the same per-view score files
        with ProcessPoolExecutor(audit.WORKERS) as ex:
            list(ex.map(_tta_v2_job, [(arm, p, m, r) for p, m, _ in audit.MATCHES for r in bench.REPLICATES]))
    tta = audit.read(audit.OUT / "tta.json")
    card = {}

    def original(arm, mode):
        if arm in ("adop", "cand"):
            entry = tta[f"{arm}_{mode}"]
            return {v: dict(spell=entry["spell"][v]["equal_provider_mean"], macro=entry["macro"][v]["equal_provider_mean"])
                    for v in ("clean", "feed_like")}
        rows = {}
        for f in glob.glob(str(OUT / "original_tta" / f"{arm}__*.json")):
            name = Path(f).name[len(arm)+2:-5]
            match = next(m for m in audit.PROVIDER if name.startswith(audit.FILE[m]))
            view = "clean" if name.endswith("clean_clean") else "feed_like_" + name.rsplit("feed_like_", 1)[1]
            rows[(audit.PROVIDER[match], match, view)] = audit.read(f)[mode]
        s, m = audit.summary(rows), audit.summary(rows, "macro_f1")
        return {v: dict(spell=s[v]["equal_provider_mean"], macro=m[v]["equal_provider_mean"]) for v in ("clean", "feed_like")}
    v2arms = bench.summarize()
    for arm in ("adop", "cand", "raw", "v2train"):
        for mode in ("phase0", "phase_tta"):
            key = arm if mode == "phase0" else f"{arm}_phase_tta"
            o = original(arm, mode)
            card[key] = dict(clean=o["clean"], v1_feed_like=o["feed_like"],
                             v2=dict(spell=v2arms[key]["spell_f1"], macro=v2arms[key]["macro_f1"],
                                     full_pitch_spell=v2arms[key]["full_pitch_spell_f1"]))
    audit.write(audit.ROOT / "experiments/results/feed_like_v2_scorecard.json",
                dict(study=STUDY, arms=card, public_scored_arms_reused={
                    "adop": "short_quality_public_v1 adopted_baseline_reused: spell .630303, macro .864181",
                    "cand": "short_quality_public_v1 candidate: spell .611765, macro .851742"},
                     organizer_reference_access=False, public_scoring=False))
    return card


if __name__ == "__main__":
    import sys
    print(scorecard() if sys.argv[1:] == ["scorecard"] else run())
