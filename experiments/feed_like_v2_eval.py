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


if __name__ == "__main__":
    print(run())
