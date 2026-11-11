"""Shared roots and cell helpers for the `sc2020_realfeed_v1` study.

The study adds six unused SkillCorner 2019/20 broadcast-tracking TRAIN matches
(real missing-ball and partial-visibility defects, provider H/A silver labels)
to the frontier CE continuation. This module only binds restored archive roots
and exposes input/emission helpers; it never reads evaluation labels into
inference and does not modify any frozen runner.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT.parent
os.environ.setdefault("MTAH_DATA_DIR", str(HOME / "mtah-data"))
os.environ.setdefault("MTAH_SQD_DIR", str(HOME / "mtah-sqd"))
os.environ.setdefault("MTAH_DT_DIR", str(HOME / "mtah-dt"))
os.environ.setdefault("MTAH_RF_ARCHIVE_DIR", str(HOME / "mtah-rf/experiments/feed_robustness_v1/artifacts"))
os.environ.setdefault("V2_ITERATIVE_GAR_ARCHIVE", str(
    HOME / "mtah-dt/experiments/dense_targeted_followups_v1/artifacts/train_records"))
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from experiments import v2_spell_target_runtime as runtime  # noqa: E402,F401
from experiments import v2_spell_target as base  # noqa: E402
from experiments import v2_data_capacity_eval as capacity  # noqa: E402
from experiments import v2_short_original_diversity as recipe  # noqa: E402
from experiments import tracking_corruption  # noqa: E402

STUDY = "sc2020_realfeed_v1"
OUT = ROOT / "data/experiments" / STUDY
FRONTIER = Path(os.environ.get("FRONTIER_RECOVERY_DIR", str(HOME / "fr/data/experiments/frontier_recovery_v1")))
SEEDS = (2026, 2027)
FOLDS = ("leave_out_metrica", "leave_out_skillcorner", "leave_out_sportec")
VIEWS = (("clean", (0,)), ("feed_like_v1", (1, 2)), ("feed_like_v2", (1, 2, 3, 4)))


def configure():
    data = Path(os.environ["MTAH_DATA_DIR"]).resolve()
    recipe.rf.old.DATA = data
    recipe.rf.sc.DATA = data
    recipe.OUT = OUT / "train_views"
    recipe.ARCHIVE_ROOT = Path(os.environ["MTAH_RF_ARCHIVE_DIR"])
    recipe.TRAIN_RECORDS = OUT / "gar_records"
    link = recipe.TRAIN_RECORDS / "train"
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path(os.environ["V2_ITERATIVE_GAR_ARCHIVE"]), target_is_directory=True)
    base._roots()
    runtime.frozen._roots()
    runtime.frozen.tc = tracking_corruption
    return data


def cells():
    return [(p, m, v, r) for p, m, _ in base.audit.MATCHES for v, reps in VIEWS for r in reps]


def cell_key(provider, match, view, rep):
    return f"{provider}__{match}__{view}__{rep}"


def frontier_cell(key):
    with np.load(FRONTIER / "cells" / f"{key}.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files if not k.startswith("identity__")}


def prepared(provider, match, view, rep):
    """Native inputs and 5 Hz phase records for one development cell (cached)."""
    path = OUT / "cells" / f"{cell_key(provider, match, view, rep)}.joblib"
    if path.is_file():
        return joblib.load(path)
    _, inputs, labels, phases, model_period = capacity._raw_cache(provider, match, view, rep)
    payload = dict(inputs=inputs, labels=labels, phases=phases, model_period=model_period)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(payload, tmp, compress=3)
    os.replace(tmp, path)
    return payload


def predict(payload, model, teacher):
    """Per-phase native H/A/D probabilities for one fold model."""
    out = []
    for rec in payload["phases"]:
        record = base.old.inference_record(rec)
        record["names"] = list(record["names"])
        out.append(base.old.upsample(base.sn.predict(record, model, teacher), record,
                                     base.tc.inputs_only(payload["inputs"])))
    return np.stack(out)
