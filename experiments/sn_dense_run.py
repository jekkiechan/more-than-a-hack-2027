"""Phase-C experiment runner for the dense-supervision SoccerNet study (``sn_dense_v1``).

Commands (``python -m experiments.sn_dense_run <cmd>``):

* ``declare``   write the frozen declaration (refuses to overwrite).
* ``reference`` score fold teachers (dev cells + GAR-valid) and pooled references (GAR-valid).
* ``fit``       one fold fit + dev-cell scoring + GAR-valid scoring (json receipt, joblib model).
* ``queue``     run a list of fits with N worker processes.
* ``ensemble``  score the seed-2026 + seed-2027 probability ensemble of a tag.
* ``report``    aggregate receipts into ``experiments/results/sn_dense_v1_progress.md``.

SoccerNet-derived arrays and weights stay under ignored ``data/experiments/sn_dense_v1``;
only aggregates are written to ``experiments/results``. GAR *test* is never read here,
organizer public ground truth is never touched.

GAR-valid validator: the official (frozen) ``supervised.decode`` is a Python Viterbi loop
(about 1M native frames of valid). ``viterbi_c`` below is a C re-implementation with the same
arithmetic and tie-breaking (first maximum); ``tests/test_sn_dense_run.py`` checks it is
bit-identical to ``supervised.decode``. Dev-cell scoring keeps the frozen ``supervised.decode``.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import (factorized_had as fh, learned_temporal as lt, soccernet_capacity as sc,  # noqa: E402
                         soccernet_dense_labels as dl, soccernet_extension as old,
                         sn_dense_train as sn, supervised)

DATA = sc.DATA
OUT = DATA / "experiments/sn_dense_v1"
MODELS = OUT / "models"
EVALS = OUT / "evals"
RESULTS = ROOT / "experiments/results"
FOLDS = sc.FOLDS
FOLD_SHORT = {f: f.removeprefix("leave_out_") for f in FOLDS}
POOLED_DIR = Path("/home/user/mtah-data/models/joint_gru_soccernet_capacity_ensemble_v1")
BASELINE_CLEAN_SPELL = 0.626425      # saved seed-2026 combined-teacher baseline (declared)
PUBLISHED_ENSEMBLE = dict(clean=0.641790, feed_like=0.502586)

ARMS = {
    "R": dict(ha_target="anchors", train_d=False, steps=2048, early=False),
    "D1": dict(ha_target="dense", train_d=False, steps=2048, early=False),
    "D2": dict(ha_target="dense", train_d=True, steps=2048, early=False),
    "D3": dict(ha_target="dense", train_d=True, steps=8192, early=True),
}
DROP_KEYS = ("full_player_observed", "full_ball_observed", "full_prov", "full_y_literal",
             "full_game_event_type", "full_frame_numbers", "sampled_frame_numbers",
             "anchor_frame_mapping", "touch_runs", "touch_run_columns", "prov", "out_frames")


# ------------------------------------------------------------------ fast exact Viterbi
_C_SOURCE = r"""
#include <stdlib.h>
void viterbi(const double *logp, long n, int k, double pen, int *out, signed char *back) {
    double value[8], nv[8];
    for (int j = 0; j < k; j++) value[j] = logp[j];
    for (long r = 1; r < n; r++) {
        for (int j = 0; j < k; j++) {
            int best = 0;
            double bv = (j == 0) ? value[0] : value[0] - pen;
            for (int i = 1; i < k; i++) {
                double t = (i == j) ? value[i] : value[i] - pen;
                if (t > bv) { bv = t; best = i; }
            }
            back[r * k + j] = (signed char)best;
            nv[j] = bv + logp[r * k + j];
        }
        for (int j = 0; j < k; j++) value[j] = nv[j];
    }
    int state = 0;
    for (int j = 1; j < k; j++) if (value[j] > value[state]) state = j;
    for (long r = n - 1; r >= 0; r--) {
        out[r] = state;
        state = back[r * k + state];
    }
}
"""
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        OUT.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(_C_SOURCE.encode()).hexdigest()[:12]
        so = OUT / f"viterbi_{digest}.so"
        if not so.exists():
            src = OUT / f"viterbi_{digest}.c"
            src.write_text(_C_SOURCE)
            tmp = so.with_suffix(f".{os.getpid()}.tmp")
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(tmp), str(src)], check=True)
            os.replace(tmp, so)
        lib = ctypes.CDLL(str(so))
        lib.viterbi.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_double,
                                ctypes.c_void_p, ctypes.c_void_p]
        _LIB = lib
    return _LIB


def viterbi_c(probabilities, period, fps, switch_cost=0):
    """Drop-in, bit-identical replacement for ``supervised.decode``."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    valid_classes = np.flatnonzero(probabilities.sum(axis=0) > 0)
    probabilities = probabilities[:, valid_classes]
    if switch_cost <= 0:
        return supervised.LABELS[valid_classes[np.argmax(probabilities, axis=1)]]
    logp = np.ascontiguousarray(np.log(np.maximum(probabilities, 1e-7)))
    penalty = switch_cost * fps / 10
    k = len(valid_classes)
    result = np.empty(len(period), dtype=np.int8)
    lib = _lib()
    for start, stop in supervised.blocks(period):
        n = stop - start
        block = np.ascontiguousarray(logp[start:stop])
        out = np.empty(n, dtype=np.int32)
        back = np.zeros(n * k, dtype=np.int8)
        lib.viterbi(block.ctypes.data, n, k, float(penalty), out.ctypes.data, back.ctypes.data)
        result[start:stop] = valid_classes[out]
    return supervised.LABELS[result]


# ------------------------------------------------------------------------ GAR-valid
class ValidScorer:
    """Dense-silver GAR-valid scoring; mirrors ``sn.make_gar_validator`` with the fast decode.

    ``native(model, teacher)`` -> per-record native-rate probabilities; ``score(natives)``
    decodes and scores them (average several models' natives first for an ensemble).
    """

    def __init__(self, records: list[dict]):
        from experiments import objective_ensemble as scoring
        self.scoring = scoring
        self.records = records
        self.fps = None
        self.truth, self.periods, self.stamps, self.infos, self.rec_fps = [], [], [], [], []
        text = np.array(["H", "A", "D", "X"])
        offset = 0
        for r in records:
            fp = np.asarray(r["full_period"])
            fps = sn._native_fps(r)
            self.fps = fps if self.fps is None else self.fps
            if abs(fps - self.fps) > 1e-4 * self.fps:   # per-game fps estimates jitter at 1e-7
                raise ValueError("Validation records must share one native fps")
            self.rec_fps.append(fps)
            self.infos.append((dict(indices=np.asarray(r["indices"]), native_n=len(r["full_y"])), fp))
            self.truth.append(text[np.where(np.asarray(r["full_y"]) < 0, 3, r["full_y"])])
            self.periods.append(fp - fp.min() + offset)
            offset = int(self.periods[-1].max()) + 1
            self.stamps.append(np.asarray(r["full_timestamps"], dtype=float))

    def native(self, model: dict, teacher: dict) -> list[np.ndarray]:
        out = []
        for r, (info, fp) in zip(self.records, self.infos):
            prob = sn.predict(r, model, teacher)
            out.append(np.column_stack([lt.upsample(prob[:, k], info, fp) for k in range(3)]))
        return out

    def teacher_native(self, teacher: dict) -> list[np.ndarray]:
        out = []
        for r, (info, fp) in zip(self.records, self.infos):
            prob = fh.predict({k: r[k] for k in ("x", "names", "period")}, teacher)
            out.append(np.column_stack([lt.upsample(prob[:, k], info, fp) for k in range(3)]))
        return out

    def score(self, natives: list[np.ndarray]) -> dict:
        pred = [np.asarray(viterbi_c(nat, fp, rf, lt.COST)).astype(str)
                for nat, (_, fp), rf in zip(natives, self.infos, self.rec_fps)]
        inputs = dict(period=np.concatenate(self.periods), fps=float(np.mean(self.rec_fps)),
                      timestamps=np.concatenate(self.stamps))
        row = self.scoring.score_prediction(np.concatenate(self.truth), np.concatenate(pred),
                                            inputs, "gar_valid", {}, "n/a")
        keys = ("spell_f1", "macro_f1", "weighted_70_30", "d_f1", "d_precision", "d_recall",
                "short_spell_recall", "short_reference_spells", "n_gt_spells", "n_pred_spells",
                "graded_frames")
        out = {k: row[k] for k in keys if k in row}
        out["score"] = float(row["spell_f1"])
        return out

    def validator(self, teacher: dict):
        return lambda model: self.score(self.native(model, teacher))


def load_valid() -> ValidScorer:
    return ValidScorer(strip(dl.load_dense_records("valid")))


def strip(records: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in DROP_KEYS} for r in records]


# ------------------------------------------------------------------------- dev cells
def dev_rows(model: dict | None, teacher: dict, fold: str, arm: str, member: str,
             return_native: bool = False):
    """Score the fold's provider dev cells; ``model=None`` scores the teacher itself."""
    rows, natives = [], {}
    for item in old.selection_items():
        if "leave_out_" + item["source"] != fold:
            continue
        for view, replicate in old.VIEWS:
            cached = joblib.load(old.eval_path(item, view, replicate))
            inputs, labels = cached["inputs"], cached["labels"]
            record = old.inference_record(cached["record"])
            if model is None:
                prob = fh.predict(record, teacher)
            else:
                prob = sn.predict(record, model, teacher)
            native = old.upsample(prob, record, inputs)
            row, _ = old.metric_row(native, inputs, labels, item, view, replicate, arm, member)
            rows.append(row)
            natives[(item["match"], view, replicate)] = native
    return (rows, natives) if return_native else rows


def teacher_parity(fold: str, seed: int) -> float:
    """Max abs probability difference of rebuilt teacher dev probabilities vs saved arrays."""
    teacher = joblib.load(sc.initial_path(fold, seed))
    worst = 0.0
    for item in old.selection_items():
        if "leave_out_" + item["source"] != fold:
            continue
        for view, replicate in old.VIEWS:
            cached = joblib.load(old.eval_path(item, view, replicate))
            record = old.inference_record(cached["record"])
            native = old.upsample(fh.predict(record, teacher), record, cached["inputs"])
            with np.load(old.historical_path(item, view, replicate, seed), allow_pickle=False) as s:
                worst = max(worst, float(np.abs(native - s["probability"]).max()))
    return worst


def aggregate(rows: list[dict], arm: str, member: str) -> dict:
    return {view: sc._aggregate_provider(rows, arm, view, member) for view in ("clean", "feed_like")}


# --------------------------------------------------------------------------- fitting
def tag_of(arm: str, fraction: float, games: int, seed: int, steps: int | None = None) -> str:
    extra = "" if steps is None else f"_n{steps}"
    return f"{arm}_f{int(round(fraction * 100)):03d}_g{games:02d}{extra}_s{seed}"


def model_path(tag: str, fold: str) -> Path:
    return MODELS / f"{tag}__{FOLD_SHORT[fold]}.joblib"


def eval_path(tag: str, fold: str) -> Path:
    return EVALS / f"{tag}__{FOLD_SHORT[fold]}.json"


def load_train(anchor_mode: bool) -> list[dict]:
    records = strip(dl.load_dense_records("train"))
    if not anchor_mode:
        for r in records:
            r.pop("anchor_y", None); r.pop("anchor_full_y", None)
        return records
    kept = []
    for r in records:
        if np.isin(r["anchor_y"], (0, 1)).any():
            kept.append(dict(r, y=r["anchor_y"], full_y=r["anchor_full_y"]))
    return kept


def fit(spec: dict) -> None:
    arm, fold, seed = spec["arm"], spec["fold"], int(spec["seed"])
    params = dict(ARMS[arm])
    steps = int(spec.get("steps") or params["steps"])
    early = bool(spec.get("early", params["early"]))
    fraction, games = float(spec["fraction"]), int(spec["games"])
    tag = spec.get("tag") or tag_of(arm, fraction, games, seed, spec.get("steps"))
    out_json = eval_path(tag, fold)
    if out_json.exists():
        print("SKIP", out_json.name, flush=True); return
    MODELS.mkdir(parents=True, exist_ok=True); EVALS.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    teacher = joblib.load(sc.initial_path(fold, seed))
    old_records = old.load_old(fold) if fraction < 1 else []
    train_records = load_train(params["ha_target"] == "anchors")
    ids = sorted({str(np.asarray(r["match"]).reshape(()).item()) for r in train_records})
    subset = None if games >= len(ids) else sn.nested_games(ids, games)
    valid = load_valid()
    kwargs = {}
    if early:
        kwargs = dict(eval_every=512, patience=3, validate=valid.validator(teacher))
    out = sn.train(teacher, old_records, train_records, width=96, lr=3e-4, seed=seed,
                   steps=steps, fraction=fraction, batch_size=32, ha_target=params["ha_target"],
                   train_d=params["train_d"], soccernet_games=subset, require_full_native=True,
                   log_every=256, **kwargs)
    model = out["model"]
    joblib.dump(model, model_path(tag, fold))
    del old_records, train_records
    gc.collect()
    rows = dev_rows(model, teacher, fold, tag, str(seed))
    valid_metrics = valid.score(valid.native(model, teacher))
    receipt = dict(tag=tag, fold=fold, seed=seed, spec=spec, steps_run=int(model["step"]),
                   best_step=out["best_step"], early=early, fraction=fraction, games=games,
                   history=out["history"], config={k: v for k, v in model["training"].items()
                                                   if k != "soccernet_games"},
                   n_games=len(model["training"].get("soccernet_games", [])),
                   model_sha256=hashlib.sha256(model_path(tag, fold).read_bytes()).hexdigest(),
                   dev_rows=rows, valid=valid_metrics, wall_seconds=time.monotonic() - start)
    if not np.isfinite(valid_metrics["score"]):
        raise RuntimeError("non-finite validation score")
    tmp = out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, default=float))
    os.replace(tmp, out_json)
    print("FIT_DONE", tag, fold, "%.0fs" % receipt["wall_seconds"],
          "valid_spell=%.4f" % valid_metrics["spell_f1"], flush=True)


def queue(specs: list[dict], workers: int) -> None:
    pending = [s for s in specs if not eval_path(
        s.get("tag") or tag_of(s["arm"], float(s["fraction"]), int(s["games"]), int(s["seed"]),
                               s.get("steps")), s["fold"]).exists()]
    running: list[tuple[subprocess.Popen, dict]] = []
    log_dir = OUT / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    while pending or running:
        running = [(p, s) for p, s in running if p.poll() is None]
        while pending and len(running) < workers:
            spec = pending.pop(0)
            name = "{}_{}_{}_{}".format(spec["arm"], spec["fraction"], spec["games"], spec["seed"])
            name += "_" + FOLD_SHORT[spec["fold"]] + (f"_n{spec['steps']}" if spec.get("steps") else "")
            log = open(log_dir / f"{name}.log", "ab")
            proc = subprocess.Popen([sys.executable, "-m", "experiments.sn_dense_run", "fit",
                                     "--spec", json.dumps(spec)], stdout=log, stderr=subprocess.STDOUT,
                                    env=env, cwd=ROOT)
            running.append((proc, spec))
            print("STARTED", name, flush=True)
        time.sleep(5)
    print("QUEUE_DONE", flush=True)


# ---------------------------------------------------------------------- references
def reference() -> None:
    EVALS.mkdir(parents=True, exist_ok=True)
    valid = load_valid()
    result = dict(folds={}, pooled={})
    for fold in FOLDS:
        teacher = joblib.load(sc.initial_path(fold, 2026))
        parity = teacher_parity(fold, 2026)
        rows = dev_rows(None, teacher, fold, "teacher", "2026")
        vm = valid.score(valid.teacher_native(teacher))
        result["folds"][fold] = dict(dev_rows=rows, valid=vm, teacher_probability_parity=parity)
        print("REF", fold, "parity", parity, "valid", vm["spell_f1"], flush=True)
    members = {}
    for seed in (2026, 2027):
        teacher = joblib.load(POOLED_DIR / f"teacher_{seed}.joblib")
        capacity = joblib.load(POOLED_DIR / f"capacity_{seed}.joblib")
        t_nat = valid.teacher_native(teacher)
        c_nat = valid.native(capacity, teacher)
        members[seed] = (t_nat, c_nat)
        result["pooled"][f"teacher_{seed}"] = valid.score(t_nat)
        result["pooled"][f"capacity_{seed}"] = valid.score(c_nat)
        print("REF pooled", seed, result["pooled"][f"teacher_{seed}"]["spell_f1"],
              result["pooled"][f"capacity_{seed}"]["spell_f1"], flush=True)
    for i, name in enumerate(("teacher", "capacity")):
        avg = [(a + b) * 0.5 for a, b in zip(members[2026][i], members[2027][i])]
        result["pooled"][f"{name}_ensemble"] = valid.score(avg)
    (EVALS / "reference.json").write_text(json.dumps(result, default=float))
    print("REFERENCE_DONE", flush=True)


def ensemble(tag_a: str, tag_b: str, name: str) -> None:
    """Probability-average of two seeds' fold models on dev cells and GAR-valid."""
    valid = load_valid()
    result = dict(name=name, tags=[tag_a, tag_b], folds={})
    all_rows = []
    for fold in FOLDS:
        teachers = {s: joblib.load(sc.initial_path(fold, s)) for s in (2026, 2027)}
        models = {2026: joblib.load(model_path(tag_a, fold)), 2027: joblib.load(model_path(tag_b, fold))}
        per = {s: dev_rows(models[s], teachers[s], fold, name, str(s), return_native=True)
               for s in (2026, 2027)}
        rows = []
        for key in per[2026][1]:
            item = next(i for i in old.selection_items() if i["match"] == key[0])
            cached = joblib.load(old.eval_path(item, key[1], key[2]))
            avg = (per[2026][1][key] + per[2027][1][key]) * 0.5
            row, _ = old.metric_row(avg, cached["inputs"], cached["labels"], item, key[1], key[2],
                                    name, "ensemble")
            rows.append(row)
        all_rows += rows
        nat = [valid.native(models[s], teachers[s]) for s in (2026, 2027)]
        vm = valid.score([(a + b) * 0.5 for a, b in zip(*nat)])
        result["folds"][fold] = dict(dev_rows=rows, valid=vm)
        print("ENS", fold, vm["spell_f1"], flush=True)
    result["dev"] = aggregate(all_rows, name, "ensemble")
    (EVALS / f"ensemble_{name}.json").write_text(json.dumps(result, default=float))
    print("ENSEMBLE_DONE", name, flush=True)


# ------------------------------------------------------------------------ reporting
def load_tag(tag: str) -> dict | None:
    receipts = []
    for fold in FOLDS:
        p = eval_path(tag, fold)
        if not p.exists():
            return None
        receipts.append(json.loads(p.read_text()))
    rows = [r for rc in receipts for r in rc["dev_rows"]]
    seed = str(receipts[0]["seed"])
    dev = aggregate(rows, tag, seed)
    keys = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall", "short_spell_recall",
            "n_pred_spells", "n_gt_spells")
    valid = {k: float(np.mean([rc["valid"][k] for rc in receipts])) for k in keys}
    return dict(tag=tag, dev=dev, valid=valid, receipts=receipts,
                steps=[rc["steps_run"] for rc in receipts],
                best=[rc["best_step"] for rc in receipts],
                wall=sum(rc["wall_seconds"] for rc in receipts))


def reference_summary():
    path = EVALS / "reference.json"
    if not path.exists():
        return None
    ref = json.loads(path.read_text())
    rows = [r for f in FOLDS for r in ref["folds"][f]["dev_rows"]]
    dev = aggregate(rows, "teacher", "2026")
    valid = {k: float(np.mean([ref["folds"][f]["valid"][k] for f in FOLDS]))
             for k in ("spell_f1", "macro_f1", "d_f1", "short_spell_recall")}
    return dict(dev=dev, valid=valid, ref=ref)


def guard_ok(cand: dict, base: dict) -> bool:
    checks = sc.guard_checks(cand["dev"], base["dev"])
    return all(checks.values())


def select(tags: list[str], base_tag: str) -> dict:
    """Predeclared selection rule among ``tags`` relative to ``base_tag`` (the R arm)."""
    base = load_tag(base_tag)
    loaded = {t: load_tag(t) for t in tags}
    eligible = [t for t, v in loaded.items() if v and guard_ok(v, base)]
    pool = eligible or [base_tag]
    best = max(pool, key=lambda t: (loaded[t]["valid"]["spell_f1"] if loaded[t] else -1,
                                    loaded[t]["dev"]["clean"]["equal_provider_mean"]["spell_f1"]))
    return dict(best=best, eligible=eligible)


def fmt(v, d=4):
    return "n/a" if v is None else f"{v:.{d}f}"


def table(tags: list[str], base_tag: str, title: str) -> str:
    base = load_tag(base_tag)
    ref = reference_summary()
    lines = [f"### {title}", "",
             "| tag | steps (best) | dev clean spell | d vs R | d vs .626425 | dev clean macro | dev clean D-F1 | "
             "dev feed spell | feed d vs R | valid spell | valid macro | valid D-F1 | valid short-rec | guard | min |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if ref:
        c, f = ref["dev"]["clean"]["equal_provider_mean"], ref["dev"]["feed_like"]["equal_provider_mean"]
        v = ref["valid"]
        lines.append(f"| teacher (fold) | 0 | {fmt(c['spell_f1'])} | | | {fmt(c['macro_f1'])} | {fmt(c['d_f1'])} | "
                     f"{fmt(f['spell_f1'])} | | {fmt(v['spell_f1'])} | {fmt(v['macro_f1'])} | {fmt(v['d_f1'])} | "
                     f"{fmt(v['short_spell_recall'])} | | |")
    for t in tags:
        r = load_tag(t)
        if r is None:
            lines.append(f"| {t} | (incomplete) |" + " |" * 13); continue
        c, f = r["dev"]["clean"]["equal_provider_mean"], r["dev"]["feed_like"]["equal_provider_mean"]
        bc = base["dev"]["clean"]["equal_provider_mean"]["spell_f1"]
        bf = base["dev"]["feed_like"]["equal_provider_mean"]["spell_f1"]
        v = r["valid"]
        steps = "/".join(str(s) for s in r["steps"]) + ("" if not any(r["best"]) else
                " (" + "/".join(str(b) for b in r["best"]) + ")")
        lines.append(f"| {t} | {steps} | {fmt(c['spell_f1'])} | {c['spell_f1']-bc:+.4f} | "
                     f"{c['spell_f1']-BASELINE_CLEAN_SPELL:+.4f} | {fmt(c['macro_f1'])} | {fmt(c['d_f1'])} | "
                     f"{fmt(f['spell_f1'])} | {f['spell_f1']-bf:+.4f} | {fmt(v['spell_f1'])} | {fmt(v['macro_f1'])} | "
                     f"{fmt(v['d_f1'])} | {fmt(v['short_spell_recall'])} | {'ok' if guard_ok(r, base) else 'FAIL'} | "
                     f"{r['wall']/60:.0f} |")
    return "\n".join(lines) + "\n"


def report(sections: list[dict], extra: str = "") -> None:
    parts = ["# sn_dense_v1 progress (aggregates only; seed 2026 unless stated)", "",
             "Dev = 15 development cells (provider-held-out fold model on its provider, equal-provider "
             "mean). Valid = GAR-valid dense silver, mean of the 3 fold models. Aggregates only.", ""]
    ref = reference_summary()
    if ref:
        parts += ["### GAR-valid references (dense silver)", "", "| model | spell F1 | macro | D-F1 | short-rec |",
                  "|---|---|---|---|---|"]
        for k, v in ref["ref"]["pooled"].items():
            parts.append(f"| pooled {k} | {fmt(v['spell_f1'])} | {fmt(v['macro_f1'])} | {fmt(v['d_f1'])} | "
                         f"{fmt(v['short_spell_recall'])} |")
        parts.append("")
    for sec in sections:
        parts.append(table(sec["tags"], sec["base"], sec["title"]))
    parts.append(extra)
    (RESULTS / "sn_dense_v1_progress.md").write_text("\n".join(parts) + "\n")


# ------------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--spec", required=True)
    sub.add_parser("reference")
    q = sub.add_parser("queue"); q.add_argument("--specs", required=True); q.add_argument("--workers", type=int, default=3)
    e = sub.add_parser("ensemble"); e.add_argument("a"); e.add_argument("b"); e.add_argument("name")
    args = parser.parse_args()
    if args.cmd == "fit":
        fit(json.loads(args.spec))
    elif args.cmd == "reference":
        reference()
    elif args.cmd == "queue":
        queue(json.loads(Path(args.specs).read_text()), args.workers)
    elif args.cmd == "ensemble":
        ensemble(args.a, args.b, args.name)


if __name__ == "__main__":
    main()
