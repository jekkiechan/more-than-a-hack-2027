"""Bounded capacity/data-mix training arms for the v2 short-possession recipe.

The frozen PR82 reconstruction recipe remains the source of records, four v2
views, original-diversity sampler, GAR inputs and normalization. A scoped AST
adapter broadens only the shared trainer's width guard for the 192-unit arm.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import ast
from collections import Counter
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import time

import joblib
import numpy as np

from experiments import v2_short_original_diversity as recipe

ROOT = recipe.ROOT
STUDY = "v2_data_capacity_v1"
OUT = ROOT / "data/experiments" / STUDY
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
FOLDS, SEEDS = recipe.FOLDS, recipe.SEEDS
ARMS = ("width96_control", "width192_capacity", "mixed_v1_v2_width96")


def fit_path(arm: str, fold: str, seed: int) -> Path:
    _validate(arm, fold, seed)
    return OUT / "models" / arm / f"{fold}_{seed}.joblib"


def receipt_path(arm: str, fold: str, seed: int) -> Path:
    return Path(str(fit_path(arm, fold, seed)) + ".json")


def _bind_recipe_namespace() -> None:
    # Keep PR82 view seeds (recipe._view_seed's original namespace) while putting
    # all newly materialized artifacts under this study ID.
    recipe.OUT = OUT
    if os.environ.get("MTAH_RF_ARCHIVE_DIR"):
        recipe.ARCHIVE_ROOT = Path(os.environ["MTAH_RF_ARCHIVE_DIR"])
    elif os.environ.get("MTAH_RF_DIR"):
        recipe.ARCHIVE_ROOT = Path(os.environ["MTAH_RF_DIR"]) / "experiments/feed_robustness_v1/artifacts"
    if os.environ.get("SOCCERNET_GAR_TRAIN_RECORDS"):
        recipe.TRAIN_RECORDS = Path(os.environ["SOCCERNET_GAR_TRAIN_RECORDS"])


def _validate(arm, fold, seed):
    if arm not in ARMS or fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Unknown arm, provider fold, or fixed seed")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _network(n_features: int, hidden: int, input_dropout: float):
    if hidden not in (96, 192):
        raise ValueError("This study permits width 96 or 192 only")
    import torch
    from torch import nn
    class ConditionalNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_dropout = nn.Dropout(float(input_dropout))
            self.sequence = nn.GRU(int(n_features), int(hidden), batch_first=True,
                                   bidirectional=True)
            self.output = nn.Linear(2 * int(hidden), 2)
        def forward(self, x):
            return self.output(self.sequence(self.input_dropout(x))[0])
    return ConditionalNetwork()


class _WidthGuard(ast.NodeTransformer):
    """Widen only sn.train's fixed width tuple after exact source verification."""
    def __init__(self):
        self.matches = 0
    def visit_Compare(self, node):
        self.generic_visit(node)
        if (isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Name)
                and node.left.func.id == "int" and node.left.args
                and isinstance(node.left.args[0], ast.Name)
                and node.left.args[0].id == "width"
                and any(isinstance(op, ast.NotIn) for op in node.ops)
                and node.comparators and isinstance(node.comparators[0], ast.Tuple)
                and tuple(e.value for e in node.comparators[0].elts
                          if isinstance(e, ast.Constant)) == (24, 48, 96)):
            node.comparators[0] = ast.Tuple(elts=[ast.Constant(v) for v in (24, 48, 96, 192)],
                                            ctx=ast.Load())
            self.matches += 1
        return node


def _adapt_train():
    import experiments.sn_dense_train as sn
    source = textwrap.dedent(inspect.getsource(sn.train))
    tree = ast.parse(source)
    adapter = _WidthGuard()
    tree = adapter.visit(tree)
    if adapter.matches != 1:
        raise RuntimeError("Shared trainer width guard changed; refusing adaptation")
    ast.fix_missing_locations(tree)
    namespace = dict(sn.train.__globals__)
    exec(compile(tree, inspect.getsourcefile(sn.train), "exec"), namespace)
    return namespace["train"]


@contextmanager
def _training_adapter(width: int):
    """Temporarily route shared trainer construction through width-aware GRU."""
    import experiments.sn_dense_train as sn
    cap = sn.cap
    prior_network = cap.network
    prior_train = sn.train
    try:
        cap.network = _network
        if width == 192:
            sn.train = _adapt_train()
        yield sn.train
    finally:
        sn.train = prior_train
        cap.network = prior_network


def training(fold: str, arm: str = "width96_control") -> list[dict]:
    _bind_recipe_namespace()
    if fold not in FOLDS or arm not in ARMS:
        raise ValueError("Unknown fold or arm")
    records = recipe.training(fold)
    if arm == "mixed_v1_v2_width96":
        manifest, _ = recipe._v1_view_manifest()
        for record in records:
            views = record.get("_v2_feature_views")
            if views is None:
                continue
            match = str(record["_v2_view_match"])
            v1, _, _ = recipe._verified_v1_view(match, manifest)
            direct = np.asarray(v1["x"])
            if len(record.get("_v2_feature_views", ())) != recipe.N_REALIZATIONS:
                raise ValueError("Unexpected v2 feature view count")
            # Original and swap records get aligned, correctly oriented views.
            same_labels = np.array_equal(np.asarray(record["y"]), np.asarray(v1["y"]))
            flipped_labels = np.array_equal(
                np.asarray(record["y"]),
                np.where(np.isin(v1["y"], (0, 1)), 1-np.asarray(v1["y"]), v1["y"]))
            if same_labels == flipped_labels:
                raise ValueError(f"Cannot establish unique team orientation for {match}")
            if flipped_labels:
                from experiments import supervised
                direct = supervised.swap_team_features(direct, record["names"])
            record["_v2_feature_views"] = [direct, direct, direct, direct, *views]
            record["_v2_mix_families"] = ["v1"] * 4 + ["v2"] * 4
    return records


@contextmanager
def _view_draws(records, mixed: bool):
    cap = recipe.rf.sn.cap
    original = cap._prepare_window
    counts: Counter[str] = Counter()
    used: Counter[str] = Counter()
    def prepare_window(record, start, size, full, alternate, *, median, scale, swap):
        views = record.get("_v2_feature_views")
        if views is None:
            return original(record, start, size, full, alternate, median=median,
                            scale=scale, swap=swap)
        match = str(record["_v2_view_match"])
        choices = len(views)
        choice = counts[match] % choices
        counts[match] += 1
        label = ("v1" if choice < 4 else f"v2r{choice-4}") if mixed else f"v2r{choice}"
        used[f"{match}/{label}"] += 1
        selected = dict(record, x=views[choice], alternate=views[choice])
        return original(selected, start, size, full, False, median=median,
                        scale=scale, swap=swap)
    cap._prepare_window = prepare_window
    try:
        yield used
    finally:
        cap._prepare_window = original



def prepare(workers: int = 3) -> dict:
    _bind_recipe_namespace()
    if not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("Preparation worker count must be between 1 and 4")
    return recipe.prepare(workers=workers)


def _source_hashes() -> dict[str, str]:
    paths = {Path(__file__), *recipe._source_files(), *recipe._view_pipeline_files()}
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): _sha(path)
            for path in sorted(paths, key=str)}


def _input_hashes() -> dict[str, str]:
    _bind_recipe_namespace()
    preparation_path = OUT / "preparation.json"
    preparation = json.loads(preparation_path.read_text())
    if len(preparation.get("train_matches", {})) != 13:
        raise ValueError("Expected four prepared v2 views for all 13 original TRAIN matches")
    if preparation.get("calibration_sha256") != _sha(recipe.CALIBRATION):
        raise ValueError("Prepared views are bound to a different v2 calibration")
    if preparation.get("calibration_sha256") != _sha(recipe.CALIBRATION):
        raise ValueError("Prepared views are bound to a different v2 calibration")
    items = recipe.prep.split_plan()["train"]
    spec = recipe.rf.old.old_spec()
    inputs = {str(preparation_path.relative_to(ROOT)): _sha(preparation_path)}
    v1_manifest = recipe._archive_root().parents[2] / "feed_robustness_v1_cache_manifest.json"
    inputs[str(v1_manifest)] = _sha(v1_manifest)
    for item in items:
        raw = recipe.prep.data_path(item, recipe.rf.old.DATA)
        inputs[str(raw)] = _sha(raw)
        cache_entry = spec["prepared_training_caches"][item["match"]]
        cache = recipe.rf.old.DATA / Path(cache_entry["cache"]).relative_to("data")
        inputs[str(cache)] = _sha(cache)
        v1 = recipe._archive_root() / "views" / f"{item['match']}.joblib"
        inputs[str(v1)] = _sha(v1)
        for rep in range(recipe.N_REALIZATIONS):
            view = recipe._view_path(item["match"], rep)
            inputs[str(view.relative_to(ROOT))] = _sha(view)
            inputs[str(Path(str(view)+".json").relative_to(ROOT))] = _sha(Path(str(view)+".json"))
    manifest = recipe.TRAIN_RECORDS / "train/manifest.json"
    inputs[str(manifest)] = _sha(manifest)
    manifest_data = json.loads(manifest.read_text())
    for name in sorted(manifest_data["game_files"]):
        path = recipe.TRAIN_RECORDS / "train/games" / name
        inputs[str(path)] = _sha(path)
    return {"files": inputs,
            "teachers": {f"{fold}/{seed}": _sha(recipe.rf.sc.initial_path(fold, seed))
                         for fold in FOLDS for seed in SEEDS}}


def freeze() -> dict:
    _bind_recipe_namespace()
    if DECLARATION.exists():
        raise FileExistsError(f"Preserve the existing immutable declaration: {DECLARATION}")
    preparation = json.loads((OUT / "preparation.json").read_text())
    source_hashes, inputs = _source_hashes(), _input_hashes()
    value = {
        "study_id": STUDY, "status": "frozen_before_fit_or_candidate_scoring",
        "question": "Do a width-192 v2 model or an equal-mass v1/v2 training-input mixture improve the PR82 width-96 four-view recipe while meeting strict no-regression guards?",
        "arms": list(ARMS), "folds": list(FOLDS), "seeds": list(SEEDS), "fits": 18,
        "training": {"updates": 2048, "batch_size": 32, "fraction_gar": .5,
                     "learning_rate_ha": 3e-4, "learning_rate_d": 1e-4,
                     "train_d": True, "ha_target": "dense",
                     "normalization": "fold teacher training-only median/scale",
                     "sampler": "PR82 OriginalDiversitySampler; unchanged GAR quality/diversity and original short-possession bag policy",
                     "views": "four deterministic PR82-seeded feed_like_v2 training views per original match",
                     "widths": {"width96_control": 96, "width192_capacity": 192,
                                "mixed_v1_v2_width96": 96},
                     "mixed_view_mass": "equal v1/v2 family mass; four v2 realizations equally represented"},
        "cohort": {"provider_folds": {fold: recipe.prep.split_plan(fold=fold) for fold in FOLDS},
                   "heldout_rule": "evaluated provider excluded from all fit inputs and teacher checkpoint"},
        "source_sha256": source_hashes, "input_bindings": inputs,
        "preparation_sha256": _sha(OUT / "preparation.json"),
        "width_adapter": "Exact AST rewrite of one inspected sn.train width guard, with scoped cap.network and sn.train restoration; no frozen source file changes.",
        "controls": {"width96_adapter_parity": "before full fitting, two-step synthetic replay must match frozen sn.train loss log and all H/A/D state arrays exactly"},
        "evaluation_guards": {"minimum": "candidate clean/v1/v2 spell, macro, D-F1 each >= matched adopted controls; v2 spell >= PR82 eligible 0.5927491913246999; zero tolerance"},
        "budget": {"fits_per_arm": 6, "total_fits": 18, "steps_per_fit": 2048,
                   "candidate_scoring": False, "organizer_reference_access": False},
        "promotion": False,
        "limitations": "Five exposed original-provider development matches and synthetic feed_like views; no independent continuous-match confirmation."}
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    DECLARATION.write_text(json.dumps(value, indent=1, sort_keys=True)+"\n")
    return value


def check() -> dict:
    _bind_recipe_namespace()
    if not DECLARATION.is_file():
        raise FileNotFoundError("Freeze the capacity declaration before fitting/evaluation")
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("arms") != list(ARMS):
        raise ValueError("Training declaration identity or arm inventory mismatch")
    if declaration.get("source_sha256") != _source_hashes():
        raise RuntimeError("Training source closure changed after freeze")
    if declaration.get("input_bindings") != _input_hashes():
        raise RuntimeError("Training inputs changed after freeze")
    if declaration.get("preparation_sha256") != _sha(OUT / "preparation.json"):
        raise RuntimeError("Prepared training view manifest changed after freeze")
    return declaration


def load_fit(arm: str, fold: str, seed: int) -> dict:
    path, receipt = fit_path(arm, fold, seed), receipt_path(arm, fold, seed)
    if not path.is_file() or not receipt.is_file():
        raise FileNotFoundError(path)
    row = json.loads(receipt.read_text())
    if row.get("sha256") != _sha(path) or row.get("arm") != arm:
        raise ValueError(f"Fit receipt mismatch: {path}")
    return joblib.load(path)


def fit(fold: str, seed: int, arm: str = "width96_control") -> dict:
    _validate(arm, fold, seed)
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze the v2_data_capacity_v1 declaration before fitting")
    declaration = check()
    if (declaration.get("study_id") != STUDY or
            declaration.get("status") != "frozen_before_fit_or_candidate_scoring" or
            declaration.get("arms") != list(ARMS)):
        raise ValueError("Capacity study declaration identity/status/arms mismatch")
    path = fit_path(arm, fold, seed)
    receipt = receipt_path(arm, fold, seed)
    if path.exists() or receipt.exists():
        raise FileExistsError(path)
    records = training(fold, arm)
    teacher = joblib.load(recipe.rf.sc.initial_path(fold, seed))
    gar = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    sampler = recipe.short.OriginalDiversitySampler(
        records, gar, list(teacher["feature_names"]), seed)
    width = 192 if arm == "width192_capacity" else 96
    mixed = arm == "mixed_v1_v2_width96"
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with _view_draws(records, mixed) as draws:
        with recipe.short.parent.base.installed(sampler):
            with _training_adapter(width) as trainer:
                result = trainer(teacher, records, gar, width=width, lr=3e-4,
                    seed=seed, steps=2048, fraction=.5, batch_size=32,
                    ha_target="dense", train_d=True, require_full_native=True,
                    log_every=256, verbose=True)
    joblib.dump(result["model"], path)
    row = dict(study=STUDY, arm=arm, fold=fold, seed=seed, sha256=_sha(path),
        wall_seconds=time.monotonic()-start, config=result["config"],
        sampler=sampler.receipt(), exposures=result["exposures"],
        input_view_draws=dict(draws), updates=2048,
        training_matches=sorted({str(r["match"]) for r in records}),
        training_sources=sorted({str(r["source"]) for r in records}),
        heldout_provider=fold.removeprefix("leave_out_"),
        declaration_sha256=_sha(DECLARATION),
        teacher_sha256=_sha(recipe.rf.sc.initial_path(fold, seed)),
        organizer_reference_access=False)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(row, indent=1, sort_keys=True, default=float)+"\n")
    return row


def _fit_job(job):
    arm, fold, seed = job
    return fit(fold, seed, arm)


def fit_all(workers: int = 2) -> dict:
    check()
    if not isinstance(workers, int) or not 1 <= workers <= 2:
        raise ValueError("Use one or two CPU fit workers")
    jobs = [(arm, fold, seed) for arm in ARMS for fold in FOLDS for seed in SEEDS]
    if workers == 1:
        rows = [_fit_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_fit_job, jobs))
    result = {"study_id": STUDY, "declaration_sha256": _sha(DECLARATION),
              "fits": rows, "all_18_fits_complete": len(rows) == 18,
              "organizer_reference_access": False, "candidate_scoring": False}
    path = OUT / "fit_receipt.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=1, sort_keys=True, default=float)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--workers", type=int, default=3)
    fits = sub.add_parser("fit-all"); fits.add_argument("--workers", type=int, default=2)
    sub.add_parser("freeze"); sub.add_parser("check")
    args = parser.parse_args()
    if args.action == "prepare": result = prepare(args.workers)
    elif args.action == "freeze": result = freeze()
    elif args.action == "check": result = check()
    else: result = fit_all(args.workers)
    print(json.dumps(result, indent=1, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
