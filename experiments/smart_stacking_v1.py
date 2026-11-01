"""Bounded provider-excluded stacking study; no organizer reference access.

Base models are restored, never refitted. Meta-training predictions must be
regenerated with the evaluation provider's outer checkpoints, rather than using
other providers' OOF arrays whose checkpoints saw the outer provider's TRAIN.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ["MTAH_DATA_DIR"] = str(ROOT / "data")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import joblib
import numpy as np

from experiments import smart_stacking as stack
from experiments import feed_robustness_v1 as common
from experiments import preparation, multisource_eval, supervised, tracking_corruption
from experiments import spell_mbr_run as shared, decoder_search_run as scoring
from experiments import feed_spell_push_run as selection
from experiments.eval_set.protocol import paired_source_bootstrap

STUDY = "smart_stacking_v1"
OUT = ROOT / "data/experiments" / STUDY
RESULTS = ROOT / "experiments/results"
DECL = RESULTS / f"{STUDY}_declaration.json"
PROVIDERS = ("Metrica", "SkillCorner", "Sportec")
SEEDS = (2026, 2027)
EXPERTS = ("adopted", "feed", "short", "original")
FITS = {"single_baseline": ([0], False), "single_original_temporal": ([3], True),
        "static_stack": ([0, 1, 2, 3], False), "temporal_stack": ([0, 1, 2, 3], True)}
ARMS = ("adopted", "feed", "targeted_ha", "original", "uniform",
        "single_baseline", "single_original_temporal", "static_stack",
        "temporal_ha_stack", "temporal_joint_stack", "safe_temporal")
LEARNED = ("static_stack", "temporal_ha_stack", "temporal_joint_stack", "safe_temporal")
FRESH_SEEDS = (862401, 862402)
read, sha = common.read, common.sha


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def model_path(expert, provider, seed):
    source = provider.lower()
    if expert == "adopted":
        return ROOT / f"data/experiments/dense_label_weight_v1/models/control_D2__{source}__s{seed}.joblib"
    study = {"feed": "feed_robustness_v1", "short": "feed_short_sampling_v1", "original": "short_original_diversity_v1"}[expert]
    return ROOT / f"data/experiments/{study}/models/leave_out_{source}_{seed}.joblib"


def original_cells():
    return shared._load_cells(read(common.DECLARATION))


def check():
    d = read(DECL)
    for field in ("source_sha256", "input_sha256"):
        for rel, digest in d[field].items():
            if sha(ROOT / rel) != digest:
                raise ValueError("Frozen dependency changed: " + rel)
    if d["arms"] != list(ARMS):
        raise ValueError("Changed arms")
    return d


def declare():
    restored = read(OUT / "restore.json")
    dependencies = {r["path"]: r["sha256"] for r in restored}
    baseline = read(common.DECLARATION)
    teachers = read(RESULTS / "dense_label_weight_v1_declaration.json")["restore"]["fold_teacher_sha256"]
    for provider in PROVIDERS:
        fold = "leave_out_" + provider.lower()
        for seed in SEEDS:
            teacher = common.sc.initial_path(fold, seed)
            assert sha(teacher) == teachers[f"{fold}/{seed}"]
            original = joblib.load(teacher)
            for expert in EXPERTS:
                path = model_path(expert, provider, seed)
                assert sha(path) == dependencies[str(path.relative_to(ROOT))]
                model = joblib.load(path)
                for key in ("median", "scale"):
                    np.testing.assert_array_equal(model[key], original[key])
                assert model["feature_names"] == original["feature_names"]
    plans = preparation.split_plan()
    base_train = {r["match"] for r in plans["train"]}
    meta_matches = common.old.selection_items()
    assert not base_train & {r["match"] for r in meta_matches}
    for rel, digest in dependencies.items():
        assert rel.startswith("data/") and sha(ROOT / rel) == digest
    paths = list((ROOT / "experiments").glob("*.py")) + [
        ROOT / "evaluation/possession_metrics.py", ROOT / "experiments/eval_set/protocol.py",
        ROOT / "experiments/eval_set/manifest.json", ROOT / "experiments/baseline_manifest.json",
        ROOT / "experiments/model_decision_policy.json", ROOT / "experiments/tests/test_smart_stacking.py"]
    write(DECL, dict(study=STUDY, utc=datetime.now(timezone.utc).isoformat(),
        question="Can learned static or temporal stacking combine near-miss models to improve official spell F1 without losing provider robustness?",
        experts=list(EXPERTS), arms=list(ARMS), fits=FITS,
        budget="24 small meta-models / 48 binary fits; zero base fits; 360 base inferences; 495 original score cells. No post-result arms or parameters.",
        meta_training="Outer provider excluded from every checkpoint, teacher, normalizer and meta-training label. Generate predictions for OTHER providers' selection matches with the SAME outer-fold models. Those matches are absent from base TRAIN. Never reuse other folds' OOF predictions for meta-training.",
        plans={p:dict(base=preparation.split_plan(fold="leave_out_"+p.lower()),
                      meta_train=[r for r in meta_matches if r['source'] != p.lower()],
                      evaluation=[r for r in meta_matches if r['source'] == p.lower()]) for p in PROVIDERS},
        extra_supervision="Stacks use other providers' development labels beyond base TRAIN; single-model calibration controls receive exactly the same extra labels, weights and exclusion. All matches have historical researcher exposure; not independent confirmation.",
        features="Per-expert conditional H/A log odds; optional current-minus-centered 1s local mean. D uses only player-only D log odds plus intercept, with no ball/H-A/missingness/provider cue. No H/A intercept ensures team symmetry; never cross model_period boundaries.",
        objective=dict(penalty=stack.PENALTY, optimizer="L-BFGS-B; max200; ftol1e-12; gtol1e-8",
            anchor="Equal logit expert weights, zero temporal terms and D intercept", native_stride_hz=5,
            weighting="Equal provider, match, clean/feed half, replica; conditional H/A short-span sqrt(3/duration), clipped [1,4]; D unweighted within cell. Unknown X excluded. Weights are TRAIN-only.",
            parameter_tuning=False),
        inference="Each seed has its own meta-fit, average the TWO resulting native H/A/D probabilities before fixed D-temperature4/cost3. Preserve native masks, model periods and grading. Temporal H/A arm keeps raw-feed D; safe arm is fixed .5 temporal H/A + .5 adopted native probabilities.",
        selection="Rank original feed spell F1, macro exact tie; learned winner must beat both original full and both single-calibration controls under guards. Auxiliary 70/30 descriptive. .55 stretch; no promotion.",
        guards=dict(clean_spell_delta=-.005, provider_clean_macro_d_delta=-.02, provider_feed_spell_delta=-.01,
                    historical_clean_gain=.01, reference="adopted"),
        replay=dict(probability_atol=2e-6, changed_adopted_labels=0, saved_score_atol=1e-12),
        fresh=dict(seeds=list(FRESH_SEEDS), rule="If any learned arm passes guards and beats original full plus single-model calibration controls, lock its original winner and evaluate only it, adopted, original full and uniform on two new raw-feed corruption realizations per match. No reselection.",
                   evidence="Previously unused synthetic realizations on exposed matches; not new independent matches"),
        perturbation="For the same fixed eligible winner, replace each expert in turn with adopted probabilities on original views; do not retrain or reselect. Operational sensitivity diagnostic only.",
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}, input_sha256=dependencies,
        prediction_files=baseline["prediction_files"], evaluation_input_hashes=baseline["evaluation_input_hashes"],
        model_checkpoint_hashes=baseline["model_checkpoint_hashes"],
        organizer_reference_access=False, promotion=False))
    print("DECLARED", STUDY, flush=True)


def prepare_fresh():
    d = check()
    write(OUT / "fresh_prepare_lock.json", dict(declaration_sha256=sha(DECL)))
    rows = []
    for item in common.old.selection_items():
        path = preparation.data_path(item, ROOT / "data")
        assert sha(path) == item["sha256"]
        inputs, labels = multisource_eval.view(path)
        for replica, seed_base in enumerate(FRESH_SEEDS, 1):
            seed = int(hashlib.sha256(f"{STUDY}/{seed_base}/{item['match']}".encode()).hexdigest()[:8], 16)
            view = common.raw_view(inputs, common.sn.default_ball_calibration(), seed)
            record = preparation.build_record(view, labels, item["source"], item["match"])
            dest = OUT / "fresh_inputs" / f"{item['match']}__{replica}.joblib"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError(dest)
            joblib.dump(dict(inputs=view, labels=labels, record=record), dest)
            rows.append(dict(path=str(dest.relative_to(ROOT)), sha256=sha(dest), seed=seed,
                             match=item["match"], provider=shared._provider_for_match(item["match"]),
                             view=f"feed_like_fresh_{replica}"))
            print("FRESH PREPARED", item["match"], replica, flush=True)
    write(OUT / "fresh_preparation.json", dict(files=rows, declaration_sha256=sha(DECL)))


def cached_input(c):
    return joblib.load(shared._resolve_artifact(shared._scalar(c["record"], "evaluation_input_path")))


def infer_record(cached, models, teachers):
    inputs = tracking_corruption.inputs_only(cached["inputs"])
    record = common.old.inference_record(cached["record"])
    record["names"] = np.asarray(record["names"]).tolist()
    return np.asarray([[common.old.upsample(common.sn.predict(record, models[s, e], teachers[s]), record, inputs)
                        for e in EXPERTS] for s in SEEDS])


def load_models(provider):
    teachers = {s:joblib.load(common.sc.initial_path("leave_out_"+provider.lower(), s)) for s in SEEDS}
    models = {(s,e):joblib.load(model_path(e,provider,s)) for s in SEEDS for e in EXPERTS}
    return models, teachers


def infer(provider):
    check()
    models, teachers = load_models(provider)
    cells = [c for c in original_cells() if c["member"] == "ensemble"]
    files = {}
    differences = []
    for c in cells:
        dest = OUT / "level1" / provider / f"{c['match']}__{c['view']}.npz"
        if dest.exists():
            raise FileExistsError(dest)
        started = time.monotonic()
        prob = infer_record(cached_input(c), models, teachers)
        # Target cells retain exact archived probabilities after numerical replay.
        # Meta-training cells use newly generated SAME-OUTER-FOLD predictions.
        if c["provider"] == provider:
            for i, seed in enumerate(SEEDS):
                for j, expert in enumerate(EXPERTS):
                    expected = archived(c, expert, seed)
                    difference = float(np.max(np.abs(prob[i,j]-expected)))
                    if difference > 2e-6:
                        raise ValueError("Base inference replay differs: " + str((provider,expert,seed,difference)))
                    if expert == "adopted":
                        np.testing.assert_array_equal(common.decode(prob[i,j], c["model_period"], c["fps"]),
                                                      common.decode(expected, c["model_period"], c["fps"]))
                    differences.append(dict(expert=expert, seed=seed, match=c["match"], view=c["view"], max_difference=difference))
                    prob[i,j] = expected
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, probabilities=prob, **{k:c[k] for k in ("labels","period","timestamps","model_period","fps")})
        files[str(dest.relative_to(ROOT))] = sha(dest)
        print("LEVEL1", provider, c["match"], c["view"], round(time.monotonic()-started,2), flush=True)
    write(OUT / f"level1_{provider}.json", dict(files=files, replay=differences, outer_provider=provider,
          declaration_sha256=sha(DECL), meta_target_provider_excluded=True))


def archived(c, expert, seed):
    if expert == "adopted":
        candidate = next(x for x in original_cells() if all(x[k] == c[k] for k in ("provider","match","view")) and x["member"] == str(seed))
        return candidate["probabilities"]
    folder = {"feed":"feed_robustness_v1/predictions/feed_augmented", "short":"feed_short_sampling_v1/predictions/original/short_sampling",
              "original":"short_original_diversity_v1/predictions/original_full"}[expert]
    path = ROOT / "data/experiments" / folder / f"{c['provider']}__{c['match']}__{c['view']}__{seed}.npz"
    with np.load(path) as src:
        for k in ("labels","period","timestamps","model_period"):
            np.testing.assert_array_equal(src[k], c[k])
        return src["probabilities"].copy()


def level1(provider, member):
    receipt = read(OUT / f"level1_{provider}.json")
    cells = []
    index = SEEDS.index(member)
    for rel, digest in receipt["files"].items():
        path = ROOT / rel
        assert sha(path) == digest
        match, view = path.stem.split("__", 1)
        with np.load(path) as src:
            cells.append(dict({k:src[k].copy() for k in src.files if k != "probabilities"},
                              probabilities=src["probabilities"][index].copy(), outer_provider=provider,
                              provider=shared._provider_for_match(match), match=match, view=view, member=str(member)))
    return cells


def fit():
    check()
    write(OUT / "fit_lock.json", dict(declaration_sha256=sha(DECL)))
    files = {}
    for provider in PROVIDERS:
        for seed in SEEDS:
            train = [c for c in level1(provider, seed) if c["provider"] != provider]
            for name, (indices, temporal) in FITS.items():
                model = stack.fit(train, provider, indices, temporal)
                dest = OUT / "meta_models" / f"{provider}__{seed}__{name}.json"
                write(dest, model)
                files[str(dest.relative_to(ROOT))] = sha(dest)
                print("FIT", provider, seed, name, flush=True)
    write(RESULTS / f"{STUDY}_model_seal.json", dict(files=files, declaration_sha256=sha(DECL), models=24,
          base_refits=0, fit_on_evaluation_provider=False))


def arm_probabilities(p, period, fps, models):
    temporal = stack.predict(p, period, fps, models["temporal_stack"])
    ha = stack.predict(p, period, fps, models["temporal_stack"], dead_expert=1)
    return dict(adopted=p[0], feed=p[1], targeted_ha=stack.cross(p[2],p[1]), original=p[3],
        uniform=np.mean(p,axis=0), single_baseline=stack.predict(p,period,fps,models["single_baseline"]),
        single_original_temporal=stack.predict(p,period,fps,models["single_original_temporal"]),
        static_stack=stack.predict(p,period,fps,models["static_stack"]), temporal_ha_stack=ha,
        temporal_joint_stack=temporal, safe_temporal=.5*(ha+p[0]))


def meta_models(provider, seed):
    seal = read(RESULTS / f"{STUDY}_model_seal.json")
    result = {}
    for name in FITS:
        path = OUT / "meta_models" / f"{provider}__{seed}__{name}.json"
        assert sha(path) == seal["files"][str(path.relative_to(ROOT))]
        result[name] = read(path)
    return result


def save_score(c, probability, arm, group, rows, files):
    row, prediction = common.score(c, probability)
    rows[arm].append(row)
    path = OUT / "predictions" / group / arm / f"{c['provider']}__{c['match']}__{c['view']}__{c['member']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, probabilities=probability, prediction=prediction,
                        **{k:c[k] for k in ("labels","period","timestamps","model_period","fps")})
    files[str(path.relative_to(ROOT))] = sha(path)


def summaries(rows):
    return {a:{m:scoring.summarize(r,m) for m in shared.MEMBERS} for a,r in rows.items()}


def evaluate():
    check()
    assert (OUT / "fresh_preparation.json").exists(), "Fresh preparation must precede candidate scores"
    seal = RESULTS / f"{STUDY}_model_seal.json"
    write(OUT / "score_lock.json", dict(declaration_sha256=sha(DECL), model_seal_sha256=sha(seal),
          fresh_preparation_sha256=sha(OUT / "fresh_preparation.json")))
    rows, files = {a:[] for a in ARMS}, {}
    for provider in PROVIDERS:
        members = {}
        for seed in SEEDS:
            models = meta_models(provider, seed)
            for c in level1(provider, seed):
                if c["provider"] != provider:
                    continue
                values = arm_probabilities(c["probabilities"], c["model_period"], float(c["fps"]), models)
                members[c["match"], c["view"], seed] = values
                for arm in ARMS:
                    save_score(c,values[arm],arm,"original",rows,files)
        for c in [c for c in level1(provider,2026) if c["provider"] == provider]:
            c["member"] = "ensemble"
            for arm in ARMS:
                prob = np.mean([members[c["match"],c["view"],s][arm] for s in SEEDS],axis=0)
                save_score(c,prob,arm,"original",rows,files)
        print("SCORED", provider, flush=True)
    s = summaries(rows)
    baseline = s["adopted"]["ensemble"]
    # Exact archived score controls catch changed aggregation or grading.
    expected = {"adopted":read(RESULTS/"feed_spell_push_v1_aggregate.json")["arms"]["f062"],
                "feed":read(RESULTS/"feed_robustness_v1_aggregate.json")["summaries"]["feed_augmented"]["ensemble"],
                "original":read(RESULTS/"short_original_diversity_v1_aggregate.json")["summaries"]["original_full"]["ensemble"]}
    for arm, value in expected.items():
        for view in ("clean","feed_like"):
            for metric in ("spell_f1","macro_f1","d_f1"):
                assert abs(s[arm]["ensemble"][view]["equal_provider_mean"][metric]-value[view]["equal_provider_mean"][metric]) < 1e-12, (arm,view,metric)
    guards = {a:bool(selection.eligible(v["ensemble"],baseline)) for a,v in s.items()}
    rivals = ("original", "single_baseline", "single_original_temporal", "uniform")
    candidates = [a for a in LEARNED if guards[a] and all(selection.rank(s[a]["ensemble"]) > selection.rank(s[b]["ensemble"]) for b in rivals)]
    winner = max(candidates,key=lambda a:selection.rank(s[a]["ensemble"])) if candidates else None
    bootstraps = {}
    for a in LEARNED:
        for b in ("adopted","original"):
            def match_rows(arm):
                values = []
                for provider in PROVIDERS:
                    for match in sorted({r["match"] for r in rows[arm] if r["provider"] == provider}):
                        use = [r["spell_f1"] for r in rows[arm] if r["member"] == "ensemble" and r["provider"] == provider and r["match"] == match and r["view"] != "clean"]
                        values.append(dict(eval_source=provider, match=match, spell_f1=float(np.mean(use))))
                return values
            bootstraps[a+"_vs_"+b] = paired_source_bootstrap(match_rows(b),match_rows(a))
    write(RESULTS/f"{STUDY}_aggregate.json",dict(study=STUDY, rows=rows, summaries=s, guards_pass=guards,
        winner=winner, score_cells=sum(map(len,rows.values())), controls_exact=True,
        historical_clean_screen_pass={a:bool(v["ensemble"]["clean"]["equal_provider_mean"]["spell_f1"] >= baseline["clean"]["equal_provider_mean"]["spell_f1"]+.01) for a,v in s.items()},
        paired_whole_match_bootstrap=bootstraps, prediction_sha256=files, organizer_reference_access=False,
        independent_match_confirmation=False, model_promoted=False))
    for a in ARMS:
        print(a,{v:s[a]["ensemble"][v]["equal_provider_mean"]["spell_f1"] for v in ("clean","feed_like")},guards[a],flush=True)
    print("WINNER",winner,flush=True)


def fresh():
    check()
    result = read(RESULTS/f"{STUDY}_aggregate.json")
    winner = result["winner"]
    if winner is None:
        write(RESULTS/f"{STUDY}_fresh_skipped.json",dict(reason="No eligible learned improvement over fixed and single-model controls",scored=False))
        return
    write(OUT/"fresh_score_lock.json",dict(winner=winner, aggregate_sha256=sha(RESULTS/f"{STUDY}_aggregate.json")))
    arms = (winner,"adopted","original","uniform")
    rows, files = {a:[r for r in result["rows"][a] if r["view"] == "clean"] for a in arms}, {}
    for entry in read(OUT/"fresh_preparation.json")["files"]:
        provider = entry["provider"]
        path = ROOT/entry["path"]
        assert sha(path) == entry["sha256"]
        cached = joblib.load(path)
        inputs = tracking_corruption.inputs_only(cached["inputs"])
        prob = infer_record(cached,*load_models(provider))
        c = dict(provider=provider,match=entry["match"].removeprefix("skillcorner_had_"),view=entry["view"],
            labels=cached["labels"],period=inputs["period"],timestamps=inputs["timestamps"],
            model_period=supervised.model_periods(inputs),fps=float(inputs["fps"]))
        values = {s:arm_probabilities(prob[i],c["model_period"],c["fps"],meta_models(provider,s)) for i,s in enumerate(SEEDS)}
        for member in (*SEEDS,"ensemble"):
            c["member"] = str(member)
            for arm in arms:
                p = np.mean([values[s][arm] for s in SEEDS],axis=0) if member == "ensemble" else values[member][arm]
                save_score(c,p,arm,"fresh",rows,files)
        print("FRESH SCORED",provider,c["match"],c["view"],flush=True)
    write(RESULTS/f"{STUDY}_fresh.json",dict(winner_fixed=winner, rows=rows, summaries=summaries(rows),
        prediction_sha256=files,new_score_cells=len(files),independent_match_confirmation=False,reselection=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("declare","prepare_fresh","infer","fit","evaluate","fresh"))
    parser.add_argument("--provider",choices=PROVIDERS)
    args = parser.parse_args()
    infer(args.provider) if args.action == "infer" else globals()[args.action]()
