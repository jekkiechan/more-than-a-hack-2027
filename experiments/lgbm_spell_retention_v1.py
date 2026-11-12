"""Bounded development-only refiner training and possession-topology study.

No organizer inputs, labels, predictions or scoring gates are read by this study.
The prior public result motivates investigation, never the choice of an arm.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import ball_exit_runtime, supervised, tracking_corruption, feed_like_v2
from experiments import feed_benchmark_audit_v1 as audit
from experiments import lgbm_public_v1 as previous
from experiments import v2_target_push_refiner as original
from experiments import sc2020_realfeed_score as scoring
from experiments import ball_exit_dmask_v1 as guard_source
from experiments import lgbm_retention_features as feature
from experiments import lgbm_retention_topology as topology

ROOT = Path(__file__).resolve().parents[1]
STUDY = "lgbm_spell_retention_v1"
OUT = ROOT / "data/experiments" / STUDY
RES = ROOT / "experiments/results"
DECL = RES / f"{STUDY}_declaration.json"
PREP = OUT / "preparation.json"
CONTROL = RES / f"{STUDY}_control.json"
TRAIN = RES / f"{STUDY}_training.json"
RESULT = RES / f"{STUDY}.json"
SEEDS = (2026, 2027)
PROVIDERS = ("Metrica", "SkillCorner", "Sportec")
RECIPES = ("full", "equalview", "nomiss")
ARMS = ("full", "equalview", "nomiss", "topology_both", "topology_start", "topology_end")
MEMBERS = ("ensemble", "2026", "2027")
METRICS = tuple(scoring.METRICS) + ("d_precision", "d_recall", "weighted_70_30",
    "graded_frames", "short_reference_spell_count")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def sources():
    paths = list((ROOT / "experiments").glob("*.py")) + list((ROOT / "evaluation").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def declare():
    import lightgbm
    write(DECL, dict(study=STUDY, utc=datetime.now(timezone.utc).isoformat(),
        hypothesis="Balanced view exposure or topology-preserving D boundary updates retain D-F1 gains without development spell/short-recall loss",
        arms=list(ARMS), recipes=list(RECIPES), seeds=list(SEEDS), fits=18,
        training="Leave one provider out of LightGBM; clean/v1x2/v2x4; stride5; X dropped",
        recipe_details={"full":"Original row weighting 1:2:4 view masses",
            "equalview":"Same full features with raw per-row weights 1,.5,.25 normalized to mean1; clean/v1/v2 mass1:1:1",
            "nomiss":"Original missingness ablation: non-ball columns plus within-block forward-filled speed and signed boundary distance; original row weighting"},
        topology="Only full-model proposals: local reciprocal one-to-one D overlap; both/start/end edges; keep baseline H/A and exact ordered live-spell sequence; block fallback on topology change",
        params=original.PARAMS, lightgbm_version=lightgbm.__version__, blend=.5,
        model_seed_ensemble="Mean P(D) before decoding, never mean scores",
        ranking="Ensemble v2 official spell descending; exact ties macro; 70/30 auxiliary only",
        eligibility="Original retention guards plus aggregate D-F1 >= baseline in clean/v1/v2 and strictly > in v2; report both constituent seeds and all provider effects",
        control_expected=read(ROOT / "experiments/baseline_manifest.json")["evidence"]["original_development"],
        control_tolerance=1e-9, matches=[list(x) for x in audit.MATCHES], cells=35,
        input_restore_sha256=sha(RES / "lgbm_public_v1_restore.json"),
        baseline_manifest_sha256=sha(ROOT / "experiments/baseline_manifest.json"),
        source_sha256=sources(),
        limitations=["Five heavily exposed development matches; no fresh evidence",
            "Refiner-level provider exclusion only: upstream features use each row's own provider fold, not nested outer-fold cross-fitting",
            "Synthetic v2 defects are label/model-conditioned; balanced views do not establish missingness transport",
            "New block-safe features differ from historical R2"],
        stop="18 fixed fits and six arms only; no post-result tweaks; no public/GAR scoring or model promotion"))


def check():
    d = read(DECL)
    if (d["source_sha256"] != sources() or d["input_restore_sha256"] != sha(RES / "lgbm_public_v1_restore.json")
            or d["baseline_manifest_sha256"] != sha(ROOT / "experiments/baseline_manifest.json")):
        raise ValueError("Frozen sources or restored inputs changed")
    return d


def aggregate(rows):
    out = {}
    for view in scoring.VIEW_NAMES:
        group = {}
        for key, row in rows.items():
            provider, match, v, _ = key.split("__")
            if v == view:
                group.setdefault(provider, {}).setdefault(match, []).append(row)
        out[view] = {}
        for metric in METRICS:
            per = {p:float(np.mean([np.mean([r[metric] for r in rs]) for rs in matches.values()]))
                   for p,matches in group.items()}
            out[view][metric] = float(np.mean(list(per.values())))
            out[view][metric+"/providers"] = per
    return out


def prepare():
    d = check()
    if PREP.exists():
        raise FileExistsError(PREP)
    old = read(previous.PREP)
    restored = read(RES / "lgbm_public_v1_restore.json")
    calpath = ROOT / "experiments/results/feed_like_v2_calibration.json"
    assert sha(calpath) == restored["calibration_sha256"]
    cal = read(calpath)["calibration"]
    entries, rows, names = [], {}, {}
    for r in old["cells"]:
        key = r["key"]; provider, match, view, rep = key.split("__")
        raw, native_path = Path(r["raw_path"]), Path(r["cell_path"])
        assert sha(raw) == restored["raw_inputs"][raw.name] == r["raw_sha256"]
        assert sha(native_path) == restored["cells"][native_path.name] == r["cell_sha256"]
        payload = joblib.load(raw)
        inputs = tracking_corruption.inputs_only(payload["inputs"])
        labels = np.asarray(payload["labels"]).astype("U1")
        if view == "feed_like_v2":
            seed = int(hashlib.sha256(f"feed_like_v2_benchmark/{rep}/{match}".encode()).hexdigest()[:8],16)
            inputs = feed_like_v2.corrupt(inputs, labels, cal, seed)
        with np.load(native_path) as z:
            native = {k:z[k] for k in z.files if not k.startswith("identity__")}
        assert np.array_equal(labels,native["labels"])
        assert np.array_equal(supervised.model_periods(inputs),native["model_period"])
        comp = previous.components(native)
        arrays = {}
        for fs in ("full", "nomiss"):
            nm, matrix = feature.features(inputs, comp, fs)
            if fs in names and names[fs] != nm:
                raise ValueError("Feature schema changed")
            names[fs] = nm; arrays[fs] = matrix
        base = ball_exit_runtime.decode_components(inputs, comp)
        row = audit.score(native, base)
        rows[key] = {m:float(row[m]) for m in METRICS}
        arrays.update({"labels":labels,"baseline":base,"period":native["period"],
            "model_period":native["model_period"],"timestamps":native["timestamps"],
            "fps":native["fps"],"ball":inputs["ball"],"pitch":inputs["pitch"]})
        arrays.update(comp)
        path = OUT / "cells" / f"{key}.npz"
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise FileExistsError(path)
        np.savez_compressed(path,**arrays)
        entries.append(dict(key=key,path=str(path.relative_to(ROOT)),sha256=sha(path),
            raw_sha256=sha(raw),native_sha256=sha(native_path)))
        print("PREPARED",key,flush=True)
    means = aggregate(rows)
    for v in scoring.VIEW_NAMES:
        for m in scoring.METRICS:
            assert abs(means[v][m]-d["control_expected"][v][m]) <= d["control_tolerance"], (v,m)
    write(CONTROL,dict(study=STUDY,rows=rows,aggregate=means,exact_baseline_replay=True))
    write(PREP,dict(declaration_sha256=sha(DECL),cells=entries,feature_names=names,control_sha256=sha(CONTROL)))


def load_cells():
    p=read(PREP)
    if p["declaration_sha256"]!=sha(DECL) or p["control_sha256"]!=sha(CONTROL):
        raise ValueError("Preparation changed")
    roster={f"{provider}__{match}__{view}__{rep}" for provider,match,_ in audit.MATCHES
            for view,reps in previous.VIEWS for rep in reps}
    if len(p["cells"])!=35 or {r["key"] for r in p["cells"]} != roster:
        raise ValueError("Incomplete cohort")
    out={}
    for r in p["cells"]:
        path=ROOT/r["path"]
        assert sha(path)==r["sha256"]
        with np.load(path) as z: out[r["key"]]={k:z[k] for k in z.files}
    return p,out


def train():
    import lightgbm as lgb
    check()
    if TRAIN.exists(): raise FileExistsError(TRAIN)
    p,cells=load_cells(); fits=[]
    for recipe in RECIPES:
        fs="nomiss" if recipe=="nomiss" else "full"
        for provider in PROVIDERS:
            xs,ys,ws,keys=[],[],[],[]
            for key,c in cells.items():
                if key.split("__")[0]==provider: continue
                idx=np.arange(0,len(c["labels"]),5);idx=idx[c["labels"][idx]!="X"]
                xs.append(c[fs][idx]);ys.append((c["labels"][idx]=="D").astype(np.int8))
                weight=feature.view_weight(key.split("__")[2]) if recipe=="equalview" else 1.
                ws.append(np.full(len(idx),weight));keys.append(key)
            X,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w=w/w.mean()
            for seed in SEEDS:
                path=OUT/"models"/f"{recipe}_{provider}_{seed}.txt"
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():raise FileExistsError(path)
                model=lgb.LGBMClassifier(**dict(original.PARAMS,random_state=seed)).fit(
                    X,y,sample_weight=w,feature_name=p["feature_names"][fs])
                model.booster_.save_model(str(path))
                for key,c in cells.items():
                    if key.split("__")[0]!=provider:continue
                    target=OUT/"probabilities"/f"{recipe}_{seed}"/f"{key}.npy"
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():raise FileExistsError(target)
                    np.save(target,model.predict_proba(c[fs])[:,1])
                fits.append(dict(recipe=recipe,held_out_provider=provider,seed=seed,
                    training_rows=len(y),training_keys=keys,path=str(path.relative_to(ROOT)),sha256=sha(path)))
                print("FITTED",recipe,provider,seed,len(y),flush=True)
    probabilities={str(path.relative_to(OUT)):sha(path) for path in sorted((OUT/"probabilities").rglob("*.npy"))}
    assert len(fits)==18 and len(probabilities)==210
    write(TRAIN,dict(study=STUDY,declaration_sha256=sha(DECL),preparation_sha256=sha(PREP),
        fits=fits,probabilities=probabilities,fit_count=18,provider_exclusion_checked=True,
        nested_upstream_exclusion=False))


def decode(c, refined):
    from experiments import spell_frontier_baseline as front
    pd=.5*refined+.5*c["adopted"][:,2]
    ha=c["adopted"][:,:2];denom=ha.sum(axis=1,keepdims=True)
    cond=np.divide(ha,denom,out=np.full_like(ha,.5),where=denom>0)
    prob=np.column_stack((cond*(1-pd)[:,None],pd))
    mask=audit.decode(prob,c["model_period"],float(c["fps"]),dict(audit.CONFIG,dt=4,dead=3))=="D"
    inside,outside=guard_source.ball_state(c["ball"],c["pitch"],0.)
    dead=guard_source.edit_mask(mask,inside,outside,c["model_period"],float(c["fps"]),"snap")
    ha=front.blend.recipe_probability({k:c[k] for k in ("ce","adopted","calibrated")},
        front.RECIPE["ce_weight"],front.RECIPE["calibrated_weight"],front.RECIPE["pooling"])
    return front.blend.decode_fixed_d(ha,np.where(dead,"D","H"),c["model_period"],float(c["fps"]),front.RECIPE["ha_cost"])


def evaluate():
    check();p,cells=load_cells();t=read(TRAIN)
    if RESULT.exists():raise FileExistsError(RESULT)
    assert t["declaration_sha256"]==sha(DECL) and t["preparation_sha256"]==sha(PREP)
    for rel,h in t["probabilities"].items():assert sha(OUT/rel)==h
    rows={f"{a}/{m}":{} for a in ARMS for m in MEMBERS};diagnostics={};pred_hashes={}
    control=read(CONTROL)
    for key,c in cells.items():
        preds={};diags={}
        for recipe in RECIPES:
            probs={str(s):np.load(OUT/"probabilities"/f"{recipe}_{s}"/f"{key}.npy") for s in SEEDS}
            probs["ensemble"]=np.mean(list(probs.values()),axis=0)
            for member,pd in probs.items():preds[f"{recipe}/{member}"]=decode(c,pd)
        for mode in ("both","start","end"):
            for member in MEMBERS:
                name=f"topology_{mode}/{member}"
                preds[name],diags[name]=topology.topology_repair(c["baseline"],preds[f"full/{member}"],c["model_period"],mode)
        for name,pred in preds.items():
            r=audit.score(c,pred)
            for pop in ("graded_frames","reference_spell_count","short_reference_spell_count"):
                assert r[pop]==control["rows"][key][pop]
            rows[name][key]={m:float(r[m]) for m in METRICS}
        path=OUT/"predictions"/f"{key}.npz";path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,**{name.replace('/','__'):pred for name,pred in preds.items()})
        pred_hashes[key]=sha(path);diagnostics[key]=diags
        print("SCORED",key,flush=True)
    base=read(CONTROL)["aggregate"];means={name:aggregate(r) for name,r in rows.items()}
    verdicts={}
    for name,a in means.items():
        fail=guard_source.guards(a,base)
        for v in scoring.VIEW_NAMES:
            if a[v]["d_f1"]<base[v]["d_f1"] or (v=="feed_like_v2" and a[v]["d_f1"]==base[v]["d_f1"]):
                fail.append(f"{v}/d_f1_retention")
        verdicts[name]=dict(eligible=not fail,failed=fail,
            delta={v:{m:a[v][m]-base[v][m] for m in METRICS} for v in scoring.VIEW_NAMES})
    eligible=[n for n in means if n.endswith('/ensemble') and verdicts[n]["eligible"]]
    winner=max(eligible,key=lambda n:(means[n]["feed_like_v2"]["spell_f1"],means[n]["feed_like_v2"]["macro_f1"])) if eligible else None
    write(RESULT,dict(study=STUDY,control=base,aggregate=means,verdicts=verdicts,
        eligible=eligible,selected_for_confirmation=winner,cell_rows=rows,topology_diagnostics=diagnostics,
        prediction_sha256=pred_hashes,declaration_sha256=sha(DECL),training_sha256=sha(TRAIN),
        public_access=False,gar_access=False,adoption=False))
    for n in means:
        if n.endswith('/ensemble'):
            print(n,{v:round(means[n][v]["spell_f1"],6) for v in scoring.VIEW_NAMES},verdicts[n]["failed"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('declare','prepare','train','evaluate'))
    args=p.parse_args();globals()[args.action]()
