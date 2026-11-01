"""Separate algebra, official-score and exclusion replay of saved stacking output."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.special import expit, logit

from experiments import smart_stacking_v1 as study, supervised, metrics
from experiments import smart_stacking_runtime_v1 as runtime
from experiments import smart_stacking_scalar_runtime_v1 as scalar_runtime


def forward(p, period, fps, model, dead_expert=None):
    p = np.asarray(p, dtype=float)
    p = p/p.sum(axis=2,keepdims=True)
    chosen = p[model["expert_indices"]]
    live = chosen[:,:,:2].sum(axis=2)
    q = np.divide(chosen[:,:,0],live,out=np.full_like(live,.5),where=live>0)
    ha = logit(np.clip(q,1e-6,1-1e-6)).T
    if model["temporal"]:
        means = np.empty_like(ha)
        radius = max(1,round(.5*fps))
        kernel = np.ones(radius*2+1)/(radius*2+1)
        for a,b in supervised.blocks(period):
            for j in range(ha.shape[1]):
                means[a:b,j] = np.convolve(np.pad(ha[a:b,j],radius,mode="edge"),kernel,mode="valid")
        ha = np.column_stack((ha,ha-means))
    home = expit(ha @ np.asarray(model["ha"]["coef"]))
    dead = (p[dead_expert,:,2] if dead_expert is not None else
            expit(np.column_stack((logit(np.clip(chosen[:,:,2],1e-6,1-1e-6)).T,np.ones(p.shape[1]))) @ np.asarray(model["d"]["coef"])))
    return np.column_stack((home*(1-dead),(1-home)*(1-dead),dead))


def probabilities(p, period, fps, fitted):
    # Standalone algebra deliberately does not call stack.predict/arm_probabilities.
    conditional = p[2,:,:2]/p[2,:,:2].sum(axis=1,keepdims=True)
    d = p[1,:,2]/p[1].sum(axis=1)
    targeted = np.column_stack((conditional*(1-d[:,None]),d))
    ha = forward(p,period,fps,fitted["temporal_stack"],dead_expert=1)
    return dict(adopted=p[0],feed=p[1],targeted_ha=targeted,original=p[3],uniform=p.mean(axis=0),
        single_baseline=forward(p,period,fps,fitted["single_baseline"]),
        single_original_temporal=forward(p,period,fps,fitted["single_original_temporal"]),
        static_stack=forward(p,period,fps,fitted["static_stack"]),
        temporal_ha_stack=ha,temporal_joint_stack=forward(p,period,fps,fitted["temporal_stack"]),
        safe_temporal=.5*(ha+p[0]))


def decode(p, period, fps):
    p = np.asarray(p,dtype=float)
    p = p/p.sum(axis=1,keepdims=True)
    q = p[:,:2]/np.maximum(p[:,:2].sum(axis=1,keepdims=True),1e-30)
    odds = np.log(np.maximum(q[:,0],1e-7))-np.log(np.maximum(q[:,1],1e-7))
    home = expit(odds)
    dead = expit(logit(np.clip(p[:,2],1e-7,1-1e-7))/4)
    transformed = np.column_stack((home*(1-dead),(1-home)*(1-dead),dead))
    transformed /= transformed.sum(axis=1,keepdims=True)
    return supervised.decode(transformed,period,fps,3)


def verify():
    study.check()
    for path in (runtime.AMENDMENT,scalar_runtime.AMENDMENT):
        amendment = study.read(path)
        for rel,digest in amendment["source_sha256"].items():
            assert study.sha(ROOT/rel)==digest
    result = study.read(study.RESULTS/f"{study.STUDY}_aggregate.json")
    lock = study.read(study.OUT/"score_lock.json")
    assert lock["model_seal_sha256"]==study.sha(study.RESULTS/f"{study.STUDY}_model_seal.json")
    assert lock["fresh_preparation_sha256"]==study.sha(study.OUT/"fresh_preparation.json")
    replay_max = 0.
    cells = 0
    for provider in study.PROVIDERS:
        all_inputs = {seed:study.level1(provider,seed) for seed in study.SEEDS}
        for seed in study.SEEDS:
            allowed = {c["match"] for c in all_inputs[seed] if c["provider"]!=provider}
            for model in study.meta_models(provider,seed).values():
                assert set(model["train_matches"])==allowed
                assert provider not in model["train_providers"]
                assert model["excluded_provider"]==provider
        target = [c for c in all_inputs[2026] if c["provider"]==provider]
        for c in target:
            values = {}
            for seed in study.SEEDS:
                other = next(x for x in all_inputs[seed] if x["match"]==c["match"] and x["view"]==c["view"])
                values[seed] = probabilities(other["probabilities"],c["model_period"],float(c["fps"]),study.meta_models(provider,seed))
            for arm in study.ARMS:
                for member in (*study.SEEDS,"ensemble"):
                    expected = np.mean([values[s][arm] for s in study.SEEDS],axis=0) if member=="ensemble" else values[member][arm]
                    path = study.OUT/"predictions/original"/arm/f"{provider}__{c['match']}__{c['view']}__{member}.npz"
                    rel = str(path.relative_to(ROOT))
                    assert study.sha(path)==result["prediction_sha256"][rel]
                    with np.load(path) as src:
                        delta = float(np.max(np.abs(expected-src["probabilities"])))
                        replay_max = max(replay_max,delta)
                        assert delta<3e-12,(arm,member,delta)
                        np.testing.assert_array_equal(decode(expected,c["model_period"],float(c["fps"])),src["prediction"])
                        for key in ("labels","period","timestamps","model_period"):
                            np.testing.assert_array_equal(src[key],c[key])
                        official = metrics.evaluate_tracking(src["labels"],src["prediction"],period=src["period"],fps=float(src["fps"]),timestamps=src["timestamps"])
                    row = next(r for r in result["rows"][arm] if r["provider"]==provider and r["match"]==c["match"] and r["view"]==c["view"] and r["member"]==str(member))
                    assert abs(official["spell_f1_mean"]-row["spell_f1"])<1e-12
                    assert abs(official["macro_f1"]-row["macro_f1"])<1e-12
                    assert abs(official["per_class"]["D"]["f1"]-row["d_f1"])<1e-12
                    assert official["n_graded"]==row["graded_frames"]
                    base = next(r for r in result["rows"]["adopted"] if all(r[k]==row[k] for k in ("provider","match","view","member")))
                    for key in ("graded_frames","reference_spell_count","short_reference_spell_count"):
                        assert row[key]==base[key]
                    cells += 1
        print("VERIFIED",provider,flush=True)
    assert cells==495==result["score_cells"]
    # Independently repeat equal provider aggregation from the saved row scalars.
    for arm in study.ARMS:
        for member in ("2026","2027","ensemble"):
            for view in ("clean","feed_like"):
                for metric in ("spell_f1","macro_f1","d_f1"):
                    providers = []
                    for provider in study.PROVIDERS:
                        matches = []
                        for match in {r["match"] for r in result["rows"][arm] if r["provider"]==provider}:
                            values = [r[metric] for r in result["rows"][arm] if r["member"]==member and r["provider"]==provider and r["match"]==match and (r["view"]=="clean" if view=="clean" else r["view"].startswith("feed_like"))]
                            matches.append(np.mean(values))
                        providers.append(np.mean(matches))
                    assert abs(np.mean(providers)-result["summaries"][arm][member][view]["equal_provider_mean"][metric])<1e-12
    study.write(study.RESULTS/f"{study.STUDY}_verification.json",dict(passed=True,original_cells=cells,
        independent_probability_max_difference=replay_max,changed_decoded_labels=0,
        independent_official_score_replay=True,equal_provider_aggregation=True,grading_unchanged=True,
        target_provider_absent_from_all_meta_models=True,
        declaration_sha256=study.sha(study.DECL),aggregate_sha256=study.sha(study.RESULTS/f"{study.STUDY}_aggregate.json"),
        verifier_sha256=study.sha(Path(__file__)),organizer_reference_access=False))


if __name__=="__main__":
    verify()
