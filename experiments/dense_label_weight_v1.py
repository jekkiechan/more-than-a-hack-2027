"""Matched dense-silver reliability weighting study.

The D2 control delegates to frozen ``sn_dense_train.train``. The candidate
uses the same model, sampler, optimizer, and target builders with a small,
explicitly adapted per-window loss loop to apply native-frame reliability
weights. No organizer labels or GAR VALID/TEST labels are read.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MTAH_DATA_DIR", str(ROOT.parent / "mtah-data"))
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import joblib
import numpy as np
from experiments import factorized_had as fh, learned_temporal as lt
from experiments import objective_losses as ol, soccernet_capacity as sc
from experiments import soccernet_dense_labels as dl, soccernet_extension as old
from experiments import sn_dense_train as sn, supervised

OUT = ROOT / "data/experiments/dense_label_weight_v1"
RECORDS = OUT / "records"
MODELS = OUT / "models"
FITS = OUT / "fits"
PREDICTIONS = OUT / "predictions"
RESULTS = ROOT / "experiments/results"
FOLDS = sc.FOLDS
SEEDS = (2026, 2027)
ARMS = ("control_D2", "weight_025")
STUDY_ID = "dense_label_weight_v1"
RECONSTRUCTION_STATUS = "reconstructed from exact frozen fold teachers, verified source records, and the declared D2 recipe; original trained D2 fold weights were unavailable"
WEIGHT = 0.25
RADIUS_SECONDS = 0.5
QUALIFYING_PROVENANCE = (dl.P_TOUCH, dl.P_PROPAGATED, dl.P_HELD)
EXPECTED_CONTROL = {"clean": 0.6472145591, "feed_like": 0.5058107894}
CONTROL_TOLERANCE = 0.000051
EXPECTED_TRAIN_MANIFEST = "d2639fa1a47b38bff1b293e8b2592fb1d7392c09700a2c9873c1a90a72397b51"
EXPECTED_TEACHERS = {
    "leave_out_metrica/2026": "b2a41684850435bceb78db2176188a2e21a4b8adc0073b55cceec4302d95ef91",
    "leave_out_metrica/2027": "260cc80cff50a1a280af2423a77d8f7a66a372b5dc762994d1fc2df6b8d741b3",
    "leave_out_skillcorner/2026": "faf05644439996b8afd005d2b65ba3e853f2c7c20f7be24840e1055f72dab56",
    "leave_out_skillcorner/2027": "f67e4a659daa60204be66fea1774d74e062a2a14f38b8835136d4109705f5933",
    "leave_out_sportec/2026": "18e5dbe2da2e1d6f7659118889c72259bf7abed07d4ac927d46745dcff9e8676",
    "leave_out_sportec/2027": "026c291ba5ac5d0093c2bbd5de500374f925567f5e91bc25cc974e2379ab73a4",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, value: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Refusing to overwrite frozen artifact: {path}")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def ambiguous_native_mask(record: dict) -> np.ndarray:
    """Mask ±round(0.5*native_fps) frames around eligible H/A/D changes.

    A change is eligible only when labels on both sides are known, differ,
    remain within the same prepared period, and at least one adjacent frame
    carries touch, propagated, or held-uncontrolled provenance. Each record is
    processed independently, so masks cannot cross match, period, or time gaps.
    """
    y = np.asarray(record["full_y"], dtype=np.int8)
    period = np.asarray(record["full_period"])
    provenance = np.asarray(record["full_prov"], dtype=np.int8)
    if y.ndim != 1 or period.shape != y.shape or provenance.shape != y.shape:
        raise ValueError("Native labels, periods, and provenance must align")
    fps = float(np.asarray(record.get("native_fps", record["fps"])))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Native frame rate must be positive")
    centers = np.flatnonzero((y[1:] >= 0) & (y[:-1] >= 0) & (y[1:] != y[:-1]) &
                             (period[1:] == period[:-1]) &
                             (np.isin(provenance[1:], QUALIFYING_PROVENANCE) |
                              np.isin(provenance[:-1], QUALIFYING_PROVENANCE))) + 1
    radius = max(1, int(round(RADIUS_SECONDS * fps)))
    mask = np.zeros(len(y), dtype=bool)
    for center in centers:
        lo, hi = max(0, int(center) - radius), min(len(y), int(center) + radius + 1)
        same_period = period[lo:hi] == period[center]
        mask[lo:hi] |= same_period
    return mask


def sample_weights(record: dict, native_mask: np.ndarray) -> np.ndarray:
    indices = np.asarray(record["indices"], dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(native_mask)):
        raise ValueError("Sample-to-native indices are invalid")
    return np.where(native_mask[indices], WEIGHT, 1.0).astype(np.float32)


def control_cache(fold: str, seed: int) -> Path:
    return sc.initial_path(fold, seed)


def declaration() -> dict:
    data_root = Path(os.environ["MTAH_DATA_DIR"])
    train_manifest = RECORDS / "train" / "manifest.json"
    if not train_manifest.is_file():
        raise FileNotFoundError("TRAIN dense-record preparation is incomplete")
    prep_manifest = json.loads(train_manifest.read_text())
    if len(prep_manifest["games"]) != 45:
        raise ValueError("Expected exactly 45 frozen TRAIN games")
    if sha256(train_manifest) != EXPECTED_TRAIN_MANIFEST:
        raise ValueError("Restored dense TRAIN record manifest differs from the frozen D2 input")
    teacher_hashes = {}
    for fold in FOLDS:
        for seed in SEEDS:
            path = control_cache(fold, seed)
            if not path.is_file() or path.stat().st_size < 10_000:
                raise FileNotFoundError(f"Missing restored full fold teacher: {path}")
            actual=sha256(path); expected=EXPECTED_TEACHERS[f"{fold}/{seed}"]
            if actual != expected: raise ValueError(f"Restored fold teacher hash mismatch: {fold}/{seed}")
            teacher_hashes[f"{fold}/{seed}"] = actual
    old_spec=old.old_spec()
    old_manifest=old.DATA/"experiments/objective_alignment_v1/predeclared.json"
    old_cache_hashes={match:sha256(data_root/Path(entry["cache"]).relative_to("data"))
                      for match,entry in old_spec["prepared_training_caches"].items()}
    eval_hashes={entry["path"]:entry["sha256"]
                 for entry in old_spec["evaluation_feature_cache"]["files"].values()}
    for rel,expected in eval_hashes.items():
        if sha256(data_root/Path(rel).relative_to("data"))!=expected:
            raise ValueError(f"Evaluation cache hash mismatch before declaration: {rel}")
    raw_receipt=RESULTS/"soccernet_huggingface_restore.json"
    return {
        "schema": 1,
        "study": "dense_label_weight_v1",
        "question": "Does reducing dense-silver loss near uncertain touch/inferred transitions improve full-provider transfer while preserving interior supervision?",
        "arms": {
            "control_D2": "Frozen sn_dense_v1 D2 objective: dense H/A/D supervision, no reliability reweighting.",
            "weight_025": "Same D2 recipe and samples; frame-level H/A, D, and live-transition losses receive weight 0.25 within the mask below and 1.0 elsewhere.",
        },
        "common": {
            "folds": list(FOLDS), "seeds": list(SEEDS), "updates_per_fit": 2048,
            "batch_size": 32, "soccernet_fraction": 0.5, "width": 96,
            "learning_rate": 0.0003, "d_learning_rate": 0.0001,
            "input_dropout": 0.1, "initialization": "matching full-provider fold teacher",
            "old_data": "all original TRAIN caches from the two non-held-out providers; provider excluded from every learned component",
            "soccernet_data": "45 TRAIN games only; no GAR VALID/TEST labels are read",
            "cohort": "15 previously used full-provider selection cells: 5 matches x clean, corrupt_1, corrupt_2; official scorer and unchanged cost-3 decoder",
            "scoring": "two seed members plus native-probability average ensemble; equal-provider means; paired by source/match/view",
            "cpu_threads_per_fit": 1,
        },
        "weight_formula": {
            "boundary": "t is the first frame of a changed label; evaluate adjacent frames t-1,t",
            "eligible": "both labels known, labels differ, full_period equal, and either adjacent provenance in P_TOUCH=1/P_PROPAGATED=2/P_HELD=4",
            "mask": "native frame i is ambiguous iff abs(i-t) <= round(0.5*native_fps) for any eligible boundary, clipped to this record",
            "weights": {"ambiguous": WEIGHT, "interior_and_other": 1.0},
            "loss_normalization": "for each sampled window and each dense frame loss, sum(weight*loss)/sum(weight over that loss's valid mask); transition terms retain their frozen stable/change group balance with per-edge weights multiplied by the same frame weight",
            "unknowns": "unknown/X (-1) targets remain masked and never create a boundary; target-derived weights are used only during training",
            "interiors": "all frames outside eligible ±0.5-second masks retain weight 1.0",
        },
        "guards": {
            "replay": "reconstructed control ensemble must match source-head PR47 D2 clean/feed-like aggregates to four-decimal rounding tolerance 0.000051",
            "control_score_expected": EXPECTED_CONTROL,
            "provider_exclusion": "enforced by old.load_old(fold), split plan, and fold teacher identity",
            "no_public": True, "no_gar_valid_or_test": True,
            "no_promotion": True, "no_shared_docs_or_frozen_changes": True,
        },
        "restore": {
            "dataset": "OpenSportsLab/SoccerNet-GAR tracking-full",
            "revision": "32c840210b7d08d06976e7f9457dfff79fb2da99",
            "raw_train_restore_receipt": str(RESULTS / "soccernet_huggingface_restore.json"),
            "raw_train_restore_receipt_sha256": sha256(raw_receipt),
            "dense_train_manifest_sha256": sha256(train_manifest),
            "expected_dense_train_manifest_sha256": EXPECTED_TRAIN_MANIFEST,
            "dense_train_game_files": prep_manifest["game_files"],
            "fold_teacher_sha256": teacher_hashes,
            "original_training_cache_sha256": old_cache_hashes,
            "original_training_manifest_sha256": sha256(old_manifest),
            "external_evaluation_cache_sha256": eval_hashes,
            "old_cache_root": str(data_root / "experiments/objective_alignment_v1"),
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (ROOT / "experiments/sn_dense_train.py", ROOT / "experiments/soccernet_dense_labels.py",
                      ROOT / "experiments/soccernet_capacity_model.py",
                      ROOT / "experiments/objective_losses.py", ROOT / "experiments/soccernet_extension.py",
                      ROOT / "experiments/soccernet_capacity.py",
                      ROOT / "experiments/preparation.py", ROOT / "experiments/combined_objective_deployment.py",
                      ROOT / "experiments/objective_ensemble.py", ROOT / "experiments/metrics.py",
                      ROOT / "experiments/spell_diagnostics.py",
                      ROOT / "experiments/supervised.py", ROOT / "experiments/learned_temporal.py",
                      ROOT / "experiments/factorized_had.py", ROOT / "evaluation/possession_metrics.py",
                      ROOT / "experiments/tests/test_dense_label_weight_v1.py", Path(__file__))
        },
    }


def verify_declared_sources() -> dict:
    path=RESULTS/"dense_label_weight_v1_declaration.json"
    if not path.is_file(): raise RuntimeError("Freeze declaration before fitting or scoring")
    value=json.loads(path.read_text())
    for rel,expected in value["source_sha256"].items():
        actual=sha256(ROOT/rel)
        if actual!=expected: raise RuntimeError(f"Declared source changed after freeze: {rel}")
    return value


def verify_declared_inputs() -> dict:
    value=verify_declared_sources()
    restore=value["restore"]
    if sha256(RECORDS/"train/manifest.json")!=restore["dense_train_manifest_sha256"]:
        raise RuntimeError("Dense TRAIN record manifest changed after declaration")
    old_manifest=old.DATA/"experiments/objective_alignment_v1/predeclared.json"
    if sha256(old_manifest)!=restore["original_training_manifest_sha256"]:
        raise RuntimeError("Original training-cache manifest changed after declaration")
    for key,expected in restore["fold_teacher_sha256"].items():
        fold,seed=key.rsplit("/",1)
        if sha256(control_cache(fold,int(seed)))!=expected:
            raise RuntimeError(f"Fold teacher changed after declaration: {key}")
    for rel,expected in restore["external_evaluation_cache_sha256"].items():
        path=old.DATA/Path(rel).relative_to("data")
        if sha256(path)!=expected:
            raise RuntimeError(f"External evaluation cache changed after declaration: {rel}")
    return value


def frame_weight_vectors(records: list[dict]) -> tuple[list[np.ndarray], dict]:
    vectors = []
    summary = {"records": 0, "native_frames": 0, "sampled_frames": 0,
               "ambiguous_native_frames": 0, "known_sampled_frames": 0,
               "weighted_sampled_frames": 0, "weighted_ha_frames": 0,
               "weighted_d_frames": 0,
               "qualifying_boundary_count": 0, "native_mask_sha256": {}}
    for r in records:
        mask = ambiguous_native_mask(r)
        y=np.asarray(r["full_y"],dtype=np.int8); period=np.asarray(r["full_period"])
        prov=np.asarray(r["full_prov"],dtype=np.int8)
        eligible=(y[1:]>=0)&(y[:-1]>=0)&(y[1:]!=y[:-1])&(period[1:]==period[:-1])&(
            np.isin(prov[1:],QUALIFYING_PROVENANCE)|np.isin(prov[:-1],QUALIFYING_PROVENANCE))
        w = sample_weights(r, mask)
        key = f"{str(np.asarray(r['match']).reshape(()).item())}/{r['game_period']}/{int(np.asarray(r['full_frame_numbers'])[0])}"
        summary["native_mask_sha256"][key] = hashlib.sha256(mask.tobytes()).hexdigest()
        summary["records"] += 1
        summary["native_frames"] += int(len(mask))
        summary["sampled_frames"] += int(len(w))
        summary["ambiguous_native_frames"] += int(mask.sum())
        y_sample=np.asarray(r["y"])
        known=np.isin(y_sample,(0,1,2))
        ha=np.isin(y_sample,(0,1)); dead=(y_sample==2)
        summary["known_sampled_frames"] += int(known.sum())
        summary["weighted_sampled_frames"] += int(((w<1)&known).sum())
        summary["weighted_ha_frames"] += int(((w<1)&ha).sum())
        summary["weighted_d_frames"] += int(((w<1)&dead).sum())
        summary["qualifying_boundary_count"] += int(eligible.sum())
        vectors.append(w)
    summary["ambiguous_native_fraction"]=(summary["ambiguous_native_frames"]/summary["native_frames"]
                                           if summary["native_frames"] else 0.0)
    summary["weighted_sampled_fraction"]=(summary["weighted_sampled_frames"]/summary["known_sampled_frames"]
                                           if summary["known_sampled_frames"] else 0.0)
    return vectors, summary


def train_weighted(initial_model: dict, old_records: list[dict], new_records: list[dict],
                   vectors: list[np.ndarray], *, seed: int, steps: int = 2048) -> dict:
    """D2-equivalent sampler and architecture with explicit frame-weight means.

    This is a narrow new loop adaptation of sn_dense_train.train: all sampling,
    normalization, target creation, initialization, optimizers and constants use
    the shared frozen helpers. The sole objective change is the weighted mean
    on dense H/A and D frames, plus multiplying dense transition edge weights.
    """
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    names, median, scale, input_features = sc.model._model_input_width(initial_model)
    old_groups, old_ids = sc.model._window_index(old_records, names, old=True)
    old_spell, old_pairs, old_weights, _ = sc.model._old_objective_targets(
        old_records, require_full_native=True)
    new_groups, new_ids = sc.model._window_index(new_records, names, old=False)
    new_spell, new_pairs, new_weights, _ = sn.objective_targets(new_records, require_full_native=True)
    if len(vectors) != len(new_records) or any(len(w) != len(r["y"]) for w, r in zip(vectors, new_records)):
        raise ValueError("Per-record weight vectors do not align to dense labels")
    # Precompute transition weights from the same native-frame mask at each
    # sampled frame. The transition helper's edge row is frame-aligned.
    new_weights = [w * edge for w, edge in zip(new_weights, vectors)]
    torch.manual_seed(int(seed))
    model = sc.model.network(input_features, 96, sc.model.INPUT_DROPOUT)
    sc.model._copy_teacher_function(initial_model, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=sc.model.WEIGHT_DECAY)
    d_net = fh.network(input_features, list(initial_model["player_indices"]),
                       bool(initial_model["player_only"]))
    d_net._sn_player_only = bool(initial_model["player_only"])
    d_net.load_state_dict({k: torch.as_tensor(v) for k, v in initial_model["state"].items()})
    d_optimizer = torch.optim.AdamW(d_net.parameters(), lr=1e-4, weight_decay=sc.model.WEIGHT_DECAY)
    rng = random.Random(int(seed))
    batch_size, fraction = 32, 0.5
    old_count = new_count = 16
    base = dict(median=median, scale=scale, feature_names=names, input_features=input_features)
    config = dict(width=96, lr=3e-4, seed=int(seed), steps=int(steps), fraction=fraction,
                  batch_size=batch_size, input_dropout=sc.model.INPUT_DROPOUT,
                  weight_decay=sc.model.WEIGHT_DECAY, gradient_clip=sc.model.GRADIENT_CLIP,
                  ha_target="dense", train_d=True, d_lr=1e-4, d_weight=1.0,
                  target_weighting=dict(ambiguous=WEIGHT, radius_seconds=RADIUS_SECONDS),
                  teacher_sha256=sc.model._checkpoint_sha(initial_model))
    history = []; exposures = {}
    model.train(); d_net.train()
    for step in range(1, steps + 1):
        bx=[]; by=[]; bg=[]; bs=[]; bt=[]; bm=[]; bw=[]; fw=[]
        for group, count in ((0, old_count), (1, new_count)):
            for _ in range(count):
                if group == 0:
                    ri, start, size, full, alt = sc.model._draw_old(rng, old_groups, old_records)
                    record = old_records[ri]; weight = np.ones(sc.model.CORE, np.float32)
                else:
                    ri, start, size, full, alt = sc.model._draw_new(rng, new_groups)
                    record = new_records[ri]
                    weight = np.ones(sc.model.CORE, np.float32)
                    weight[:size] = vectors[ri][start:start + size]
                swap = bool(group and rng.random() < 0.5)
                x, y = sc.model._prepare_window(record, start, size, full, alt,
                                                median=median, scale=scale, swap=swap)
                if not np.isin(y, (0, 1)).any():
                    raise RuntimeError("Sampler yielded no known H/A targets")
                spell=np.ones(sc.model.CORE,np.float32); tt=np.zeros(sc.model.CORE,np.float32)
                tm=np.zeros(sc.model.CORE,bool); tw=np.zeros(sc.model.CORE,np.float32)
                if group == 0:
                    spell[:size]=old_spell[ri][start:start+size]
                    pair=old_pairs[ri]; tt[:size]=pair["live_target"][start:start+size]
                    tm[:size]=pair["live_mask"][start:start+size]
                    tw[:size]=old_weights[ri][start:start+size]
                    exposures["old/"+"/".join(old_ids[ri])] = exposures.get("old/"+"/".join(old_ids[ri]),0)+1
                else:
                    spell[:size]=new_spell[ri][start:start+size]
                    pair=new_pairs[ri]; tt[:size]=pair["live_target"][start:start+size]
                    tm[:size]=pair["live_mask"][start:start+size]
                    tw[:size]=new_weights[ri][start:start+size]
                    key="new/"+new_ids[ri][1]; exposures[key]=exposures.get(key,0)+1
                bx.append(x); by.append(y); bg.append(group); bs.append(spell); bt.append(tt); bm.append(tm); bw.append(tw); fw.append(weight)
        order=list(range(batch_size)); rng.shuffle(order)
        xt=torch.as_tensor(np.stack([bx[i] for i in order]),dtype=torch.float32)
        yt=torch.as_tensor(np.stack([by[i] for i in order]),dtype=torch.long)
        gt=torch.as_tensor([bg[i] for i in order],dtype=torch.int64)
        st=torch.as_tensor(np.stack([bs[i] for i in order]),dtype=torch.float32)
        ttar=torch.as_tensor(np.stack([bt[i] for i in order]),dtype=torch.float32)
        tmask=torch.as_tensor(np.stack([bm[i] for i in order]),dtype=torch.bool)
        twt=torch.as_tensor(np.stack([bw[i] for i in order]),dtype=torch.float32)
        frame_w=torch.as_tensor(np.stack([fw[i] for i in order]),dtype=torch.float32)
        logits_all=model(xt); logits=logits_all[:,sn.CONTEXT:sn.CONTEXT+sn.CORE]
        ha=(yt==0)|(yt==1); safe_y=torch.where(ha,yt,torch.zeros_like(yt))
        row_loss=F.cross_entropy(logits.transpose(1,2),safe_y,reduction="none")
        ha_w=frame_w*ha
        ha_den=ha_w.sum(dim=1)
        if torch.any(ha_den<=0): raise RuntimeError("Weighted H/A window has no effective label")
        factors=1.0+sc.model.SPELL_COEFFICIENT*(st-1.0)
        per_window=(row_loss*ha_w*factors).sum(dim=1)/ha_den
        old_mask,new_mask=gt==0,gt==1
        loss=0.5*per_window[old_mask].mean()+0.5*per_window[new_mask].mean()
        ql=torch.softmax(logits_all[:,sn.CONTEXT:sn.CONTEXT+sn.CORE],dim=-1)
        qr=torch.softmax(logits_all[:,sn.CONTEXT+1:sn.CONTEXT+sn.CORE+1],dim=-1)
        pchange=ql[...,0]*qr[...,1]+ql[...,1]*qr[...,0]
        edge=F.binary_cross_entropy(pchange.clamp(1e-6,1-1e-6),ttar,reduction="none")
        trans=ol.group_balanced_binary_loss(edge,ttar,tmask & new_mask[:,None],twt)
        if trans is not None: loss=loss+0.5*sc.model.LIVE_TRANSITION_COEFFICIENT*trans
        trans_old=ol.group_balanced_binary_loss(edge,ttar,tmask & old_mask[:,None],twt)
        if trans_old is not None: loss=loss+0.5*sc.model.LIVE_TRANSITION_COEFFICIENT*trans_old
        dead=sn._dead_logit(d_net,xt)[:,sn.CONTEXT:sn.CONTEXT+sn.CORE]
        known=yt>=0
        dbce=F.binary_cross_entropy_with_logits(dead,(yt==2).float(),reduction="none")
        d_w=frame_w*known; d_den=d_w.sum(dim=1)
        per_d=(dbce*d_w).sum(dim=1)/d_den.clamp(min=1e-12)
        has=known.any(dim=1)
        loss=loss+0.5*per_d[old_mask & has].mean()+0.5*per_d[new_mask & has].mean()
        if not torch.isfinite(loss): raise RuntimeError(f"Non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); d_optimizer.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),sc.model.GRADIENT_CLIP)
        optimizer.step(); torch.nn.utils.clip_grad_norm_(d_net.parameters(),sc.model.GRADIENT_CLIP); d_optimizer.step()
        if step % 32 == 0 or step == steps:
            history.append(dict(step=step,loss=float(loss.detach())))
        if step % 256 == 0: print(f"WEIGHTED step={step}/{steps} loss={float(loss.detach()):.5g}",flush=True)
    model.eval(); d_net.eval()
    exported=sn._export(model,d_net,base=base,teacher_sha=sc.model._checkpoint_sha(initial_model),config=config,step=steps)
    return dict(model=exported,history=history,exposures=exposures,best_step=steps)


def load_training(fold: str) -> tuple[dict, list[dict], list[dict]]:
    seed=2026
    teacher=joblib.load(control_cache(fold,seed))
    old_records=old.load_old(fold)
    records=dl.load_dense_records("train",output=RECORDS)
    if len({str(np.asarray(r["match"]).reshape(()).item()) for r in records}) != 45:
        raise ValueError("Expected complete 45-game SoccerNet TRAIN cohort")
    for r in records:
        if not all(k in r for k in ("full_y","full_period","full_prov","indices","y","period")):
            raise ValueError("Dense training record lacks native label/provenance fields")
    return teacher,old_records,records


def run_fit(arm: str, fold: str, seed: int, records=None) -> dict:
    if arm not in ARMS or fold not in FOLDS or seed not in SEEDS: raise ValueError("Undeclared fit identity")
    declared=verify_declared_sources()
    if arm=="weight_025": require_control_gate()
    expected_teacher=declared["restore"]["fold_teacher_sha256"][f"{fold}/{seed}"]
    if sha256(control_cache(fold,seed))!=expected_teacher:
        raise RuntimeError("Fold teacher changed after declaration")
    if sha256(RECORDS/"train/manifest.json")!=declared["restore"]["dense_train_manifest_sha256"]:
        raise RuntimeError("Dense TRAIN record manifest changed after declaration")
    old_manifest=old.DATA/"experiments/objective_alignment_v1/predeclared.json"
    if sha256(old_manifest)!=declared["restore"]["original_training_manifest_sha256"]:
        raise RuntimeError("Original training-cache manifest changed after declaration")
    fit_id=f"{arm}__{fold.removeprefix('leave_out_')}__s{seed}"
    model_path=MODELS/f"{fit_id}.joblib"; receipt_path=FITS/f"{fit_id}.json"
    if model_path.exists() or receipt_path.exists(): raise ValueError("Fit artifacts already exist; no implicit rerun")
    teacher=joblib.load(control_cache(fold,seed)); old_records=old.load_old(fold)
    if records is None: records=dl.load_dense_records("train",output=RECORDS)
    start=time.monotonic()
    if arm=="control_D2":
        mask_info={"applied":False}
        out=sn.train(teacher,old_records,records,width=96,lr=3e-4,seed=seed,steps=2048,
                     fraction=.5,batch_size=32,ha_target="dense",train_d=True,
                     require_full_native=True,log_every=256,verbose=True)
    else:
        vectors,mask_info=frame_weight_vectors(records)
        mask_info["applied"]=True
        out=train_weighted(teacher,old_records,records,vectors,seed=seed,steps=2048)
    model=out["model"]
    model_path.parent.mkdir(parents=True,exist_ok=True); FITS.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,model_path)
    receipt={"study":"dense_label_weight_v1","arm":arm,"fold":fold,"seed":seed,
             "steps_run":int(model["step"]),"model_path":str(model_path.relative_to(ROOT)),
             "model_sha256":sha256(model_path),"teacher_sha256":sha256(control_cache(fold,seed)),
             "old_training_matches":[str(v) for v in sorted({r["match"] for r in old_records})],
             "soccernet_games":sorted({str(np.asarray(r["match"]).reshape(()).item()) for r in records}),
             "target_mask_summary":mask_info,"history":out.get("history",[]),
             "wall_seconds":time.monotonic()-start,"training_config":model.get("training",{}),
             "exposures":out.get("exposures",{})}
    write_once(receipt_path,receipt)
    return receipt


def saved_inputs(item: dict, view: str, replicate: str):
    path=old.eval_path(item,view,replicate)
    if not path.is_file() or path.stat().st_size<1000:
        raise FileNotFoundError(f"Missing restored full evaluation cache: {path}")
    rel="data/"+path.relative_to(old.DATA).as_posix()
    expected=next((entry["sha256"] for entry in
                   old.old_spec()["evaluation_feature_cache"]["files"].values()
                   if entry["path"]==rel),None)
    if not expected or sha256(path)!=expected:
        raise ValueError(f"Evaluation cache hash differs from frozen dependency manifest: {path}")
    declared=verify_declared_sources()
    if declared["restore"]["external_evaluation_cache_sha256"].get(rel)!=expected:
        raise ValueError(f"Evaluation cache is not bound by the frozen declaration: {path}")
    cached=joblib.load(path)
    return path,cached


def predict_cell(arm: str, fold: str, seed: int, model: dict, teacher: dict,
                 model_sha: str, teacher_sha: str,
                 item: dict, view: str, replicate: str) -> tuple[dict,dict]:
    cache_path,cached=saved_inputs(item,view,replicate)
    inputs,labels=cached["inputs"],cached["labels"]
    record=old.inference_record(cached["record"])
    sampled=sn.predict(record,model,teacher)
    probability=old.upsample(sampled,record,inputs)
    row,pred=old.metric_row(probability,inputs,labels,item,view,replicate,arm,str(seed))
    from experiments import objective_ensemble as scoring
    pred_sha=hashlib.sha256(np.ascontiguousarray(np.asarray(pred,dtype="U1")).tobytes()).hexdigest()
    prob_sha=scoring.digest_array(probability)
    row.update(source_study_id=STUDY_ID,reconstruction_status=RECONSTRUCTION_STATUS,
               fold=fold,model_sha256=model_sha,teacher_sha256=teacher_sha,
               prediction_sha256=pred_sha,probabilities_sha256=prob_sha)
    model_period=supervised.model_periods(inputs)
    original_period=np.asarray(inputs["period"])
    ball_observed=np.asarray(inputs.get("ball_observed"),dtype=bool)
    if ball_observed.shape!=(len(probability),):
        raise ValueError("Evaluation ball_observed mask missing or not aligned")
    timestamps=np.asarray(inputs.get("timestamps"),dtype=float)
    if timestamps.shape!=(len(probability),): raise ValueError("Evaluation timestamps do not align")
    identity=dict(source=item["source"],match=item["match"],view=view,replicate=replicate,
                  seed=str(seed),arm=arm,fold=fold,source_study_id=STUDY_ID,
                  reconstruction_status=RECONSTRUCTION_STATUS,model_sha256=model_sha,
                  teacher_sha256=teacher_sha,source_arm=arm,inference_seed=f"seed{seed}",
                  model_checkpoint_hashes_json=json.dumps({str((MODELS/f"{arm}__{fold.removeprefix('leave_out_')}__s{seed}.joblib").relative_to(ROOT)):model_sha},sort_keys=True),
                  evaluation_input_path=f"data/{cache_path.relative_to(old.DATA).as_posix()}",
                  evaluation_input_sha256=sha256(cache_path),
                  fold_mapping_json=json.dumps({p:f"leave_out_{p}" for p in ("sportec","metrica","skillcorner")},sort_keys=True))
    out=PREDICTIONS/arm/fold/f"seed{seed}__{item['match']}__{view}__{replicate}.npz"
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,probabilities=probability.astype(np.float64),prediction=np.asarray(pred,dtype="U1"),
        labels=np.asarray(labels,dtype="U1"),period=original_period,model_period=model_period,
        fps=np.asarray(float(inputs["fps"])),
        timestamps=timestamps,ball_observed=ball_observed,**{k:np.asarray(v) for k,v in identity.items()})
    return row,{"path":str(out.relative_to(ROOT)),"sha256":sha256(out),"cache_sha256":sha256(cache_path)}


def score_models(arms: tuple[str,...]=ARMS) -> dict:
    declared=verify_declared_inputs()
    rows=[]; files={}
    items=old.selection_items()
    for arm in arms:
        if arm=="weight_025": require_control_gate()
        for fold in FOLDS:
            for seed in SEEDS:
                model_path=MODELS/f"{arm}__{fold.removeprefix('leave_out_')}__s{seed}.joblib"
                fit_receipt=FITS/f"{arm}__{fold.removeprefix('leave_out_')}__s{seed}.json"
                sealed=json.loads(fit_receipt.read_text())
                if sha256(model_path)!=sealed["model_sha256"]:
                    raise ValueError(f"Model hash differs from fit receipt: {model_path}")
                expected_teacher=declared["restore"]["fold_teacher_sha256"][f"{fold}/{seed}"]
                if sealed["teacher_sha256"]!=expected_teacher:
                    raise ValueError(f"Fit receipt teacher differs from frozen declaration: {model_path}")
                model=joblib.load(model_path); teacher=joblib.load(control_cache(fold,seed))
                model_sha=sha256(model_path); teacher_sha=sha256(control_cache(fold,seed))
                for item in items:
                    if "leave_out_"+item["source"]!=fold: continue
                    for view,replicate in old.VIEWS:
                        row,file=predict_cell(arm,fold,seed,model,teacher,model_sha,teacher_sha,item,view,replicate)
                        rows.append(row);files[file["path"]]=file["sha256"]
    result={"arm":arms[0] if len(arms)==1 else list(arms),"rows":rows,"prediction_files":files}
    name=arms[0] if len(arms)==1 else "all"
    write_once(RESULTS/f"dense_label_weight_v1_member_scores_{name}.json",result)
    return result


def require_control_gate() -> dict:
    path=RESULTS/"dense_label_weight_v1_control_replay.json"
    if not path.is_file(): raise RuntimeError("Weight candidate blocked until D2 replay gate is sealed")
    gate=json.loads(path.read_text())
    if gate.get("passed") is not True:
        raise RuntimeError("Weight candidate blocked because D2 replay gate failed")
    verify_declared_sources()
    member_path=RESULTS/"dense_label_weight_v1_member_scores_control_D2.json"
    ensemble_path=RESULTS/"dense_label_weight_v1_ensemble_scores_control_D2.json"
    checks={
        "declaration_sha256":sha256(RESULTS/"dense_label_weight_v1_declaration.json"),
        "member_scores_sha256":sha256(member_path),
        "ensemble_scores_sha256":sha256(ensemble_path),
        "dense_train_manifest_sha256":sha256(RECORDS/"train/manifest.json"),
        "raw_train_restore_receipt_sha256":sha256(RESULTS/"soccernet_huggingface_restore.json"),
        "original_training_manifest_sha256":sha256(old.DATA/"experiments/objective_alignment_v1/predeclared.json"),
    }
    for key,actual in checks.items():
        if gate.get(key)!=actual: raise RuntimeError(f"Control replay gate is stale or corrupted: {key}")
    return gate


def control_replay() -> dict:
    member_path=RESULTS/"dense_label_weight_v1_member_scores_control_D2.json"
    ensemble_path=RESULTS/"dense_label_weight_v1_ensemble_scores_control_D2.json"
    if not member_path.is_file() or not ensemble_path.is_file():
        raise FileNotFoundError("Control member and ensemble scoring must finish before replay gate")
    verify_declared_inputs()
    members=json.loads(member_path.read_text()); ensembles=json.loads(ensemble_path.read_text())
    rows=ensembles["rows"]
    expected_members={(i["source"],i["match"],v,r,"control_D2",str(seed))
                      for i in old.selection_items() for v,r in old.VIEWS for seed in SEEDS}
    expected_ensembles={(i["source"],i["match"],v,r,"control_D2","ensemble")
                        for i in old.selection_items() for v,r in old.VIEWS}
    member_keys={(r["source"],r["match"],r["view"],r["replicate"],r["arm"],r["member"])
                 for r in members["rows"] if r["arm"]=="control_D2"}
    ensemble_keys={(r["source"],r["match"],r["view"],r["replicate"],r["arm"],r["member"])
                    for r in rows if r["arm"]=="control_D2"}
    if len(members["rows"])!=30 or member_keys!=expected_members:
        raise ValueError("Control member replay does not cover exactly 30 declared cells")
    if len(rows)!=15 or ensemble_keys!=expected_ensembles:
        raise ValueError("Control ensemble replay does not cover exactly 15 declared cells")
    values={}
    for view in ("clean","feed_like"):
        selected=[r for r in rows if r["view"]==view and r["arm"]=="control_D2" and r["member"]=="ensemble"]
        aggregate=sc._aggregate_provider(selected,"control_D2",view,"ensemble")
        score=float(aggregate["equal_provider_mean"]["spell_f1"])
        values[view]={"observed":score,"expected":EXPECTED_CONTROL[view],
                      "abs_error":abs(score-EXPECTED_CONTROL[view]),
                      "tolerance":CONTROL_TOLERANCE,
                      "pass":abs(score-EXPECTED_CONTROL[view])<=CONTROL_TOLERANCE}
    passed=all(v["pass"] for v in values.values())
    gate={"passed":passed,"values":values,
          "member_scores_sha256":sha256(member_path),
          "ensemble_scores_sha256":sha256(ensemble_path),
          "declaration_sha256":sha256(RESULTS/"dense_label_weight_v1_declaration.json"),
          "dense_train_manifest_sha256":sha256(RECORDS/"train/manifest.json"),
          "raw_train_restore_receipt_sha256":sha256(RESULTS/"soccernet_huggingface_restore.json"),
          "original_training_manifest_sha256":sha256(old.DATA/"experiments/objective_alignment_v1/predeclared.json"),
          "expected_source":"PR47 source_conditioned_heads_report reconstructed D2 control",
          "scored_cells":len(rows),"public_labels_accessed":False}
    write_once(RESULTS/"dense_label_weight_v1_control_replay.json",gate)
    if not passed: raise RuntimeError("Reconstructed D2 control did not pass the declared replay gate")
    return gate


def ensemble_scores(arms: tuple[str,...]=ARMS) -> dict:
    verify_declared_inputs()
    from experiments import objective_ensemble as scoring
    rows=[]; files={}
    for arm in arms:
      if arm=="weight_025": require_control_gate()
      for fold in FOLDS:
       for item in old.selection_items():
        if "leave_out_"+item["source"]!=fold: continue
        for view,replicate in old.VIEWS:
         paths=[PREDICTIONS/arm/fold/f"seed{s}__{item['match']}__{view}__{replicate}.npz" for s in SEEDS]
         member_score_path=RESULTS/f"dense_label_weight_v1_member_scores_{arm}.json"
         sealed_prediction_files=json.loads(member_score_path.read_text())["prediction_files"]
         for path in paths:
            if sealed_prediction_files.get(str(path.relative_to(ROOT)))!=sha256(path):
                raise ValueError(f"Member prediction differs from sealed score receipt: {path}")
         arrays=[np.load(p,allow_pickle=False) for p in paths]
         prob=(arrays[0]["probabilities"]+arrays[1]["probabilities"])/2
         labels=arrays[0]["labels"]; periods=arrays[0]["period"]; model_period=arrays[0]["model_period"]
         if not np.array_equal(labels,arrays[1]["labels"]) or not np.array_equal(periods,arrays[1]["period"]):
            raise ValueError("Ensemble members do not share identical scored frames")
         if not np.array_equal(model_period,arrays[1]["model_period"]): raise ValueError("Model periods differ")
         prediction=supervised.decode(prob,model_period,float(arrays[0]["fps"]),lt.COST)
         inputs={"period":periods,"timestamps":arrays[0]["timestamps"],"fps":float(arrays[0]["fps"]),
                 "ball_observed":arrays[0]["ball_observed"]}
         item_labels=labels.astype("U1")
         row=scoring.score_prediction(item_labels,prediction,inputs,arm,
             dict(source=item["source"],match=item["match"],view=view,replicate=replicate,arm=arm,member="ensemble"),
             scoring.digest_array(prob))
         model_hashes={}
         for seed in SEEDS:
            model_path=MODELS/f"{arm}__{fold.removeprefix('leave_out_')}__s{seed}.joblib"
            model_hashes[str(model_path.relative_to(ROOT))]=sha256(model_path)
         eval_hashes={f"data/{old.eval_path(item,view,replicate).relative_to(old.DATA).as_posix()}":
                      sha256(old.eval_path(item,view,replicate))}
         row.update(source_study_id=STUDY_ID,reconstruction_status=RECONSTRUCTION_STATUS,
                    source_arm=arm,inference_seed="seedensemble",fold=fold,
                    model_checkpoint_hashes=model_hashes,evaluation_input_hashes=eval_hashes,
                    prediction_sha256=hashlib.sha256(np.ascontiguousarray(np.asarray(prediction,dtype="U1")).tobytes()).hexdigest(),
                    probabilities_sha256=scoring.digest_array(prob))
         rows.append(row)
         out=PREDICTIONS/arm/fold/f"seedensemble__{item['match']}__{view}__{replicate}.npz"
         np.savez_compressed(out,probabilities=prob,prediction=np.asarray(prediction,dtype="U1"),
             labels=labels,period=periods,model_period=model_period,fps=arrays[0]["fps"],timestamps=arrays[0]["timestamps"],
             ball_observed=arrays[0]["ball_observed"],source=arrays[0]["source"],match=arrays[0]["match"],
             view=arrays[0]["view"],replicate=arrays[0]["replicate"],seed=np.asarray("ensemble"),arm=np.asarray(arm),fold=np.asarray(fold),
             source_study_id=np.asarray(STUDY_ID),reconstruction_status=np.asarray(RECONSTRUCTION_STATUS),
             source_arm=np.asarray(arm),inference_seed=np.asarray("seedensemble"),
             model_checkpoint_hashes_json=np.asarray(json.dumps(model_hashes,sort_keys=True)),
             evaluation_input_hashes_json=np.asarray(json.dumps(eval_hashes,sort_keys=True)),
             fold_mapping_json=np.asarray(json.dumps({p:f"leave_out_{p}" for p in ("sportec","metrica","skillcorner")},sort_keys=True)))
         files[str(out.relative_to(ROOT))]=sha256(out)
         for a in arrays: a.close()
    value={"arm":arms[0] if len(arms)==1 else list(arms),"rows":rows,"prediction_files":files}
    name=arms[0] if len(arms)==1 else "all"
    write_once(RESULTS/f"dense_label_weight_v1_ensemble_scores_{name}.json",value)
    for arm in arms:
        model_hashes={}
        for fold in FOLDS:
            for seed in SEEDS:
                p=MODELS/f"{arm}__{fold.removeprefix('leave_out_')}__s{seed}.joblib"
                model_hashes[str(p.relative_to(ROOT))]=sha256(p)
        input_hashes={}
        for item in old.selection_items():
            for view,replicate in old.VIEWS:
                p=old.eval_path(item,view,replicate)
                input_hashes[f"data/{p.relative_to(old.DATA).as_posix()}"]=sha256(p)
        provenance={"source_study_id":STUDY_ID,"reconstruction_status":RECONSTRUCTION_STATUS,
            "source_arm":arm,"inference_seed":"seedensemble","model_checkpoint_hashes":model_hashes,
            "evaluation_input_hashes":input_hashes,
            "fold_mapping":{provider:f"leave_out_{provider}" for provider in ("sportec","metrica","skillcorner")},
            "member_prediction_files":{
                str(p.relative_to(ROOT)):sha256(p) for p in PREDICTIONS.joinpath(arm).rglob("seed20*.npz")
                if "seedensemble__" not in p.name},
            "ensemble_prediction_files":{k:v for k,v in files.items() if f"/{arm}/" in k}}
        write_once(RESULTS/f"dense_label_weight_v1_prediction_provenance_{arm}.json",provenance)
    return value


def summarize_scores(member: dict, ensemble: dict) -> dict:
    from experiments import objective_ensemble as scoring
    rows=member["rows"]+ensemble["rows"]
    summaries={}
    for arm in ARMS:
        summaries[arm]={}
        for view in ("clean","feed_like"):
            selected=[r for r in rows if r["arm"]==arm and r["view"]==view and r["member"]=="ensemble"]
            summaries[arm][view]=sc._aggregate_provider(selected,arm,view,"ensemble")
    seed_summaries={}
    for arm in ARMS:
      seed_summaries[arm]={}
      for member_id in ("2026","2027"):
       seed_summaries[arm][member_id]={}
       for view in ("clean","feed_like"):
        selected=[r for r in member["rows"] if r["arm"]==arm and r["view"]==view and r["member"]==member_id]
        seed_summaries[arm][member_id][view]=sc._aggregate_provider(selected,arm,view,member_id)
    result={"summaries":summaries,"seed_summaries":seed_summaries,
        "differences":{view:{"spell_f1":summaries["weight_025"][view]["equal_provider_mean"]["spell_f1"]-
                                  summaries["control_D2"][view]["equal_provider_mean"]["spell_f1"]}
                        for view in ("clean","feed_like")},
        "member_score_sha256":sha256(RESULTS/"dense_label_weight_v1_member_scores_all.json"),
        "ensemble_score_sha256":sha256(RESULTS/"dense_label_weight_v1_ensemble_scores_all.json")}
    write_once(RESULTS/"dense_label_weight_v1_readout.json",result)
    return result


def render_report(readout: dict) -> str:
    gate=json.loads((RESULTS/"dense_label_weight_v1_control_replay.json").read_text())
    fit_receipts=[json.loads(p.read_text()) for p in sorted(FITS.glob("*.json"))]
    mask=next((r["target_mask_summary"] for r in fit_receipts if r["arm"]=="weight_025"),None)
    lines=["# Dense silver label weighting: targeted follow-up", "",
        "## Question and design", "",
        "This study tested whether reducing dense SoccerNet silver-label weight near uncertain touch and inferred transitions improves full-provider transfer. The finite comparison was the reconstructed D2 control and a single 0.25-weight arm, each fit for 2,048 updates on three leave-one-provider-out folds and seeds 2026/2027. Each fit mixed 16 original windows and 16 SoccerNet windows per batch. Only TRAIN SoccerNet labels and provenance produced weights; no GAR VALID/TEST labels or organizer labels were read.", "",
        "The ambiguity mask marked native frames within ±round(0.5 × native FPS) of a same-period known-label transition when either side had touch, propagated, or held-uncontrolled provenance. Unknown labels remained masked. All frames outside these bands retained weight 1.0. Candidate H/A and D losses use a per-window weighted mean over valid labels. Transition edge weights use the same mask inside the existing stable/change group-balanced loss; that normalization can cancel a uniform scale within a group, so the transition term reweights edge locations rather than reducing its group-level mass.", "",
        "## Control replay", "",
        f"The reconstructed control gate {'passed' if gate['passed'] else 'failed'}. Expected clean/feed-like equal-provider spell F1 was {EXPECTED_CONTROL['clean']:.6f}/{EXPECTED_CONTROL['feed_like']:.6f}; observed was {gate['values']['clean']['observed']:.6f}/{gate['values']['feed_like']['observed']:.6f}. The declared tolerance was {CONTROL_TOLERANCE:.6f}. Candidate fits were permitted only after this receipt passed.", "",
        "## External development results", "",
        "Official metrics are paired across the same five previously exposed selection matches, with seeds averaged within each provider/match/view and then equal weight across the three providers. Candidate and control are reconstructions from exact fold teachers and source records; original trained D2 fold checkpoints were unavailable.", "",
        "| Arm | View | Spell F1 | Macro F1 | D-F1 | Short-spell recall | Δ spell vs control |", "|---|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for view in ("clean","feed_like"):
            aggregate=readout["summaries"][arm][view]["equal_provider_mean"]
            delta=readout["differences"][view]["spell_f1"] if arm=="weight_025" else 0.0
            lines.append(f"| {arm} | {view} | {aggregate['spell_f1']:.6f} | {aggregate['macro_f1']:.6f} | {aggregate['d_f1']:.6f} | {aggregate['short_spell_recall']:.6f} | {delta:+.6f} |")
    lines += ["", "### Per-provider spell F1", "", "| Arm | View | Metrica | SkillCorner | Sportec |", "|---|---|---:|---:|---:|"]
    for arm in ARMS:
        for view in ("clean","feed_like"):
            provider=readout["summaries"][arm][view]["per_provider"]
            lines.append(f"| {arm} | {view} | {provider['metrica']['spell_f1']:.6f} | {provider['skillcorner']['spell_f1']:.6f} | {provider['sportec']['spell_f1']:.6f} |")
    lines += ["", "### Weight coverage", ""]
    if mask:
        lines += [f"Across {mask['records']} dense records and {mask['native_frames']:,} native TRAIN frames, {mask['qualifying_boundary_count']:,} transitions qualified. The mask covered {mask['ambiguous_native_frames']:,} native frames ({mask['ambiguous_native_fraction']:.2%}); {mask['weighted_sampled_frames']:,} known sampled frames ({mask['weighted_sampled_fraction']:.2%}) received reduced weight, including {mask['weighted_ha_frames']:,} H/A and {mask['weighted_d_frames']:,} D targets.", ""]
    else:
        lines += ["No candidate fit receipt was available.", ""]
    lines += ["## Interpretation and limits", "",
        "This is a single predeclared weighting value on previously used development matches, with no independent broadcast gold set. It supports a decision about this specific loss reweighting under the declared full-provider protocol; it does not establish whether SoccerNet silver labels are reliable in general. The score result is reported without model promotion.", "",
        f"Declaration SHA-256: `{sha256(RESULTS/'dense_label_weight_v1_declaration.json')}`. Control replay receipt: `experiments/results/dense_label_weight_v1_control_replay.json`. Paired score readout: `experiments/results/dense_label_weight_v1_readout.json`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="cmd",required=True)
    sub.add_parser("declare")
    fitp=sub.add_parser("fit");fitp.add_argument("arm",choices=ARMS);fitp.add_argument("fold",choices=FOLDS);fitp.add_argument("seed",type=int,choices=SEEDS)
    scorep=sub.add_parser("score");scorep.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    enp=sub.add_parser("ensemble");enp.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    sub.add_parser("replay");sub.add_parser("summarize")
    sub.add_parser("report")
    args=parser.parse_args()
    if args.cmd=="declare":
        value=declaration();write_once(RESULTS/"dense_label_weight_v1_declaration.json",value)
        print("DECLARED",sha256(RESULTS/"dense_label_weight_v1_declaration.json"),flush=True)
    elif args.cmd=="fit":
        value=run_fit(args.arm,args.fold,args.seed);print(json.dumps(value,sort_keys=True),flush=True)
    elif args.cmd=="score":
        value=score_models(tuple(args.arms));print("SCORED",len(value["rows"]),flush=True)
    elif args.cmd=="ensemble":
        value=ensemble_scores(tuple(args.arms));print("ENSEMBLED",len(value["rows"]),flush=True)
    elif args.cmd=="replay":
        value=control_replay();print(json.dumps(value,sort_keys=True),flush=True)
    elif args.cmd=="summarize":
        value=summarize_scores(json.loads((RESULTS/"dense_label_weight_v1_member_scores_all.json").read_text()),
                               json.loads((RESULTS/"dense_label_weight_v1_ensemble_scores_all.json").read_text()))
        print(json.dumps(value,sort_keys=True),flush=True)
    elif args.cmd=="report":
        value=json.loads((RESULTS/"dense_label_weight_v1_readout.json").read_text())
        report=render_report(value)
        path=RESULTS/"dense_label_weight_v1_report.md"
        if path.exists() and path.read_text()!=report: raise RuntimeError("Refusing to overwrite completed report")
        path.write_text(report)
        print("REPORT",path,flush=True)


if __name__=="__main__": main()
