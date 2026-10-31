"""Write the frozen sn_dense_v1 Phase-C declaration (refuses to overwrite)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import soccernet_capacity as sc  # noqa: E402

DATA = sc.DATA
OUT = DATA / "experiments/sn_dense_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    code = ["experiments/sn_dense_train.py", "experiments/sn_dense_run.py",
            "experiments/soccernet_dense_labels.py", "experiments/soccernet_capacity.py",
            "experiments/soccernet_capacity_model.py", "experiments/supervised.py",
            "experiments/SOCCERNET_DENSE.md", "experiments/SN_DENSE_HARNESS.md"]
    data = {}
    for fold in sc.FOLDS:
        for seed in sc.SEEDS:
            p = sc.initial_path(fold, seed)
            data[str(p.relative_to(DATA))] = sha(p)
    pooled = Path("/home/user/mtah-data/models/joint_gru_soccernet_capacity_ensemble_v1")
    for name in ("capacity_2026", "capacity_2027", "teacher_2026", "teacher_2027"):
        data["pooled/" + name] = sha(pooled / f"{name}.joblib")
    for split in ("train", "valid"):
        data[f"dense_records_{split}_manifest"] = sha(OUT / "records" / split / "manifest.json")
    return dict(
        study="sn_dense_v1", phase="C", schema=1,
        statement="Frozen before any Phase-C fit. Changes go in a separate amendment file.",
        common=dict(width=96, lr=3e-4, batch_size=32, input_dropout=0.1, seed=2026,
                    initial_model="matching fold teacher (same seed)", folds=list(sc.FOLDS),
                    d_lr=1e-4, fraction_default=0.5, require_full_native=True,
                    old_data="13 historical TRAIN caches of the two non-held-out providers (old.load_old(fold))",
                    soccernet_train="45 TRAIN games (dense silver records, recipe v1)"),
        arms={
            "R": "ha_target=anchors, train_d=False, fraction .5, 2048 steps; SoccerNet records = dense-record anchor_y/anchor_full_y "
                 "(records with >=1 H/A anchor, order of load_dense_records); rebuild of the adopted recipe",
            "D1": "ha_target=dense, train_d=False, fraction .5, 2048 steps",
            "D2": "ha_target=dense, train_d=True, fraction .5, 2048 steps",
            "D3": "as D2, up to 8192 steps; validate every 512 steps on GAR-valid dense silver spell F1; "
                  "patience 3 evaluations; keep best-on-GAR-valid checkpoint"},
        phases={
            "C1": "arms R, D1, D2, D3 x 3 folds (12 fits), seed 2026",
            "C2": "best C1 recipe at fraction 0, .25, .75, 1.0 (reuse C1 for .5); 12 fits. If the best C1 recipe is D3, C2/C3 "
                  "use fixed steps = median of the three C1 best_steps rounded to the nearest 512 (minimum 512), no early stopping, "
                  "to bound compute (the C1 D3 row stays the early-stopped reference); the .5 row is then the D3 C1 fits only for "
                  "reference and the fixed-step .5 fits are added to keep the sweep like-for-like",
            "C3": "best recipe+fraction on nested SoccerNet subsets of 11 and 22 TRAIN games (sn.nested_games); 45 = existing; 6 fits",
            "C4": "re-fit only the final winner with seed 2027 (teacher seed 2027), 3 fits; score each seed and the "
                  "probability-average ensemble (average native probabilities, then decode)"},
        metrics={
            "dev_cells": "15 development cells (5 matches x clean, feed_like corrupt_1/2); provider-held-out fold model on its "
                         "provider's matches; official metrics; cost-3 Viterbi (frozen supervised.decode); equal-provider mean; "
                         "spell F1, macro F1, D-F1 (P/R), short-spell recall, five IoUs, predicted-spell counts, per-provider rows",
            "gar_valid": "dense silver GAR-valid (9 games, 2191 records, all of valid); native-rate probabilities by interpolation, "
                         "cost-3 Viterbi, official metric code; spell F1, macro, D-F1, short-spell recall; mean over the 3 fold models. "
                         "Viterbi implemented in C bit-identical to supervised.decode (tested). Validator keeps per-record native fps "
                         "(jitter 1e-7 across games; scoring uses the mean). Not subsampled.",
            "references": "adopted pooled width-96 ensemble and pooled teachers scored on GAR-valid only (they saw the dev providers)"},
        selection_rule={
            "guards": "relative to arm R of the same phase: for every provider, dev clean D-F1 and clean macro F1 >= R - .02; "
                      "dev feed_like equal-provider spell F1 >= R - .01 (soccernet_capacity.guard_checks)",
            "choose": "among arms passing the guards (R passes trivially), highest GAR-valid spell F1 (mean of 3 fold models); "
                      "tie-break dev clean equal-provider spell F1",
            "not_used": "dev clean spell deltas vs R and vs saved baseline .626425 are reported but never used to select"},
        scaling="slope 22->45 reported as delta per doubling (log2) on GAR-valid spell F1 and dev clean spell F1",
        references_to_report=dict(saved_baseline_clean_spell=0.626425,
                                  published_adopted_ensemble=dict(clean=0.641790, feed_like=0.502586)),
        stop_rule="NaN/impossible scores or label leakage -> stop, fix, amend, rerun affected fits; "
                  "if total runtime far exceeds ~6 h, finish current phase and report",
        hard_rules="no SoccerNet data/weights committed; no organizer public GT; no GAR test evaluation; no promotion; no push",
        source_sha256={p: sha(ROOT / p) for p in code},
        data_sha256=data,
    )


if __name__ == "__main__":
    value = build()
    text = json.dumps(value, indent=1, sort_keys=True) + "\n"
    target = OUT / "declaration.json"
    if target.exists():
        raise SystemExit("declaration exists; write an amendment instead")
    target.write_text(text)
    (ROOT / "experiments/results/sn_dense_v1_declaration.json").write_text(text)
    print("declared", hashlib.sha256(text.encode()).hexdigest())
