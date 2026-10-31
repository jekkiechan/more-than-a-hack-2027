# Current adopted baseline: dense-SoccerNet H/A with fine-tuned D

The current package is `joint_gru_sn_dense_ensemble_v1`, with architecture kind
`sn_dense_ensemble`, adopted by explicit user decision on 2026-10-31. Its default
prediction is `data/experiments/current_baseline/joint_gru_sn_dense_ensemble.csv`.
Each of the two pooled members (seeds 2026/2027) is a width-96 conditional H/A GRU
plus a fine-tuned hidden-24 D path, trained from the matching combined-objective
teacher with the [sn_dense_v1](SOCCERNET_DENSE.md) D2 recipe: dense touch-stream
H/A/D targets with complete-spell and transition losses on 45 SoccerNet-GAR TRAIN
games, SoccerNet batch fraction .5, 2,048 updates. Normalizer, feature schema and
the cost-3 decoder are unchanged. Members average native probabilities before
decoding. The package holds four weights: two dense members and their teachers.

Full-provider development (five previously exposed matches): two-seed ensemble
.6472/.5058 clean/feed-like versus .641790/.502586 for the previous package.
Seed 2026 scores .6355 versus .6288 for the rebuilt previous recipe; seed 2027
scores .6372. All provider D/macro and feed-like guards pass; the +.01 clean
screen is not met. The one-time public diagnostic, known before authorization and
never used for selection, scores .6303 spell / .8514 macro versus .5697 / .8613
on 17 references. Promotion performed no fitting or rescoring: the baseline CLI
reproduced the sealed public-check CSV byte for byte. Receipt:
[baseline_sn_dense_promotion.json](results/baseline_sn_dense_promotion.json);
evidence: [sn_dense_v1_progress.md](results/sn_dense_v1_progress.md).

The sealed progress report's “published adopted ensemble” comparator refers to
`joint_gru_soccernet_capacity_ensemble_v1`, the baseline **at study time**.
Its .641790/.502586 scores are the previous package, not a second estimate of
the current dense ensemble. Preserve that historical report unchanged.

## Private cache restore

Private [mtah-data PR #7](https://github.com/jekkiechan/mtah-data/pull/7) is open
as of 2026-10-31 (head `33bcf56`). Use its delivery branch `ccr-9579d51e-9073tq`, or a newer private
checkout after that delivery merges. A private `main` checkout predating the
merge still defaults to the previous capacity package. Use the versioned
manifest explicitly:

```sh
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest_joint_gru_sn_dense_ensemble_v1.json --data-dir data
python -m experiments.current_baseline predict --output data/experiments/current_baseline/sn_dense_restored.csv
python evaluation/validate_submission.py data/experiments/current_baseline/sn_dense_restored.csv
```

Verify all four checkpoint hashes against the
[active manifest](baseline_manifest.json). This public checkout contains the
matching inference code; no source snapshot overlay is needed. Existing outputs
are never overwritten: choose a new `--output` path for a replay. Restore is
separate from fitting or scoring; closed public gates stay closed. See the
[restore reference](../.agents/skills/possession-research/references/restore.md) for missing artifacts
and [data access](SOCCERNET_DATA_ACCESS.md) for raw SoccerNet inputs.

## Historical packages

The [active decision policy](model_decision_policy.json) owns current ranking
and guards. Each archived package below retains its own manifest, policy and
study decision. Old reports' generic private-cache defaults describe their
original session; restore an older model with its versioned manifest.

| Archived version | Evidence / promotion | Manifest |
| --- | --- | --- |
| `joint_gru_soccernet_capacity_ensemble_v1` | [Capacity study](SOCCERNET_CAPACITY.md), [promotion](results/baseline_soccernet_capacity_promotion.json), [verified remote delivery](results/baseline_soccernet_capacity_remote_artifacts.json) | [Package](baselines/joint_gru_soccernet_capacity_ensemble_v1.json) |
| `joint_gru_combined_ensemble_v1` | [Objective ensemble](OBJECTIVE_ENSEMBLE.md), [deployment/restore receipt](results/baseline_combined_objective_artifacts.json), [closed public diagnostic](results/combined_objective_public_check.json) | [Package](baselines/joint_gru_combined_ensemble_v1.json) |
| `joint_gru_seed_ensemble_v1` | [Seed/context study](SEED_GAP_CONTEXT.md), [promotion](results/baseline_seed_ensemble_promotion.json) | [Package](baselines/joint_gru_seed_ensemble_v1.json) |
| `joint_gru_slotdrop_v1` | [Stress components and promotion rationale](STRESS_COMPONENT_ABLATION.md), [promotion](results/baseline_joint_gru_promotion.json) | [Package](baselines/joint_gru_slotdrop_v1.json) |
| `gru_hgb_ballgap25_v1` | [Light corruption](LIGHT_CORRUPTION.md), [promotion](results/baseline_ballgap25_promotion.json) | [Package](baselines/gru_hgb_ballgap25_v1.json) |
| `gru_hgb_v1` | [Temporal ensemble](TEMPORAL_ENSEMBLE.md), [historical selection](MODEL_SELECTION.md) | [Package](baselines/gru_hgb_v1.json) |

For `<version>`, use private `baseline_cache_manifest_<version>.json` and its
matching public `experiments/baselines/<version>.json`. Archived policies are
`experiments/baselines/model_decision_policy_<version>.json`. For example:

```sh
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest_joint_gru_soccernet_capacity_ensemble_v1.json --data-dir data
python -m experiments.current_baseline predict --manifest experiments/baselines/joint_gru_soccernet_capacity_ensemble_v1.json --output data/experiments/current_baseline/capacity_replay.csv
```

If reproducing a frozen study rather than package inference, use its original
source snapshot and sealed inputs. Never rewrite its source hashes to accept
current code. The older baseline descriptions and initial setup are available
in Git history; the evidence above remains the reproducible record.
