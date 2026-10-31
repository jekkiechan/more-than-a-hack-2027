# Possession research code

Start with [current status](../docs/research/STATUS.md), then use the
[study index](../docs/research/EXPERIMENTS.md) to find the relevant evidence.
[BASELINE](BASELINE.md) owns the current package and restore/inference commands.
The first Sportec comparison is the historical [primary tracking report](PRIMARY_TRACKING.md).

## Workflow

Use the [possession research skill](../.agents/skills/possession-research/SKILL.md)
for restore, preparation, experiments and readout. It routes directly to the
reference for the task. Repo editing rules live in [root AGENTS](../AGENTS.md).

## Vocabulary

| Term | Meaning |
| --- | --- |
| Providers / dev cohort | Sportec (J03WOY), Metrica (Game 3), SkillCorner (3 IDs). Five dev matches, averaged within provider, then equally across providers. 13 TRAIN matches = 3 Sportec + 2 Metrica + 8 SkillCorner. Defined in [`eval_set/manifest.json`](eval_set/manifest.json). |
| Confirmation matches | J03WPY (original test), J03WR9 (PathCRF comparison), J03WQQ (refinement confirmation). All are now spent. No fresh independent set exists. |
| **Pooled** | Trained on all providers. Used for deployment fits. |
| **D-only provider holdout** | The D head excludes the provider, but H/A is pooled. Older studies (≤ PR #19) used this. **Not** a full-pipeline transfer claim. |
| **Full provider holdout** | Every learned H/A, D and normalization component excludes the provider. The active selection evidence uses full provider holdouts; see the current policy for guards. |
| Views | `clean`/native, `legacy` (old coarse augmentation), `composite` stress (frozen [tracking_corruption](tracking_corruption.py)), `feed_like` (composite minus per-frame sampling loss and regional blanking; **the current stress guardrail**), `ball_gaps`, `no_sampling`, `no_regional`. See [stress_components](stress_components.py). |
| Guardrails | The active rules are in [`model_decision_policy.json`](model_decision_policy.json). Frozen studies retain their original thresholds and references. Explicit promotion exceptions do not rewrite a failed screen. |
| Gate | A one-time exclusive marker for organizer public scoring. Once claimed it stays closed, even after an error. Read saved results; never re-score. |
| Shortcuts | Sportec freezes the ball during D. Metrica blanks the ball during D. SkillCorner's ≥12-visible filter drops most D frames. This is why ball-based D heads do not transfer and why the D head is player-only. |

**Never subtract across protocols.** For example, .5740 (D-only holdout) and
.5338 (full holdout) are the *same* old baseline. Public-window deltas below
~.06 spell are about one possession out of 17.

## Code map

**Shared libraries** (import these, do not fork them):

| Module | Role |
| --- | --- |
| `metrics.py` (`evaluate_tracking`), `eval_set/protocol.py` | Official scorer without joining halves; cohort and fold protocol, `sha256`, paired bootstrap, D-boundary summary |
| `supervised.py`, `refine_features.py`, `refine_domain.py` | Frame features (geometry/motion, physical, player-only D features) and the HGB+Viterbi decoder |
| `robust.py`, `stress.py`, `tracking_corruption.py`, `stress_components.py` | Ball masks, composite corruption, component-gated corruption |
| `learned_temporal.py` (`lt`), `had_gru.py`, `factorized_had.py`, `soccernet_capacity_model.py` | Sequence windows, masks, GRU/TCN and factorized H/A/D components. See `sn_dense_baseline.py` for the adopted dense H/A and fine-tuned D paths. |
| `live_gap_experiment.py` | `feature_view(...)` builds the joint-GRU input record. Also used at inference. |
| `joint_lgbm.py`, `temporal_lgbm.py`, `possession_geometry.py` | Full H/A/D LightGBM features, ordered temporal features, control geometry and anchors |
| `datasets.py`, `metrica_open.py`, `skillcorner_had.py`, `audit_dataset.py` | Source download and normalization into the shared NPZ contract |
| `preparation.py` | Validated manifest folds and hash-verified, reusable unnormalized joint feature caches for new runners; see [preparation reference](../.agents/skills/possession-research/references/preparation.md) |
| `soccernet_segments.py`, `soccernet_dense_labels.py`, `soccernet_dense_qa.py` | SoccerNet clip assembly, dense touch-stream silver labels and label QA |
| `challenge.py` | Organizer tracking prep. Never loads ground truth. |
| `sn_dense_train.py`, `sn_dense_run.py`, `sn_dense_baseline.py` | Dense silver-label training harness, bounded study runner and dense-package inference |
| `current_baseline.py` | `predict` with the active manifest; dispatches on `architecture.kind` |

**Study runners** (one per study, frozen): `*_experiment.py`,
`confidence_abstention.py`, `stress_component_ablation.py`,
`temporal_competition_{fit,check,resume}.py`, `light_corruption.py`,
`refine_*.py`, `promote_*_baseline.py`, `native_short_spell*.py`,
`turnover_decoder*.py`. The native runtime wrapper preserves its reviewed
pre-fit mass-assertion amendment; it is not a general training API. Treat these as records. Do not edit
them to run a new study; write a new runner that imports the libraries.
**Read-only audits:** `*_audit*.py`, `spell_*`, `feed_defect_profile.py`,
`headroom.py`, `error_analysis.py`, `verify_temporal_followup.py`.

## Data and outputs

Run commands from the repository root. Use pinned `requirements*.txt` in this
directory (CPU PyTorch for neural contracts); [CI](../docs/CI.md) lists checks.
Organizer assets and external arrays, checkpoints and frame predictions belong
under ignored `data/` or private storage. Commit aggregate results and receipts
in `results/`, detailed reports alongside the runner, and one entry per study
in the index. Never reset a completed scoring gate.
