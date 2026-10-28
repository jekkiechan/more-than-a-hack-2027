# Agent guide: `experiments/`

Dense map of what has been tried here, so you can resume without reading 40
reports. Repo-wide rules in [root AGENTS](../AGENTS.md) still apply. Numbers
below are snapshots as of 2026-10-28 (after PR #20). [STATUS](../docs/research/STATUS.md)
is the coordination summary for adoption and next work. Before model-choice
advice, follow [the root evidence-refresh checklist](../AGENTS.md#before-answering-model-choice-or-current-results-questions):
verify the active manifest/policy and recent PR evidence. STATUS and this guide
can both lag newer work; historical verdicts below are not current adoption rules.
Open a detailed report for the study relevant to the question.

## 1. State in five lines

- **Task:** label every 25 Hz frame of BAR-ATH H/A/D/X from tracking only.
  Ranked by possession **spell F1** (H/A intervals, greedy one-to-one matching,
  mean over IoU .3–.7). **Macro F1** breaks exact ties. The 70/30 weighted
  summary is shown but never ranks.
- **Adopted baseline:** `joint_gru_slotdrop_v1`. One BiGRU (24 hidden) outputs
  full H/A/D. Its D head sees players only. Trained pooled on 13 external TRAIN
  matches with ball-gap and player-slot-dropout augmentation, seed 2026, Viterbi
  switch cost 3. See [BASELINE.md](BASELINE.md).
- **Its evidence (full provider holdout, 5 dev matches):** clean spell .6088
  (previous baseline .5338). Feed-like stress spell .4742 (previous .4328).
  Public 5-min check: .5818 spell / .8689 macro.
- **Known weaknesses:** composite stress .318 (fails if that guardrail were
  used). Single seed. Full-match performance is unknown. All dev matches are
  previously exposed.
- **Where the error is:** mostly team attribution during long ball gaps, and
  turnover timing. On Sportec validation, oracle H/A gives .996 spell and oracle
  dead/live gives nothing ([HEADROOM](HEADROOM.md)). On SkillCorner, D recall
  is also fragile.

## 2. Vocabulary you must get right

| Term | Meaning |
| --- | --- |
| Providers / dev cohort | Sportec (J03WOY), Metrica (Game 3), SkillCorner (3 IDs). Five dev matches, averaged within provider, then equally across providers. 13 TRAIN matches = 3 Sportec + 2 Metrica + 8 SkillCorner. Defined in [`eval_set/manifest.json`](eval_set/manifest.json). |
| Confirmation matches | J03WPY (original test), J03WR9 (PathCRF comparison), J03WQQ (refinement confirmation). All are now spent. No fresh independent set exists. |
| **Pooled** | Trained on all providers. Used for deployment fits. |
| **D-only provider holdout** | The D head excludes the provider, but H/A is pooled. Older studies (≤ PR #19) used this. **Not** a full-pipeline transfer claim. |
| **Full provider holdout** | Every learned H/A, D and normalization component excludes the provider. Current preferred view (policy v3). |
| Views | `clean`/native, `legacy` (old coarse augmentation), `composite` stress (frozen [tracking_corruption](tracking_corruption.py)), `feed_like` (composite minus per-frame sampling loss and regional blanking; **the current stress guardrail**), `ball_gaps`, `no_sampling`, `no_regional`. See [stress_components](stress_components.py). |
| Guardrails | Per seed: feed-like stress spell ≥ baseline − .01. Each provider's clean D-F1 and macro within .02 of baseline. Clean screen +.01 spell. Exact rules are in [`model_decision_policy.json`](model_decision_policy.json) and each declaration. |
| Gate | A one-time exclusive marker for organizer public scoring. Once claimed it stays closed, even after an error. Read saved results; never re-score. |
| Shortcuts | Sportec freezes the ball during D. Metrica blanks the ball during D. SkillCorner's ≥12-visible filter drops most D frames. This is why ball-based D heads do not transfer and why the D head is player-only. |

**Never subtract across protocols.** For example, .5740 (D-only holdout) and
.5338 (full holdout) are the *same* old baseline. Public-window deltas below
~.06 spell are about one possession out of 17.

## 3. Study lineage

Each step reacted to the one before it. ✔ = adopted. ✖ = retained the baseline.

| # | Study (report) | Question → outcome | Verdict |
| --- | --- | --- | --- |
| 1 | Sportec baselines ([README](README.md), [STATE_MACHINE](STATE_MACHINE.md), [TEMPORAL](TEMPORAL.md), [SUPERVISED](SUPERVISED.md), [ROBUST](ROBUST.md)) | State machine, GMM+HMM, HGB+Viterbi, gap-augmented HGB. J03WPY .9298 macro / .6232 spell. D is trivial on Sportec because of the frozen ball ([BENCHMARK_AUDIT](BENCHMARK_AUDIT.md)). | ✔ gap-augmented HGB (historical) |
| 2 | Error/headroom ([ERROR_ANALYSIS](ERROR_ANALYSIS.md), [HEADROOM](HEADROOM.md)) | Possessions under 3 s cause 68% of missed spells. The bottleneck is H/A, not D. | diagnostic |
| 3 | Refinement ([REFINEMENT](REFINEMENT.md) and REFINE_*) | Physical features, duration weighting, LightGBM, scale, ensemble, decoder 3/8. J03WQQ spell .6654 → .7059. Public check only .4300. | ✔ then superseded. **Big external gains did not transfer.** |
| 4 | PathCRF ([PATHCRF](PATHCRF.md), [CHALLENGE_TRANSFER](CHALLENGE_TRANSFER.md), [PATHCRF_ENSEMBLE](PATHCRF_ENSEMBLE.md)) | The pretrained H/A model saw our Sportec val/test. Hybrid public .5067. Matched blend loses spell and macro, and adds 13–15 min per match. | ✖ archived comparator |
| 5 | Multi-source ([OPEN_DATA](OPEN_DATA.md), [eval_set](eval_set/README.md)) | Added Metrica and SkillCorner H/A/D. Exposed provider D shortcuts. Created the dev protocol used since. | infrastructure |
| 6 | Context pipeline (PR #8, [RESEARCH_LOG](RESEARCH_LOG.md)) | Control-context features +.025 spell. Preprocessing only .5251 vs .5259. Public .4190. | ✖ |
| 7 | Temporal D ([LEARNED_TEMPORAL](LEARNED_TEMPORAL.md), [TEMPORAL_PUBLIC_CHECK](TEMPORAL_PUBLIC_CHECK.md)) | Player-only D heads: HGB-ordered, LightGBM, TCN, BiGRU. D gains do not translate into spell gains. GRU public .4889. | partial |
| 8 | D ensembles ([TEMPORAL_ENSEMBLE](TEMPORAL_ENSEMBLE.md), [MODEL_SELECTION](MODEL_SELECTION.md)) | GRU+HGB P(D) 50/50 .5746. A 3-way blend scored .5750; the user chose the simpler pair. | ✔ `gru_hgb_v1` (PR #10) |
| 9 | Corruption ([REALISTIC_CORRUPTION](REALISTIC_CORRUPTION.md), [LIGHT_CORRUPTION](LIGHT_CORRUPTION.md), [BASELINE_IMPROVEMENTS](BASELINE_IMPROVEMENTS.md)) | Composite augmentation trades clean for stress. A 25% ball-gap-only team replacement gives clean −.0006 and stress +.011. Public .5257. | ✔ `gru_hgb_ballgap25_v1` (robustness tradeoff, user-chosen) |
| 10 | Tracking pilot ([TOP_THREE_PILOT](TOP_THREE_PILOT.md), [DIRECTIONS](DIRECTIONS.md)) | State-conditioned views, broadcast ball fusion, gap/de-dup features. No matched spell gain. Fusion recovers 6.8% of missing-ball frames, but more than 10% disagree by over 5 m. | ✖ |
| 11 | Confidence abstention ([CONFIDENCE_ABSTENTION](CONFIDENCE_ABSTENTION.md)) | X/hold at .50–.80 confidence. Every arm fails a guardrail. | ✖ closed |
| 12 | Joint H/A/D GRU ([LIVE_GAP_HAD_GRU](LIVE_GAP_HAD_GRU.md)) | First full-holdout study. Joint GRU clean .6272, but SkillCorner D collapses. Live-gap pipeline fails the clean screen. | ✖ |
| 13 | D repair ([DEADBALL_TRANSFER](DEADBALL_TRANSFER.md)) | Player-only D path fixes SkillCorner D (.18 → .71). Clean .6093, composite stress fails (.28/.32). Pooled public check .6194. | ✖ at the time |
| 14 | Slot dropout ([GRU_SLOT_ROBUSTNESS](GRU_SLOT_ROBUSTNESS.md)) | Raw player-slot dropout: clean .6132, composite stress .3141. Seed-inconsistent. Failure mode: D overprediction under corruption. | ✖ at the time |
| 15 | Full H/A/D LightGBM ([JOINT_LGBM](JOINT_LGBM.md), [LGBM_CAPACITY_REASONING](LGBM_CAPACITY_REASONING.md)) | Ordered context helps stress. 127 leaves/depth 10 beats 63/depth 8. Full127 .5555/.4345. Blends lower SkillCorner D recall. Earlier D-only TEMPORAL_LGBM/CAPACITY_LGBM runs were cancelled as wrong scope and never scored. | ✖ |
| 16 | Metric audit ([SPELL_METRIC_AUDIT](SPELL_METRIC_AUDIT.md)) | 30 saved rows reproduce. 70% of SkillCorner spells touch grading edges. Report all five IoUs, duration recall and fragmentation. | reporting rule |
| 17 | Competition check ([TEMPORAL_COMPETITION_CHECK](TEMPORAL_COMPETITION_CHECK.md)) | Four pooled candidates: dropout GRU .5818, robustness blend .5818, clean blend .5500, Full127 .5371. | gate closed |
| 18 | Stress components ([STRESS_COMPONENT_ABLATION](STRESS_COMPONENT_ABLATION.md), [feed profile](results/feed_defect_profile.json)) | The GRU's composite loss is mostly per-frame sampling loss (53–57%), which the organizer feed lacks. Under feed-like stress the GRU beats the baseline on every provider and seed. LightGBM adds about ±.002. | ✔ guardrail switched to feed-like (decided post hoc, disclosed) → `joint_gru_slotdrop_v1` (PR #20) |

Public-window history (17 possessions, descriptive only): .4190 context ·
.4300 refinement · .4889 GRU · .5067 PathCRF hybrid · .5257 ballgap25 ·
.5371 Full127 · .5818 slotdrop GRU · .6194 pooled player-only-D GRU.

## 4. Settled: do not re-propose without a new mechanism

- Short-gap preprocessing or interpolation (≤0.5 s). Organizer ball gaps have a
  median of 10.7 s and cover about 41% of frames ([DIRECTIONS](DIRECTIONS.md)).
- Composite-stress augmentation as a general fix. Renewal-process gaps are
  random, not game-state-dependent.
- Ball-based D features. They exploit provider shortcuts.
- PathCRF in the default pipeline. Fixed confidence abstention/hold.
- Adding LightGBM to the GRU: no feed-like value, and it lowers D recall.
- Decoder or threshold tuning, and anything else selected on the public window.
- Treating a mean of separate seeds as an ensemble.

## 5. Open threads (priorities live in STATUS)

1. SkillCorner dead-ball under-recall in tree predictors. Separate anchors,
   ordered context and persistence.
2. Native player-frame blanking in the GRU. A six-fit design is proposed in
   [GRU_SLOT_ROBUSTNESS](GRU_SLOT_ROBUSTNESS.md#proposed-native-frame-blanking-experiment).
   Not yet declared.
3. Untested on the baseline: seed averaging, and a game-state-dependent
   (turnover/aerial) gap model.
4. An independent labelled broadcast match. Data-blocked (private mtah-data issue #4).
5. Calibrated player defects and ball-fusion confidence.

Draft PRs #8, #11, #13, #16 and #18 hold evidence that is not on main. They are
pinned in [EXPERIMENTS](../docs/research/EXPERIMENTS.md). Check live PR state.

## 6. Code map

**Shared libraries** (import these, do not fork them):

| Module | Role |
| --- | --- |
| `metrics.py` (`evaluate_tracking`), `eval_set/protocol.py` | Official scorer without joining halves; cohort and fold protocol, `sha256`, paired bootstrap, D-boundary summary |
| `supervised.py`, `refine_features.py`, `refine_domain.py` | Frame features (geometry/motion, physical, player-only D features) and the HGB+Viterbi decoder |
| `robust.py`, `stress.py`, `tracking_corruption.py`, `stress_components.py` | Ball masks, composite corruption, component-gated corruption |
| `learned_temporal.py` (`lt`), `had_gru.py`, `factorized_had.py` | Sequence windows, masks, GRU/TCN; the joint H/A/D GRU with player-only D path (the current baseline's network) |
| `live_gap_experiment.py` | `feature_view(...)` builds the joint-GRU input record. Also used at inference. |
| `joint_lgbm.py`, `temporal_lgbm.py`, `possession_geometry.py` | Full H/A/D LightGBM features, ordered temporal features, control geometry and anchors |
| `datasets.py`, `metrica_open.py`, `skillcorner_had.py`, `audit_dataset.py` | Source download and normalization into the shared NPZ contract |
| `challenge.py` | Organizer tracking prep. Never loads ground truth. |
| `current_baseline.py` | `predict` with the active manifest; dispatches on `architecture.kind` |

**Study runners** (one per study, frozen): `*_experiment.py`,
`confidence_abstention.py`, `stress_component_ablation.py`,
`temporal_competition_{fit,check,resume}.py`, `light_corruption.py`,
`refine_*.py`, `promote_*_baseline.py`. Treat these as records. Do not edit
them to run a new study; write a new runner that imports the libraries.
**Read-only audits:** `*_audit*.py`, `spell_*`, `feed_defect_profile.py`,
`headroom.py`, `error_analysis.py`, `verify_temporal_followup.py`.

## 7. How a study is built here

- The runner exposes `freeze` → `run`/`predict` (per match/view) → `report`
  (or `check` for a gate). `freeze` writes
  `results/<id>_declaration.json` with source, config and input hashes before
  any fit. Never edit it afterwards; a post-hoc idea gets a new ID.
- Aggregates go to `results/<id>*.json`, and the prose report to `<ID>.md`.
  Weights, arrays and frame predictions go to ignored `data/experiments/<id>/`.
  Upload them to private `mtah-data` and record a receipt
  (`results/*_artifacts.json`).
- Every comparison re-scores or reuses the matched baseline and checks parity
  (for example, exactly 15 clean/composite cells reproduced). Report per-provider
  results, both seeds, D precision/recall, all five IoUs, and spell counts.
- Promotion is a separate, user-authorized step (`promote_*_baseline.py`). It
  writes `baseline_manifest.json` and `model_decision_policy.json`, and archives
  the old ones under `baselines/`. It never refits or re-scores.
- Use the skills: [possession-experiment](../.agents/skills/possession-experiment/SKILL.md)
  before fitting, and [possession-readout](../.agents/skills/possession-readout/SKILL.md)
  for comparisons. Copy the [template](../docs/research/templates/EXPERIMENT.md).
  After a study, update the [EXPERIMENTS](../docs/research/EXPERIMENTS.md) row and STATUS.

```sh
python -m unittest discover -s experiments/tests -v          # synthetic contracts, no private data
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest.json --data-dir data
python -m experiments.current_baseline predict              # -> data/experiments/current_baseline/joint_gru_slotdrop.csv
python evaluation/validate_submission.py <candidate.csv>
```

Run from the repo root with pinned `experiments/requirements*.txt` (CPU
PyTorch). Restore and verify hashes before retraining anything. A missing cache
is an access problem, not a model result.
