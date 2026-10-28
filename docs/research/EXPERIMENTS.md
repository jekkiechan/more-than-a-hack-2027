# Experiment index

Reviewed 2026-10-28. Find evidence here; adoption and next work live in
[STATUS](STATUS.md). Rows are study families, not a cross-protocol leaderboard.
Detailed reports own their original selection rules and aggregate JSON paths.

## Merged studies

| ID | Scope / outcome | Reports |
| --- | --- | --- |
| `primary-tracking` | Fixed Sportec match split; gap-augmented boosting selected before test. Frozen-ball D shortcut limits transfer. | [Original report](../../experiments/README.md), [state machine](../../experiments/STATE_MACHINE.md), [temporal](../../experiments/TEMPORAL.md), [supervised](../../experiments/SUPERVISED.md), [robust](../../experiments/ROBUST.md) |
| `refinement` | Features/capacity/weights/ensembles/decoder; external confirmation improved, competition transfer weak. No .90 competition target or intrinsic ceiling established. | [REFINEMENT](../../experiments/REFINEMENT.md) links individual ablations; [protocol](../../experiments/REFINE_PROTOCOL.md), [headroom](../../experiments/HEADROOM.md), [errors](../../experiments/REFINE_ERRORS.md) |
| `pathcrf-transfer` | Pretrained H/A model; audit match exposure and roster imputation. Native output lacks D. | [PATHCRF](../../experiments/PATHCRF.md), [challenge transfer](../../experiments/CHALLENGE_TRANSFER.md), [data audit](../../experiments/PATHCRF_DATA.md) |
| `multisource` | Metrica/SkillCorner reveal provider-specific D shortcuts. Historical development, not fresh confirmation. | [OPEN_DATA](../../experiments/OPEN_DATA.md), [evaluation provenance](../../experiments/eval_set/README.md), [benchmark audit](../../experiments/BENCHMARK_AUDIT.md) |
| `temporal-d` | D-only provider holdouts with pooled H/A; TCN D gains do not establish overall spell advantage. | [LEARNED_TEMPORAL](../../experiments/LEARNED_TEMPORAL.md), [closed public check](../../experiments/TEMPORAL_PUBLIC_CHECK.md) |
| `temporal-ensemble` | Fixed GRU/HGB complementary; three-way blend numerically slightly higher. Simpler pair user-adopted. | [TEMPORAL_ENSEMBLE](../../experiments/TEMPORAL_ENSEMBLE.md), [selection policy](../../experiments/MODEL_SELECTION.md) |
| `pathcrf-retention` | Matched Metrica/SkillCorner team blending loses spell/macro; Sportec excluded for checkpoint exposure. Archive comparator. | [PATHCRF_ENSEMBLE](../../experiments/PATHCRF_ENSEMBLE.md) |
| `composite-corruption` | 120 evaluations; stress gains with clean loss. No adoption. | [REALISTIC_CORRUPTION](../../experiments/REALISTIC_CORRUPTION.md) |
| `light-corruption` | 90 new + 30 reused comparisons; ball-only teams explicitly adopted as robustness tradeoff. D replacements not adopted. | [LIGHT_CORRUPTION](../../experiments/LIGHT_CORRUPTION.md), [baseline improvements](../../experiments/BASELINE_IMPROVEMENTS.md), [BASELINE](../../experiments/BASELINE.md) |
| `tracking-pilot` | 80 pooled development evaluations; matched gap/suppression features lose overall spell. Fusion quality/player calibration unresolved. | [TOP_THREE_PILOT](../../experiments/TOP_THREE_PILOT.md) |
| `confidence-abstention` | 135 evaluations with exact baseline parity; four X thresholds and matched hold controls all lose stress spell. Primary .60 improves clean spell slightly but loses macro/short-spell recall. No adoption. | [CONFIDENCE_ABSTENTION](../../experiments/CONFIDENCE_ABSTENTION.md), [declaration](../../experiments/results/confidence_abstention_declaration.json), [results](../../experiments/results/confidence_abstention.json) |

| `temporal-competition-check` | User-requested four-candidate public diagnostic, completed; gate closed. Dropout GRU .5818, robustness blend .5818, clean blend .5500, Full127 .5371 spell. | [TEMPORAL_COMPETITION_CHECK](../../experiments/TEMPORAL_COMPETITION_CHECK.md), [declaration](../../experiments/results/temporal_competition_declaration.json) |
| `stress-component-ablation` | GRU composite-stress loss is mostly per-frame sampling loss, absent from the organizer feed; under feed-like stress every GRU seed beats the baseline. Guardrail switched to feed-like (decided after composite results). | [STRESS_COMPONENT_ABLATION](../../experiments/STRESS_COMPONENT_ABLATION.md), [results](../../experiments/results/stress_component_ablation.json) |
| `joint-gru-slotdrop-promotion` | User-authorized promotion of the sealed slot-dropout joint GRU to `joint_gru_slotdrop_v1`; no refit or rescoring. | [BASELINE](../../experiments/BASELINE.md), [receipt](../../experiments/results/baseline_joint_gru_promotion.json) |

## Draft-only evidence

Links pin inspected commits. These modules/reports may not exist on main;
check live PR state before resuming.

| ID / PR | Scope / outcome | Pinned report |
| --- | --- | --- |
| `context-pipeline` / #8 | Context helps its matched external protocol; preprocessing-only does not. No adoption. | [PIPELINE at 7433fef](https://github.com/jekkiechan/more-than-a-hack-2027/blob/7433fef721a810a2f8af383ca29fad7d13fa28d1/experiments/PIPELINE.md), [merged history](../../experiments/RESEARCH_LOG.md) |
| `live-attribution` / #11 | Pooled-D study; small weighted-team gain has uncertainty spanning zero and does not reproduce on public diagnostic. | [LIVE_ATTRIBUTION at b93ce36](https://github.com/jekkiechan/more-than-a-hack-2027/blob/b93ce3600853a72aed1100c9f8919aff9341e12b/experiments/LIVE_ATTRIBUTION.md) |
| `possession-gaps` / #13 | Reports only. Short-spell weighting, boundaries, ball-hidden averaging, duration priors and provider transfer measured; no overall replacement. | [POSSESSION_GAPS at 33c03ca](https://github.com/jekkiechan/more-than-a-hack-2027/blob/33c03caaddcc4a33ebc29637858aa2b3cd2cf2a9/experiments/POSSESSION_GAPS.md), [next studies](https://github.com/jekkiechan/more-than-a-hack-2027/blob/33c03caaddcc4a33ebc29637858aa2b3cd2cf2a9/experiments/NEXT_RESEARCH.md) |
| `live-gap-joint` / #16 | Full H/A+D holdouts; live-gap pipeline fails clean screen. Original joint GRU improves clean spell with severe SkillCorner D collapse. | [LIVE_GAP_HAD_GRU at 8c704aa](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/LIVE_GAP_HAD_GRU.md) |
| `joint-d-repair` / #16 | Matched unrestricted/player-only factorization, two seeds; clean D repaired, stress guardrail fails. | [DEADBALL_TRANSFER at 8c704aa](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/DEADBALL_TRANSFER.md) |
| `joint-public-check` / #16 | Requested diagnostic completed; pooled seed 2026, 13 training matches, five development matches excluded. Gate closed, no adoption. | [JOINT_COMPETITION_CHECK at 8c704aa](https://github.com/jekkiechan/more-than-a-hack-2027/blob/8c704aa0d919a942992dd1c493ae1297e5725e30/experiments/JOINT_COMPETITION_CHECK.md) |

## Temporal robustness follow-up

All model comparisons below use previously exposed external development matches;
no new organizer scoring or baseline promotion is part of this work.

| ID and state | Scope and outcome | Evidence |
| --- | --- | --- |
| `spell-metric-audit` — completed | Unchanged official objective; 30 saved external predictions reproduce. Strict-IoU gains, stress losses and substantial SkillCorner boundary censoring motivate expanded diagnostics. | [Metric audit](../../experiments/SPELL_METRIC_AUDIT.md), [aggregate](../../experiments/results/spell_metric_audit.json) |
| `gru-slot-robustness-v2` — completed | Matched raw player-slot dropout: 12 fits / 60 cells. Mean clean spell .6092 → .6132, stress .2991 → .3141; stress gains are seed-inconsistent and both candidates fail the baseline guardrail. Aborted v1 preserves its pre-fit mask correction receipt. | [GRU study](../../experiments/GRU_SLOT_ROBUSTNESS.md) |
| `temporal-lgbm-v1` — scope-superseded, unscored | Original D-only target was incorrect for the user's intended full H/A/D task. Twelve fitted checkpoints retained for provenance; planned validation cancelled before scoring. | [Feature implementation](../../experiments/temporal_lgbm.py), [frozen runner](../../experiments/temporal_lgbm_experiment.py) |
| `capacity-lgbm-v1` — scope-superseded, unscored | D-only capacity extension stopped on target correction; no completed new checkpoints or validation scores. Frozen declaration and cancellation receipt preserved. | [Capacity extension](../../experiments/capacity_lgbm_experiment.py), [LightGBM study](../../experiments/TEMPORAL_LGBM.md) |
| `joint-lgbm-v1` — completed | Full H/A/D: 12 fits / 300 cells. Geometry alone brings no consistent gain; ordered context improves stress, and 127 leaves/depth 10 improve on 63/depth 8. Full127 standalone .5555/.4345 clean/stress spell; dropout-GRU blends average .6101/.4226 across two separate seeds. All recipes fail a provider guardrail, chiefly SkillCorner D recall. | [Full H/A/D study](../../experiments/JOINT_LGBM.md), [capacity rationale](../../experiments/LGBM_CAPACITY_REASONING.md), [independent verification](../../experiments/results/temporal_followup_verification.json) |
| `seed-gap-context-v1` — running | Actual archived-seed H/A/D probability ensemble: clean/feed-like spell .6265/.4874 versus .6088/.4742 and .6176/.4853; 30/30 individual parity. Gap audit supports H/A dominance on synthetic feed-like long gaps, while native clean live long-gap support is only 108 frames. Six matched 15s-context fits are running; no adoption. | [Study](../../experiments/SEED_GAP_CONTEXT.md), [declaration](../../experiments/results/seed_gap_context_declaration.json), [ensemble/audit](../../experiments/results/seed_gap_context.json), [verified ensemble archive](../../experiments/results/seed_gap_context_ensemble_artifacts.json) |

## Supporting references

[Data](../../experiments/DATASETS.md), [licenses](../../experiments/LICENSES.md),
[literature/metric compatibility](../../experiments/REFERENCE_METHODS.md),
[early error analysis](../../experiments/ERROR_ANALYSIS.md),
[historical task setup](../WORKPLAN.md).

Add a stable study ID, proposed/running/completed state, protocol, outcome and
evidence link for a new study. The report should link its exact JSON/receipt
paths. Replace the proposal outcome when complete; do not duplicate scorecards.
