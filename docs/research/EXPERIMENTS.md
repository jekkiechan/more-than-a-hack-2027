# Experiment index

Reviewed 2026-10-27. Find evidence here; adoption and next work live in
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
| `gru-slot-robustness-v2` — running | Restore raw player-slot dropout in latest player-only D joint GRU; two seeds, three full-pipeline provider holdouts, 12 fits and 60 new paired cells. Aborted v1 preserves its pre-fit mask correction receipt. | [GRU study](../../experiments/GRU_SLOT_ROBUSTNESS.md) |
| `temporal-lgbm-v1` — running | Matched 67 versus 625 features plus two regularization settings; 12 fits, 240 decoded D replacement/blend cells against the same GRU controls. | [Feature implementation](../../experiments/temporal_lgbm.py), [frozen runner](../../experiments/temporal_lgbm_experiment.py) |
| `capacity-lgbm-v1` — running | User-requested 63 leaves/depth 8 and 127 leaves/depth 10, learning rate .01 with 1,280 trees; six fits and 120 new cells with the same 625 features, training rows and controls. Declared before any original-study scores were opened. | [Capacity extension](../../experiments/capacity_lgbm_experiment.py), [LightGBM study](../../experiments/TEMPORAL_LGBM.md) |

## Supporting references

[Data](../../experiments/DATASETS.md), [licenses](../../experiments/LICENSES.md),
[literature/metric compatibility](../../experiments/REFERENCE_METHODS.md),
[early error analysis](../../experiments/ERROR_ANALYSIS.md),
[historical task setup](../WORKPLAN.md).

Add a stable study ID, proposed/running/completed state, protocol, outcome and
evidence link for a new study. The report should link its exact JSON/receipt
paths. Replace the proposal outcome when complete; do not duplicate scorecards.
