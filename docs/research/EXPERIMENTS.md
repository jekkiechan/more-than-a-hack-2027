# Experiment index

Reviewed 2026-10-31 through PR #43. Find evidence here; adoption and next work live in
[STATUS](STATUS.md). Rows are study families, not a cross-protocol leaderboard.
Detailed reports own their original selection rules and aggregate JSON paths.

## Dense SoccerNet supervision

| ID / state | Scope / outcome | Evidence |
| --- | --- | --- |
| `sn_dense_v1` — completed; D2 separately user-adopted | Dense touch-stream silver H/A/D labels and fine-tuned D paths. Two-seed ensemble .6472/.5058 clean/feed-like on exposed full-provider development; provider/stress guards pass, +.01 clean screen not met. Public diagnostic .6303 spell on 17 references, known before authorization; gate closed. GAR-valid silver metrics are distinct from external development and independent gold labels. | [Label recipe and QA](../../experiments/SOCCERNET_DENSE.md), [training harness](../../experiments/SN_DENSE_HARNESS.md), [frozen declaration](../../experiments/results/sn_dense_v1_declaration.json), [sealed progress/readout](../../experiments/results/sn_dense_v1_progress.md), [public check](../../experiments/results/sn_dense_v1_public_check.json), [promotion](../../experiments/results/baseline_sn_dense_promotion.json), [current package](../../experiments/BASELINE.md) |

## SoccerNet matched training follow-up

| ID | Scope / state | Evidence |
| --- | --- | --- |
| `soccernet_followup_execution_v1` — completed, retain baseline | Three Luna mechanisms, six reconstructed references and 24 matched continuations; 225 scored outputs and exact weight replay. Reconstruction .642988/.503681 clean/feed-like versus reference .641790/.502586 and control .642896/.504988. Only +.000092 clean over control, with -.001307 feed-like; all continuations miss +.01 clean, all provider/stress guards pass. No promotion. | [Completed report](../../experiments/SOCCERNET_FOLLOWUP.md), [aggregate](../../experiments/results/soccernet_followup_results.json), [readout](../../experiments/results/soccernet_followup_readout.json), [verification](../../experiments/results/soccernet_followup_verification.json), [artifacts](../../experiments/results/soccernet_followup_artifacts.json), [original implementation](../../experiments/SOCCERNET_TRAINING_IDEAS.md) |

## Merged studies

| ID | Scope / outcome | Reports |
| --- | --- | --- |
| `primary-tracking` | Fixed Sportec match split; gap-augmented boosting selected before test. Frozen-ball D shortcut limits transfer. | [Original report](../../experiments/PRIMARY_TRACKING.md), [state machine](../../experiments/STATE_MACHINE.md), [temporal](../../experiments/TEMPORAL.md), [supervised](../../experiments/SUPERVISED.md), [robust](../../experiments/ROBUST.md) |
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
| `temporal-architecture-v1` — completed; report and receipts published | Fixed-budget comparison: 18 challenger fits / 200 cells, full-provider holdouts and two seeds. Historical slot-dropout GRU mean .6132/.4798 clean/feed-like; TCN .6008/.4704, BiLSTM .5976/.4656, Transformer .5253/.4323. All challengers fail the both-seed screen. The later-adopted .6265/.4874 two-seed ensemble is separate evidence and was not compared here; implementation commits remain local. | [Report](../../experiments/TEMPORAL_ARCHITECTURE.md), [aggregate](../../experiments/results/temporal_architecture_development.json), [declaration](../../experiments/results/temporal_architecture_declaration.json), [independent verification](../../experiments/results/temporal_architecture_independent_verification.json), [restore receipt](../../experiments/results/temporal_architecture_restore.json) |
| `temporal-architecture-ensembles-v1` — completed; independent replay passed | Zero new fits / 900 cells. All 11 pooled architecture blends fail the declared screen. Closest GRU+TCN .626189/.483176 clean/feed-like versus adopted GRU .626531/.487419; GRU+BiLSTM+TCN .624938/.487455 does not improve primary clean spell. Same-seed modes are diagnostics only. Independent audit validated all 2,101 manifest entries; 184 tests passed. No promotion or organizer scoring. | [Report](../../experiments/TEMPORAL_ARCHITECTURE_ENSEMBLES.md), [declaration](../../experiments/results/temporal_architecture_ensembles_declaration.json), [aggregate](../../experiments/results/temporal_architecture_ensembles_development.json), [verification](../../experiments/results/temporal_architecture_ensembles_verification.json), [artifact manifest](../../experiments/results/temporal_architecture_ensembles_artifacts.json), [operational log](../../experiments/results/temporal_architecture_ensembles_operational_events.json) |

## Draft-only evidence

Links pin inspected commits. These modules/reports may not exist on main;
check live PR state before resuming.

| ID / PR | Scope / outcome | Pinned report |
| --- | --- | --- |
| `label-free-followup` / #35 | Completed context gaps, possession-rate diagnostic and target adaptation; all no go against the former combined-objective baseline. Open unmerged PR at review; not a matched comparison with dense SoccerNet. | [Summary at 143d010](https://github.com/jekkiechan/more-than-a-hack-2027/blob/143d01025215133238b45381c9d0f6aba0943b63/experiments/LABEL_FREE_FOLLOWUP.md) |
| `objective-alignment` / #25 | Draft loss-study work; later combined-objective adoption is separately recorded below. Check live PR scope before resuming. | [PR #25](https://github.com/jekkiechan/more-than-a-hack-2027/pull/25), [inspected head 7f70a0a](https://github.com/jekkiechan/more-than-a-hack-2027/commit/7f70a0aa55c946feca37ce12da25e22533dd1ab2) |
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
| `seed-gap-context-v1` — completed | Actual archived-seed H/A/D ensemble clean/feed-like spell .6265/.4874; 30/30 individual parity. Six matched 15s-context fits improve long-gap team accuracy +1.85/+1.36 pp and newly bracketed frames ~+3 pp, but clean spell declines in both seeds; feed-like is flat/+.0046 and seed 2026 fails the SkillCorner D guard. Native clean live long-gap support is only 108 frames. Retain package; no organizer scoring or promotion. | [Study](../../experiments/SEED_GAP_CONTEXT.md), [declaration](../../experiments/results/seed_gap_context_declaration.json), [ensemble/audit](../../experiments/results/seed_gap_context.json), [context results](../../experiments/results/seed_gap_context_long.json), [readout](../../experiments/results/seed_gap_context_readout.json), [verification](../../experiments/results/seed_gap_context_verification.json), [archive](../../experiments/results/seed_gap_context_artifacts.json) |

## Spell-F1 follow-up

| ID and state | Scope and outcome | Evidence |
| --- | --- | --- |
| `spell_opportunity_audit_v2` — completed | Read-only current combined-ensemble audit; all 15 saved cells reproduce. Short-spell recovery is the broadest supported opportunity: under-3s recall .384/.203 clean/feed-like, with strong synthetic-gap H/A errors and limited native long-gap support. | [Audit](../../experiments/SPELL_OPPORTUNITY_AUDIT.md), [results](../../experiments/results/spell_opportunity_audit.json), [corrected receipt](../../experiments/results/spell_opportunity_audit_declaration_v2.json) |
| `window_overlap_v4` — completed, retain baseline | Zero fits, one fixed shifted-core arm, 15 paired cells. Spell .633939/.498330 versus .633205/.498830 clean/feed-like; +.000734/-.000500 fails the +.01 clean screen. Earlier unscored failures and numerical replay drift are preserved. | [Study](../../experiments/WINDOW_OVERLAP.md), [aggregate](../../experiments/results/window_overlap_v4.json), [declaration](../../experiments/results/window_overlap_declaration_v4.json) |
| `contact_memory_v1` — completed, retain baseline | Six H/A residual adapters on frozen combined backbones, 15 ensemble cells and 30 seed diagnostics. Spell .628430/.499523 clean/feed-like versus .633205/.498830; -.004775/+.000694 fails the original clean-gain and stricter high-value screens. Exact archived P(D); numerical replay-guard correction preserved. | [Study](../../experiments/CONTACT_MEMORY.md), [aggregate](../../experiments/results/contact_memory_readout.json), [declaration](../../experiments/results/contact_memory_declaration.json), [runtime correction](../../experiments/results/contact_memory_runtime_correction.json) |
| `independent-data-search` — acquired sample, no possession evaluation | SoccerTrack v2 117093 public sample acquired with exact hashes; panorama/player tracking and event points, no dense possession gold. Ball paths are event-interpolated. True broadcast PFF/SNGAR access remains pending/gated. | [Search and acquisition plan](../../experiments/INDEPENDENT_DATA_SEARCH.md), [metadata receipt](../../experiments/results/soccertrack_v2_acquisition_manifest.json) |

## Short-spell continuation

All studies reuse five previously exposed development matches with full-provider
exclusion. Feed-like stress is synthetic; no organizer scoring or promotion.
The [private archive restore receipt](../../experiments/results/short_spell_recovery_remote_verification.json)
verifies all 420 files at private commit `0ff246b`; per-study local-only
receipts remain unchanged. The [next-work proposal](../../experiments/SHORT_SPELL_NEXT.md)
prioritizes blind interval labels over further local tuning.

| ID and state | Scope and outcome | Evidence |
| --- | --- | --- |
| `short_spell_recoverability_v2` — completed diagnostic | All 15 saved controls reproduce. Complete under-3s misses split into cost-suppressed evidence / insufficient intervals / no local correct top-1 signal: 15/40/45% clean and 10/39/51% feed-like. Argmax improves complete-short recall but reduces overall spell F1 to .561/.419. | [Audit](../../experiments/SHORT_SPELL_RECOVERABILITY.md), [aggregate](../../experiments/results/short_spell_recoverability_v2.json), [declaration](../../experiments/results/short_spell_recoverability_v2_declaration.json) |
| `native_short_spell_v1` — completed, retain baseline | Twelve H/A residual heads, native 10/25 Hz versus matched 5 Hz, 30 ensemble cells. Native spell .629784/.498723 versus matched .630100/.499242 and baseline .633205/.498830 clean/feed-like. Neither arm passes +.01 clean; no resolution advantage established. | [Study](../../experiments/NATIVE_SHORT_SPELL.md), [readout](../../experiments/results/native_short_spell_readout.json), [declaration](../../experiments/results/native_short_spell_declaration.json), [independent replay](../../experiments/results/native_short_spell_independent_replay_verification.json) |
| `turnover_decoder_v1` — completed, retain baseline | Three H/A edge heads; unchanged emissions and D-edge costs. Sum-one weights strongly regularize the head; clean spell unchanged, feed-like +.000166. Fails clean screen. | [Study](../../TURNOVER_DECODER.md), [declaration](../../experiments/results/turnover_decoder_declaration.json), [score receipt](../../experiments/results/turnover_decoder_score_receipt.json), [replay](../../experiments/results/turnover_decoder_independent_replay_verification.json) |
| `turnover_decoder_v2_unit_weights` — completed, retain baseline | Separately declared post-v1 mean-one-weight convention, otherwise identical three heads and fifteen cells. Short recall +.021727/+.008559, extra spells +6.889/+3.500 per match; spell falls to .631627/.495856. Fails clean screen; no sweep follows. | [Study](../../TURNOVER_DECODER_V2.md), [declaration](../../experiments/results/turnover_decoder_v2_declaration.json), [score receipt](../../experiments/results/turnover_decoder_v2_score_receipt.json), [replay](../../experiments/results/turnover_decoder_v2_independent_replay_verification.json) |

## Supporting references

The user adopted the actual original-context two-seed ensemble after this study.
The [promotion receipt](../../experiments/results/baseline_seed_ensemble_promotion.json)
records the separate deployment completion and versioned package. The
[progress Page](https://chatgpt.com/space/page_1df518ea32448191bf885da4952259a9)
tracks dated numbers without mixing evaluation protocols.

[Data](../../experiments/DATASETS.md), [licenses](../../experiments/LICENSES.md),
[literature/metric compatibility](../../experiments/REFERENCE_METHODS.md),
[early error analysis](../../experiments/ERROR_ANALYSIS.md),
[original setup in Git history](https://github.com/jekkiechan/more-than-a-hack-2027/blob/96314f8f4489ae356afea03111fa4b0dbfd40ec5/docs/WORKPLAN.md).

Add a stable study ID, proposed/running/completed state, protocol, outcome and
evidence link for a new study. The report should link its exact JSON/receipt
paths. Replace the proposal outcome when complete; do not duplicate scorecards.

## Objective alignment follow-up

| ID and state | Scope and outcome | Evidence |
| --- | --- | --- |
| `objective-ensemble-v1` — previous adopted package, now archived | Fixed two-seed combined-loss probability ensemble .633205/.498830 clean/feed-like vs .626531/.487419. Its provider guards passed; SkillCorner spell declined about .007. The separate public check tied spell and lowered macro. It now supplies the unchanged D teachers for the user-authorized SoccerNet-capacity package; its study, policy, and adoption history remain unchanged. | [Report](../../experiments/OBJECTIVE_ENSEMBLE.md), [declaration](../../experiments/results/objective_ensemble_declaration.json), [aggregate](../../experiments/results/objective_ensemble.json), [verification](../../experiments/results/objective_ensemble_verification.json), [private archive](../../experiments/results/objective_ensemble_artifacts.json) |

## Dataset expansion

| ID and state | Scope and outcome | Evidence |
| --- | --- | --- |
| `skillcorner_expansion_v1` — completed, retain active dataset/model | Six fixed-budget fits, original 13 versus expanded 19 TRAIN matches, matched seven-match evaluations. Primary clean/feed-like spell .641468/.510206 → .643523/.502104; pooled .696722/.554822 → .686765/.546531. Both clean gates fail; remaining guards pass. Reserved match unscored; nine acquired matches remain experimental. Separate fixed public diagnostic spell .557576 → .548571; no tuning or adoption change. | [Report](../../experiments/SKILLCORNER_EXPANSION.md), [declaration](../../experiments/results/skillcorner_expansion_declaration.json), [decision](../../experiments/results/skillcorner_expansion.json), [252 rows](../../experiments/results/skillcorner_expansion_rows.json), [source audit](../../experiments/results/skillcorner_expansion_data_audit.json), [local artifact receipt](../../experiments/results/skillcorner_expansion_artifacts.json) |

## SkillCorner source contribution

| ID and state | Scope and outcome | Evidence |
| --- | --- | --- |
| `skillcorner_removal_v1` — comparison complete, retain current model | Six fixed-budget fits: remove original eight SkillCorner TRAIN matches. Original-five full-provider clean/feed-like spell .633205/.498830 → .625786/.496092; both seeds decline. Separate provider-represented tactical mean improves +.013053/+.012420, concentrated in Sportec; Metrica +.002720/-.007082. Primary clean removal screen fails; other thirteen guards pass. No organizer scoring or promotion. | [Report](../../experiments/SKILLCORNER_REMOVAL.md), [declaration](../../experiments/results/skillcorner_removal_declaration.json), [readout](../../experiments/results/skillcorner_removal.json), [180 rows](../../experiments/results/skillcorner_removal_rows.json), [fit verification](../../experiments/results/skillcorner_removal_fit_verification.json), [external verification](../../experiments/results/skillcorner_removal_external_verification.json), [local artifacts](../../experiments/results/skillcorner_removal_artifacts.json) |

| `soccernet_extension_v1` — stage 1 complete; no adoption | 18 full-provider H/A-only warm-start fits: baseline .633205/.498830 clean/feed-like; matched old-only .629253/.495753; GAR25 .630438/.494076; selected GAR50 .631957/.495121. GAR50 is +.002703 clean versus old-only but -.001248/-.003709 versus baseline, so its +.01 clean improvement screen fails. Provider/feed-like guards pass. Keep the active baseline. The separately deferred public check was completed later by the capacity study after final external selection; it did not alter stage-1 results or adoption. | [Report](../../experiments/SOCCERNET_EXTENSION.md); aggregate `data/experiments/soccernet_extension_v1/report.json` (180 rows), with declaration, decision and receipts in the local ignored artifact directory |
| `soccernet_capacity_v1` — frozen study complete; separately promoted by user authorization | 18 Stage-A, 9 Stage-B, and 6 matched old-only fits. Width-96 ensemble scores .641790/.502586 clean/feed-like vs baseline .633205/.498830 and is +.008138/.008398 vs equal-budget old-only, but misses the frozen +.01 clean screen. The post-selection public diagnostic is .569697 spell F1 vs .557576 baseline on 7,500 frames/17 references; it was known when the user authorized a separate promotion, but did not inform model selection. Frozen `adopt_model: false` remains unchanged. All independent replay receipts pass. | [Methods and dated adoption note](../../experiments/SOCCERNET_CAPACITY.md), [public result](../../experiments/results/soccernet_capacity_public_check.json), [aggregate verification](../../experiments/results/soccernet_capacity_verification.json), [promotion receipt](../../experiments/results/baseline_soccernet_capacity_promotion.json), [local artifact inventory receipt](../../experiments/results/soccernet_capacity_artifacts.json) |

## Smart Tagging

| ID and state | Scope and outcome | Evidence |
| --- | --- | --- |
| `smart-tagging-validation-v1` — completed; no adoption | Genuine-XML label-free audit plus one explicitly requested fixed public diagnostic. Missing-ball H/A prior, max odds multiplier 2, 1s taper, unchanged model and cost-3 decoder. Public spell .557576 → .537143; macro .854088 → .866346; predicted possessions 16 → 18 vs 17 references. No public tuning; no independent/external efficacy claim. Luna artifact replay passes. | [Report/protocol](../../experiments/SMART_TAGGING_VALIDATION.md), [declaration](../../experiments/results/smart_tagging_declaration.json), [public result](../../experiments/results/smart_tagging_public_check.json), [label-free audit](../../experiments/results/smart_tagging_label_free_audit.json), [verification](../../experiments/results/smart_tagging_verification.json), [artifacts](../../experiments/results/smart_tagging_artifacts.json) |

## Settled directions

Revisit only with a distinct mechanism and matched external evidence. These
outcomes describe their tested recipes/cohorts, not universal impossibility.

| Direction | Evidence / limitation |
| --- | --- |
| Short-gap interpolation or broad corruption as a general fix | [Tracking pilot](../../experiments/TOP_THREE_PILOT.md), [realistic corruption](../../experiments/REALISTIC_CORRUPTION.md); coverage alone did not improve spell F1. |
| Ball-aware D heads | [Provider shortcuts](../../experiments/BENCHMARK_AUDIT.md), [player-only D repair](../../experiments/DEADBALL_TRANSFER.md). |
| PathCRF default, fixed confidence/hold gates, or LightGBM added to the GRU | [PathCRF retention](../../experiments/PATHCRF_ENSEMBLE.md), [abstention](../../experiments/CONFIDENCE_ABSTENTION.md), [joint LightGBM](../../experiments/JOINT_LGBM.md), [feed-like ablation](../../experiments/STRESS_COMPONENT_ABLATION.md). |
| More shifted-core, frozen contact-adapter, native instantaneous-geometry or turnover-cost sweeps on the same cohort | [Window overlap](../../experiments/WINDOW_OVERLAP.md), [contact memory](../../experiments/CONTACT_MEMORY.md), [native short spells](../../experiments/NATIVE_SHORT_SPELL.md), [turnover v2](../../TURNOVER_DECODER_V2.md). Short recall gains did not establish overall spell gain. |
| Smart Tagging cap/taper/taxonomy tuning on the public window | [Fixed diagnostic](../../experiments/SMART_TAGGING_VALIDATION.md) completed; no public-driven follow-up sweep. |
| Further SoccerNet objective/scale sweeps without a new hypothesis | [Matched training follow-up](../../experiments/SOCCERNET_FOLLOWUP.md) found no qualifying objective; [dense progress](../../experiments/results/sn_dense_v1_progress.md) found dense H/A alone, longer training and GAR-only gains did not transfer, and 11/22/45-game scaling was flat. |

Native player-frame blanking, calibrated player defects and ball-fusion
confidence remain historical proposals, not current priorities or completed
causal studies. See [slot robustness](../../experiments/GRU_SLOT_ROBUSTNESS.md)
and [tracking directions](../../experiments/DIRECTIONS.md) before revisiting.
