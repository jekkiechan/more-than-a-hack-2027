# Diverse postprocessing search

Status: declared before scoring, 2026-11-01. Branch `research/decoder-search-v2`,
base fork main `f7364fb`. This is a user-requested broad development search,
following the negative fixed [posterior-MBR trial](SPELL_MBR_V1.md) and
[confidence-boundary trial](DENSE_TARGETED_FOLLOWUPS.md).

## Question and fixed budget

Can a structurally different postprocessor improve official H/A possession
spell F1 using the fixed adopted dense ensemble's native probabilities?
The search isolates postprocessing: no model fits, feature changes, raw-data
corruptions, label repairs, new seeds or organizer scoring. The user explicitly
requested alternatives beyond cost-3 Viterbi and experimentation; this authorizes
a new bounded search on the exposed development cohort, despite earlier advice
to stop local sweeps. It does not create independent confirmation evidence.

The executable configuration list fixes **134 arms in ten families**:

| Family | Arms | Mechanism |
| --- | ---: | --- |
| Transition matrix | 36 | Separate H/A turnover and team/dead costs, including zero H/A cost |
| Factorized | 21 | Decode conditional H/A and alive/dead separately; team memory crosses dead intervals |
| Gaussian | 10 | Centered probability filtering, then argmax or cost 3 |
| Median | 10 | Robust per-class probability filtering, then argmax or cost 3 |
| Minimum duration | 12 | Exact semi-Markov lower bounds on H/A duration; first/last runs censored |
| Hysteresis | 9 | Require persistent confident challengers; backdate accepted boundaries |
| Clipped evidence | 9 | Cap conditional H/A log odds to reduce domination by extreme emissions |
| Adaptive edge | 9 | Lower transition costs where local probability trends change |
| Island pruning | 12 | Remove short uncertain runs when both adjacent states agree |
| Dead calibration | 6 | Symmetric team/dead log-prior adjustment |

All constants are in `decoder_search.configurations()`. Maximum budget:
2,010 ensemble cells and up to 90 component-seed cells for three finalists.
No extra arms follow the results. Stop on control, population or contract failure.

## Evidence, selection and limits

Restore the exact 45 reconstructed `control_D2` arrays and six source fold
checkpoints from private archive commit
`64b6eaafef0f25bd3229aae2021efc9e13a8a1c9`, manifest
`dense_targeted_followups_cache_manifest_v1.json`. These are the same controls
used in the previous dense follow-ups and MBR study; they reconstruct the D2
recipe and reproduce its scores, without claiming original fold checkpoint byte
identity. Source receipts bind each NPZ to the provider-excluded heads, teachers,
normalization and evaluation inputs. Preserve those historical receipts.

Use Metrica 3, Sportec J03WOY and SkillCorner 1996435/1996436/2006229, clean and
both frozen `feed_like` views. All five matches have prior researcher exposure;
some labels are derived silver supervision and stress is synthetic. No fresh
holdout or real-broadcast confirmation exists. New decoders receive only H/A/D
probabilities, original period/time break identifiers and fps; they cannot use
truth, truth validity masks, match IDs or provider IDs. Native unknown targets
remain excluded by the official evaluator. Preserve all reference spell counts.

Before any candidate scoring, tests cover exact baseline recurrence parity,
exhaustive short-sequence duration optimality, team swaps, unavailable classes,
period isolation, input validation and immutability. Freeze source/configuration,
provenance, model and input hashes before running. Replay all 45 control paths
and official scores, requiring clean/feed-like ensemble spell
0.6472145590743718 / 0.5058107894247519 within 1e-10.

Rank by clean equal-provider spell F1, macro for exact ties. Average stress
replicates within match, matches within provider, then providers equally.
Report the unconstrained maximum and the highest candidate passing feed-like
spell >= control-.01 and every clean provider D-F1/macro >= control-.02.
The policy's +.01 clean screen remains explicit. The 70/30 weighted score is
descriptive only. Evaluate the top three guard-passing arms on both component
seeds; these are consistency checks on the same cohort, not confirmation.

Additionally choose an arm using only two providers' scores/guards and evaluate
the omitted provider. Repeat for each provider using the frozen rule. This
checks whether one global setting transfers; it is still exposed development
evidence. The single highest full-cohort score is a search maximum subject to
selection optimism, not a universal optimum or automatic adoption decision.

The selection scorer calls official spell/confusion functions directly, skipping
unused transition statistics. It must exactly match the complete official
wrapper on all controls and finalist ensemble cells. Save full official finalist
metrics including D precision/recall, short-spell recall and spell counts.

## Reproduction

```sh
.venv/bin/python -m experiments.decoder_search_run restore --archive /workspace/decoder-archive
.venv/bin/python -m unittest experiments.tests.test_decoder_search -v
.venv/bin/python -m experiments.decoder_search_run freeze
.venv/bin/python -m experiments.decoder_search_run run
```

Optional study accelerator: Numba 0.61.2 (same Python recurrence if unavailable).
All outputs use exclusive creation. Per-arm scalar receipts and finalist frame
arrays remain in ignored `data/experiments/decoder_search_v2`; the public
aggregate is `results/decoder_search_v2_aggregate.json`. No public scoring gate
is touched. The adopted package is unchanged pending an explicit decision.

## Completed comparison and decision

Completed 2026-11-01: 2,010 ensemble evaluations, 90 component-seed evaluations, and 45 finalist ensemble replays through the complete official wrapper. All 45 baseline paths match exactly; selection and full-wrapper scores match to 1e-12. All 134 per-arm aggregates and the frozen selection rule independently recompute from saved scalar receipts.

**Best observed development decoder: H/A switch cost 3, team/dead cost 12** (`a019`). It improves clean spell F1 by +0.004127 and feed-like by +0.002371. This beats the fixed cost-3 baseline on this cohort, but misses the +0.01 clean screen. The provider-selection experiment is negative, so keep this as an opt-in experimental candidate and retain the adopted baseline.

| Decoder | Clean spell | Feed-like spell | Clean macro | Feed-like macro |
| --- | ---: | ---: | ---: | ---: |
| Cost 3 for all switches | 0.647215 | 0.505811 | 0.857677 | 0.815535 |
| H/A 3, dead 12 (winner) | 0.651341 | 0.508182 | 0.856791 | 0.814835 |
| H/A 3, dead 6 | 0.650941 | 0.507474 | 0.858044 | 0.815756 |
| H/A 2, dead 12 | 0.650837 | 0.503990 | 0.856879 | 0.814629 |

### Why the winner helps, and its costs

It makes entering or leaving dead ball four times as costly while retaining the original H/A turnover penalty. This reduces predicted possessions by an equal-provider mean 5.22 per clean match and 7.22 under stress. These are aggregate observations consistent with fewer interruptions, not a label-based boundary correction. Clean macro falls 0.000886, D-F1 falls 0.002132, D precision rises 0.002982, and D recall falls 0.004437. The corresponding feed-like changes are −0.000700 macro, −0.001668 D-F1, +0.003974 D precision and −0.004937 D recall.

| Provider | Clean spell delta | Feed-like spell delta | Clean D-F1 delta | Clean macro delta |
| --- | ---: | ---: | ---: | ---: |
| Metrica | +0.009526 | +0.004930 | +0.001699 | +0.001190 |
| Sportec | +0.001692 | -0.002552 | -0.000134 | -0.000515 |
| SkillCorner | +0.001163 | +0.004735 | -0.007962 | -0.003334 |

All three providers improve clean spell F1. Sportec loses 0.002552 under stress. The largest clean D-F1 loss is SkillCorner −0.007962, inside the .02 guard; its D recall drops 0.015932. The aggregate stress and all clean provider macro/D guards pass.

| Realization | Control clean spell | Winner clean spell | Control stress spell | Winner stress spell |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 0.635521 | 0.637287 | 0.501847 | 0.500836 |
| 2027 | 0.637245 | 0.639642 | 0.495759 | 0.499293 |
| ensemble | 0.647215 | 0.651341 | 0.505811 | 0.508182 |

Both component seeds improve clean spell, but seed 2026 loses feed-like spell F1 (0.501847 → 0.500836). Seed 2027 improves under stress (0.495759 → 0.499293). Component robustness is mixed; the ensemble benefit is not universal.

Ensemble short-spell recall (<3 s, IoU .5): clean 0.402475 → 0.400066; feed_like 0.217097 → 0.211673. These are complete-spell recall metrics, not frame accuracy.

### Other families and provider selection

| Family | Best clean spell | Its feed-like spell | Guards pass |
| --- | ---: | ---: | --- |
| matrix | 0.651341 | 0.508182 | True |
| dead_bias | 0.648073 | 0.508382 | True |
| median | 0.647362 | 0.505998 | True |
| duration | 0.647215 | 0.505811 | True |
| prune | 0.647215 | 0.505811 | True |
| adaptive | 0.646537 | 0.508262 | True |
| clip | 0.646298 | 0.505983 | True |
| gaussian | 0.646090 | 0.506571 | True |
| hysteresis | 0.645653 | 0.498697 | True |
| factorized | 0.635374 | 0.491912 | False |

The more complex alternatives did not beat the best matrix decoder. Several duration/pruning configurations reproduce the original path; matching baseline does not establish a new gain. The factorized family winner fails the aggregate stress guard. Each row is that family’s clean-selected maximum, not its maximum stress score.

**Provider selection fails to transfer:** selecting on two providers and applying to the omitted provider scores **0.639413 clean / 0.495678 feed-like**, versus the fixed baseline 0.647215 / 0.505811. The deltas are −0.007801 / −0.010132. The omitted Metrica selection is hysteresis `a091`, omitted SkillCorner is H/A 1/dead 12 `a011`, and omitted Sportec is H/A 4/dead 12 `a023`. The selection rule uses no scores or guards from the omitted provider. This negative result must accompany the global search maximum: the search has found a development improvement, not a reliably transferable setting.

### Verification and delivery

All **501 experiment tests**, 26 integrity/gate tests, the synthetic file-backed fitting contract and bundled 199,319-row CSV validation pass. The first suite attempt had three missing-`pyarrow` errors; installing the pinned 20.0.0 dependency resolved them and the complete suite then passed. No code defect or arm change followed candidate scores. Numba 0.61.2 used NumPy 2.2.6; exact baseline replay and official-score parity passed in that runtime.

[Frozen declaration](results/decoder_search_v2_declaration.json), [51-file verified restore](results/decoder_search_v2_restore.json), [control replay](results/decoder_search_v2_control_replay.json), [all-arm aggregate](results/decoder_search_v2_aggregate.json), and [scalar selection replay](results/decoder_search_v2_readout.json). The baseline manifest, model policy, historical runners, source receipts and all organizer gates remain unchanged.

After evaluation, `experiments.decoder_search_predict` provides an opt-in, reference-free CSV generator using the exact four hash-verified adopted deployment weights. It selects the frozen best guard-passing configuration from the aggregate and preserves the baseline manifest. It does not score the organizer match:

```sh
.venv/bin/python -m experiments.decoder_search_predict --models-dir ../mtah-data/models/joint_gru_sn_dense_ensemble_v1 --output data/experiments/decoder_search_v2_new.csv
```

The 134-way full-cohort maximum is optimistically selected on repeatedly exposed data. The +.01 screen fails and provider-selected transfer is negative. Stronger evidence requires independent dense matches; more tuning on these five cannot turn them into fresh confirmation.
