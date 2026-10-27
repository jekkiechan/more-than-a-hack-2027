# Joint GRU player slot robustness study

**Retain `gru_hgb_ballgap25_v1`.** Restoring player slot dropout preserves the
joint GRU's clean spell gain, but does not reliably fix robustness. Mean clean
spell F1 changes from **.6092 to .6132** and mean stress spell F1 from **.2991
to .3141**. The stress effect is **+.0413 for seed 2026 and −.0114 for seed
2027**. Both augmented seeds fail the frozen stress guardrail against the
matched baseline's **.4129**. These are completed, previously exposed external
development results from 2026-10-27; the baseline is unchanged.

The failure remains dead-ball overprediction under corruption. Sportec stress
D precision worsens from .6291 to .5968 while recall remains near .99. Metrica
precision improves only from .4741 to .4901, with recall near .98. SkillCorner
clean D recall improves from .5980 to .6609, contributing to the clean macro
gain. The provider differences and opposite seed effects prevent a reliable
robustness claim from the small overall mean improvement.

The experiment follows the failure diagnosed in
[DEADBALL_TRANSFER](DEADBALL_TRANSFER.md) and isolates restoration of raw player
slot dropout while keeping architecture, optimization, labels and decoding fixed.

## Frozen comparison

The matched control and candidate use the same player-only D factorization,
shared bidirectional GRU, 24 hidden units, 5 Hz samples, 10 second prediction
cores and 5 seconds of context on each side. Both train for eight epochs with
AdamW learning rate .001, weight decay .0001, batch size 64 and gradient clipping
at 1. Their H/A swaps, target masks, provider and match weights, window order,
25% whole-window ball-gap replacement and decoder cost 3 are identical. Medians
and scales are fitted exclusively on each fold's training records.

The candidate's only changed mechanism is raw player slot dropout in the second
training view. The original first clean view remains intact. The second
coarse-ball view and its ball-gap alternative also drop approximately 25% of
each team's player slots within each continuous tracking block, retaining at
least five tracked slots per team. This is the existing `drop_player_slots`
mechanism, using seed 2026 plus the match index within its source. All team and
player motion features are recomputed from the altered raw tracking. Native
missingness is preserved; no target or event field chooses dropout. Half the
training records receive this augmentation. Training targets, sample counts,
H/A swap pairs and optimizer updates stay matched.

Exactly two arms, two fixed training seeds (2026 and 2027), and three full
provider holdouts yield 12 checkpoints. Each held-out provider is excluded from
both H/A and D learning and normalization. The full declared cohort is 13
training matches and five previously inspected development matches across
Sportec, Metrica and SkillCorner. Exact IDs and original input hashes are frozen
in `data/experiments/gru_slot_robustness_v2/predeclared.json`. Each evaluation
match uses the native clean view plus the same two label-free composite-stress
views as the archived joint study. This gives 60 new scored cells. All 12 models
must be sealed before any new scoring. No early stopping, additional arm,
threshold change, seed selection or decoder tuning is part of this study.

The prior player-only D checkpoints were not present in the verified remote
cache. Therefore both control seeds are reproduced in a distinct new directory;
this study does not silently substitute the original unrestricted GRU. The
archived full-pipeline baseline scores are reused without retraining or scoring.

## Metric and decision rule

Official possession spell F1 is primary; macro F1 breaks exact ties. Provider
means weight matches equally, and the overall score weights the three providers
equally. The two synthetic stress views are averaged within each match. The
fixed 70% spell and 30% macro summary is auxiliary and does not select a winner.
Both seeds must independently meet these historical development guardrails:

- Clean spell F1 at least .01 above the matched archived current recipe.
- Every provider's clean macro F1 and D F1 no more than .02 below that baseline.
- Stress spell F1 no more than .01 below that baseline.

The readout also shows paired candidate-minus-control effects and whether
clean spell loss against the joint control exceeds .01. These are descriptive
screens on reused development data, not significance tests or an adoption rule.
Seeds are separate models; averaging their reported scores is not an inference
ensemble. The model's D probability remains player-only, but decoded D labels
also depend on H/A evidence through the fixed sequence decoder.

New rows include official per-IoU counts, D precision/recall, short-spell recall,
false possessions, fragmentation and the supplemental spell diagnostic audit.
Native grading and reference spell counts must match the archived baseline.
Unknown targets never contribute training loss. Tracking discontinuities reset
windows and interpolation. No organizer labels are read or scored, and all
historical organizer gates remain closed.

## Reproduction and artifacts

Restore the original private `live_gap_had_cache_manifest.json` archive and its
18 external arrays from private `mtah-data` branch `experiments/live-gap-had-v1`.
Use the dependencies pinned in `requirements-temporal.txt` and CPU PyTorch.
The runner has no organizer prediction or scoring entrypoint.

```sh
python -m experiments.gru_robustness_experiment freeze
python -m experiments.gru_robustness_experiment prepare
for provider in sportec metrica skillcorner; do
  for arm in control slot_dropout; do
    for seed in 2026 2027; do
      python -m experiments.gru_robustness_experiment fit --source "$provider" --arm "$arm" --seed "$seed"
    done
  done
done
python -m experiments.gru_robustness_experiment seal
python -m experiments.gru_robustness_experiment evaluate
python -m experiments.gru_robustness_experiment report
```

Feature caches, model checkpoints, probabilities and predictions remain under
ignored `data/experiments/gru_slot_robustness_v2/`. Source, configuration, raw
input, feature-cache and checkpoint hashes are checked before dependent steps.
The two focused synthetic contracts verify target-independent raw masking and
identical paired targets, window masks, swap pairing and optimizer budgets.
Fresh independently labeled broadcast confirmation remains necessary.

## Completed development readout

Retain the adopted baseline. The augmented GRU does not clear every frozen development guardrail in both seeds.

All 12 checkpoints were sealed before 60 new evaluations. The 15 archived baseline cells were reused. These are full H/A and D provider holdouts on previously exposed development matches, with two fixed synthetic stress views. Separate-seed means below are not an inference ensemble.

| Model | Clean spell | Clean macro | Clean D F1 | Stress spell | Stress macro | Stress D F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.5338 | 0.8300 | 0.8338 | 0.4129 | 0.7779 | 0.8168 |
| control | 0.6092 | 0.8410 | 0.8302 | 0.2991 | 0.6465 | 0.7073 |
| slot_dropout | 0.6132 | 0.8489 | 0.8440 | 0.3141 | 0.6504 | 0.7044 |

| Model | Seed | Clean spell | Stress spell | Clean spell delta versus baseline | Worst clean provider D or macro delta | Stress spell delta versus baseline | Passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 2026 | 0.6083 | 0.2767 | +0.0745 | -0.0106 | -0.1362 | No |
| control | 2027 | 0.6102 | 0.3215 | +0.0764 | -0.0102 | -0.0915 | No |
| slot_dropout | 2026 | 0.6088 | 0.3180 | +0.0750 | -0.0155 | -0.0949 | No |
| slot_dropout | 2027 | 0.6176 | 0.3101 | +0.0838 | -0.0007 | -0.1028 | No |

### Matched augmentation effects

| Seed | Clean spell delta | Clean macro delta | Stress spell delta | Stress macro delta | Clean spell loss exceeds .01 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2026 | +0.0005 | +0.0027 | +0.0413 | +0.0405 | No |
| 2027 | +0.0074 | +0.0130 | -0.0114 | -0.0326 | No |

### Provider behavior

Values average matches and stress views within provider, then the two fixed seeds.

| Provider | View | Arm | Spell | Macro | D precision | D recall | D F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| sportec | clean | control | 0.6201 | 0.8628 | 0.9092 | 0.8952 | 0.9017 |
| sportec | clean | slot_dropout | 0.6112 | 0.8629 | 0.9212 | 0.8892 | 0.9038 |
| sportec | corrupted | control | 0.2326 | 0.6187 | 0.6291 | 0.9887 | 0.7671 |
| sportec | corrupted | slot_dropout | 0.2103 | 0.5875 | 0.5968 | 0.9915 | 0.7450 |
| metrica | clean | control | 0.7565 | 0.9085 | 0.8760 | 0.8917 | 0.8834 |
| metrica | clean | slot_dropout | 0.7669 | 0.9132 | 0.9095 | 0.8710 | 0.8896 |
| metrica | corrupted | control | 0.2623 | 0.6029 | 0.4741 | 0.9799 | 0.6389 |
| metrica | corrupted | slot_dropout | 0.3142 | 0.6329 | 0.4901 | 0.9801 | 0.6534 |
| skillcorner | clean | control | 0.4511 | 0.7518 | 0.8648 | 0.5980 | 0.7055 |
| skillcorner | clean | slot_dropout | 0.4614 | 0.7705 | 0.8406 | 0.6609 | 0.7386 |
| skillcorner | corrupted | control | 0.4025 | 0.7178 | 0.7818 | 0.6642 | 0.7159 |
| skillcorner | corrupted | slot_dropout | 0.4177 | 0.7308 | 0.6807 | 0.7532 | 0.7146 |

### Spell diagnostics and cost

Short-spell recall here uses reference possessions shorter than three seconds and official matching at IoU .5. False possessions are unmatched predicted H/A spells at IoU .5.

| Arm | View | Short-spell recall | False possessions per minute | Inference and features seconds per match | Auxiliary 70 and 30 summary |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | clean | 0.2917 | 2.9165 | 52.8636 | 0.6227 |
| baseline | corrupted | 0.1521 | 2.9446 | 51.2980 | 0.5224 |
| control | clean | 0.3497 | 2.1080 | 63.5806 | 0.6788 |
| control | corrupted | 0.1082 | 2.5089 | 59.7495 | 0.4033 |
| slot_dropout | clean | 0.3356 | 1.9781 | 63.0463 | 0.6839 |
| slot_dropout | corrupted | 0.1009 | 2.0970 | 59.5042 | 0.4150 |

The auxiliary summary uses 70% spell F1 and 30% macro F1; it does not rank models. Timings come from concurrent CPU work, while baseline timings are archived, so they are descriptive rather than a controlled speed benchmark.

Aggregate evidence: [complete results](results/gru_slot_robustness_development.json). The aggregate includes the frozen declaration, all provider and seed rows, official IoU threshold counts, and the supplemental fragmentation, merging, boundary and duration diagnostics. Native predictions and fitted weights remain in ignored/private storage.

Before fitting, code review found an observation-mask edge case in the first preparation-only declaration: finite coordinates with a native false player mask could be restored during dropout. The corrected routine first applies native masks, then drops slots. A targeted synthetic regression test covers this case. The first declaration and source are preserved with an aborted receipt; no model fitting or scoring occurred in that run. The completed v2 declaration is separate and unchanged.

### Official IoU thresholds and counts

Mean F1 uses the declared equal-provider aggregation. Counts sum the ten clean scored cells or twenty stress scored cells, including repeated matches across seeds and stress views; they are diagnostics and are not the denominator of the mean F1 column. TP means matched spells, FP unmatched predictions and FN unmatched references.

| View | Arm | IoU | Mean F1 | Summed TP | Summed FP | Summed FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clean | control | 0.3 | 0.7084 | 2511 | 1446 | 1415 |
| clean | control | 0.4 | 0.6747 | 2369 | 1588 | 1557 |
| clean | control | 0.5 | 0.6198 | 2161 | 1796 | 1765 |
| clean | control | 0.6 | 0.5573 | 1931 | 2026 | 1995 |
| clean | control | 0.7 | 0.4860 | 1685 | 2272 | 2241 |
| clean | slot_dropout | 0.3 | 0.7097 | 2495 | 1327 | 1431 |
| clean | slot_dropout | 0.4 | 0.6755 | 2356 | 1466 | 1570 |
| clean | slot_dropout | 0.5 | 0.6228 | 2148 | 1674 | 1778 |
| clean | slot_dropout | 0.6 | 0.5662 | 1929 | 1893 | 1997 |
| clean | slot_dropout | 0.7 | 0.4917 | 1688 | 2134 | 2238 |
| corrupted | control | 0.3 | 0.4219 | 3306 | 3240 | 4546 |
| corrupted | control | 0.4 | 0.3599 | 2921 | 3625 | 4931 |
| corrupted | control | 0.5 | 0.2958 | 2507 | 4039 | 5345 |
| corrupted | control | 0.6 | 0.2345 | 2090 | 4456 | 5762 |
| corrupted | control | 0.7 | 0.1834 | 1730 | 4816 | 6122 |
| corrupted | slot_dropout | 0.3 | 0.4323 | 3235 | 2614 | 4617 |
| corrupted | slot_dropout | 0.4 | 0.3684 | 2868 | 2981 | 4984 |
| corrupted | slot_dropout | 0.5 | 0.3089 | 2468 | 3381 | 5384 |
| corrupted | slot_dropout | 0.6 | 0.2545 | 2083 | 3766 | 5769 |
| corrupted | slot_dropout | 0.7 | 0.2063 | 1755 | 4094 | 6097 |

### Control reproduction and next experiment

The six regenerated controls have identical parameter, window and optimizer-update counts to the prior player-only D controls. Their checkpoint byte hashes differ. Across all thirty matched scored cells, the maximum absolute spell F1 difference from the archived results is .000435, macro F1 .000458 and D F1 .000607; reference spell counts are identical, and predicted spell count differs by at most one. The new augmentation effects use the newly reproduced paired controls. See the [saved aggregate reproduction audit](results/gru_control_reproduction.json).

The next robustness hypothesis should isolate transient regional player loss or track fragmentation from whole-block slot removal, with each feature recomputed from the corrupted raw inputs. Compare the corruption components individually before combining them. Evaluate whether the model mistakes loss of visible player motion for dead play using D precision/recall, erased possessions, short-spell recall and per-IoU matches. Keep clean and per-provider guardrails; mean stress spell alone hid a Sportec regression and an unfavorable second seed here. This is a proposed follow-up, not another completed experiment.

No additional GRU fits, threshold adjustments, seed selection or organizer scoring followed this readout. The twelve checkpoints, fifteen shared input views and sixty native predictions/probabilities are verified locally under the ignored v2 run directory. Remote artifact preservation is handled separately; this report does not claim a remote restore until a verified receipt exists.
