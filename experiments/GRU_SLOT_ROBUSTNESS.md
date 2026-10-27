# Joint GRU player slot robustness study

Status: frozen and running, 2026-10-27. The adopted package remains
`gru_hgb_ballgap25_v1`. This study tests whether restoring the legacy player
slot dropout mechanism repairs the latest player-only D joint GRU's stress
failure while preserving its clean spell gain. It follows the failure diagnosed
in [DEADBALL_TRANSFER](DEADBALL_TRANSFER.md), rather than repeating its
information-restriction comparison.

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

The readout will also show paired candidate-minus-control effects and whether
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
