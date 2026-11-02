# PCBAS reserve: adopted vs runner-up package

Study `pcbas_package_comparison_v1`, completed 2 November 2026. Adopted package
retained; no fitting, threshold or offset search, organizer access or promotion.

## Question

Calibrated dense + phase averaging still leads GAR TEST spell F1 by a margin
whose interval includes zero ([phase-average readout](ORIGINAL_FULL_PHASE_AVERAGE.md)).
On fresh, never-scored PCBAS matches, which package attributes the acting team
better under the player-only regime? Does the adopted package's ~72% Drive
agreement replicate?

## Protocol (frozen before inference)

- Cohort: the 36 TRAIN games left unscored by
  [pcbas_expanded_anchor_v2](PCBAS_EXPANDED_ANCHOR_V2.md), ranked by
  `sha256("pcbas_package_comparison_v1:<game>")`. The first 18 are compared; **18
  stay sealed** for a future confirmation. Selection read no targets.
- Arms: `original_full_phase` (adopted, exact deployed path),
  `calibrated_phase` (calibrated checkpoints, per-member five-phase mean, frozen
  f062 decoder) and secondary `calibrated_phase0` (archived package as deployed).
- Parity gate: on 750 real frames, our adopted arm equals
  `original_full_phase_baseline.predict_labels` and `calibrated_phase0` equals
  `calibrated_dense_baseline.predict_labels` exactly. **Passed.**
- Frozen v1 adapter and anchor rules: ball entirely missing, ROI visibility mask,
  standardized 105×68 m, replay excluded, D/X wrong. Predictions were sealed
  before the single scoring pass.
- Primary: paired equal-game Drive agreement difference, 10,000 whole-game
  bootstrap draws, seed 20261103. Informative only: no ball, so this can't
  change the package by itself.
- PCBAS inputs match the hashes pinned by the earlier studies.

## Results (18 games, 36 halves, 2.7M frames)

| Arm | Drive | Pass | Cross | Shot |
| --- | ---: | ---: | ---: | ---: |
| original_full_phase (adopted) | **71.97%** | **69.86%** | 61.43% | 53.67% |
| calibrated_phase | 71.52% | 69.24% | 61.01% | 54.83% |
| calibrated_phase0 | 71.57% | 69.24% | 60.30% | 53.45% |

Equal-game agreement; 11,901 Drive anchors. Adopted Drive 95% whole-game
interval: **69.40–74.43%**. Pooled cross/shot predicted D: 31.5%/40.6%.

Paired Drive difference, calibrated_phase − adopted: **−0.45 points, 95% CI
(−0.86, −0.02)**. The adopted package is better in 12 games and worse in 4.
calibrated_phase0 − adopted: −0.40 points (−0.89, +0.06).

## Interpretation

- The adopted package's player-only team attribution replicates on 18 more
  matches: 71.97% here, 71.57% on 12 games and 71.66% on 3 VAL games. All
  three PCBAS studies agree within half a point.
- The runner-up does not beat the adopted package on any primary metric here,
  and is slightly worse on Drive with a 95% interval just below zero. Combined
  with GAR TEST, where the calibrated clean lead has an interval including zero
  and no-ball spell is equal (.4917 vs .4906, see
  [GAR_NOBALL_BRIDGE_V1](GAR_NOBALL_BRIDGE_V1.md)), there is no case for
  switching the submission package.
- The cross/shot dead-play pattern is present in both packages and matches GAR
  with the ball removed. It reflects missing ball, not a package difference.

## Reproduction

```sh
python -m experiments.pcbas_package_comparison_v1 declare   # done; immutable
python -m experiments.pcbas_package_comparison_v1 parity
python -m experiments.pcbas_package_comparison_v1 predict --half game_32_H1   # each of 36 halves
python -m experiments.pcbas_package_comparison_v1 seal
python -m experiments.pcbas_package_comparison_v1 score
python -m unittest experiments.tests.test_pcbas_package_comparison_v1 -v
```

[Declaration](results/pcbas_package_comparison_v1_declaration.json) ·
[result](results/pcbas_package_comparison_v1_result.json). The declaration,
seal and score gate are closed. Raw tracks and native outputs stay under ignored
`data/`. The 18 sealed games are the last untouched PCBAS TRAIN games and are
reserved for one final confirmation.
