# GAR TEST no-ball bridge for the PCBAS evidence

Study `gar_noball_bridge_v1`, completed 2 November 2026. Adopted package retained;
no fitting, threshold search, organizer-label access or promotion.

## Question

The [PCBAS checks](PCBAS_EXPANDED_ANCHOR_V2.md) score the adopted package at
~72% Drive / ~69% Pass team agreement with **no ball input**. Is that a 2023–24
European-cohort transfer penalty, or simply what this model does without a ball?
GAR TEST has dense silver H/A/D labels, so removing the ball there measures the
same input regime on the training cohort, and also measures what the ball is worth.

## Protocol (frozen before scoring)

- Ten GAR TEST games, the exact adopted `original_full` and runner-up `calibrated`
  packages, six-phase averaging, frozen f062 decoder. Same inference path as
  [GAR_DEPLOYMENT_EVAL_V2](GAR_DEPLOYMENT_EVAL_V2.md).
- Views: `clean` and `noball` (every ball coordinate NaN, `ball_observed` false;
  players, masks, periods and time breaks unchanged).
- Parity gate: clean per-game spell/macro F1 must equal the saved
  `gar_deployment_eval_v2` TEST values. **Max difference 0.0 for both packages.**
- PCBAS-like anchors: controlled GAR touch runs, anchor at the run's last frame;
  end codes PA→pass, CR→cross, SH→shot, BC→drive. GAR clips centre on passes,
  so carries are rare (217 drive vs 10,126 pass anchors); **pass is primary**.
  The primary class was switched from drive to pass after one game's anchor
  class counts were inspected, before the declaration was written and before any
  predictions were scored. All 10,968 anchors agree with the silver team label.
- Decision rule (declared): GAR no-ball pass agreement within 5 points of PCBAS
  68.80% means there's no cohort penalty, and PCBAS team-head training is not
  prioritized. More than 5 points above means a gap, and an H/A-only PCBAS
  fine-tune becomes the recommended experiment.
- Whole-game paired bootstrap, 2,000 draws, seed 20261102.

## Results (adopted package, equal-game means over 10 games)

| Metric | Clean | No ball | Clean − no ball (95% CI) |
| --- | ---: | ---: | ---: |
| Spell F1 | .7016 | .4906 | **+.211 (+.180, +.238)** |
| Macro F1 | .7791 | .6307 | +.148 (+.127, +.166) |
| Live-frame team agreement | 87.5% | 68.2% | +19.3 pts (+16.9, +21.1) |
| Predicted D on silver-live frames | 5.3% | 11.0% | |
| Pass anchor agreement | 87.1% | **67.5%** (65.6–69.3) | +19.5 pts |
| Cross / shot agreement | 74.3% / 78.4% | 67.0% / 56.0% | |
| Cross / shot predicted D | 21.1% / 19.6% | 26.8% / 38.0% | |

**Bridge outcome: consistent.** GAR no-ball pass agreement is 67.5% versus PCBAS
68.8% (−1.3 points). The PCBAS cross/shot dead-play pattern also reproduces on
GAR once the ball is removed: shots go from 20% to 38% predicted D, compared
with 33–40% on PCBAS. It's a ball-absence effect, not a new-cohort failure. GAR
has about 12 visible players per frame and 61% ball coverage. It is
broadcast-like, like PCBAS.

The calibrated package behaves the same way: clean .7048 → no ball .4917 spell.
The no-ball spell difference between packages is .001.

## Interpretation

- The PCBAS numbers are a measure of player-only inference, not evidence of
  weaker transfer to 2023–24 club football. This removes the main worry the
  PCBAS result could have raised about the submission.
- Ball information is worth about .21 spell F1 on GAR. On the organizer feed the
  ball is missing on 41% of graded frames, mostly during dead play, which the
  model correctly reads as a dead cue. Only about 12% of live frames have no
  ball. So the no-ball view is a stress bound, not an expected challenge score.
- Per the declared rule, an H/A-only PCBAS fine-tune is **not prioritized**.
  The remaining lever for no-ball live play is the cross/shot dead gate. Any
  attempt must be judged on the original-provider `feed_like_v2` guards, not on
  exposed PCBAS/GAR views.

## Reproduction

```sh
python -m experiments.gar_noball_bridge_v1 prepare   # GAR TEST tracking-full, revision 32c84021
python -m experiments.gar_noball_bridge_v1 games --workers 2
python -m experiments.gar_noball_bridge_v1 report
python -m unittest experiments.tests.test_gar_noball_bridge_v1 -v
```

[Declaration](results/gar_noball_bridge_v1_declaration.json) ·
[result](results/gar_noball_bridge_v1.json). Raw GAR inputs (hashes match
`soccernet_dense_v1_download_receipt.json`) and native predictions stay under
ignored `data/`.
