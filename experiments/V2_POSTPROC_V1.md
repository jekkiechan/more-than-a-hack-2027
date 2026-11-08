# V2 postprocessing and frontier rebuild

Study: `v2_postproc_v1` · 8 November 2026 · complete; no candidate retained.

The question was whether postprocessing, or more continuation diversity on the
existing training data, can lift exposed-development v2 spell F1 to at least
0.60 without regressing clean or v1 spell against the adopted spell-frontier
package (0.670137 / 0.532128 / 0.593619).

## Result

**No.** 197 scored arms (including repeated controls) cluster at 0.580–0.595 v2.
The numerical v2 maximum is **0.594992** (continuation 0.5 / calibrated 0.5,
H/A cost 2.5), which loses clean spell (0.667679) and short recall. No arm
beats the frontier on v2 while retaining all three spell means and all three
short-recall means; the only arm passing that check is the frontier itself.

| Lever | Best v2 spell | Clean / v1 at that arm | Verdict |
| --- | ---: | --- | --- |
| Frontier replica (control) | 0.593619 | 0.670137 / 0.532128 | exact replay |
| H/A cost × mixture weights (cost 1.5–3) | 0.594992 | 0.667679 / 0.533355 | clean loss |
| H/A cost 3.5–5 | 0.594007 | 0.669824 / 0.532319 | clean loss, short recall loss |
| D-mask decoder (temperature 3/4/6, cost 1.5/3/6, cand or cand+cal D) | 0.588284 (cal 0.5 H/A) | — | flat, ±0.002 |
| Label-free short-spell insertion (threshold 0.3–0.45) | 0.568383 | 0.658245 / 0.500042 | large loss |
| Stage-512 + stage-1024 snapshot members | 0.593443 | 0.670465 / 0.533002 | no v2 gain |
| Four continuation seeds (two extra draws) | 0.594209 | 0.670725 / 0.531252 | within seed noise |
| New calibrated-warm-start continuation family | 0.594386 | 0.668646 / 0.529875 | v1 loss |

Seed noise bounds what this protocol can detect: the two extra continuation
seeds alone give 0.594030 versus 0.593619 for the original pair, and small
decoder changes move individual providers by about ±0.005.

## Diagnostics

The decoder recovers only part of the reference H↔A turnovers. Summed over
cells for the adopted original-full emissions:

| Provider / view | Reference H↔A switches | Predicted |
| --- | ---: | ---: |
| Sportec clean | 344 | 214 |
| Sportec v2 (4 reps) | 1376 | 781 |
| Metrica clean | 280 | 263 |
| Metrica v2 (4 reps) | 1120 | 1021 |

Lowering the switch cost barely changes spell counts, and threshold insertion of
opposing-team runs adds far more false spells than it recovers. The missing
Sportec turnovers are not near the decision boundary of the current emissions;
they need better emissions (data or features), not a different decoder.

## Method

- **Emissions.** Native per-phase, per-seed H/A/D probabilities were cached for
  all 35 development cells (five exposed matches × clean, two v1, four v2
  realizations). Families: `cand` (adopted original-full folds, short-quality
  archive) and `adop` (calibrated dense D2 folds, dense-targeted archive), each
  with its original paired teacher and provider-excluded fold. Seeds average
  within phase, then phases average. Labels enter only the official scorer.
- **Control replay.** Original-full f062 reproduces 0.667556 / 0.531576 /
  0.583571 and calibrated f062 reproduces 0.588020 v2, exactly.
- **Frontier rebuild.** The frontier's provider-excluded continuation weights
  were local-only. They were refit from the archived fold warm starts with the
  recorded recipe (512 CE updates at 3e-4, then 512 at 1e-4, original-diversity
  sampler and four round-robin v2 TRAIN views, 45 GAR TRAIN games). The rebuilt
  blend reproduces **0.670137 / 0.532128 / 0.593619** and the stage-512 blend
  reproduces 0.590454, matching the sealed results to six decimals.
- **New families.** Extra continuation seeds 2126/2127 warm-start from the
  seed-2026/2027 checkpoints with different Torch and sampler seeds. The
  calibrated-warm-start family runs the same two-stage recipe from the
  calibrated D2 fold checkpoints. D state and normalization stay fixed.
- **Fixed-D decoding.** Unless a D-mask arm says otherwise, H/A is decoded
  inside the adopted original-full f062 live spans, as in the frontier package.

Aggregation is realizations within match, matches within provider, then
providers equally. No organizer labels, GAR VALID/TEST data or public scoring
were used. All weights, emissions and frame outputs remain in ignored `data/`.

## Evidence limits and decision

These are the same five exposed development matches already used for every
prior v2 study, so the null result is strong only for this cohort. The
retention rule here (all spell and short-recall view means at or above the
frontier) is stricter than a v2-only ranking. Keep the adopted frontier package.
Further gains likely need new emissions: independently annotated continuous
matches, or features that expose the short Sportec turnovers.

## Reproduction

Restore the private archives (`mtah-data` main, `research/short-quality-diversity-v1`,
`research/dense-targeted-archive-v1`, `research/feed-robustness-v1`), set
`MTAH_DATA_DIR`, `MTAH_SQD_DIR`, `MTAH_DT_DIR`, `MTAH_RF_ARCHIVE_DIR` and
`V2_ITERATIVE_GAR_ARCHIVE`, then run:

1. `python -m experiments.v2_postproc_v1_emit` (base emissions).
2. `python -m experiments.v2_postproc_v1_train prepare`, then `fit` (seeds 2026/2027;
   `--seeds 2126,2127` for extra seeds; `V2_POSTPROC_WARM=adop` for the calibrated warm start).
3. `CE_STAGE=1024 python -m experiments.v2_postproc_v1_emit --family ce`
   (`CE_STAGE=512`, `EMIT_SEEDS=2126,2127 CE_SUFFIX=__x` or `V2_POSTPROC_WARM=adop` as needed).
4. `python -m experiments.v2_postproc_v1_grid <controls|ce_cost|high_cost|snapshot|seeds4|adopwarm>`.

The base-cost, D-mask and insertion grids were run interactively from the same
arm functions (`v2_postproc_v1_arms.py`); their aggregates are saved alongside.

Results: [base cost](results/v2_postproc_v1_base_cost.json),
[D mask](results/v2_postproc_v1_d_mask.json),
[short insertion](results/v2_postproc_v1_short_insertion.json),
[continuation cost grid](results/v2_postproc_v1_ce_cost.json),
[high cost](results/v2_postproc_v1_high_cost.json),
[snapshot](results/v2_postproc_v1_snapshot.json),
[four seeds](results/v2_postproc_v1_seeds4.json),
[calibrated warm start](results/v2_postproc_v1_adopwarm.json).
