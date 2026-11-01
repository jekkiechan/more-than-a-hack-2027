# Expanded held-out evaluation: 26 matches on the challenge-calibrated feed

Studies `gar_heldout_eval_v1`, `sportec_extra_eval_v1` and `expanded_eval_summary_v1`,
1 November 2026. No fitting, no organizer labels and no promotion. One user-requested
public diagnostic is reported separately in
[`feed_like_v2_public_result.json`](results/feed_like_v2_public_result.json).

## Why

Five development matches give an absolute spell-F1 standard error of about ±.04. The SD
between matches is .087 on `feed_like_v2`, against about .01 between realizations, so
more matches matter more than more realizations. The user authorized two additions:

* **Sportec J03WPY and J03WQQ:** full-pitch like the challenge, and spent confirmation
  matches. The Sportec-excluded fold models never saw Sportec. See the
  [Sportec report](SPORTEC_EXTRA_EVAL_V1.md).
* **19 SoccerNet-GAR VALID (9) and TEST (10) matches:** PFF World Cup 2022 tracking with
  silver dense labels derived from trained-annotator events. These matches had never been
  used for training or selection; all arms train on the 45 disjoint GAR TRAIN games.

## Method

* **Arms:**
  * `adop`: calibrated dense, the previously adopted package.
  * `cand`: `original_full`, now adopted.
  * `raw`: raw-feed recipe trained on v1 views.
  * `v2train`: the same recipe trained on v2 views.
  * Each arm is run without (phase 0) and with 5 Hz phase averaging (TTA).
* **Models per dataset:**
  * Development and Sportec matches use each arm's provider-excluded fold models.
  * GAR matches use the equal average of all 6 fold models (3 folds × 2 seeds).
* **GAR records:** downloaded VALID/TEST files are hash-recorded in the
  [download receipt](results/gar_heldout_eval_v1_download_receipt.json). The rebuilt dense
  records match the documented 9/2,191 and 10/2,439 games/records.
* **GAR game inputs:** the frozen assembler's per-segment tracking is captured and joined
  per game, with a distinct period for each record so breaks are preserved. The phase-0
  features equal the stored dense records exactly.
* **Sanity checks:** a development replay matches within 3 × 10⁻⁸, and reruns are deterministic.
* **Scoring:** v2 views are 2 realizations per GAR match and 4 per development or Sportec
  match. The decoder is the unchanged f062 decoder, and spell/macro F1 come from the
  official metric.

## GAR VALID/TEST (19 matches, about 17 graded hours)

| Arm | Clean | Clean + TTA | v2 | v2 + TTA |
| --- | ---: | ---: | ---: | ---: |
| Calibrated dense | .7073 | .7067 | .7058 | .7050 |
| `original_full` | .7083 | **.7088** | .7069 | **.7087** |
| Raw-feed | .7060 | .7052 | .7049 | .7052 |
| v2-trained | .7069 | .7072 | .7049 | .7040 |

All arms fall within .005 of each other. The SD between games is about .045. On v2,
`original_full` + TTA minus calibrated dense + TTA is +.0037 (90% CI −.0001 to +.0076,
P(>0) .94). GAR is in-domain for every arm, because they all train on GAR TRAIN, which
explains the high absolute level. The full per-split tables and paired bootstraps are in
[`gar_heldout_eval_v1.json`](results/gar_heldout_eval_v1.json).

## Pooled over 26 matches (`feed_like_v2`, each match once)

Group means use a plain mean per match; the earlier development tables use an
equal-provider mean, so the numbers differ.

| Group (matches) | Calibrated dense | + TTA | `original_full` | + TTA |
| --- | ---: | ---: | ---: | ---: |
| Development (5) | .5466 | **.5492** | .5408 | .5438 |
| Extra Sportec (2) | .5759 | .5808 | .5905 | **.6001** |
| GAR VALID/TEST (19) | .7058 | .7050 | .7069 | **.7087** |
| **All 26** | .6652 | .6655 | .6660 | **.6686** |

Paired bootstrap over matches (4,000 draws, matches resampled within each group):

| Comparison | All 26 Δ | 90% CI | P(>0) | Development / Sportec / GAR Δ |
| --- | ---: | --- | ---: | --- |
| `original_full` − calibrated dense | +.0008 | [−.0022, +.0041] | .67 | −.0059 / +.0146 / +.0011 |
| `original_full` + TTA − calibrated dense + TTA | +.0032 | [−.0001, +.0064] | .95 | −.0054 / +.0193 / +.0037 |
| `original_full` + TTA − calibrated dense | +.0035 | [+.0005, +.0064] | .97 | −.0028 / +.0242 / +.0029 |
| TTA effect on `original_full` | +.0026 | [+.0009, +.0044] | 1.00 | +.0030 / +.0096 / +.0018 |
| TTA effect on calibrated dense | +.0003 | [−.0016, +.0023] | .58 | +.0026 / +.0050 / −.0008 |
| v2-trained − raw-feed | +.0009 | [−.0002, +.0020] | .91 | +.0033 / +.0028 / .0000 |

## Public window (17 spells, one-time diagnostics)

| CSV | Spell | Macro | Matches at IoU .3–.7 |
| --- | ---: | ---: | --- |
| Calibrated dense | .6303 | .8642 | 12 12 12 9 7 (16 predictions) |
| `original_full` | .6118 | .8517 | 12 12 12 10 6 (17 predictions) |
| `original_full` + TTA | .6118 | .8495 | 12 12 12 10 6 (17 predictions) |

TTA changes no spell match on the 5-minute window. The public difference amounts to one
spell at IoU .6 versus .7 plus one extra prediction. That is within noise: both 90%
bootstrap intervals are about .32 wide.

## Interpretation

* **The broader evidence favors `original_full` + phase TTA.** Across 26 matches it is
  best on v2: +.0035 over calibrated dense (P .97) and +.0032 over calibrated dense with
  TTA (P .95). TTA reliably helps `original_full` (+.0026, CI excludes 0) but not calibrated
  dense.
* **The effect differs by data source.** The candidate wins strongly on all three Sportec
  matches and narrowly on GAR. It loses on the original development set (Metrica −.023;
  SkillCorner narrowly), and on the public window, where n is tiny. The challenge feed is
  full-pitch, as Sportec and GAR are, whereas SkillCorner is broadcast tracking.
* **Effect sizes are small:** a few thousandths of spell F1. The level differences between
  datasets are far larger: development .55, Sportec .59, GAR .71. So these results rank the
  models; they do not predict the competition score.
* **Limits:**
  * GAR labels are event-derived silver labels, not the organizer's possession definition.
  * The Sportec matches were exposed earlier.
  * There is still no independent full-match organizer confirmation.
  * Phase TTA is not part of the adopted package; adding it requires a declared integration.

## Reproduce

```sh
export MTAH_DATA_DIR=../mtah-data MTAH_FSP_DIR=../mtah-fsp MTAH_DT_DIR=../mtah-dt
python -m experiments.sportec_extra_eval_v1 sanity   # then j03woy, cells, aggregate
python -m experiments.gar_heldout_eval_v1 sanity      # then games, report
python -m experiments.expanded_eval_summary_v1
```
