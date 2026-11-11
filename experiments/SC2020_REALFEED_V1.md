# SkillCorner 2019/20 real-feed windows in the frontier continuation

Study `sc2020_realfeed_v1` · 11 November 2026 · complete; no candidate retained.

## Result and decision

**No. Keep the approved spell-frontier package.** Adding the six unused
SkillCorner 2019/20 broadcast-tracking TRAIN matches does not raise exposed-development
`feed_like_v2` spell F1 above 0.60, and no candidate passes the retention rule.

| Candidate (CE component) | Clean spell | V1 spell | V2 spell | Failed retention checks |
| --- | ---: | ---: | ---: | --- |
| Approved frontier (control, exact replay) | 0.670137 | 0.532128 | 0.593619 | — |
| Round 1 best v2 with new data: `sc8_256/half` | 0.671081 | 0.530982 | 0.594473 | v1 spell, v1 short recall |
| Round 1 closest to retention: `sc8_256/replace` | 0.671345 | 0.533599 | 0.594168 | v1 and v2 short recall |
| Round 1 best v2 without new data: `ctrl_512/replace` | 0.668768 | 0.529968 | 0.594575 | clean spell and short recall, v1 spell and short recall |
| Round 2 rebuild: `rb8_1024/replace` | 0.669857 | 0.528659 | 0.591410 | all three spell means, all three short-recall means |
| Round 2 rebuild: `rb8_1024/half` | 0.670646 | 0.532018 | 0.593806 | v1 spell, v1 short recall |

All 14 scored candidates sit between **0.591410 and 0.594575** v2 spell. The 0.60
target remains unmet by at least 0.0054. Complete scorecards:
[round 1](results/sc2020_realfeed_v1.json), [round 2](results/sc2020_realfeed_v1_round2.json).

### The data effect is within draw noise

Arm minus the matched no-new-data continuation at the same snapshot and pipeline:

| Arm | Δ clean spell | Δ v1 spell | Δ v2 spell | Δ v2 Metrica | Δ v2 Sportec |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sc4_256/replace` | -0.001139 | +0.000163 | -0.000089 | -0.0012 | +0.0009 |
| `sc4_256/half` | -0.001741 | +0.000554 | +0.000469 | +0.0009 | +0.0005 |
| `sc4_512/replace` | +0.000324 | +0.002193 | -0.000644 | -0.0007 | -0.0012 |
| `sc4_512/half` | -0.000918 | -0.000093 | -0.000764 | -0.0022 | -0.0001 |
| `sc8_256/replace` | +0.001159 | +0.003688 | +0.000195 | +0.0009 | -0.0003 |
| `sc8_256/half` | +0.000805 | +0.000548 | +0.000841 | +0.0015 | +0.0011 |
| `sc8_512/replace` | +0.001347 | -0.000421 | -0.001362 | -0.0015 | -0.0026 |
| `sc8_512/half` | +0.000426 | +0.000421 | -0.000736 | -0.0025 | +0.0003 |

The v2 effect ranges from -0.0014 to +0.0008 with no consistent sign across share,
budget or provider. A fresh continuation draw alone moves v2 by up to 0.0010
(`ctrl` rows versus the frontier), so these differences are not distinguishable from
sampling noise. The stronger dose in round 2 (the frontier's full two-stage CE recipe
with the new windows) lowers v2 by 0.0022 and v1 by 0.0035 when it replaces the CE
component, driven by Metrica v2 (0.7058 → 0.6980).

## Question and data

Can real tracking defects, rather than more synthetic corruption or more GAR weight,
improve the frontier's H/A emissions under the challenge-calibrated v2 stress view?

`mtah-data` has held nine processed SkillCorner 2019/20 open-data matches since
25 October 2026 (source pin `1b99d401`, MIT licence with SkillCorner attribution). No
training or evaluation code referenced them before this study. They are broadcast
tracking with real missing-ball and partial-visibility defects and provider possession
labels (`H`/`A`; null is `X`; no `D`).

- Six TRAIN matches (2068, 2269, 2417, 2440, 2841, 3442) per the pre-existing
  `split_skillcorner2020.json`. The three evaluation matches were never loaded.
- The shared `multisource_eval.view` rule applies: unobserved positions are masked and
  frames with fewer than twelve visible players become `X`. That leaves **271 labelled
  minutes** (378 before the rule), ball observed on 89% of them, 15.8 visible players
  on average, and 81,419 known H/A samples at 5 Hz.
- The source is treated as SkillCorner for provider exclusion. The SkillCorner-held-out
  fold never sees it, so any gain had to come from Sportec and Metrica.

## Declared design

Both rounds were frozen and committed before their fits:
[round 1](results/sc2020_realfeed_v1_declaration.json),
[round 2](results/sc2020_realfeed_v1_round2_declaration.json). Round 2 was declared
after round 1's results and says so.

- **Control.** The approved frontier replays exactly from the `frontier_recovery_v1`
  archive: 0.6701365499557103 / 0.532128306931943 / 0.5936187198896373. The emission
  path reproduces the archived CE probabilities to 8e-7.
- **Round 1.** Continue the reconstructed stage-1024 CE fold checkpoints at 1e-4 with
  fresh draws, snapshots at 256 and 512 updates. Each batch keeps 16 original-provider
  windows; of 16 external windows, 0 (`ctrl`), 4 (`sc4`) or 8 (`sc8`) come from
  SkillCorner 2019/20 and the rest from GAR. External loss mass 0.5 splits in proportion.
  Fourteen fits.
- **Round 2.** The frontier CE recipe itself (original-full warm start, 512 updates at
  3e-4 then 512 at 1e-4) with 8 SkillCorner 2019/20 windows per batch. Four fits; its
  matched control is the approved CE.
- **Candidates.** The arm's CE either replaces the frontier CE component or is mixed
  50/50 with it. Mixture weights 0.5/0.375/0.125, H/A cost 3 and the exact predecessor
  D mask are fixed. D state and normalization are unchanged in every fit.
- **Retention rule** versus the approved frontier: clean and v1 spell not lower, v2
  spell higher, short-spell recall not lower in any view, macro within 0.001, provider
  clean macro within 0.02.

With no SkillCorner 2019/20 windows the new loop is bitwise identical to the shared
`v2_iterative_train` objective (contract test).

## Evidence limits

- These are the same five exposed development matches used by every prior v2 study.
- One draw schedule per arm; the comparison cannot resolve effects below about 0.001.
- Only the tested recipe is covered: the shared ≥12-visible-player rule discards the
  lowest-visibility 28% of labelled frames, which are the most feed-like ones, and the
  new source can reach only two of three provider folds.
- No organizer labels, public scoring, GAR VALID/TEST or pooled deployment fit was used.

Stop SkillCorner 2019/20 share, budget and mixture arms. This agrees with the earlier
extra-data, capacity and turnover studies: more H/A supervision of the existing kind
does not move v2 spell on this cohort.

## Reproduction

Restore the files listed in `results/frontier_reconstruction_v1_restore.json` from
`mtah-data` (`main`, `research/short-quality-diversity-v1`,
`research/dense-targeted-archive-v1`, `research/feed-robustness-v1`), unpack
`frontier_recovery_v1.tar.gz` from `research/frontier-recovery-v1`, and pull
`external/processed/skillcorner2020_*`. Defaults expect sibling checkouts `mtah-data`,
`mtah-sqd`, `mtah-dt`, `mtah-rf` and `fr`; override with `MTAH_*_DIR`,
`V2_ITERATIVE_GAR_ARCHIVE` and `FRONTIER_RECOVERY_DIR`.

```sh
python -m experiments.sc2020_realfeed_prepare cells
python -m experiments.sc2020_realfeed_prepare train
python -m experiments.sc2020_realfeed_train records
python -m experiments.sc2020_realfeed_train fit
python -m experiments.sc2020_realfeed_emit
python -m experiments.sc2020_realfeed_run
python -m experiments.sc2020_realfeed_train rebuild --job leave_out_sportec:2026   # and 2027, leave_out_metrica
python -m experiments.sc2020_realfeed_emit --arms rb8 --steps 1024
```

Weights, emissions and prepared inputs are local-only under ignored `data/`; nothing
was uploaded to private storage. The research site was not updated from this session.
