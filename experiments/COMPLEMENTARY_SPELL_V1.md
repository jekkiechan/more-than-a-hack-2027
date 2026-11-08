# Complementary spell pilot: seed weights and boundary uncertainty

Status: complete; all 280 saved predictions independently verified.
Date: 8 November 2026.

This study asks whether unequal seed weighting or posterior-median boundary
placement improves spell F1 under the same frozen probability model. It uses
Luna implementation agents and an independently reviewed execution path.

**Keep equal seed weighting. Retain the approved frontier package.** The
0.8-second posterior-median boundary arm improves the historical predecessor
by **+.000242 clean / +.000214 v1 / +.000207 v2 spell F1** and passes all
declared pilot checks. This is a small development signal, concentrated in
SkillCorner, not established superiority or a verified frontier upgrade.

## Results

The weight column assigns that fraction to seed 2026, with the remainder to
seed 2027. Every arm retains the predecessor's decoded D/X mask. The boundary
arms operate on the equal-seed control only; there is no combined search.

| Arm | Clean spell | V1 spell | V2 spell | V2 macro | V2 70/30 | Pilot checks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Seed 2026 weight 0 | .668287 | .527157 | .578848 | .849413 | .660017 | Fail |
| Seed 2026 weight .25 | .668353 | .532154 | .583004 | .849787 | .663039 | Fail |
| Equal .50/.50 control | .667556 | .531576 | .583571 | .849975 | .663492 | Reference |
| Seed 2026 weight .75 | .667075 | .530909 | .583469 | .849800 | .663368 | Fail |
| Seed 2026 weight 1 | .666526 | .528809 | .580068 | .849248 | .660822 | Fail |
| Boundary median, .2-second cap | .667730 | .531667 | .583615 | .849973 | .663523 | Pass |
| Boundary median, .4-second cap | .667683 | .531610 | .583670 | .849982 | .663563 | Pass |
| Boundary median, .8-second cap | .667798 | .531789 | .583778 | .849993 | .663643 | Pass |

Equal weighting has the highest v2 spell F1 among the five tested ratios.
This does not prove that .50 is the global optimum. The .25 arm improves clean
and v1 spell but loses v2 spell and short recall, and fails the v1 macro floor.
The .75 arm loses spell in every view. Choosing a ratio on the other two
providers and evaluating the excluded provider gives **.581506 v2**, below
the fixed equal blend's .583571 by .002065. This diagnostic does not create
fresh confirmation from already exposed matches.

| View, .8-second boundary arm | Spell delta | Macro F1 | Macro delta | Short-recall delta |
| --- | ---: | ---: | ---: | ---: |
| Clean | +.000242 | .871279 | +.000007 | +.000741 |
| V1 | +.000214 | .826924 | +.000071 | .000000 |
| V2 | +.000207 | .849993 | +.000018 | +.000729 |

The improvement is not uniform across providers. Clean spell changes are
Metrica .000000, SkillCorner +.001246 and Sportec −.000519. V1 changes are
−.000300 / +.000668 / +.000274 respectively; v2 changes are
+.000002 / +.000619 / .000000. All declared provider macro/D checks pass.
No interval or independent-match superiority claim is made for this tiny gain.
D precision/recall/F1 remain exact. The scorecard retains official predicted
and reference counts; preserving raw boundary runs does not assume that
reference-masked official predicted counts are invariant.

The auxiliary 70/30 summary is derived from the reported spell and macro
scores; it does not change the spell-first ordering. No alternate radius,
prior strength or seed ratio was added after seeing these results.

## Scope and current model

The latest explicitly approved package on [PR #87](https://github.com/jekkiechan/more-than-a-hack-2027/pull/87)
is the [spell-frontier ensemble](SPELL_FRONTIER_PROMOTION.md). Its conditional
H/A family weights are 0.5 CE, 0.375 calibrated and 0.125 predecessor, with
equal seed weights. The 50% GAR training objective is a separate quantity.
The [extra-data comparison](V2_EXTRA_DATA.md) already tested increasing GAR
objective weight to 0.625; that did not retain all the previous checks.

The frontier's new weights and native prediction caches were delivered locally
in a previous environment, not uploaded to the available private cache. They
are absent from this workspace. This pilot therefore uses the exact archived
**original-full phase predecessor**, with all six provider-excluded research
checkpoints and their paired teachers. Deployment weights are not substituted
for provider-held-out models. A pilot gain cannot establish a gain over the
approved frontier, whose clean/v1/v2 spell scores are .670137/.532128/.593619.
The [missing-artifact inventory](results/complementary_spell_v1_frontier_missing.json)
identifies the exact 35 stage-1024 and 35 adopted/calibrated caches needed for
that follow-up: **0/70 are available here**. Restoring only the ten pooled
deployment checkpoints would not supply provider-held-out research evidence.

The remote branch scan found three older Claude branches (`great-knuth`,
`preprocessing-review` and `sweet-hopper`), all already behind fork main.
No published `claude/fervent-*` branch was visible at inspection. Consequently
this report cannot describe unpublished work on that branch.

## Fixed comparison before scoring

- Five exposed development matches: Metrica `metrica_3`, Sportec `J03WOY`,
  and SkillCorner `1996435`, `1996436`, `2006229`.
- Clean, two historical v1 realizations, and four fixed historical v2
  realizations: 35 cells. Replicates average within match, matches within
  provider, then providers equally. No GAR or organizer scoring.
- Seed-2026 weights 0, 0.25, 0.5, 0.75 and 1, with complementary seed-2027
  weights. Pool seeds within each native phase, then phases. Hold the exact
  historical decoded D mask fixed and use H/A transition cost 3.
- Three boundary-only arms move existing H/A transitions by at most 0.2,
  0.4 or 0.8 seconds. A local posterior median minimizes expected absolute
  boundary error; the displacement prior costs 0.1 natural-log units per
  moved frame. Nonoverlapping corridors preserve run order and count, D/X
  frames and period/time breaks. No boundary/seed cross product is tested.
- Replay every historical control before scoring candidates. Expected
  clean/v1/v2 spell is .6675555328266306/.5315755112553304/.5835712076382747,
  with numerical tolerance 1e-8. Exact equal-seed decoded-label parity is
  required, not just similar scores.
- Rank by official v2 spell F1, macro F1 for exact ties. Report the auxiliary
  70/30 score separately. A promising pilot must strictly improve v2 spell,
  retain clean/v1/v2 spell and short-spell recall, keep each view's macro within
  .001, preserve D exactly and keep provider clean macro/D-F1 within .02.
  These checks do not authorize promotion or erase historical failed screens.
- As a descriptive transfer check, choose the seed weight using the other two
  providers' v2 means and report its result on the excluded provider. This
  does not create fresh holdout evidence or select another deployable arm.
- Stop these fixed comparisons after the declared budget. No follow-up weight,
  shift-radius or displacement-prior tuning on their results. Current-frontier
  verification requires restoring its exact missing research artifacts.

## Evidence limitations

These matches have already informed many experiments. Synthetic v2 corruption
is model-conditioned; additional realizations are not new independent matches.
Neither an aggregate gain nor a favorable seed establishes a full-match gold
performance ceiling. The organizer's 17-spell reference remains out of tuning
and its closed diagnostic gates stay closed.

## Execution, verification and reproduction

The [restore receipt](results/complementary_spell_v1_restore.json) verifies
84 selected historical artifacts / 570,302,741 bytes against private manifests,
including six provider-excluded research checkpoint/teacher pairs and 15 input
caches. No models were refit. The phase package manifest identifies the
historical procedure; the receipt identifies the actual fold checkpoints used.

All 35 native caches were prepared with two workers: five phases at 25 Hz and
two phases at 10 Hz. The final [declaration](results/complementary_spell_v1_declaration.json)
and [manifest](results/complementary_spell_v1_manifest.json) bind the complete
cache cohort and 64 source/manifest files after baseline inference, before
candidate scoring. The preparation receipt's narrower source inventory is
preserved; the broader source binding occurs at freeze time. No inference
source changed during preparation.

The [controls-only replay](results/complementary_spell_v1_preflight.json)
reproduces all three saved spell/macro means and exact equal-seed decoded
labels on every cell. The [scorecard](results/complementary_spell_v1_result.json)
contains all eight arms. The [independent verification](results/complementary_spell_v1_verification.json)
replays **280/280** saved predictions with the official metric implementation,
rebuilds provider means and guards, and checks grading, reference counts, exact
D/X masks and all 105 boundary run-order cases. Metric comparisons use 1e-10
tolerance. The [code review](results/complementary_spell_v1_code_review.json)
records the separate invariant and posterior-median checks.

```sh
# Restore the exact archived subset identified by the restore receipt first.
MTAH_DATA_DIR=/workspace/mtah-data MTAH_SQD_DIR=/workspace/mtah-sqd \
  .venv/bin/python -m experiments.complementary_pilot_prepare prepare --workers 2
# The completed study's declarations and outputs are immutable.
# To verify its saved native artifacts without rerunning the search:
.venv/bin/python -m experiments.complementary_spell_readout \
  --manifest experiments/results/complementary_spell_v1_manifest.json \
  --declaration experiments/results/complementary_spell_v1_declaration.json \
  --preflight experiments/results/complementary_spell_v1_preflight.json \
  --result experiments/results/complementary_spell_v1_result.json \
  --predictions-dir data/experiments/complementary_spell_v1/predictions \
  --output /tmp/complementary_spell_replay.json
```

All 17 new focused tests, 26 root tests, the file-backed fit contract and bundled
CSV format check pass. The full experiment suite runs 858 tests: 852 pass,
one is skipped, and five inherited frontier-public fixture tests fail because
the previous workspace's local-only prediction seal/CSV is absent. They do not
establish a regression in this pilot. Documentation link checks pass.

New native caches and predictions are locally verified in ignored storage;
no new private upload or independent remote restore is claimed. The archived
input subset remains remotely restorable under its existing private manifests.
The approved package, policy and closed organizer gates are unchanged.

Stop this bounded comparison. The .8-second boundary rule is worth checking
on the exact frontier caches once restored; these results do not justify
another fine search on the same exposed matches or establish a performance
ceiling. Independently annotated continuous matches remain the stronger route
to assessing further improvements.
