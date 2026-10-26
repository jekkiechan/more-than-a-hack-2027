# Current GRU plus HGB baseline

The adopted development baseline is **50/50 bidirectional GRU plus ordered HGB
for P(D)**, with the unchanged pooled contextual HGB for conditional H/A and
Viterbi cost 3. The user authorized promotion after it beat the previous
ordered-HGB baseline on the identical external cohort.

| External D-head provider holdouts | Spell F1 | Macro F1 |
| --- | ---: | ---: |
| Previous ordered HGB | .5652 | .8363 |
| Adopted GRU + HGB | **.5746** | **.8449** |

The promotion follows spell-first ordering, with macro F1 for exact ties.
It compares five historical development matches across three providers, with
equal provider weights. It is not fresh independent confirmation. The
three-way GRU/HGB/LightGBM blend remains numerically slightly higher; the
two-model baseline is the authorized simplicity choice.

The deployment package uses models trained on the pooled training matches.
Provider-holdout scores describe the family with its corresponding held-out
D heads, rather than the pooled deployment checkpoint itself.

## Organizer five-minute pipeline check

At the user's request, the already generated, externally selected candidate was
checked once against the public window. Candidate, source, model and promotion
manifest hashes were verified before reading the reference labels. Earlier
experiment gates were preserved; this check has its own exclusive marker.

| Frozen candidate | Spell F1 | Macro F1 |
| --- | ---: | ---: |
| Adopted GRU + HGB | .4889 | .8568 |
| Historical GRU alone | .4889 | .8617 |
| Historical ordered HGB | .4205 | .8653 |
| Historical selected refinement | .4300 | .6487 |
| Historical PathCRF teams + original robust D | .5067 | .6795 |

These are the identical 7,500 graded frames with 17 reference possessions;
the ensemble predicts 19 possessions. Its D precision/recall/F1 are
.8918/.8359/.8630. The historical scores are previously recorded results,
not regenerated candidates. The ensemble improves spell F1 over ordered HGB
by .0684, but ties GRU alone and has slightly lower macro F1. There is no
demonstrated ensemble advantage over GRU alone on this window. The old
PathCRF hybrid is a different architecture from the external PathCRF
retention experiment and does not establish a benefit from adding PathCRF
to GRU/HGB.

This is a descriptive pipeline check, not a tuning or selection set. The
externally chosen baseline remains fixed, and full-match performance is
unknown. Aggregate results and frozen candidate provenance are saved in
`results/current_baseline_public_check.json`; local guard/receipts are under
`data/experiments/current_baseline/`. Do not repeat this check to iterate on
the public labels.

`baseline_manifest.json` binds the comparison report, cohort, model hashes,
architecture and promotion time. Previous benchmark/competition selection
manifests remain historical records; they are not overwritten by this update.

## Restore and run

The model package is saved in private `jekkiechan/mtah-data` at
`models/gru_hgb_v1/`. It includes the GRU D head, ordered HGB D head, H/A model,
and promotion manifest. The package is about 7 MB and was verified by restoring
from a separate clone and loading all three models.

From the code checkout, with the private cache alongside it:

```sh
python ../mtah-data/scripts/competition_cache.py fetch --data-dir data
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest.json --data-dir data
python -m experiments.current_baseline predict
```

The first command restores organizer inputs, the second restores trained models.
The last generates `data/experiments/current_baseline/gru_hgb.csv` and validates
its 199,319 rows. Inference deserializes tracking fields only; it does not read
reference labels or score the organizer window. The generated file is a current
baseline candidate, not an organizer-submitted entry.

Install the experiment requirements and optional CPU PyTorch package as in
`LEARNED_TEMPORAL.md`. `--models-dir` can point directly at the verified private
cache; `--inputs` and `--output` select alternate paths. Existing generated
candidates are not overwritten, so use a new output filename for another run.

## Verification

The new baseline's D probabilities reproduced the evaluated pooled GRU/HGB
cache exactly on external Metrica game 3. Its organizer CSV passes the official
validator. Contract tests protect spell-first promotion, separation of PathCRF
team attribution from the shared D head, and exclusion of empty period-zero
segments. The full synthetic suite passes with optional neural dependencies.

The separate PathCRF experiment uses the same GRU/HGB D head and compares the
original team model, PathCRF team attribution, and a fixed 50/50 team blend on
Metrica/SkillCorner only. Sportec J03WOY is excluded because it was PathCRF
checkpoint training data. Both PathCRF variants reduce spell and macro F1, so
PathCRF remains an archived comparator and is excluded from the default
baseline. See [PATHCRF_ENSEMBLE.md](PATHCRF_ENSEMBLE.md) for matched results.
