# Improvements since the GRU/HGB baseline in PR #10

This follows [PR #10](https://github.com/jekkiechan/more-than-a-hack-2027/pull/10),
which adopted 50/50 GRU and ordered HGB dead-ball probabilities with the original
pooled contextual HGB team model. The new working package is
`gru_hgb_ballgap25_v1`. Its only changed inference component is team attribution;
the D checkpoints, probability blend, physical feature schema and Viterbi
switch cost 3 remain unchanged.

## What changed and why

The previous augmentation modeled short Markov ball gaps. Aggregate unlabeled
organizer tracking shows a median missing-ball run of about 10.7 seconds. The
new team training replaces 25% of each original clean/coarse view with a fixed
longer-gap visibility mask, preserving player coordinates and direct/team-swap
pairs. Training-row counts and HGB capacity stay fixed. The resulting mixture
is approximately 37.5% clean, 37.5% original coarse augmentation and 25% targeted
ball-gap examples. No organizer possession labels enter training.

The broader first corruption batch tested ball gaps, regional/individual player
loss, smooth coordinate error, jumps, track fragmentation and sample loss before
feature extraction. Augmenting both team and D components improved stress spell
F1 but cost 1.29 clean percentage points. The follow-up lowered the replacement
share and separated team ball visibility from D player/coordinate corruption.
The adopted ball-only team arm has a much smaller clean change while improving
stress robustness. The experimental D replacements are not adopted.

## Matched external comparison

These are the same five historically used development matches: Sportec J03WOY,
Metrica game 3, and SkillCorner 1996435/1996436/2006229. Scores average matches
within a provider, then the three providers equally. Provider holdouts apply to
D training only; both team models remain pooled. Stress views reuse two fixed
composite-corruption seeds per match, with original grading masks unchanged.

| Condition | Previous spell F1 | Current spell F1 | Previous macro F1 | Current macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Clean | .5746 | .5740 | .8449 | .8439 |
| Composite-corrupted | .4205 | .4315 | .7699 | .7866 |

Corrupted spell improves by 1.10 percentage points; clean spell decreases by
0.06 points. Stress conditional H/A accuracy on reference-live frames increases
from 80.10% to 82.88%. The clean delta's conditional paired interval is
[-0.20,+0.16] points. These reused matches and singleton providers cannot establish
independent significance or real broadcast transfer.

The original clean spell-first rule retained the old baseline. The user later
explicitly authorized this working-baseline robustness tradeoff; the frozen
experimental selection was not rewritten. The promotion manifest records that
distinction and that public diagnostic results were already known.

## Recorded competition diagnostic

| Candidate | Spell F1 | Macro F1 | Dead-ball F1 | Predicted possessions |
| --- | ---: | ---: | ---: | ---: |
| PR #10 baseline | .4889 | .8568 | .8630 | 19 |
| Current ball-gap team model | .5257 | .8588 | .8568 | 18 |

The new spell point estimate is 3.68 percentage points higher. Both refer to the
same public five minutes: 7,500 frames and 17 reference possessions. The new
diagnostic was frozen after external evaluation and checked once at the user's
request; the old score was reused from its archive. Its 90% spell interval is
[.3657,.6743]. Full-match competition performance remains unknown, and no model
or setting changed after the diagnostic. Promotion performed no new scoring.

D probabilities are unchanged, but the joint decoder can change D labels as
team confidences change; that explains why D F1 can move despite fixed heads.
The public window is not a tuning set or independent confirmation of a winner.

## Reproducibility and verification

- The default `python -m experiments.current_baseline predict` now uses the
  versioned ball-gap team package and writes `gru_hgb_ballgap25.csv`.
- Its complete 199,319-row CSV reproduces the tested diagnostic byte for byte;
  inference reads no reference labels and passes the official validator.
- The previous package remains available, with its manifest and decision policy
  archived under `experiments/baselines/`; `--manifest` supports historical replay.
- All 69 synthetic/protocol tests pass. The model package and matching source
  snapshot were restored independently and their hashes verified.
- Organizer inputs, trained weights, corrupted arrays and frame-level predictions
  remain outside this public repository. Authorized users restore the adopted
  package through the private cache's default `baseline_cache_manifest.json`.
- Existing experiment declarations, reports and completed public scoring gates
  are preserved. No organizer submission is made by this update.

Methods and full aggregate evidence: [lighter corruption](LIGHT_CORRUPTION.md),
[broader corruption](REALISTIC_CORRUPTION.md), and [active baseline](BASELINE.md).
