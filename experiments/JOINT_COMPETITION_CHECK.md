# Joint GRU: requested competition pipeline check

This is one user-requested diagnostic of the **latest player-only-D joint GRU**,
not a promotion or a preprocessing experiment. The baseline stays
`gru_hgb_ballgap25_v1`. The earlier plain joint GRU is not scored in this check.
The full-match competition labels are unavailable; only the official five-minute
pipeline check can be scored.

## Declaration before fitting or reading labels

The user's request is: "calculate comp too. i think we are moving in the right
direction. perhaps we just need better preprocessing for corrupted tracking".

Fit one pooled model on the original 13 external training matches. Exclude all
five development matches, all historical held-out matches, and organizer labels
from fitting and normalization. Use the unchanged D-repaired architecture,
eight epochs and original clean/coarse plus 25% ball-gap training recipe. Fix
seed **2026**, the original default, not the better-performing seed 2027. There
is no seed ensemble, hyperparameter search or further external scoring.

This is a pooled counterpart of the externally tested recipe, not one of the
provider-holdout checkpoints. Its prior two-seed external means are .6093 clean
spell/.8411 macro and .2991 corrupted spell/.6464 macro. The seed-2026 holdout
values are .6083/.8408 clean and .2767/.6133 corrupted. Both seeds failed the
predeclared stress guardrail; a public diagnostic cannot override that finding.

Use the exact baseline organizer tracking-input cache and observation masks,
existing feature functions, 5 Hz sampling, H/A swap symmetry, and Viterbi cost 3.
No new interpolation, identity repair, threshold or per-game correction enters
this check. Generate and format-validate all 199,319 output frames without
opening reference labels, then seal model/input/code/CSV hashes. A separate
exclusive marker must be claimed before reference-label access. Existing public
gates and reports remain closed and untouched.

Compare with the baseline's **saved** result (.5257 spell, .8588 macro, .8568 D
F1), using the same reference-file hash. Do not rerun the baseline. Report spell,
macro and D F1, possession counts, and the official 90% bootstrap interval. The
70/30 spell/macro figure is auxiliary only. No variant selection, preprocessing
tuning, model promotion or full-match accuracy claim follows this diagnostic.

## Execution

Restore the original live-gap cache and its 13 training-feature caches (or
regenerate those caches before freezing a new training declaration), plus the
existing label-free `data/challenge/inputs.npz` organizer input cache. Use the
pinned temporal environment. The check has one fixed output directory:

```sh
python -m experiments.joint_competition_check freeze
python -m experiments.joint_competition_check fit
python -m experiments.joint_competition_check predict
python -m experiments.joint_competition_check seal
python -m experiments.joint_competition_check check
```

The code prevents fitting/generation after the prediction seal and prevents a
second label check. After completion, read the saved report rather than running
the final command again. Organizer inputs and frame-level predictions remain
local and ignored. Only aggregate results/provenance belong in the public repo.

## Preprocessing hypothesis

Better preprocessing is plausible but untested by this check. The external
stress experiment found excessive D predictions under corrupted player tracking;
that points to player visibility/motion handling and the missing legacy D
player-dropout augmentation as separate hypotheses. Ball-only interpolation
cannot be assumed to resolve that failure. Test any repair on external matches
with native missingness masks preserved and frozen controls, before another
candidate is considered. Do not use this public window to choose repairs.
