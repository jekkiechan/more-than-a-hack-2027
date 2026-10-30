# SoccerNet matched training follow-up

Study: `soccernet_followup_execution_v1`, 30 October 2026.
This execution completes the three Luna implementations described in
[the training-ideas protocol](SOCCERNET_TRAINING_IDEAS.md).

## Protocol and provenance

The selected width-96 fold checkpoints and their private inventory were not
available in the restored archive. This study therefore explicitly rebuilds
six reference heads from the original, hash-verified combined-objective fold
teachers. It does not claim to restore the promoted package's original bytes.
The rebuild uses the recorded selected recipe: width 96, learning rate 3e-4,
2,048 updates, batch size 32, and equal old-data/SoccerNet windows. Original
normalization and D teachers remain frozen. A separate immutable rebuild
declaration precedes these fits; a second declaration freezes continuation
inputs and reference hashes before the 24 continuation fits.

SoccerNet-GAR TRAIN data were recovered from
[Hugging Face](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR/tree/tracking-full)
at revision `32c840210b7d08d06976e7f9457dfff79fb2da99`. All seven source-file
hashes match the historical capacity declaration. The new acquisition receipt
has its own timestamp and hash. Continuous segments are rebuilt using the
unchanged historical assembler. Sparse PASS/CROSS/SHOT actor-team anchors are
H/A silver labels; unlabeled rows remain unknown. Neither context duration nor
event labels provide dense possession or D ground truth.

Each reference is continued for 512 updates at learning rate 1e-4 under four
fixed arms: the unmodified continuation control, event-balanced anchors,
raw ball-gap consistency, and masked context reconstruction. Supervised draws
are matched across arms and checked by hash. The auxiliary arms require extra
forward passes, so the budget matches updates, not wall-clock compute.

The event-balancing recipe normalizes positive anchor weights to mean one
before clipping at five. Clipping can reduce their final mean, and therefore
the effective scale of the GAR loss. This arm tests that combined weighting
recipe; it does not isolate event composition from loss scale. The declaration
records the actual post-cap mean and fraction at the cap.

Gap views remove ball observations in raw tracking before recomputing features.
Each recovered clean view must reproduce the base segment exactly. A fixed
identity-derived seed determines the prepared gap for each segment, shared
across model seeds. Only newly hidden, originally observed, valid core rows
with frozen H/A teacher confidence at least .8 receive the auxiliary KL loss.
Context reconstruction hides valid core rows and reconstructs naturally
observed normalized features through a disposable decoder. Neither mechanism
adds possession labels or changes inference decoding.

## Evaluation

Every learned component excludes the evaluated provider: H/A, D, and the
original normalizer. The five previously exposed development matches are
Sportec J03WOY, Metrica Game 3, and SkillCorner 1996435/1996436/2006229.
The canonical manifest, exact match identifiers, and cache hashes are frozen
in the declarations. SoccerNet's 45 TRAIN fixtures are shared across folds.

The grid comprises clean and two fixed synthetic `feed_like` views for each
match: 15 cells, five arms including the rebuilt reference, and three members
(seed 2026, seed 2027, actual two-seed probability ensemble). All 30 models are
sealed before scoring. The evaluator first checks all 30 original teacher
outputs against archived probabilities (absolute tolerance 1e-6) and requires
exact decoded prediction parity. Conditional H/A inference preserves each
teacher's native P(D). Ensemble probabilities are averaged before cost-3
decoding using the historical arithmetic.

Official spell F1 ranks first, with macro F1 breaking exact ties; the 70/30
summary is auxiliary. Scores average within provider, then equally across
providers. The predeclared screen is +.01 clean spell over the rebuilt
reference, with `feed_like` spell no worse than -.01 and each provider's clean
macro/D-F1 no worse than -.02. Both seeds, all five IoU thresholds, short-spell
recall, spell counts, fragmentation, and D precision/recall are reported.
The rebuilt reference is compared numerically with the published capacity
report at tolerance 1e-9; any discrepancy remains explicit.

These are reused development matches and synthetic stress, not independent
confirmation. This execution does not access organizer labels, reopen a public
scoring gate, or automatically promote a model.

## Reproduction and artifact availability

Raw data, cached arrays, weights, frame predictions, and private fit receipts
remain under ignored local directories. The new study's inventory establishes
local integrity; it is not a remote restoration archive. Original fold teachers
and old-data caches came from the verified private LFS archive.

Use `MTAH_DATA_DIR=/workspace/mtah-data` for the restored old cache and
`MTAH_EXECUTION_DIR=<repository>/data` for new outputs. The executor provides
`freeze-rebuild`, `fit-reference --fold <fold> --seed <seed>`,
`freeze-followup`, `fit-followup --fold <fold> --seed <seed> --arm <arm>`, and
`seal`. The separate evaluator provides `evaluate` and `verify`, both with
`--execution-dir <repository>/data/experiments/soccernet_followup_execution_v1`.
Completed outputs and changed declared inputs fail closed.
