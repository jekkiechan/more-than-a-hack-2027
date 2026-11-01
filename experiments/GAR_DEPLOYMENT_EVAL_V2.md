# Exact deployment-package GAR evaluation

Study `gar_deployment_eval_v2`, declared 1 November 2026 before model scoring.
The [frozen declaration](results/gar_deployment_eval_v2_declaration.json) binds
the source, checkpoint pairs, match identities, corruption calibration and guards.
This continues the unfinished GAR work on `claude/sweet-hopper-seosxy` while
preserving its original runner and all completed evidence.

## Question and fixed comparison

Does uniform sampling-phase probability averaging improve the exact adopted
`original_full` two-seed pooled package on GAR TEST? Compare it with phase0 and
with the exact previous calibrated-dense pooled package, also with/without
phase averaging. No models are fitted and no defaults are changed.

The same frozen f062 temperature-4/cost-3 decoder is used throughout. Native
probabilities average phases within each sealed seed/teacher pair. Original_full
then pools raw seeds before normalization; calibrated dense passes both seeds
to its decoder, which normalizes them before pooling. GAR's approximately 29.97 Hz tracking has six phases; the
competition feed's 25 Hz tracking has five. This tests the same averaging rule
at each provider's native rate, rather than dropping a GAR phase to force five.

## Corrections to the earlier proposal

GAR VALID's nine games were already used for dense-model selection: see
`choose` in the [original declaration](results/sn_dense_v1_declaration.json)
and `select` in [the training runner](sn_dense_run.py). They are development
evidence. TEST's ten games are TRAIN-disjoint and had prior label/schema QA,
with no known prior model scoring/selection in the reviewed saved lineage.
Report TEST separately; the combined 19-game mean is not a confirmation metric.

The unfinished [earlier runner](gar_heldout_eval_v1.py) averages six provider-fold
checkpoints. That is a different ensemble from the deployed two pooled seeds.
The continuation uses exact manifest-verified deployment weights and each
member's sealed teacher normalization. All 45 GAR training IDs are inspected
in both packages and disjointness is asserted against evaluation IDs.

## Protocol

- Ten TEST games are primary; nine exposed VALID games are secondary.
- Each game has clean tracking and four fixed v2 corruption realizations.
- Average realizations within each game, then average games equally.
- Compare spell F1 first and macro F1 for exact ties. Report D precision,
  recall/F1, short-spell recall, reference/predicted spell counts and coverage
  wherever the shared scorer supplies them.
- Use paired whole-game bootstrap intervals with 2,000 draws; replicas and clips
  are not independent observations.
- Check clean spell delta at least −.005 and clean macro/D-F1 deltas at least
  −.02 for phase averaging against the same original_full package. Preserve the
  historical +.01 clean improvement screen and report its failure separately.
- Verify phase0 features exactly against the prepared dense records and native
  probabilities against the ordinary deployment path within 1e-6. Verify
  unchanged grading and reference counts across all arms and views.
- Preserve segment, half and timestamp breaks and unknown targets. Never infer
  full-match coverage from concatenated, separately bounded event clips.

True silver states place synthetic defects. Features and inference receive
tracking only. V2 calibration uses the predecessor's predicted states and is
not independent truth. No organizer reference labels are read, and the saved
public phase-average diagnostic is reused without rescoring or tuning.

## Promotion consultation

The user requested Astra consultation on 1 November 2026. Astra recommends
retaining original_full with phase averaging as the leading upgrade candidate
and completing this exact-package TEST check before changing the default.
Original aggregate clean/v1/v2 gains are positive, but per-provider effects are
mixed, calibrated-dense plus phase averaging still leads the original v2
benchmark, and the historical +.01 clean screen fails. The public window has
only 17 spells and provides no promotion vote. No promotion is authorized by
this research declaration.

## Execution and artifacts

All seven downloaded VALID/TEST inputs independently match the existing pinned
receipt's SHA-256 values (2.07 GB); raw data and native outputs remain ignored.
Source revision is `32c840210b7d08d06976e7f9457dfff79fb2da99`.

```sh
.venv/bin/python -m experiments.gar_deployment_eval_v2 declare
.venv/bin/python -m experiments.gar_deployment_eval_v2 prepare
.venv/bin/python -m experiments.gar_deployment_eval_v2_execution amend
.venv/bin/python -m experiments.gar_deployment_eval_v2_execution games --workers 2
.venv/bin/python -m experiments.gar_deployment_eval_v2_execution report
```

Execution is in progress. The declaration is immutable. The
[pre-score execution amendment](results/gar_deployment_eval_v2_execution_amendment.json)
preserves the first runner and records Astra's sampling and normalization
corrections before any model scoring. GAR parity uses the independently
assembled native dense records, retaining the actual fps and verifying decoded
labels as well as per-seed probabilities. The frozen simulator treats unknown
X context as live for defect placement; these frames remain ungraded.
Native records,
predictions, coverage and replay receipts are under
`data/experiments/gar_deployment_eval_v2/`. Remote availability of these new
native artifacts has not been established.
