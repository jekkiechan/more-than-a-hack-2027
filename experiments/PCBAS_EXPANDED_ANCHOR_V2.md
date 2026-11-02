# Expanded PCBAS match evaluation

Study `pcbas_expanded_anchor_v2`, 2 November 2026. Execution is being prepared;
the declaration will be frozen before inference. No model changes or fitting.

## Question and evidence scope

Does the exact adopted submission package reliably identify the acting team at
live action moments across more previously unevaluated broadcast matches?
This extends the [three-game VAL diagnostic](PCBAS_CONFIDENCE_V1.md), whose
results are already exposed. It tests the same player-only input regime across
more matches and estimates uncertainty across games. It does not supply
continuous possession labels or official possession spell F1.

The newly approved source is
[SoccerNet/SN-PCBAS-2026](https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026).
Its TRAIN archive contains 48 matches from a 2023–24 European cohort. TRAIN is
the publisher's action-recognition partition; none of these inputs is used to
fit, calibrate, select or normalize the possession package in this study.
Anonymized game IDs prevent exhaustive fixture-level deduplication. Results
must therefore remain external event-anchor evidence, rather than certified
independent full-pipeline gold confirmation.

## Fixed protocol

Rank complete TRAIN game IDs by SHA-256 of
`pcbas_expanded_anchor_v2:<game_id>`; take the first twelve, both native halves.
Exclude the three previously scored VAL IDs. The remaining 36 games stay in
reserve and receive no predictions or model scores. The cohort is selected
without reading event targets, class balance, visibility or predictions.

Use the unchanged
`joint_gru_sn_dense_original_diversity_phase_ensemble_v1` package: both sealed
seed/teacher pairs, five-phase averaging and its frozen decoder. Input reads
exclude the HDF5 action-class column. The event JSON, actor and target team do
not enter inference. Ball input is entirely missing, as this source has no ball
entity. Preserve finite-ROI player visibility, native timestamps and half
boundaries. The inherited 105×68 m coordinate standardization remains an
assumption; there is no scale, offset, decoder or threshold search.

First replay 750 frames through the direct deployed API and study wrapper and
require exact decoded equality. Verify input, source, manifest and checkpoint
hashes; then predict each complete selected half. Seal all predictions before
the one-use event-scoring gate is claimed and labels are joined. An interrupted
run may resume only from matching complete per-half artifacts and receipts.

Drive is primary. Pass, Cross and Shot are separate secondary diagnostics.
Replay events, ambiguous same-frame opposing-team events, exact duplicates and
anchors without exact tracking frames are excluded using the frozen rules
from the first diagnostic. D/X at a live anchor counts as incorrect. Neither
background events nor unannotated frames constitute dead-ball ground truth.

Sum halves within each match, then weight matches equally. Also report pooled
counts, per-game direction balance and the descriptive majority-direction-alias
reference. These aliases are half-local attacking directions, not persistent
home/away teams. The reference uses target frequencies and is not a deployable
trained comparator. Primary uncertainty uses 10,000 paired whole-game bootstrap
draws, fixed seed `20261102`, for Drive accuracy and its difference from that
reference. Halves, frames and action anchors are never independent resampling
units. Intervals describe match variation in this source, not unseen-provider
uncertainty or qualification probability.

Prespecified visibility diagnostics use 0–5, 6–11 and 12+ visible players at the
anchor frame. They pool the four reported action classes and are descriptive;
their class mix can differ. Source-imputed off-screen positions remain masked.
Predicted D frequency at live anchors measures a false dead-play pattern, not
D precision/recall or overall live/dead accuracy.

## Source and reproduction

The tactical archive is pinned to revision
`a16e373c8c8d26b1af4874f4208810baf337e709`, expected 2,457,804,618 bytes and
SHA-256 `160ecf571b5eb8298b231dcb003973702dc413469216d13d849e2c7ef6fe44ee`.
Official TRAIN events are pinned to FOOTPASS commit
`f63f37c1a0c6bcebf568a8ce716495a459376af5`,
`playbyplay_GT/playbyplay_train.json`. Access-controlled raw data and native
outputs remain under ignored `data/`; aggregate evidence contains no raw tracks
or reconstructive predictions. No raw-data upload is part of this work.

Install the [PCBAS dependency file](requirements-pcbas.txt), then run from the
repository root:

```sh
.venv/bin/python -m unittest experiments.tests.test_pcbas_expanded_anchor_v2 -v
.venv/bin/python -m experiments.pcbas_expanded_anchor_v2 freeze
.venv/bin/python -m experiments.pcbas_expanded_anchor_v2 parity
# For each of the 24 keys in the frozen declaration:
.venv/bin/python -m experiments.pcbas_expanded_anchor_v2 predict --half GAME_H1
.venv/bin/python -m experiments.pcbas_expanded_anchor_v2 seal
.venv/bin/python -m experiments.pcbas_expanded_anchor_v2 score
.venv/bin/python scripts/verify_pcbas_expanded_anchor_v2.py
```

Completed declarations, predictions, seals and score gates are immutable. Do
not recreate them to obtain another result. The independent verifier recomputes
counts, exclusions, visibility strata and bootstrap intervals from the sealed
arrays without repeating inference or reopening a score gate.

## Decision and remaining gap

Keep the submission package unchanged. A larger sparse-anchor sample can
strengthen or weaken confidence in player-only team attribution, but cannot
answer continuous possession segmentation, boundary accuracy, short-spell
recovery, or official spell F1. The more representative input source still
lacks measured ball input and continuous H/A/D/X targets.

The next stronger possession evaluation requires independently annotated
continuous broadcast intervals on separate matches, using video-only blind
annotation with last-controlled-team semantics, preserved unknown regions and
uncertain boundaries. Freeze labels and the model comparison before scoring
with the official evaluator; keep at least one whole match untouched by model
selection. Do not interpolate PCBAS action points into possession gold.
