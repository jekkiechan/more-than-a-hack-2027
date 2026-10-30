# SoccerNet training follow-ups

30 October 2026 — **implemented and synthetically tested; real-data fits and
comparisons blocked by missing local-only artifacts**. The adopted baseline
remains `joint_gru_soccernet_capacity_ensemble_v1` from main `a792b09`.
No new possession scores, promotion, or organizer diagnostic are claimed.

Three Luna agents implemented separate mechanisms in parallel. A fourth Luna
agent reviewed artifact availability and the integrated training contracts.
The parent integrated them into [soccernet_training_ideas.py](soccernet_training_ideas.py).

## Why these three mechanisms

The [capacity study](SOCCERNET_CAPACITY.md) supplies 42,919 sparse H/A anchors
from 45 TRAIN fixtures and about 44.67 hours of retained context. PASS accounts
for 94.8% of eligible events. Most sampled rows remain unknown, and 27% of
eligible anchor frames lack an observed ball. These are opportunities for new
training objectives; the context duration is not dense possession ground truth.

| Arm | Falsifiable question | Fixed implementation |
| --- | --- | --- |
| `balanced_anchors` | Does reducing PASS dominance improve transfer beyond the same-budget continuation control? | Within each fixture, inverse event-frequency weights for PASS/CROSS/SHOT; mean-one normalization before a row cap of 5. Multiple actions at one row are averaged. Existing fixture-balanced sampling remains. Only verified mapped anchors receive weights. |
| `gap_consistency` | Does stable teacher H/A evidence teach the student to tolerate missing ball observations? | Corrupt raw TRAIN tracking before recomputing features; gap lengths 1–300 native frames, start probability 1/600 at approximately 30 Hz. Frozen starting H/A teacher supplies detached soft targets only on newly corrupted, originally observed, valid core rows with confidence ≥.8. KL coefficient .05. |
| `context_reconstruction` | Can unlabeled context improve the recurrent representation without inventing possession labels? | Hide a rounded 15% of valid core rows (minimum one), in spans up to three sampled rows. Reconstruct naturally observed normalized features with a disposable linear decoder; auxiliary MSE coefficient .05. No target for naturally missing values or copied endpoint padding. |

These are separate arms, not a combined package or a hyperparameter sweep.
The gap arm uses only external TRAIN data, unlike organizer target adaptation.
The reconstruction arm trains temporal representations rather than the earlier
instantaneous native-rate residual. Neither creates D labels, propagates event
labels through time, or alters the dead-ball teacher.

## Matched execution design

All four arms (including `control`) continue the same restored width-96 head
for 512 updates, AdamW learning rate 1e-4, batch size 32, with 50/50 old/GAR
supervised windows. The original complete-spell weighting, native transition
objective, normalizer, dropout, teacher pairing, and cost-3 inference decoder
are preserved. Identical supervised window draws are recorded by hash; auxiliary
RNGs do not perturb those draws or supervised dropout. Auxiliary forward passes
add compute, so this is update-matched, **not wall-time-matched**.

The design is 24 fits: four arms × three full-provider folds × seeds 2026/2027.
Auxiliaries draw fixture-balanced windows from retained segments, including
windows without H/A anchors; supervised draws still require an anchor. The
retained corpus does not include segments discarded by historical preparation
for containing no anchors at all. This limitation remains explicit.

Future comparison must use the same five exposed development matches and
clean/two fixed `feed_like` views. Average actual member probabilities before
cost-3 decoding. Compare each arm with both the adopted width-96 outputs and
the matched continuation control. Spell F1 ranks first; macro breaks exact
ties. Report the +.01 clean screen, feed-like floor of baseline minus .01,
provider clean macro/D-F1 floors of minus .02, both seeds, all five IoUs, D
precision/recall, spell counts and short-spell recall. These are development
results, not independent confirmation. Evaluation orchestration and actual
score replay are pending the artifact restore; this runner implements fitting,
not a new scoring system. No public scoring command is provided.

## Interfaces and restore requirements

```sh
.venv/bin/python -m experiments.soccernet_training_ideas plan
.venv/bin/python -m experiments.soccernet_training_ideas freeze \
  --bundle data/experiments/soccernet_training_ideas_v1/leave_out_metrica_2026.joblib \
  --output data/experiments/soccernet_training_ideas_v1/leave_out_metrica_2026.json
.venv/bin/python -m experiments.soccernet_training_ideas fit \
  --declaration data/experiments/soccernet_training_ideas_v1/leave_out_metrica_2026.json \
  --arm balanced_anchors \
  --output data/experiments/soccernet_training_ideas_v1/balanced_anchors_metrica_2026.joblib
```

`freeze` requires a **trusted local joblib** bundle, prepared from verified
restored artifacts, with the following fields. It seals that bundle, code,
dependency versions, canonical fold, checkpoint receipts and run configuration
before fitting. Changed declarations, data, versions or code fail closed;
completed checkpoints cannot be overwritten. The low-level `train` API accepts
already validated objects and is used by synthetic tests; use the CLI provenance
checks for real runs.

| Bundle key | Required value |
| --- | --- |
| `fold`, `seed` | One canonical `leave_out_<provider>` fold; 2026 or 2027 |
| `initial`, `initial_path` | Restored selected width-96 fold head and its local file path; sibling `.json` capacity fit receipt required |
| `teacher`, `teacher_path` | Matching original combined-objective fold teacher and path; sibling `.json` historical receipt required |
| `capacity_inventory_path` | Original private capacity `artifact_inventory.json`, matching the publicly recorded SHA-256; both head and receipt must match its entries |
| `old_records` | Verified original paired training records, as returned by `soccernet_extension.load_old(fold)` with its data root configured; full native objective fields required |
| `gar_records` | Verified merged records from `soccernet_segments.load_records`, covering all 45 TRAIN fixtures, with `segment_id` and `anchor_frame_mapping` intact |
| `gar_split`, `fixtures` | Literal `train`; exactly IDs 3812–3840 and 10502–10517 as strings |

For the gap arm, reconstruct native raw inputs for each retained continuous
TRAIN segment and call
`soccernet_gap_consistency.attach_consistency_view(record, raw_inputs, seed=...)`
before serializing the bundle. Use a fixed identity-derived seed and freeze the
resulting bundle once. Raw fields are `players`, `teams`, `ball`, `period`,
`timestamps`, `fps`, `pitch`, `ball_observed`, explicit `tracking_valid`, and
optional `player_observed`. Inputs must use the existing metric coordinate
convention and native visibility policy. The helper refuses discontinuities,
invalid rows, schema/time-grid mismatch or any clean-feature difference from
the saved record. It cannot reconstruct raw tracking from feature columns.
The historical cache does not store these raw segment arrays; extracting them
from the authorized raw shards is an additional restore/preparation step.

Checkpoints retain the `soccernet_capacity_model.predict(feature_record,
candidate, original_teacher)` interface and exact original P(D). They are
experimental continuations; the adopted baseline loader intentionally does
not accept them as the original 2,048-update package. No promotion is implied.

## Current verification and blocker

The [preflight receipt](results/soccernet_training_ideas_preflight.json) records
the checked availability. The six original fold teachers were initially LFS
pointers; selective restore downloaded the real objects and verified all six
against the published capacity declaration. The SoccerNet width-96 heads,
inventory, prepared merged segments and raw data are absent. Their historical
inventory explicitly says no remote archive was uploaded. No new private data
or model upload was attempted.

Synthetic tests cover all four optimizing arms, deterministic continuation,
identical supervised sampling, exact D retention, target-independent inference,
team symmetry, sparse-anchor conflicts, held-out fixture rejection, native
time breaks, raw-to-feature parity, auxiliary gradients, checkpoint round trips,
declaration/bundle tampering, and output overwrite prevention. The experiment
suite, root suite, file-backed fit parity and bundled CSV validation results
are recorded in [verification](results/soccernet_training_ideas_verification.json).

Next: supply the original local SoccerNet archive location, verify its inventory
and selected fold checkpoints, prepare paired raw gap views, freeze the six
fold/seed bundles, then execute the matched study. No arm is recommended for
adoption until that comparison completes.
