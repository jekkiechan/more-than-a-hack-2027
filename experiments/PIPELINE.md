# Possession pipeline takeover

This work starts from `60b89dd` on the fork's `main`. Claude's latest branch,
`ccr-400015d3-yz5xkl` (`b5d0b1b`), contributed the multi-source validation results
merged in PR #7. The earlier LightGBM comparison and research plan are also
already on `main`. The new work is on `possession/robust-pipeline`.

The immediate research question is whether the existing data can support better
possession attribution through observation repair, control context, ensembling
and decoding. Adding another provider remains useful, but is not a prerequisite
for this bounded experiment.

## Implemented changes

`possession_pipeline.py` keeps all inference behind the existing tracking-field
allowlist. Historical models and feature definitions are unchanged.

- **Preprocessing:** respect observation flags, normalize incomplete XY pairs,
  reject isolated impossible position spikes, and interpolate internal player
  gaps of at most 0.2 seconds when the endpoints imply at most 13 m/s. Ball-spike
  rejection uses a generous 65 m/s bound. It requires both adjacent jumps to be
  implausible and the bridge between neighbours to be plausible; persistent
  displacements are not silently rewritten. There is no long-gap extrapolation.
- **Features:** add unambiguous proximity evidence, its continuous duration,
  recent and upcoming control evidence with a five-second expiry, agreement
  across an observation gap, team support fractions, and repair indicators.
  New control features use speed-checked ball interpolation up to 0.5 seconds.
  The historical physical feature block retains its original 0.5-second ball
  interpolation, without a speed gate. Every temporal operation resets at a
  half, clock reset or timestamp gap. Centred/future context is legitimate for
  this offline task.
- **Augmentation:** give every ablation the same clean and degraded copies,
  each with H/A swap. Degradation adds the existing 41% generated ball mask,
  8% per-player missingness with 0.6-second mean gaps, and same-team slot
  switches. These rates describe synthetic masks, not final missing fractions
  or a calibrated simulation of the challenge camera.
- **Ensemble:** combine the baseline and contextual model's *conditional H/A
  probabilities* 50/50, separately from D. A confident D prediction therefore
  cannot suppress an expert's team vote. Compare full-model D, a 50/50 blend
  with player-only D, and player-only D.
- **Postprocessing:** use the existing exact Viterbi decoder, comparing 3/3
  team/dead transition costs with 3/8 for the D blends. No forced minimum
  possession duration or manual boundary correction is introduced.

The fixed physical thresholds are experimental choices, not thresholds fitted
to organizer labels. They need external evidence, just like the model changes.

## Validation protocol

`pipeline_eval.py` declares its complete protocol and data/code hashes before
feature extraction or fitting. A changed dataset, dependency, protocol or source
file invalidates resume. Feature caches and completed match evaluations can be
resumed only under the matching manifest. Partial or duplicate score matrices
cannot select a model.

Use the whole-match assignments in `dead_transfer.SPLITS`:

| Provider | Training pool | Validation |
| --- | --- | --- |
| Sportec | J03WMX, J03WN1, J03WOH | J03WOY |
| Metrica | Games 1, 2 | Game 3 |
| SkillCorner | First eight fixed IDs | 1996435, 1996436, 2006229 |

There are **three leave-one-provider-out folds**. Each trains on the training
pool of the other two providers and evaluates the excluded provider's validation
matches. Validation matches are never training examples in any fold. Selection
averages official spell F1 equally across providers, then equally across their
matches and clean/degraded conditions; macro F1 breaks ties. Three SkillCorner
matches do not receive triple the weight of one Sportec match.

Eight candidates isolate preprocessing, added context, ensembling, D-head
mixing and the decoder cost. All use the same bounded HGB capacity: 31 leaves,
120 iterations, learning rate .08, seed 17, approximately 2 Hz training samples,
native-rate feature computation/evaluation, and training-only short-spell weights
with beta .5. Weight is normalized first per match and then per provider. The
player-only D head uses equal match/provider weighting without live-spell weights.
This is a controlled comparison at a smaller capacity than the previous
63-leaf/260-iteration suite; scores must not be compared as if capacity and
training protocol were unchanged.

The selected configuration is refitted on all **13 training matches**. The final
refit does not consume any of the five validation matches. Neither historical
Sportec tests J03WPY/J03WR9/J03WQQ nor organizer possession labels are opened.

Limitations: these validation matches have already supported earlier research;
this is not fresh independent confirmation. There is one fitting seed per run.
SkillCorner's silver-label timing uncertainty remains, and scoring still excludes
frames with fewer than 12 detected players, as in the earlier suite. Synthetic
degradation does not reproduce camera occlusion. The selected result cannot
establish full-match competition performance.

## Reproduction

Install `experiments/requirements.txt`. Restore the processed external arrays to
their existing paths under ignored `data/`, or use the public conversion commands
in [OPEN_DATA.md](OPEN_DATA.md). The private `jekkiechan/mtah-data` cache can be
cloned separately with Git LFS; keep it private and never add it as a remote of
the public code repository. Only the train/validation files listed above are
needed. An unresolved LFS pointer is rejected before a run starts.

```sh
python -m unittest discover -s experiments/tests -v
python -m experiments.pipeline_eval --feature-workers 3 --threads 2 --fit-final
```

Default outputs are in `data/experiments/pipeline_v1/`: the predeclared protocol,
feature caches, three fold models, complete validation results, selected settings,
and the final model with checksums. The run may take tens of minutes on a CPU.
Use the identical command to resume. Use a new `--out` directory after changing
code, data or seed. Preserve earlier runs rather than overwriting their protocol.

Once the organizer's tactical tracking metadata/text files are present under
`data/tactical/` (or `MTAH_DATA_DIR`):

```sh
python -m experiments.pipeline_predict
python evaluation/validate_submission.py data/experiments/pipeline_v1/competition.csv
```

Alternatively pass `--inputs data/challenge/inputs.npz` for an existing tracking
export. The loader deserializes only allowed tracking fields, even if the NPZ
contains target fields. Model/source checksums are verified before inference.
The output command masks frames outside the playing halves, validates all
199,319 rows, writes a provenance sidecar and refuses to overwrite a submission.
It does not score public labels or submit anything to the organizers.

Organizer assets were not present in this fresh environment at takeover. Dual-feed
ball fusion, homography confidence and Smart Tagging integration therefore remain
separate follow-up experiments; no results for those are claimed here.
