# Full-match coverage source audit

Completed 31 October 2026. Study: `full_match_coverage_execution_v1`. This is
a source and label-coverage audit; it is not a model result.

## Finding

The available SoccerNet-GAR TRAIN source does not support an hours-matched
full-match stoppage-coverage comparison. It has many event clips and dense
silver labels within their merged contiguous regions, but the regions leave
large gaps and nearly all observed OUT runs are censored before a restart touch.
The 369 OUT-to-touch pairs that are bounded inside one record last at most
4.805 seconds, shorter than the fixed five-second context halo. They cannot
represent complete stoppages of longer duration. No coverage model was fitted;
that would test a different question on clipped event windows.

The exact missing asset is continuous full-match tracking/video with full
stoppage-to-restart timelines and matching event semantics. The restored
`tracking-full` source still consists of 144-frame action-centered clips. The
legacy whole-match SoccerNet tracking/video release remains access-gated; no
such full-match files were present in this workspace. No organizer labels were
read.

A live, read-only Hub check on 31 October confirmed that the documented legacy
tracking/video releases return HTTP 403 for the configured account. The current
action-spotting repository also returns 403 while its access request awaits
author review. The accessible GAR branch trees expose sharded records, with no
separate aligned continuous asset found. The
[availability receipt](results/full_match_coverage_availability.json) records
exact repository IDs, current branch revisions and responses; no files were
downloaded for this check.

## Measurement

The seven pinned TRAIN source files were restored from
`OpenSportsLab/SoccerNet-GAR`, branch `tracking-full`, revision
`32c840210b7d08d06976e7f9457dfff79fb2da99`. All seven SHA-256 values match the
frozen capacity declaration. The separate restore record is
[`full_match_coverage_source_restore.json`](results/full_match_coverage_source_restore.json).

Using the existing `soccernet_dense_labels.py` recipe on TRAIN only, the
rebuild produced 45 games, 62,159 source clips, 11,249 contiguous records, and
5,063,339 native frames (46.9297 observed record hours). The sealed dense
record manifest SHA-256 is
`d2639fa1a47b38bff1b293e8b2592fb1d7392c09700a2c9873c1a90a72397b51`.

| Measure | Result |
| --- | ---: |
| Union of observed clips | 46.9297 h |
| First-to-last observed clip envelope, summed by game half | 77.5153 h |
| Union coverage within that observed envelope | 60.5425% |
| Gaps between records within observed envelopes | 11,153 |
| Gap duration, median / p90 / maximum | 2.402 / 28.529 / 498.098 s |
| Record duration, median / p90 / maximum | 11.545 / 30.070 / 124.491 s |
| OUT-derived D provenance | 2.937 h |

The envelope starts at the first observed clip and ends at the last observed
clip in each half; it is not a full-match denominator. Inter-record gaps may
reflect clip sampling, invalid frames, or time breaks. We preserve each
`full_period` block and never propagate labels across these gaps.

The dense records contain 4,352 consecutive OUT-event runs. Only 369 (8.4789%)
have a later touch run inside the same contiguous record. The other 3,983 end
at the record boundary without an observed restart touch. For the bounded
pairs, total OUT-to-touch duration is 1,217.584 seconds (20.3 minutes; 0.72%
of all observed record-hours). Duration is 2.402 seconds at the median and
p90, with a maximum of 4.805 seconds. A stoppage whose OUT occurs before a
clip starts is not identifiable from that clip, so this audit does not
estimate left-censored stoppage counts. OUT and touch events are the dataset's
silver metadata; the audit does not infer unobserved events.

## Reproduction and artifacts

The raw TRAIN files were recovered with the normal Hugging Face client at the
pinned revision using the paths and expected hashes in the restore receipt;
`HF_HUB_DISABLE_XET=1` routed around the read-only default Xet log directory.
The dense-record preparation and audit commands were:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m experiments.soccernet_dense_labels --splits train --workers 4
.venv/bin/python -m experiments.full_match_coverage_execution
.venv/bin/python -m unittest experiments.tests.test_full_match_coverage_execution -v
```

The coverage runner verifies each game's dense-record SHA-256 against the
TRAIN manifest before loading it. Its aggregate receipt is
[`full_match_coverage_execution.json`](results/full_match_coverage_execution.json)
(SHA-256 `ed6ae6e56f437e6557c0767015e6cdb9fe1b58f343f6421afbdb2a07b7657172`).
The raw TRAIN restore receipt has SHA-256
`1511e145b15adef7715ff2a1465d71115a3630e0bc97e0d690b3688178e1f5df`. The
synthetic coverage contracts pass 8/8, including per-half gap accounting,
record-edge censoring, no cross-record stoppage pairing, and internal time-break
rejection. Data files and generated dense records remain in ignored local
storage. No training, external scoring, public scoring, or promotion was done.

The next useful action is to obtain or restore the actual whole-match tracking
asset and its authorized event metadata, then repeat this coverage audit before
freezing any hours-matched arm. The existing five exposed external development
matches can only be a development readout, not fresh independent confirmation.
