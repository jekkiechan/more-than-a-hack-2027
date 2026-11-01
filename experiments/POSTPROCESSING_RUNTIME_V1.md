# Postprocessing runtime experiment

Completed 2026-11-01 against fork main `73a0d08`. Decision: retain the adopted
pipeline; keep an optional faster decoder for integration follow-up.

Two Luna agents audited prediction-quality opportunities and tested an
output-preserving runtime candidate. This is an exploratory synthetic benchmark,
not a predeclared quality-selection experiment. No training, external scoring,
organizer reference access, or model promotion was performed.

## Mechanism and matched comparison

The current dense ensemble averages native probabilities, then calls
`supervised.decode` with switch cost 3. The candidate in
[postprocessing_runtime.py](postprocessing_runtime.py) replaces repeated 3×3
transition-matrix construction with comparisons of staying versus penalized
alternative states. It imports the shared period-block and label contracts.
The dtype, subtraction-before-argmax order, globally unavailable classes, period
breaks, and first-source-index tie rule are retained.

The first scratch prototype promoted float32 recurrence values to float64.
Independent near-tie testing exposed divergent paths. That prototype was rejected;
the final implementation preserves the reference array arithmetic. The original
~25% timing observation does not describe the final implementation below.

| Final synthetic benchmark | Reference | Candidate |
| --- | ---: | ---: |
| Median of three runs, 199,319 frames | 1.489988 s | 1.253095 s |
| Decoder time reduction | — | 15.90% (1.19× faster) |
| Output parity | Reference | Exact |

The fixed-seed synthetic input has three classes, equal-probability rows and two
periods at 25 Hz, cost 3. The paired calls ran serially; shared-runtime timing varies. An earlier corrected
run measured 1.472/1.177 seconds (20.02% reduction), retained separately. This saves about 0.237 seconds in this decoder benchmark;
end-to-end latency, peak memory and real cached probability inputs were not
measured. The candidate is **not wired into the adopted prediction CLI**.

## Verification and reproduction

The benchmark checks 2,880 deterministic parity cases and the full-length array.
A second Luna agent independently checked 27,594 cases: 20,000 near-tie float32
sequences, 7,560 dtype/class-mask/period/fps/cost combinations and 34 subtraction
rounding probes. All matched. A committed regression also covers 300 near-tie
float32 sequences. These tests provide bounded evidence, not a proof for every
possible input. Inputs outside the baseline decoder's contract are not broadened.

```sh
.venv/bin/python -m unittest experiments.tests.test_postprocessing_runtime -v
.venv/bin/python -m experiments.postprocessing_runtime --json-output /tmp/runtime-new.json
```

The JSON destination is exclusive-create. Use a fresh path for reruns.
[Final benchmark receipt](results/postprocessing_runtime_v1_final_benchmark.json),
[earlier corrected benchmark](results/postprocessing_runtime_v1_benchmark.json) and
[independent review](results/postprocessing_runtime_v1_independent.json) retain
aggregate results only. The reviewed source SHA-256 is
`a3dcc553807d320335aa0dbf2e2c678c44945f2cde5a1131ceb25d8bf5dce377`.
The reference source digest remains
`b9ae95c3826a60417971961231d7bf5cf3c16897ec48010bf02cbf016bcdddee`.

## Prediction quality and next action

The quality audit found no supported new decoder sweep on the exposed cohort.
The [dense boundary comparison](DENSE_BOUNDARY_V1.md) reduced clean/feed-like
spell F1 by .002939/.003794; [gap memory](DENSE_GAP_MEMORY.md) gained .000367
clean and lost .001133 stressed. [Turnover v2](../TURNOVER_DECODER_V2.md)
also traded improved short-spell recall for worse overall spell F1.

Pooled deployment weights are locally present, but matched dense provider-holdout
probabilities are absent. The private `research/dense-targeted-archive-v1` archive,
manifest `dense_targeted_followups_cache_manifest_v1.json`, holds reconstructed
D2 controls and native probabilities. Restore those for a real-input parity test
before considering integration; they are not original dense fold checkpoints.
Independent dense labels remain the stronger next step for quality improvement.
No new private artifacts were produced, and all frozen records remain unchanged.

## Repository checks

All 485 experiment tests and 26 integrity/gate tests passed. Documentation checks
passed (107 Markdown files, 686 links/anchors), as did the synthetic file-backed
fit contract and bundled 199,319-row CSV format validation. The first suite run
reported three missing-`pyarrow` errors; installing pinned `pyarrow==20.0.0`
resolved them. No organizer labels were scored.

## Delivery

The research-site update remains outstanding. The existing restricted Site was
successfully inspected, but this runtime has no installed Sites
`site-workflow.mjs` helper, so its required source-opening/packaging workflow and
browser QA could not be completed. No site content or access settings were changed.
The intended update is this runtime result and its quality-audit limitations on
[the research site's Experiments page](https://more-than-a-hack-progress.buffhen.chatgpt.site/experiments.html).
