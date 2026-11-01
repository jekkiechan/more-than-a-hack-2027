# Spell-F1 decision-rule experiment

Status: completed, retain the adopted decoder. Date: 2026-11-01. Base: fork main `a5b6bf2`.
Primary objective: improve official possession spell F1. Runtime is secondary.

## Hypothesis and fixed comparison

The adopted cost-3 Viterbi decoder chooses the most probable complete state path,
which need not maximize expected spell F1. Test a minimum-Bayes-risk (MBR)
decision rule under the same model distribution. This changes the decision
objective, unlike the earlier confidence-boundary and turnover-cost experiments.

One fixed candidate uses 16 posterior path samples plus the Viterbi path as
possible predictions, evaluated against 32 independent posterior reference
samples. Select the candidate with highest mean official spell F1 across IoU
.3/.4/.5/.6/.7, keeping the Viterbi candidate first for exact utility ties.
Sampling uses exact forward-filter/backward-sampling under the cost-3 Potts
objective with seed 20261101 and separate candidate/reference random streams.
Select a whole-match path using global spell counts; do not average independent
block decisions. Preserve period/time discontinuities and unavailable classes.
Real target labels are unavailable to inference and its expected-utility rule.

No model fits, cost/temperature sweeps, sample-budget sweeps or post-result arms.
This is an approximate expected-utility decision, not an optimizer of unknown
true labels or a promise of an accuracy improvement. Model probability
miscalibration and finite Monte Carlo error are explicit limitations. Synthetic
exact enumeration found both improvements and a case where Monte Carlo selection
reduced true expected utility; the external comparison decides usefulness.

## Matched development protocol

Use only hash-verified `control_D2` native predictions from
`dense_label_weight_v1`, retaining reconstructed-control provenance. Five matches:
Sportec J03WOY, Metrica 3, SkillCorner 1996435/1996436/2006229. Three views:
clean and two existing feed-like corruptions. Full provider exclusion applies to
both heads and normalization in the source study. All matches are already exposed
development evidence; Metrica/SkillCorner include event-derived silver labels,
and feed-like stress is synthetic. No independent confirmation is claimed.

Run 45 paired cells: seeds 2026/2027 and their probability ensemble separately.
The ensemble is primary; individual seeds quantify component consistency. Mean
corruptions within match, then matches within provider, then providers equally.
Rank by official spell F1, macro F1 for exact ties; show the fixed 70/30 auxiliary
summary without selecting on it. Clean improvement screen +.01; feed-like spell
no worse than baseline minus .01; clean provider D-F1 and macro within .02.
No automatic promotion. A positive but sub-screen gain is exploratory evidence.

Before candidate scoring, all 15 ensemble controls must reproduce clean
.6472145590743718 and feed-like .5058107894247519 within 0.000051. Compare
identical grading populations, reference spell counts, original periods and
timestamps; preserve unknown labels and validity masks. Report D precision,
recall/F1, predicted/reference spell counts, short-spell recall and runtime.

Freeze executable source/configuration/input hashes in the declaration before
running. Claim an exclusive output directory before evaluation; retain failures.
Prediction arrays stay in ignored private data; publish aggregate receipts only.
No organizer references, public diagnostic, retraining or model promotion.

## Results and delivery

Restoration and the frozen control replay passed. All 45 paired evaluations completed.
The ensemble replay exactly reproduced both expected spell-F1 values.
All 497 experiment tests and 26 integrity/gate tests pass, as do the file-backed
fit contract and bundled CSV format validation. No organizer labels were scored.

Reproduction (fresh study outputs must not replace the frozen run):

```sh
.venv/bin/python -m experiments.spell_mbr_run freeze
.venv/bin/python -m experiments.spell_mbr_run run
```

The declaration is [spell_mbr_v1_declaration.json](results/spell_mbr_v1_declaration.json).
The runner refuses an existing output directory and retains its lock on failure.
The [restore receipt](results/spell_mbr_v1_restore_v4.json) binds 66 verified
artifacts: 45 prediction arrays, 15 evaluation inputs and six reconstructed
checkpoints. Earlier restore receipts preserve and correct a duplicated local
`data/` path; the canonical restored files are hash-identical.


## Matched results

The spell-F1 decision rule did not improve the ensemble or either component seed.
The clean +.01 screen fails. The feed-like −.01 tolerance and provider clean
D-F1/macro guards pass, but passing safety margins does not make a negative
spell-F1 result an upgrade. No promotion or post-result sample-budget sweep.

| Realization | View | Cost-3 spell F1 | MBR spell F1 | Delta | Cost-3 macro | MBR macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ensemble | clean | 0.647215 | 0.642980 | -0.004235 | 0.857677 | 0.857310 |
| ensemble | feed_like | 0.505811 | 0.503210 | -0.002601 | 0.815535 | 0.815527 |
| 2026 | clean | 0.635521 | 0.635489 | -0.000033 | 0.850631 | 0.850792 |
| 2026 | feed_like | 0.501847 | 0.499105 | -0.002742 | 0.809087 | 0.808453 |
| 2027 | clean | 0.637245 | 0.635140 | -0.002106 | 0.856672 | 0.856127 |
| 2027 | feed_like | 0.495759 | 0.493920 | -0.001839 | 0.809711 | 0.809758 |

For the ensemble, clean spell changes are Metrica −.009897, Sportec −.002807,
and SkillCorner unchanged. Under stress they are Metrica −.009379,
Sportec +.001576, and SkillCorner unchanged. This is not a consistent provider
improvement. Clean provider D-F1 changes are between 0 and +.000079; macro
changes range from −.001251 to +.000147.

The [aggregate report](results/spell_mbr_v1_aggregate.json) includes unrounded
spell/macro/70–30 results, D precision/recall/F1, provider results, both component
seeds, runtime and every declared guard. The
[control replay](results/spell_mbr_v1_control_replay.json) preserves the exact
15-cell ensemble gate and its grading/spell counts.

The [independent scalar readout](results/spell_mbr_v1_readout.json) verifies all
45 paired grading/reference-spell populations and independently reproduces the
hierarchy and guards. The ensemble adds an equal-provider mean 5.67 predicted
spells per clean match and 14.83 under stress. Short-spell (<3s, IoU .5) recall
changes by −.006244 clean and +.001061 stressed. Every Metrica and Sportec path
changes; every SkillCorner candidate remains exactly the baseline path, for
both component seeds and the ensemble.

The [independent artifact review](results/spell_mbr_v1_independent_review.json)
confirms frozen source/input hashes, all 45 prediction hashes and the saved
control replay. Neither review reran candidate inference or reference scoring.

Selection under a model's own path posterior does not establish better agreement
with reference possessions. Finite-sample utility error and imperfect model
probabilities are plausible explanations, not separately identified causes.
Do not tune the sample count, temperature or seed on these results. A stronger
next quality experiment needs new supervision or independent dense evidence;
this trial gives no reason to replace the current postprocessing rule.

The separate [runtime experiment](POSTPROCESSING_RUNTIME_V1.md) remains an
optional output-preserving speed candidate; its speed result is not spell-F1
evidence. MBR is slower: the saved real-input calls take roughly 3–7 seconds per
cell on this CPU. No matched end-to-end runtime claim is made. The test suite
shared the runtime during control replay; timing is diagnostic only.

## Artifact and site delivery

Candidate frame outputs and detailed per-cell receipts are under ignored
`data/experiments/spell_mbr_v1`. Private delivery is complete in [mtah-data PR #10](https://github.com/jekkiechan/mtah-data/pull/10),
branch `research/spell-mbr-archive-v1`, commit
`9a270280110c3bbd525da460b4a136567b87f105`. A fresh isolated clone fetched and
restored all 57 manifest entries with matching SHA-256 and size, including 45
candidate arrays, private reports, the declaration and six source snapshots.
[Remote verification receipt](results/spell_mbr_v1_private_archive_remote_verification.json).
The archive uses `spell_mbr_cache_manifest_v1.json`; its receipt binds the exact
manifest and remote commit. This restores run artifacts without reopening the
exclusive scoring directory or requiring another evaluation.
No organizer predictions or labels were used. The adopted manifest, decision
policy, historical runners and scoring gates are unchanged.

The research-site update remains outstanding: this environment lacks the Sites
`site-workflow.mjs` source-opening/packaging helper. The existing restricted
[research site](https://more-than-a-hack-progress.buffhen.chatgpt.site/) was
inspected during the preceding runtime work, but no site source or access
settings were changed. The intended Experiments update is this completed
negative result, its matched table and this report/PR.
