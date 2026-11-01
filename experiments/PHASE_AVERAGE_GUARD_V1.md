# Phase-average active-policy guard completion

Study `phase_average_guard_v1` fills missing metrics for the already-fixed
original_full phase-average candidate. It does not propose another model,
decoder, phase weight or training seed, and supplies no fresh confirmation.
All five original development matches and their v1 corruption replicas have
prior exposure. The additional GAR TEST result prompted this audit; it does
not change the fixed comparison or select a new configuration.

## Frozen scope

The [declaration](results/phase_average_guard_v1_declaration.json) binds 27
independently restored assets, source hashes, the existing published scorecard,
and exactly 90 scalar score cells: five matches × three views × two phase
modes × ensemble/two component seeds. All original provider-excluded fold
weights, paired teachers and cached tracking inputs restore byte for byte.
There is no refitting or new normalization.

The views are clean and the two original state-blind feed-like replicas.
Use every native sampling phase with the unchanged f062 decoder. Ensemble
probabilities average raw seeds within phase, then phases; component seeds are
reported separately. Labels enter scoring only. Preserve unknown targets,
original observation masks and native match/half/time breaks.

## Checks and interpretation

First reproduce every published provider and aggregate clean/v1 spell and macro
score for both phase0 and phase averaging within 1e-12. A replay mismatch is a
failure to resolve, not permission to loosen the tolerance after seeing scores.

Then report D precision/recall/F1, macro, spell, short-spell recall, reference
and predicted spell counts, coverage and both component seeds. The active
v1 stress guard requires ensemble spell at least current original_full phase0
minus .01; each provider's clean macro/D-F1 must remain within .02. The
historical +.01 clean improvement screen still fails and is reported as failed.
Seed diagnostics do not become new selection arms.

Native decoded arrays remain ignored under
`data/experiments/phase_average_guard_v1/`. No organizer labels, closed public
gates, old frozen results or adoption metadata are accessed or changed.

## Execution

The pinned assets have been restored and hash-verified (27 files, 504,652,102
bytes). The fixed audit is executing on the remaining CPU while GAR VALID
finishes. Its declaration already exists and must not be recreated.

```sh
.venv/bin/python -m experiments.phase_average_guard_v1 run
.venv/bin/python -m experiments.phase_average_guard_v1 report
```

This audit closes reporting/guard gaps; the
[GAR comparison](GAR_DEPLOYMENT_EVAL_V2.md) owns the new match evidence and the
[phase-average readout](ORIGINAL_FULL_PHASE_AVERAGE.md) owns promotion advice.
