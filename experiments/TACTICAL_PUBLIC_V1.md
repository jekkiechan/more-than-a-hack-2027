# Tactical H/A deployment and public diagnostic

16 November 2026. The [tactical expert](TACTICAL_HA_V1.md) from merged
[PR #98](https://github.com/jekkiechan/more-than-a-hack-2027/pull/98) is already
adopted on main (`6073483`). This delivery verifies that package and completes
the user's separately requested public diagnostic. No fitting or selection occurs.
The active manifest/policy and historical promotion evidence are unchanged.

## Exact restore and inference

The private cache branch `claude/vigilant-einstein-0y14sc` at
`18eb031f6048dac9c36c88c6805fab511ec617be` contains the versioned manifest
`baseline_cache_manifest_joint_gru_sn_dense_spell_frontier_ball_exit_tactical_ensemble_v1.json`.
All 47 entries pass SHA-256 and byte-count checks, including eleven deployment
model roles. This is an exact private remote restore, not reconstruction.

```sh
python ../mtah-data/scripts/competition_cache.py fetch --manifest baseline_cache_manifest_joint_gru_sn_dense_spell_frontier_ball_exit_tactical_ensemble_v1.json --data-dir data
python -m experiments.current_baseline predict --output data/experiments/tactical_public_v1/replay.csv
python evaluation/validate_submission.py data/experiments/tactical_public_v1/replay.csv
```

The default CLI reproduces the pre-public sealed CSV byte-for-byte:
`8e26f823a3e933f1feca2a64dfe6d7f728a4741548c8c7cd95d51746b6a355b9`.
It has 199,319 consecutive rows and 281 full-match predicted possessions under
the format validator's counting convention. Models, source, input, manifest,
policy and predictions are sealed before reference access in the
[declaration](results/tactical_public_v1_declaration.json).
The new [exclusive gate](results/tactical_public_v1_gate.json) permits one
fixed candidate score; historical gates remain closed.

## Public diagnostic

Official evaluator, 1,000 bootstrap draws, 7,500 graded frames, 17 reference
possessions. Control values are reused from the saved reconstructed Fable
result; the control is not rescored.

| Metric | Fable | Tactical H/A | Delta |
| --- | ---: | ---: | ---: |
| Mean spell F1, IoU .3–.7 | .623529412 | .623529412 | .000000000 |
| Macro F1 | .864234427 | .870218351 | +.005983924 |
| D-F1 | .889372281 | .889372281 | .000000000 |
| D precision | .818181818 | .818181818 | .000000000 |
| D recall | .974132063 | .974132063 | .000000000 |
| Auxiliary 70/30 | .695740916 | .697536094 | +.001795177 |
| Predicted possessions | 17 | 17 | 0 |

Spell F1 at IoU .5 is .705882353; the mean-spell 5–95 bootstrap interval is
[.458823529, .776470588]. Macro's interval is [.721658088, .909663488].
These intervals do not establish significance of the paired difference.
The complete [result](results/tactical_public_v1_public.json) binds the
exclusive gate and declaration. Reproduction of the scoring command is for
inspection only: do not reopen or reset this gate.

## Interpretation and decision

Retain the already adopted tactical package. Public spell is unchanged and
frame macro improves; public results do not select, tune or change adoption.
The adoption rationale remains the protocol-matched development and external
spell evidence in PR #98: development v2 .598690 → .603079; GAR TEST10
clean/v2 .709186/.709176 → .716648/.715728, with 9/10 games improving.

Preserve the failed clean short-recall guard (−.000151) and unmet +.01 clean
screen. The fusion choice followed about 56 exploratory development arms;
repeatedly exposed development, model-conditioned v2 corruption, reused GAR
silver segments and this tiny public diagnostic do not establish independent
continuous-match gold performance. Obtain independent gold evidence next.

## Delivery

The research site is updated with the tactical adoption, protocol-separated
comparisons, public diagnostic, architecture and limitations. The separate
boundary-refiner GAR failure from PR #97 is also recorded. [Version 46](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site) is published with the existing restricted audience. Desktop/mobile checks at 1440/390 px pass. All 1,015 experiment tests (one skip), 26 root tests, documentation checks and file-backed fit parity pass.
Publication and verification details are in the [delivery receipt](results/tactical_public_v1_delivery.json).
