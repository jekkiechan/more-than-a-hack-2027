# Original-full phase-average promotion

On 1 November 2026 the user requested: “update AGENTS.md as necessary, update the
site, and promote the latest model”. This authorizes the completed original_full
phase-average upgrade and its explicit exception to the failed +.01 clean screen.
The prior phase0 package, policy, frozen studies and closed public gates remain
preserved. This is a provisional working baseline, not a claim of overall superiority.

## Package and evidence

`joint_gru_sn_dense_original_diversity_phase_ensemble_v1` uses the same four
checkpoint hashes and teacher normalization as original_full. On the 25 Hz feed,
all five 5 Hz sampling offsets contribute equally. Native probabilities average
seeds within phase, then phases, before the unchanged f062 temperature-4/cost-3
decoder. No models are fitted and no organizer labels are accessed or rescored.

| Matched protocol | Phase0 spell | Promoted phases spell |
| --- | ---: | ---: |
| Original five, clean | .662590447 | .667555533 |
| Original five, v1 | .528179859 | .531575511 |
| Original five, v2 | .578498209 | .583571208 |
| GAR TEST ten, clean | .698836485 | .701640993 |
| GAR TEST ten, v2 | .699133249 | .701288850 |

All seven historical active guards pass and both component seeds improve on
clean/v1. Clean provider effects are mixed. The historical +.01 clean screen
still fails. TEST v2 gains in 8/10 games, but the paired 95% interval is
[−.000348, +.004637]. Calibrated+phase retains the TEST/v2 spell point lead.
The [complete scorecard](ORIGINAL_FULL_PHASE_AVERAGE.md) owns class, provider,
seed and per-IoU details. The saved public diagnostic remains .611765 spell /
.849473 macro on 17 possessions; it supplies no selection vote.

## Evaluation policy

The user-requested guidance update adopts original-provider clean + `feed_like_v2`
with at least four fixed realizations for development. Rank official spell F1,
macro for exact ties; retain equal-provider aggregation and clean macro/D-F1
floors. The default feed spell floor remains −.01 against the matched adopted
package. This policy change occurs after v2 results; historical v1 guards are
not relabelled as v2 evidence.

Evaluate frozen deployment candidates separately on ten GAR TEST games. Keep
nine previously selected-on VALID games separate. Do not pool 19/26-match
headlines or different checkpoint ensembles into confirmation. Record future
TEST reuse; repeated tuning makes it development exposure. GAR silver labels
and discontinuous segments are not independent continuous-match gold. V2's
model-conditioned calibration is also imperfect. New independently annotated
continuous matches remain the evidence priority.

## Verification and delivery

The [promotion runner](promote_original_full_phase.py) verifies the input,
checkpoint identities and saved candidate seal, then runs the real baseline CLI.
Promotion requires byte-exact replay of all 199,319 rows, SHA-256
`eb011268af14b4de77b4b1c823916631442ae2090efbb9526314007ddf27ceea`, and successful
format validation. The [promotion receipt](results/baseline_original_full_phase_promotion.json)
binds the active manifest/policy, predecessor and output. The
[baseline guide](BASELINE.md) owns restore and inference commands.

Local promotion, private remote delivery, source merge and site publication are
separate states. Subsequent delivery receipts record completed remote verification
and publication; no remote-native research archive is implied by package delivery.
