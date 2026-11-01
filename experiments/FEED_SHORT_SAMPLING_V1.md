# Short-possession exposure under raw-feed training

Status: TRAIN audit complete; fixed experiment being prepared, 1 November 2026.

The current experimental leader is the two-seed raw-feed augmented model:
clean/feed-like spell F1 .654740/.525240. The adopted calibrated baseline is
.658625/.518527. The next stretch remains .55 on the original frozen benchmark.

## Hypothesis and intervention

The earlier subsecond weighting study changed the loss contribution of rare
examples but did not change their exposure. This study replaces 25% of window
draws with an existing window containing a complete H/A possession lasting at
most three seconds, within the same already-selected record. It uses a separate
seeded RNG and falls back to the original window when no eligible pool exists.
Native unknown spans, time breaks and record-edge censoring remain intact.
This is explicitly label-conditioned TRAIN sampling; inference never uses it.

Preserving the record preserves source, match, clean/defective variant and
paired H/A orientation selection. Selected windows retain their original
alternate-feature flags. GAR swap and batch-shuffle RNG states are unchanged.
The new window distribution trains both H/A and D; D is not held fixed.
The audit replays every batch and compares source/match/game exposure with the
archived augmented-control receipts before any candidate fit.

Six fits use the same provider-excluded teachers and normalizers, seeds
2026/2027, width96, 2,048 steps, batch32, half original/half GAR, unchanged
losses and raw-feed views. No new labels, training data, architecture or decoder.
The decoder remains D temperature4 followed by cost3.

## Frozen comparison and stopping rule

Two fixed arms: the new two-seed probability ensemble and an equal native
probability blend with the two-seed raw-feed control. All six checkpoints are
sealed before scoring. Rank by original feed-like spell F1; macro breaks ties.
A candidate must exceed .525240 and pass the promoted-baseline guards: clean
spell within .005; each provider clean macro and D-F1 within .02; each provider
feed spell within .01. Report the historical +.01 clean screen separately.
There is no replacement-rate, duration-threshold, blend-weight or seed sweep.

If an eligible arm improves the original benchmark, evaluate that fixed winner
on the previous ten extra corruption views. Those views and their results are
already exposed; this is an additional development diagnostic, not independent
confirmation and not a new selection set. No organizer reference scoring or
promotion occurs. The same five exposed development matches, synthetic stress
and silver-label limitations continue to apply.

## Reproduction

Restore the parent raw-feed study and its pinned original dependencies first.
New frozen source/input bindings, TRAIN audit, checkpoints and score locks must
not be overwritten. From repository root:

```sh
PYTHONPATH=. .venv/bin/python scripts/audit_feed_short_sampling_v1.py
.venv/bin/python -m experiments.feed_short_sampling_v1 declare
.venv/bin/python -m experiments.feed_short_sampling_v1 preflight
.venv/bin/python -m experiments.feed_short_sampling_v1 fit --fold leave_out_metrica --seed 2026
# Complete the other five fixed pairs, then:
.venv/bin/python -m experiments.feed_short_sampling_v1 seal
.venv/bin/python -m experiments.feed_short_sampling_v1 evaluate
```

The target-mask, same-record/RNG, disabled-training parity and exception cleanup
contracts are in `experiments/tests/test_feed_short_sampler.py`.

## Predeclared component check

Before parent scores, `feed_short_head_v1` fixes one conditional follow-up if
resampling misses .55: use resampled conditional H/A with the existing raw-feed
D probabilities, cross within each seed and then average. This adds zero fits
and 45 original-cohort cells. It isolates the team-attribution component from
D-distribution changes caused by sampling. Same guards; no other crosses,
weights or decoding settings. No new independent confirmation is claimed.

## Conditional game-level access experiment

Before any within-record scores, `feed_short_game_v1` declares six additional
fits only if that comparison and its fixed H/A cross both miss .55 under the
guards. The hypothesis is that GAR record fragmentation blocks access to useful
short examples: within-record replacement actually fires on only about 4% of
GAR draws, despite a 25% request rate.

Keep the original-provider windows exactly matched to the first study. For
GAR, use a separate RNG and select eligible windows across records from the
same already-selected game. Consume the prior within-record RNG identically so
old-provider sampling remains exact. Game/source balance, labels, native
eligibility, all training settings and decoder stay fixed. The TRAIN audit
finds eligible windows in all 45 games and raises GAR short-row exposure to
about 7.8%; it verifies identical original-provider counts and source/game draws.

The two fixed candidates are the complete game-sampled ensemble and its
conditional H/A crossed with raw-feed control D, within each seed before
averaging. This is a change in access to existing windows, not a new threshold
or rate sweep. There is no extra-corruption selection; all evidence remains
original exposed development. All models are sealed before scores, and the
same promoted-baseline guards and .55 stretch apply.
