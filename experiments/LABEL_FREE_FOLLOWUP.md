# Label-free follow-up: stress realism, possession rate, target adaptation

Study date: 29 October 2026. Three bounded studies ran in parallel against the adopted
`joint_gru_combined_ensemble_v1`. Each pre-declared its rule before scoring, reproduced the
saved holdout scores exactly (clean .6332 / feed-like .4988 spell, max diff 0.0), and never
read organizer ground truth or scored the competition match. Holdout = five previously
exposed full-provider development matches; spell F1 is equal-provider mean.

## Decision summary

| Study | Pre-declared rule | Result | Decision |
| --- | --- | --- | --- |
| [Context-conditioned ball gaps](CONTEXT_GAPS.md) | Replace feed-like if context gaps score ≥ .02 lower and hide more transitions | .5114 vs .4988 (+.0126); hides 39.4% vs 42.1% of transitions | **No go** — keep feed-like as the stress view |
| [Possession-rate diagnostic](POSSESSION_RATE.md) | Flag fragmenting / merging / gap-driven switching | Merging (2.44 poss/min vs 3.44 band floor), weak; consistent with known feed-like under-segmentation | **No go** — no decoder change; cost 3 is the holdout optimum |
| [Target adaptation](TARGET_ADAPTATION.md) | ≥ +.01 feed-like, ≥ −.005 clean, no provider < −.02 | Rescaling −.078 / −.094; self-training −.003 / −.005 | **No go** — neither arm |

**Overall: no go. Keep the adopted package, the feed-like stress view, decoder cost 3 and the
current deployment path unchanged.**

## Key evidence

### Context gaps
- Gap-onset hazard fitted on competition tracking only (149 onsets; AUC .907 on held-out half).
  Onsets concentrate on slow, non-turning balls (2.15× rate in slowest speed quintile, 0.20× in
  fastest) — the opposite of the "ball lost in flight / duels" hypothesis.
- Placement alone (context vs random at equal count/duration): −.0009 spell. Gaps hide
  transitions at ~their coverage rate in every view (hidden/covered .98–1.03).
- The +.012 of both new views over feed-like is unexplained by placement; they lack the real
  feed's ~12% sub-second gaps. Untested.

### Possession rate (full competition match, label-free)
| | poss/min | share < 3 s | switches/min | poss/min ball missing / visible |
| --- | ---: | ---: | ---: | ---: |
| adopted | 2.44 | .194 | 1.54 | .41 |
| previous seed ensemble | 2.35 | .185 | 1.40 | .47 |

- Holdout predicted/true possession ratio on 25 Hz providers under feed-like: .62–.72. Applied
  to the 25 Hz truth range 3.4–5.3, that predicts 2.1–3.8 — the competition's 2.44 sits inside.
- Closest holdout condition: feed-like (RMS z 1.42 vs clean 1.90).
- Decoder cost sweep on holdout feed-like: 2 → .4958, **3 → .4988**, 4 → .4961, 6 → .4904, 8 → .4852.

### Target adaptation
| Arm | Clean spell | Feed-like spell | Worst provider Δ |
| --- | ---: | ---: | ---: |
| none | .6332 | .4988 | — |
| target_norm | .5388 | .4207 | −.1008 |
| self_train | .6283 | .4960 | −.0166 |

Per-match normalization removes between-match feature shifts the model relies on.

## What this changes

- Real-feed ball gaps sit on near-stationary balls, so they are more likely loose/dead-ball
  moments than possession hand-overs. Missing-ball handling is less of a spell-F1 risk than
  assumed; do not prioritise ball re-detection or gap-specific repairs.
- The consistent signal across studies is **under-segmentation**: on 25 Hz data the model predicts
  ~0.6–0.9× true possessions and holds one team through gaps (competition switches drop to
  .39/min while the ball is missing). A cost decrease does not fix it on holdout. A model-side
  change to short-spell/transition recall evaluated on feed-like holdout is the remaining lever.
- Untested follow-up: add the real feed's sub-second gaps to feed-like and re-check the .012 gap.

## Files
Code: `context_gaps.py`, `possession_rate.py`, `target_adaptation.py` with tests in
`experiments/tests/` (26 pass). Aggregates: `results/context_gaps.json`,
`results/possession_rate.json`, `results/target_adaptation.json`. Declarations and per-frame
outputs are private under `data/experiments/{context_gaps,possession_rate,target_adaptation}_v1/`
and were not uploaded to the private cache.

Caveats: context-gap replicate noise is large (J03WOY .547 vs .487 across replicates); provider
truth is event-derived silver; one accidental shape listing of the all-`X` placeholder `labels`
key in `data/challenge/inputs.npz` (no values used).
