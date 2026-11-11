# Ball-exit promotion and crossing-time follow-up

11 November 2026. Fable's `snap_m0.0` remains the selected candidate. Its
development improvement reproduces exactly; two fixed extensions do not improve
v2 spell. **Promotion and GAR/public regression results are blocked on two
missing pooled deployment checkpoints. The active default has not changed.**

The user explicitly requested promotion, further development, and GAR TEST/public
regression checks. This work builds on [Fable's report](BALL_EXIT_DMASK_V1.md)
at public PR #91, commit `9a4e7f0f2e5c347b679322a9483a44b5db3490a1`.
The new study is `ball_exit_crossing_v1`; the pending delivery is
`ball_exit_promotion_v1`.

## Matched development result

| Rule | Clean spell | V1 spell | V2 spell | V2 short-spell recall | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Existing frontier | .670137 | .532128 | .593619 | .288604 | Control |
| Fable snap | **.681508** | **.534731** | **.598690** | **.307543** | Retain unchanged |
| Geometric crossing | .681579 | .534764 | .598687 | .307120 | Fails v2 spell and short-recall retention |
| Observation midpoint | .681574 | .534797 | .598188 | .306697 | Fails v2 spell and short-recall retention |

Snap improves clean/v1/v2 spell by .011371/.002602/.005071 over the frontier.
Its macro F1 is .876204/.827310/.854915, and D-F1 is
.875558/.861220/.869770. The >.60 v2 target remains unmet by .001310.

Snap's v2 spell is .714679 Metrica, .478743 SkillCorner and .602647 Sportec
(see the JSON for full precision). Fable's provider tradeoff remains: Metrica
and Sportec improve, while SkillCorner declines under its earlier silver
dead-ball convention. No provider-specific switch is introduced.

The extension asks whether missing observations make the first outside sighting
late. Both variants retain snap's D run ends and adjust only a start bracketed
by the immediately preceding inside observation, within the inherited five-second
window and the same time/period block. One estimates the straight-line crossing
of the pitch rectangle; the other uses the observation midpoint. There is no
fitting, provider condition, threshold sweep, or new time-window search.
The geometric variant's v2 difference is only −.000003; this is a failure of the
declared improvement rule, not evidence of a practically meaningful degradation.

All five already-exposed matches, clean, two historical v1 realizations, and
four fixed v2 realizations are included: 35 cells per arm. Realizations are
averaged within match, then matches within provider, then providers equally.
The original frontier and Fable snap both replay all saved cell metrics within
the predeclared 1e-12 tolerance. A separate process rechecked every saved output:
**140 prediction sets and 840 scalar comparisons, maximum error 0**.

The [declaration](results/ball_exit_crossing_v1_declaration.json) was committed
before scoring at `22ad321`; it records the user's explicit further-work request
after Fable's prior stop rule. The
[result](results/ball_exit_crossing_v1.json) and
[verification](results/ball_exit_crossing_v1_verification.json) preserve both failed
extensions. Stop this bounded comparison. No GAR/public results informed it.
V2's label-conditioned synthetic gaps and predecessor-conditioned calibration
remain limitations; these are not independent continuous-match gold results.

## Exact deployment artifact blocker

The [asset audit](results/ball_exit_promotion_v1_asset_audit.json) restores and
verifies eight of the ten checkpoint roles against the active frontier manifest.
The two missing files are:

| Role | Required SHA-256 |
| --- | --- |
| `ce_2026.joblib` | `d9691336672b13191bf2ce47715fc904f3bc6c14b8477b2f6ff2425d9f78d9c2` |
| `ce_2027.joblib` | `4592c10dc06c947077f6d8569833c6d229d3a5620350740edd26fada6d83b0fa` |

They belong in `data/models/joint_gru_sn_dense_spell_frontier_ensemble_v1/`.
The original archive is
`joint_gru_sn_dense_spell_frontier_ensemble_v1_local.tar.gz`, hash
`44c13a81d2facf56007e91ae0706309aa8b7b44aabdd4a832ea30b45e86c3422`.
The historical [local packaging receipt](results/baseline_spell_frontier_local_artifacts.json)
records it, but it is absent from the available private refs.

After the user reported new availability, private refs were refreshed. The new
upload at `1c722275098c4c13e14d4783cdf90a0faaa0e34e` contains four
`sc2020_realfeed_v1/rb8_1024` provider-fold checkpoints. These are the different,
negative SkillCorner data experiment, not the pooled frontier CE checkpoints.
They cannot substitute for the deployment package. The separate 1.08 GB frontier
recovery archive and all 160 members were restored and verified; it contains
the development folds and predictions, explicitly excluding pooled weights.

Running `python -m experiments.ball_exit_delivery stage` fails at the missing
`ce_2026.joblib` before writing a staged manifest or accessing organizer labels.
No public gate has been claimed, no GAR TEST was scored, and no new weights or
frame outputs were uploaded. Historical reports, declarations, runner code,
active manifest and active policy are preserved.

## Implementation and continuation

[The runtime adapter](ball_exit_runtime.py) calls Fable's frozen functions with
the exact `snap`, zero-margin, five-second delay and three-second advance rule.
It applies observation masks and time discontinuities, then uses the unchanged
frontier H/A mixture. The normal baseline dispatcher supports the new versioned
architecture. The crossing variants are experimental only.

[The delivery runner](ball_exit_delivery.py) requires exact checkpoint hashes,
replays the old full CSV before sealing a new one, and provides a separate
exclusive public gate. [The GAR runner](ball_exit_gar.py) checks the exact ten
TEST games using clean plus the same four corruption realizations; it replays
the saved frontier metrics before candidate scoring and verifies saved native
predictions before reporting. VALID is not pooled into the result. These
deployment paths are implemented and contract-tested where possible, but their
end-to-end numerical behavior is unverified until the missing files are restored.

After restoring the exact two files and the declaration-pinned GAR TEST raw
files from the [authorized source](SOCCERNET_DATA_ACCESS.md), run:

```sh
python -m experiments.ball_exit_delivery stage
python -m experiments.ball_exit_gar prepare
python -m experiments.ball_exit_gar run
python -m experiments.ball_exit_delivery predict
python -m experiments.ball_exit_delivery public
python -m experiments.ball_exit_delivery activate
```

Activation requires the bound GAR/public reports. A spell regression stops
activation for a disclosed decision; it never triggers tuning on those labels.
The public baseline is the saved .611765 spell / .851285 macro score on 17
possessions. No conclusion about regression is available yet.

## Delivery checks and site update

The 21 focused ball-exit tests pass, covering Fable parity, observation masks,
time breaks, target independence, geometric crossing, missing-weight failure,
source/seal checks and exclusive scoring gates. Full-suite results and the PR
are recorded in the final delivery receipt after completion.

The research site was readable at version 37 with its custom restricted
audience. The packaged Sites source helper is absent from this environment.
Automatic approval review rejected an ordinary Git clone using the site's
scoped credential, including after the repository's explicit delivery instruction
was cited: it required direct authorization for the separate private site.
No site source was accessed or edited, and the audience is unchanged.

The pending site update is this report's result table and interpretation on
Experiments, plus a Progress note: “Fable's .598690 v2 result reproduced; two
crossing-time extensions did not improve it. Promotion and GAR/public regression
checks await the exact pooled CE weights.” It must not mark the model adopted
or display unmeasured GAR/public gains. Publication remains blocked on approval.
