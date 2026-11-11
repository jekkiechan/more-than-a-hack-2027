# Ball-exit promotion and crossing-time follow-up

11 November 2026. Fable's `snap_m0.0` remains the selected candidate. Its
development improvement reproduces exactly; two fixed extensions do not improve
v2 spell. **Fable is now the user-authorized provisional default. Exact deployment
verification and GAR/public regression results remain blocked on two missing
pooled checkpoints.**

The user explicitly requested promotion, further development, and GAR TEST/public
regression checks. This work builds on [Fable's report](BALL_EXIT_DMASK_V1.md)
at public PR #91, commit `9a4e7f0f2e5c347b679322a9483a44b5db3490a1`.
The new study is `ball_exit_crossing_v1`; the pending delivery is
`ball_exit_promotion_v1`.

After the user requested complementary improvements instead, the separate
[turnover composition study](BALL_EXIT_COMPLEMENT.md) retained a small research
gain to .598780 v2. It remains research-only. The user then delegated the choice
with “promote the version you think is best.” We adopted unchanged Fable as a
simplicity and deployment-cost tradeoff: the complement adds only .000090 on
exposed data and requires additional pooled models. See the
[adoption receipt](results/ball_exit_adoption_v1.json).

The active manifest and policy now select Fable, while the predecessor manifest
and policy remain archived. This explicitly separates the adoption decision from
exact deployment readiness. No GAR/public pass, new full-match CSV hash, or
restored missing weight is implied. The earlier delivery/publication receipts
remain unchanged as records of their original states.

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

Before adoption, `python -m experiments.ball_exit_delivery stage` failed at the
missing `ce_2026.joblib`. After adoption, use `python -m experiments.ball_exit_adoption stage`;
it preserves the same exact-weight requirement before writing a diagnostic
manifest or accessing organizer labels.
No public gate has been claimed, no GAR TEST was scored, and no new weights or
frame outputs were uploaded. Historical reports, declarations, runner code,
predecessor manifest and predecessor policy are preserved; the active pointers
were explicitly advanced by the later user-authorized provisional adoption.

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
python -m experiments.ball_exit_adoption stage
python -m experiments.ball_exit_gar prepare
python -m experiments.ball_exit_gar run
python -m experiments.ball_exit_delivery predict
python -m experiments.ball_exit_delivery public
```

The adopted manifest is staged byte-for-byte only after its exact sources and
weights verify. The existing prediction, exclusive public gate and GAR runners
remain unchanged. Do not call the older `activate` action: adoption has already
changed the default. Record a separate deployment-verification receipt after
reviewing the bound reports. Any regression must be disclosed and reviewed; it
never triggers tuning on those labels or a silent change of candidate.
The public baseline is the saved .611765 spell / .851285 macro score on 17
possessions. No conclusion about regression is available yet.

## Delivery checks and site update

The 28 focused ball-exit tests pass, covering Fable parity, observation masks,
time breaks, target independence, geometric crossing, missing-weight failure,
source/seal checks, exclusive scoring gates and complementary H/A composition.
The final full suite has 958 tests: 952 pass, one skips, and five inherited
public-contract tests require absent historical local-only seal/CSV fixtures.
The 26 root tests, documentation check, file-backed fit contract and bundled
submission format check pass.

The user explicitly approved promotion, then delegated the final version choice.
[The adoption tool](ball_exit_adoption.py) changes only the default manifest/policy
and archives their predecessors; it preserves the frozen deployment gates.
Its separate `stage` action still requires all exact weights.

The user explicitly approved the prepared delivery. The implementation
and aggregate evidence are published in [PR #92](https://github.com/jekkiechan/more-than-a-hack-2027/pull/92).
The [research site](https://more-than-a-hack-progress.tanapat-ratana.chatgpt.site/)
is published as version 38 with its existing restricted audience. Experiments,
Progress and the model timeline now distinguish the approved Fable target,
the small complementary research candidate, and pending GAR/public checks.

The packaged Sites source helper was absent, so ordinary Git with an ephemeral
scoped credential and a matching static deployment archive was used. Desktop
(1440 px) and mobile (390 px) browser checks pass on both changed pages: no page
overflow or script errors, links/IDs valid, and method disclosure works.
The [publication receipt](results/ball_exit_delivery_publication_v1.json) records
the source, version and successful deployment. Earlier automatic-review blockers
were resolved by the explicit user approval. That version-38 publication predates
the later provisional adoption; the adoption site receipt records the subsequent
update. Exact deployment and GAR/public regression checks remain pending.

The post-adoption [check receipt](results/ball_exit_adoption_v1_checks.json) records
961 experiment tests: 955 pass, one skips and the same five historical public
fixture errors remain. All 26 root tests, file-backed fit parity, bundled CSV
format, documentation links and adopted source/predecessor hashes pass.

The subsequent [site publication](results/ball_exit_adoption_v1_site_publication.json)
is version 39, covering the provisional default and historical architecture
labels. The restricted audience is unchanged. PR #92 is ready for review; it
has not been merged, and no required CI checks were bypassed.

## Verification before final promotion — Luna review

The user requested verification before final promotion, using Luna subagents.
The [new verification receipt](results/ball_exit_verification_v2.json) supersedes
the earlier software-test blocker: **964 tests run, 963 pass, one skip, zero
errors**. The historical public contract tests now use isolated synthetic CSV,
seal and recipe fixtures, with additional tamper/closed-gate/overwrite checks.
Production runners, hard-coded historical hashes and saved public artifacts
are unchanged. All 26 root tests, file-backed fit parity and format checks pass.

A fresh Luna replay independently verifies **140 saved prediction sets and 840
scalar metrics, zero error**, including aggregates, guards and snap selection.
The active/versioned manifests match and all 16 declared source hashes verify.
The separate GAR restore recovered all four declaration-pinned TEST files
(1,093,146,606 bytes) at revision `32c840210b7d08d06976e7f9457dfff79fb2da99`.
Every raw SHA-256 was checked again by the parent agent. No GAR/public scoring.

Luna refreshed every private branch and PR ref and searched both tree paths
and LFS identities. Neither original CE checkpoint nor the original deployment
archive was found. The user confirmed that PR #19 uploaded the wrong experiment:
`rb8_1024` belongs to the non-retained SkillCorner study; Fable fitted no new
weights. An originating agent is now locating the correct frontier files.
The original pooled training seal identifies these exact source paths:

- `data/experiments/v2_spell_balanced_v1/train/models/pooled/stage1024_2026.joblib`
  → deployment `ce_2026.joblib`, SHA-256 `d9691336672b13191bf2ce47715fc904f3bc6c14b8477b2f6ff2425d9f78d9c2`.
- `data/experiments/v2_spell_balanced_v1/train/models/pooled/stage1024_2027.joblib`
  → deployment `ce_2027.joblib`, SHA-256 `4592c10dc06c947077f6d8569833c6d229d3a5620350740edd26fada6d83b0fa`.

Each file is 1,432,445 bytes. Preserve the unrelated rb8 archive as its own
experiment; it cannot satisfy these identities. PR #92 is held as a draft until
exact-package verification completes. The existing provisional default remains
recorded, but no merge or completed deployment promotion is claimed. The exact
next step is restore these two files, then run the adopted staging, GAR and
public workflow above, review its results and record final promotion separately.
