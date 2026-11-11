# Ball-exit promotion and crossing-time follow-up

11 November 2026. Fable's `snap_m0.0` remains the selected candidate. Its
development improvement reproduces exactly; two fixed extensions do not improve
v2 spell. The frozen pooled recipe has now been reconstructed in four fits, and
the guarded runner installed the reconstructed ten-role package as the working
default on this research branch. The original pooled files remain unavailable
and the reconstructed weights are not byte-identical to them. The
[promotion receipt](results/ball_exit_reconstructed_v1.json) binds the active
manifest/policy and completed control-CSV, public and GAR checks. PR #92 is open and unmerged; this branch's active package has not reached main.
The normal default CLI reproduces the sealed Fable CSV byte-for-byte
(`56425990299fcdda15bf799cfc7384d53f444885e8141ebdd600965f64c5274f`), and
post-promotion checks pass 977 tests with one skip, plus all 26 root tests,
file-backed parity, submission format and the independent fit audit.

The GAR TEST10 result is a repeated descriptive check, not fresh confirmation:
all four aggregate measures improve in clean and v2, with spell/macro/D-F1/short
recall `.709186/.784717/.593243/.264521` clean and
`.709176/.784865/.596512/.263570` v2, against `.708496/.780580/.583474/.263398`
and `.708619/.782179/.589851/.262769` controls. Two games decline in each view.
The reused public diagnostic moves spell/macro/D-F1 from
`.611765/.851285/.862870` to `.623529/.864234/.889372` on 7,500 frames and 17
possessions. No tuning followed either result; TEST labels are silver and this
is not independent continuous-match gold evidence. Full detail is in the
[reconstruction report](FRONTIER_POOLED_RECONSTRUCTION.md).

The user explicitly requested promotion, further development, and GAR TEST/public
regression checks. This work builds on [Fable's report](BALL_EXIT_DMASK_V1.md)
at public PR #91, commit `9a4e7f0f2e5c347b679322a9483a44b5db3490a1`.
The development study is `ball_exit_crossing_v1`; deployment continuation was
recorded later in `ball_exit_reconstructed_v1`.

After the user requested complementary improvements instead, the separate
[turnover composition study](BALL_EXIT_COMPLEMENT.md) retained a small research
gain to .598780 v2. It remains research-only. The user then delegated the choice
with “promote the version you think is best.” We adopted unchanged Fable as a
simplicity and deployment-cost tradeoff: the complement adds only .000090 on
exposed data and requires additional pooled models. See the
[adoption receipt](results/ball_exit_adoption_v1.json).

The earlier adoption receipt selected the Fable rule while keeping the
predecessor manifest and policy archived. The later guarded promotion installed
the reconstructed pooled package; it did not recover the original CE weights.
The reconstruction and current active manifest are recorded in the receipt linked
at the top. Earlier delivery/publication receipts remain unchanged as historical
records of their original states.

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

Historical record: the [asset audit](results/ball_exit_promotion_v1_asset_audit.json)
restored eight of the ten roles against the archived frontier manifest. These
two original files remained unavailable; the current package uses separately
identified reconstructed replacements:

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

Earlier exact-weight staging attempts failed because the original
`ce_2026.joblib` was missing. The reconstructed models were then fitted in the
separate bounded study, evaluated under the already-closed public gate and on
the fixed GAR TEST10 cohort, and activated only after their receipts passed.
The staged private manifest remains unchanged. The historical original hashes
above remain useful provenance; they are not claimed for reconstructed files.
No closed gate was reset, and no public/GAR result informed fitting or candidate
selection. The fresh private-branch package excludes raw inputs and frame
predictions; see the reconstruction report for its restore instructions.

## Implementation and continuation

[The runtime adapter](ball_exit_runtime.py) calls Fable's frozen functions with
the exact `snap`, zero-margin, five-second delay and three-second advance rule.
It applies observation masks and time discontinuities, then uses the unchanged
frontier H/A mixture. The normal baseline dispatcher supports the new versioned
architecture. The crossing variants are experimental only.

The fixed full-control prediction CSV matches its saved reference byte for byte.
The separate [reconstruction study](FRONTIER_POOLED_RECONSTRUCTION.md) records
all four fit identities, the complete fixed public diagnostic and GAR TEST10
replay. Its promotion receipt records installation on this research branch.
The public baseline remains the saved `.611765` spell / `.851285` macro score on
17 possessions; the observed Fable result is reported above. Any future
evaluation remains descriptive unless supported by distinct gold data, and
must not trigger tuning on these reused labels.

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

At that time, the packaged Sites source helper was absent, so ordinary Git with
an ephemeral scoped credential and a matching static deployment archive was used. Desktop
(1440 px) and mobile (390 px) browser checks pass on both changed pages: no page
overflow or script errors, links/IDs valid, and method disclosure works.
The [publication receipt](results/ball_exit_delivery_publication_v1.json) records
the source, version and successful deployment. Earlier automatic-review blockers
were resolved by the explicit user approval. That version-38 publication predates
the later provisional adoption; the adoption site receipt records that subsequent
update. The later reconstructed deployment checks are documented above.

The post-adoption [check receipt](results/ball_exit_adoption_v1_checks.json) records
961 experiment tests: 955 pass, one skips and the same five historical public
fixture errors remain. All 26 root tests, file-backed fit parity, bundled CSV
format, documentation links and adopted source/predecessor hashes pass.

The subsequent [site publication](results/ball_exit_adoption_v1_site_publication.json)
was version 39, covering the then-provisional default and historical architecture
labels. The restricted audience is unchanged. PR #92 remains unmerged; its current
branch state is summarized at the top of this report.

## Historical verification before pooled reconstruction — Luna review

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
Every raw SHA-256 was checked again by the parent agent. At that stage there had
been no GAR/public scoring; the later fixed results are reported above.

Luna refreshed every private branch and PR ref and searched both tree paths
and LFS identities. Neither original CE checkpoint nor the original deployment
archive was found. The user confirmed that PR #19 uploaded the wrong experiment:
`rb8_1024` belongs to the non-retained SkillCorner study; Fable fitted no new
weights. The original files were never found; the separately trained replacements are
identified in the reconstruction report.
The original pooled training seal identifies these exact source paths:

- `data/experiments/v2_spell_balanced_v1/train/models/pooled/stage1024_2026.joblib`
  → deployment `ce_2026.joblib`, SHA-256 `d9691336672b13191bf2ce47715fc904f3bc6c14b8477b2f6ff2425d9f78d9c2`.
- `data/experiments/v2_spell_balanced_v1/train/models/pooled/stage1024_2027.joblib`
  → deployment `ce_2027.joblib`, SHA-256 `4592c10dc06c947077f6d8569833c6d229d3a5620350740edd26fada6d83b0fa`.

Each file is 1,432,445 bytes. Preserve the unrelated rb8 archive as its own
experiment; it cannot satisfy these identities. This was the pre-reconstruction
handoff state: PR #92 was held as a draft while exact-package checks remained
blocked. The later reconstruction completed the fixed checks and was promoted on
the research branch, without changing these historical identities.

The [verification site update](results/ball_exit_verification_v2_site_publication.json)
is published as version 40 with the same restricted audience. It reports the
passing software/development checks and restored GAR inputs, and explicitly
left exact-package checks and final promotion pending at that time. Desktop/mobile checks
pass on Overview, Experiments and Architecture.
