# Provider definition alignment and touch-rule calibration plan

**State: protocol scaffold only; no calibration has run.** The 31 October 2026
research status identifies fitting SoccerNet's touch rule to provider possession
definitions and testing source-conditioned heads as the next step. This note
establishes the evidence boundary and a bounded calibration protocol. It does
not report fitted parameters, model scores, or confirmation evidence.

## What the sources actually label

| Source | Available possession evidence | What it does not provide |
| --- | --- | --- |
| SoccerNet-GAR `tracking-full` | Per-frame `game_event_type`, `game_event_id`, `possession_event_type`, and `home_team`; sparse PASS/CROSS/SHOT clip-centre team anchors. Dense labels in `soccernet_dense_labels.py` infer touch runs from `OTB` and `*KICKOFF`, then interpret CH/RE/CL and OUT. | A documented `controlled` boolean or a provider possession-run annotation. The run-control flag is inferred by this repo's rule. |
| Sportec Open DFL | Framewise `BallStatus` and `BallPossession` in DFL XML; official live/dead and team target used for the Sportec benchmark. The evaluation manifest lists three TRAIN matches (J03WMX/J03WN1/J03WOH) and one previously evaluated selection match (J03WOY). | An event-run stream with touch IDs and controlled/uncontrolled run-end semantics. |
| Metrica | Human event stream (CSV for Games 1–2, JSON for Game 3) with team, event type/subtypes, period, start and end frames. `metrica_open.py` derives H/A/D from possession and stoppage event semantics. | GAR-style OTB touch IDs or controlled/uncontrolled touch-run labels. Its target is an event-derived interval interpretation. |
| SkillCorner | `possession.group` H/A/null per tracking frame and `dynamic_events.csv` possession events with interruption-before/after tags. `skillcorner_had.py` derives conservative D cores with X margins. | GAR-style touch-run IDs or controlled/uncontrolled touch-run labels; null possession is not itself D. |

Evidence is in `soccernet_dense_labels.py` and `SOCCERNET_DENSE.md`,
`BENCHMARK_AUDIT.md`, `metrica_open.py`, `datasets.py`, `skillcorner_had.py`,
and `OPEN_DATA.md`. Historical reports, manifests, and acquisition receipts
document previously restored or processed data. In this session's checkout,
there are no current provider event or label caches under `data/external` or
`data/independent`; the only challenge-derived input present is the known
`data/challenge/inputs.npz`. Therefore this task did not inspect provider frame
outputs or run a calibration. This is a session-local availability statement,
not a claim that these datasets were never acquired or inspected historically.

These streams cannot be joined as paired observations: the GAR games and
provider benchmark matches are different fixtures. Nor can a provider's dense
reference be converted into GAR run metadata without inventing unobserved event
semantics. In particular, Sportec BallStatus is a useful possession-definition
reference, not a source of touch-run control labels. Any future mapping must
keep native provider targets as references and report the semantic assumptions
separately from measurable agreement.

## Bounded next study

The first stage should be a mapping audit, not parameter fitting. For each
source, document event/target field meanings, clock alignment, team identity,
stoppage start/restart timing, unknown intervals, and which semantics are
directly observed versus inferred. Implement adapters only for documented
fields, then verify candidate labels against source references on each source's
own whole matches. Report H/A/D frame confusion, live possession-spell
boundaries with the official scorer, coverage/unknown rate, and boundary timing
errors; do not call cross-fixture agreement paired calibration.

Once an adapter supplies comparable candidate outputs and a declared loss,
`provider_touch_calibration.py` exposes a fixed 24-arm grid: three end-code
rules (v1's CH/RE on any run and CL on single-frame runs; CH/RE/CL on any run;
none), two opponent-touch policies (hold or literal), censored-run handling
on/off, and OUT tail grace of 0 or 3 seconds. The exact v1 recipe (CH/RE any
run; CL single-frame only; hold; censored as controlled; 3 s) is included.
The grid is a sensitivity design, not evidence that all arms are semantically
valid. Select by equal-weight mean provider loss using eligible training-source
matches; when a provider is held out, the available mapped source set may have
only one eligible provider (for example, holding out Sportec can leave Metrica
as the only source with a usable event mapping; SkillCorner currently has no
touch-run mapping). The selector accepts one or more eligible providers.
Exclude the held-out provider and every one of its matches from selection,
normalization, and any learned head. The aggregate selector cannot enforce
whole-match separation within an included provider: callers must provide
training-match-only scores, record match IDs, and use an inner whole-match
split where candidate choice depends on match-level validation. Preserve
match/half boundaries, native unknowns, and provider-native evaluation
definitions. Report each provider separately and do not pool frame counts
across providers.

Because the three providers have substantially different schemas and only a
small match cohort, begin with two distinct questions: (1) whether the fixed
GAR-derived rule's observable outputs are consistent with each source's
reference semantics, and (2) whether a source-conditioned model head improves
within-provider transfer under whole-match exclusion. Do not combine those
claims. A source label/head feature may encode dataset identity; audit whether
the inference source is known at deployment and whether each head is trained
with enough independent matches. The present five development matches are
already exposed and cannot serve as fresh confirmation.

The helper `select_by_provider_mean` only enforces training-provider exclusion,
matching candidate sets, finite numeric losses, and equal weighting across the
eligible providers supplied after losses have been computed by a future
adapter. It intentionally cannot verify match-level provenance or manufacture
losses or provider touch annotations. Synthetic contracts are in
`tests/test_provider_touch_calibration.py`.
