# Agent guide: possession research

## Start here

1. Read [current status](docs/research/STATUS.md) for adoption, next work and blockers.
2. Use [the experiment index](docs/research/EXPERIMENTS.md) to find prior evidence;
   read the relevant report rather than historical handoffs.
3. Working in `experiments/`? Read its [AGENTS](experiments/AGENTS.md) and
   task-relevant [code map/vocabulary](experiments/README.md).
4. [BASELINE](experiments/BASELINE.md) owns restore/inference commands.
   `experiments/baseline_manifest.json` identifies the adopted package;
   `experiments/model_decision_policy.json` defines active ranking and guards.
   Check current `main` and relevant open PRs before current-results advice;
   follow the readout skill below. Saved Pages and historical verdicts may lag.

## Load a skill when its task applies

| Task | Skill |
| --- | --- |
| Missing datasets, weights or private cache | [possession-restore](.agents/skills/possession-restore/SKILL.md) |
| New feature preparation or provider splits | [possession-preparation](.agents/skills/possession-preparation/SKILL.md) |
| New experiment, fitting or evaluation | [possession-experiment](.agents/skills/possession-experiment/SKILL.md) |
| Scores, model choice or adoption advice (including chat) | [possession-readout](.agents/skills/possession-readout/SKILL.md) |

## Constraints that always apply

- [CHALLENGE](CHALLENGE.md), [EVALUATION](EVALUATION.md),
  [SUBMISSION](SUBMISSION.md), and [RULES](RULES.md) define the task:
  199,319 consecutive `frame,label` rows at 25 Hz, H/A/D/X. Loose or contested
  ball stays with the last controlling team. Use the official evaluator.
- Organizer public labels are pipeline checks only (7,500 frames, 17 possessions).
  Never train, tune, select candidates, thresholds, repairs or seeds on them.
  Do not manually label/correct this match or use human-produced event/possession
  data about it. Label-free tracking improvements are allowed.
- Read saved diagnostics; never reset a closed scoring gate or rewrite a frozen
  declaration. A fresh directory does not make inspected data fresh evidence.
- Split by whole match/provider. Identify which learned components are held out;
  D-only holdout with pooled H/A is not full-pipeline holdout. Audit pretrained
  checkpoint exposure and source-specific label/missingness shortcuts.
- Keep organizer assets, reconstructive derivatives, weights and frame outputs
  out of this public repo. Use ignored `data/`, `MTAH_DATA_DIR`, or the authorized
  private cache. Publish aggregate evidence. Restore verified artifacts before
  retraining; preserve native masks, unknown targets and match/half/time breaks.

## Delivery and checks

Work on a branch in `jekkiechan/more-than-a-hack-2027`; PRs target the fork's
`main`. Preserve `upstream` as `metrica-sports/more-than-a-hack-2027` when configured.
Publish an authorized PR without a separate confirmation step. Documentation
maintenance does not promote models. Existing [CI/auto-merge](docs/CI.md) applies.
[Documentation ownership](docs/research/README.md) explains what to update.

After a promising experiment completes, update the
[experiment progress Page](https://chatgpt.com/space/page_1df518ea32448191bf885da4952259a9)
with dated scores, visualizations, protocol and adoption status.

For code changes, run `python -m unittest discover -s experiments/tests -v`
with pinned dependencies (including CPU PyTorch for neural contracts), plus
applicable checks in [CI](docs/CI.md). Validate generated CSVs with
`python evaluation/validate_submission.py <candidate.csv>`.
For docs-only changes, check links, evidence and frozen-file preservation;
CI runs the synthetic suite and bundled format check. Do not fit or score
organizer labels to verify documentation.
