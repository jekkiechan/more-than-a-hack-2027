# Initial Challenge submission

Deadline: **17 November 2026, 23:59 CET** (22:59 UTC; 18 November 07:59 JST).
The organizer's 13 November email provides the
[submission form](https://docs.google.com/forms/d/e/1FAIpQLSdmiQBwk8vOMETo44FSbvb_by4fz37zAGNgk_voGd5kP9uBKA/viewform).
One response per team; the last edit before the deadline is scored.

## Reproduce

Use the [technical document](TECHNICAL_DOCUMENT.md#how-to-reproduce) for setup.
From the repository root, with `uv` 0.12.23 installed and tactical ATD in
`data/tactical/`, run on Linux x86-64:

```sh
uv sync --project submission --locked
uv run --project submission --locked bash submission/make_submission.sh 'YOUR_REGISTERED_TEAM_NAME'
```

This downloads and verifies the public
[release weights](https://github.com/jekkiechan/more-than-a-hack-2027/releases/tag/submission-2026-v1),
uses the [frozen model manifest](model_manifest.json), regenerates the input from
raw ATD, runs inference and the official validator, and checks exact CSV bytes.
The [release metadata](weights.json) pins the archive and every member's hash.
No private cache, training-data download or organizer scoring is needed.

The standalone [project](pyproject.toml), [Python version](.python-version) and
[lockfile](uv.lock) pin Python 3.12.14 and all inference dependencies, including
CPU PyTorch. `uv` creates `submission/.venv` without changing the research
environment. The original release tag predates these environment files; use a
checkout containing them and record `git rev-parse HEAD` for subsequent reruns.

Verified on 16 November 2026 in a fresh `submission/.venv`: the locked environment
downloaded and verified the public weights, rebuilt inputs from raw tactical ATD,
and reproduced all 199,319 CSV rows with the exact SHA-256 below. No organizer
reference labels were loaded. A pip `--require-hashes` dry run resolved the same
19 dependency versions as the lockfile.

The archive contains eleven model roles, a manifest and a readme, approximately
9.4 MB compressed. It contains no organizer assets, predictions, training records
or research archive. Learned checkpoint bytes, normalizers and training metadata
are retained unchanged. Download requires no GitHub account or access token.

## Environment maintenance and pip fallback

Treat `pyproject.toml` and `uv.lock` as the dependency source of truth.
`requirements.txt` is a generated export with exact versions and hashes; do not
edit it by hand. After an intentional dependency change, run from the repo root:

```sh
uv lock --project submission
uv export --project submission --locked --no-dev --emit-index-url --format requirements-txt --output-file submission/requirements.txt
```

Commit the project, lockfile and export together. Revalidate the full pipeline's
CSV hash before adopting an environment change. CI checks lock freshness and
export parity without accessing organizer data.

If `uv` is unavailable, install Python **3.12.14**, then on Linux x86-64:

```sh
python3.12 -m venv submission/.venv-pip
submission/.venv-pip/bin/python -m pip install --require-hashes -r submission/requirements.txt
PYTHON=submission/.venv-pip/bin/python bash submission/make_submission.sh 'YOUR_REGISTERED_TEAM_NAME'
```

The export includes the CPU PyTorch index and package hashes, so no separate
torch installation is needed. Unlike `uv`, pip does not enforce the project's
Python patch version or per-package index mapping; use the stated Python/platform
and retain `--require-hashes`. Keep fallback environments out of version control.

## Final checklist

- Confirm the exact registered team name, all members' registrations under that
  name (required before November 10), and team size of 3–5 for full eligibility.
- Produce `TEAM_NAME.csv`; verify 199,319 rows, frames 0–199318, H/A/D/X, and
  SHA-256 `8e26f823a3e933f1feca2a64dfe6d7f728a4741548c8c7cd95d51746b6a355b9`.
- Copy [TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md) to `TEAM_NAME.md`. The rules
  allow Markdown; no PDF conversion is needed.
- Confirm the public repository URL and release downloads work without login.
- Use a Google account to upload the CSV and technical document, enter the exact
  team name and public repository URL, and submit once per team.
- Save the submission confirmation and edit-response link privately. Revalidate
  the exact CSV before any replacement upload; there is no leaderboard or feedback.

Organizer requirements remain authoritative: [SUBMISSION](../SUBMISSION.md) and
[RULES](../RULES.md). Never publish the match data or frame-level predictions to
GitHub. The CSV is delivered through the organizer form only.
