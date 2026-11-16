# Initial Challenge submission

Deadline: **17 November 2026, 23:59 CET** (22:59 UTC; 18 November 07:59 JST).
The organizer's 13 November email provides the
[submission form](https://docs.google.com/forms/d/e/1FAIpQLSdmiQBwk8vOMETo44FSbvb_by4fz37zAGNgk_voGd5kP9uBKA/viewform).
One response per team; the last edit before the deadline is scored.

## Reproduce

Use the [technical document](TECHNICAL_DOCUMENT.md#how-to-reproduce) for setup.
From the repository root, with the pinned environment active and tactical ATD in
`data/tactical/`, run:

```sh
bash submission/make_submission.sh 'YOUR_REGISTERED_TEAM_NAME'
```

This downloads and verifies the public
[release weights](https://github.com/jekkiechan/more-than-a-hack-2027/releases/tag/submission-2026-v1),
uses the [frozen model manifest](model_manifest.json), regenerates the input from
raw ATD, runs inference and the official validator, and checks exact CSV bytes.
The [release metadata](weights.json) pins the archive and every member's hash.
No private cache, training-data download or organizer scoring is needed.

The archive contains eleven model roles, a manifest and a readme, approximately
9.4 MB compressed. It contains no organizer assets, predictions, training records
or research archive. Learned checkpoint bytes, normalizers and training metadata
are retained unchanged. Download requires no GitHub account or access token.

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
