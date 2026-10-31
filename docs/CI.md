# PR checks and automatic merges

`CI` runs on every PR targeting `main` and on pushes to `main`. Its `tests`
check validates local documentation links, heading anchors and skill routing,
installs the pinned experiment dependencies and CPU PyTorch 2.8.0, runs the synthetic/contract
test suite, and validates the bundled example submission format. It does not
download organizer assets or score the public five-minute ground truth. A newer
commit on the same PR cancels its older CI run.

After a successful PR run, `Auto-merge passing PRs` finds the open PR attached
to the tested commit and squash-merges it only if its current head is that same
commit, its base is `main`, and its head branch is in this repository. Fork PRs
are deliberately excluded. The privileged workflow does not check out or run
PR code. GitHub's branch protection remains the authority for required checks.
If a PR has conflicts, fails CI, or changes after the successful run, it will
not be merged. Resolve or update it and let CI run again.

For direct-push protection, import
[`.github/rulesets/main.json`](../.github/rulesets/main.json) in the fork's
GitHub **Settings → Rules → Rulesets** page. The JSON targets only `main`, has
no bypass actors, requires a PR and the GitHub Actions `tests` check, and
permits squash merging with zero required human reviews. Importing the file is
a repository setting change; merely committing it does not activate it.

An administrator can instead apply the equivalent branch-protection policy
with [`scripts/configure-main-protection.sh`](../scripts/configure-main-protection.sh):

```sh
gh auth login
bash scripts/configure-main-protection.sh jekkiechan/more-than-a-hack-2027
```

Use either the JSON import or the script, not both. Both require a PR, the
`tests` check, and linear history, and prohibit force pushes and deletion.
The JSON has no bypass actors; the script applies protection to admins. Check
the resulting policy in GitHub Settings. The repository-level GitHub “Allow
auto-merge” toggle is not required because the second workflow performs the
merge after CI succeeds.

Until branch protection has been applied and verified, direct pushes to `main`
remain possible. Agents should still work on separate branches and open PRs.
