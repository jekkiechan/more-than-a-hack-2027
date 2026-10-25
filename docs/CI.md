# PR checks and automatic merges

`CI` runs on every PR targeting `main` and on pushes to `main`. Its `tests`
check installs the pinned experiment dependencies, runs the synthetic/contract
test suite, and validates the bundled example submission format. It does not
download organizer assets or score the public five-minute ground truth.

After a successful PR run, `Auto-merge passing PRs` finds the open PR attached
to the tested commit and squash-merges it only if its current head is that same
commit, its base is `main`, and its head branch is in this repository. Fork PRs
are deliberately excluded. The privileged workflow does not check out or run
PR code. GitHub's branch protection remains the authority for required checks.
If a PR has conflicts, fails CI, or changes after the successful run, it will
not be merged. Resolve or update it and let CI run again.

For direct-push protection, an administrator must apply
[`scripts/configure-main-protection.sh`](../scripts/configure-main-protection.sh)
after the first `CI` run exists on `main`:

```sh
gh auth login
bash scripts/configure-main-protection.sh jekkiechan/more-than-a-hack-2027
```

This requires a PR, the `tests` check, linear history, and applies the rules
to administrators. It disables force pushes and deletion. It permits zero
required reviews so passing same-repository PRs can merge without a person
approving each one. Check the resulting protection page under GitHub repository
Settings → Branches. The repository-level GitHub “Allow auto-merge” toggle is
not required because the second workflow performs the merge after CI succeeds.

Until branch protection has been applied and verified, direct pushes to `main`
remain possible. Agents should still work on separate branches and open PRs.
