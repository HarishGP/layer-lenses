---
name: repo-upload-rules
description: What gets pushed to GitHub (1 GB cap), which runs keep tracked replay weights, the Oct 23 history rewrite and its backup branch, gitignore pitfalls
metadata:
  type: project
---

The pushed repository (github.com/HarishGP/layer-lenses, branch main) must stay under 1 GB. On 2026-10-23 it was 0.88 GiB after
rewriting the 128 unpushed commits to drop every replay weights.npz / optimizer_state.npz except those of runs 60, 70, 73, 74 (the runs
the two_layer_network_anatomy notebook must load from a fresh clone, 173 MB). The pre-rewrite history is kept locally as the branch
`backup/main-before-filter` (2.9 GiB reachable; delete it to reclaim space). Local replay files of the other runs stay on disk, ignored.

**Why:** the user said "every other stored run can be ignored in the upload" and asked for a 1 GB repository; GitHub also warns above
50 MB per file (run 73's weights are 57 MB, accepted).

**How to apply:** never commit replay weights of a new run unless the user names it; `.gitignore` ignores `results/designed_init/*/replay/
weights.npz` and `optimizer_state.npz` globally, with `!` negations for the four runs after that rule. Git has no brace expansion in
.gitignore, so use one character-class pattern per decade (`hd2_run10[7-9]_a13_*`). Clear all notebook outputs before committing
(`notebooks/old notebooks/` is read-only and was not touched). Before a push, measure with
`git rev-list --objects origin/main..main | git cat-file --batch-check='%(objectsize:disk) %(rest)'`. See [[folder-size-watch]].
