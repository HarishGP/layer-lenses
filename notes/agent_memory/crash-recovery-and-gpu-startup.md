---
name: crash-recovery-and-gpu-startup
description: The machine crashed on 2026-10-23 (git objects left empty, runs died); how the repo was repaired and why at most two runs may start on the GPU at the same time
metadata:
  type: project
---

On 2026-10-23 ~19:06 IST the machine went down (rebooted 2026-11-03): the commit made a minute earlier lost its object files (empty
files under .git/objects), tmux sessions and runs died, and /tmp (the session scratchpad, tectonic binary) was wiped.

**Why:** the WSL2 VM has only 15 GB of RAM (+4 GB swap) and resets when it runs out; a runner process needs ~5 GB of host RAM while
it builds its 10M-point data set (steady state 1.5 GB since the Nov 3 patch that frees the host copy; 5 GB before), so three runs
starting together crashed the machine twice (Oct 23 and Nov 3). Git objects written just before a crash can be zero-length (ext4
delayed allocation). On the GPU each run holds ~4.7 GB, so at most three fit on the 16 GB card anyway.

**How to apply:** after a crash run `find .git/objects -type f -size 0 -delete`, `git fsck`, point main at the last intact commit
(`git update-ref`), `rm .git/index && git reset -q`, and delete reflog lines naming the lost commit; the working tree keeps the latest
files, so recommit them. Never start two runs within the same two minutes; keep at most two concurrent (three is possible on the GPU but leaves no
margin for the host build peak); use a chain script that waits for `pgrep -f run_symmetric_gd.py` to drop before each start; check
`tmux ls` and `free -g` after launching. Do not `pkill -f run_symmetric_gd.py` from the agent shell (it matches the shell itself). Re-download tectonic into the scratchpad when it is gone ([[handout-and-latex-toolchain]]).
