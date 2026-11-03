---
name: folder-size-watch
description: Tell the user when the project folder reaches 100 GB (30 GB on 2026-10-23; 300 GB available)
metadata:
  type: feedback
---

The project folder `/home/harish/researchcodes/layer-lenses` was 30 GB on 2026-10-23 (9.4 GB in results/, the rest .git with the
backup branch and loose objects). The user has 300 GB for the project and wants to be told when the folder reaches 100 GB.

**Why:** replay snapshots of every run are kept on disk (gitignored), so the folder grows by 50-150 MB per run.

**How to apply:** check `du -sh` of the folder when launching new run sweeps; say so plainly when it passes 100 GB. See [[repo-upload-rules]].
