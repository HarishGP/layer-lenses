---
name: handout-and-latex-toolchain
description: The collaborator handout lives in notes/handout (tex + make_figures.py); LaTeX is compiled with a downloaded tectonic binary (no TeX installed, no sudo)
metadata:
  type: project
---

`notes/handout/handout.tex` is the 10-page introduction for students and collaborators (path decomposition, ODT data, runs 60 / 70 inits
and results, learned representation, symmetrised flow with runs 73 / 97 / 74), written 2026-10-23. Its figures come from
`notes/handout/make_figures.py` (reads the replay files of runs 60, 70, 73, 97, 74, 114; writes figs/*.pdf and facts.json).

**Why:** the machine has no TeX distribution and sudo needs a password, so LaTeX is compiled with the static `tectonic` binary
(GitHub releases, `tectonic-<ver>-x86_64-unknown-linux-musl.tar.gz`, ~30 MB, fetches packages on first use); it lives in the session
scratchpad, so re-download it in a new session. Page checks: `uv run --no-project --with pymupdf python` to rasterise pages (the Read
tool cannot render multi-page PDFs here, pdftoppm is missing).

**How to apply:** when the user asks to update the handout, edit the tex, rerun make_figures.py if figures change, compile with
tectonic, rasterise and look at every page before reporting. See [[repo-upload-rules]].
