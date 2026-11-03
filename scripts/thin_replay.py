"""Thin a run's replay weights so that the file fits the repository's size budget: keep every epoch up to 100 (the captures), every 5th
up to 1500 (the release of the cues and the rotation), every 20th afterwards, plus the last epoch (506 snapshots for a 4000-epoch run,
~20 MB instead of 160 MB). The full file is kept beside it as replay/weights_full.npz (ignored by git); the anatomy notebook and the
handout script read whatever epochs weights.npz holds. Usage: .venv/bin/python scripts/thin_replay.py <run folder> [<run folder> ...]"""
import sys, shutil, numpy as np
from pathlib import Path

def keep_epoch(e: int, last: int) -> bool:
    return e <= 100 or (e <= 1500 and e % 5 == 0) or e % 20 == 0 or e == last

for folder in sys.argv[1:]:
    f = Path(folder) / "replay"; full = f / "weights_full.npz"; thin = f / "weights.npz"
    if not full.exists(): shutil.copy2(thin, full)                                # keep the full trajectory locally
    z = np.load(full); ep = z["epochs"]; last = int(ep[-1]); keep = np.array([keep_epoch(int(e), last) for e in ep])
    np.savez(thin, epochs=ep[keep], **{k: z[k][keep] for k in z.files if k.startswith("w__")})
    print(f"{folder}: {len(ep)} -> {int(keep.sum())} snapshots, {full.stat().st_size / 1e6:.0f} -> {thin.stat().st_size / 1e6:.0f} MB")
