import json, hashlib, sys
import numpy as np, torch
from layer_lenses.relu_analysis import run_single_relu_seed, run_single_relu_seed_replayable

def h(t):
    return hashlib.sha256(t.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()[:16]

common = dict(dim=20, depth=3, n_train=4000, hidden_dims=[8, 8, 8], total_epochs=5, lr=2e-3,
              batch_size=500, weight_decay=0.01, snapshot_epochs=[0, 1, 3, 5], bias=False,
              show_progress=False)
ref = {}
for device in ["cpu", "cuda"]:
    for opt, mom in [("adam", 0.0), ("sgd", 0.9)]:
        for runner in [run_single_relu_seed, run_single_relu_seed_replayable]:
            key = f"{device}|{opt}|{runner.__name__}"
            r = runner(11, optimizer=opt, momentum=mom, device=device, **common)
            out = r["out"]
            ref[key] = {
                "losses": [round(float(v), 12) for v in out["epoch_losses"]],
                "lr_history": [round(float(v), 12) for v in out["lr_history"]],
                "ckpt": {str(e): {k: h(v) for k, v in sd.items()}
                         for e, sd in out["checkpoint_snapshots"].items()},
            }
json.dump(ref, open(sys.argv[1], "w"), indent=1)
print("captured", len(ref), "reference runs")
