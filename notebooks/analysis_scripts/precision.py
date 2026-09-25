import numpy as np, torch
from layer_lenses.relu_analysis import (
    run_single_relu_seed_replayable, replay_epoch_from_snapshot,
    regenerate_seed_result_data, flatten_named_tensors,
)
common = dict(dim=20, depth=3, n_train=4000, hidden_dims=[8,8,8], total_epochs=2, lr=2e-3,
              batch_size=500, snapshot_epochs=[0,1,2], bias=False, show_progress=False, device="cpu")

for lr in [2e-3, 1.0]:
    c = dict(common); c["lr"] = lr
    sr = run_single_relu_seed_replayable(11, optimizer="sgd", momentum=0.0, weight_decay=0.0, **c)
    r = replay_epoch_from_snapshot(sr, 0, data=regenerate_seed_result_data(sr))
    s = r["steps"][0]
    g = flatten_named_tensors(s["grad"]); u = flatten_named_tensors(s["update"])
    theta = flatten_named_tensors(s["params_before"])
    exact = -lr * g
    cos = float(u @ g) / float(u.norm() * g.norm())
    print(f"\nSGD lr={lr}, wd=0, momentum=0")
    print(f"  cos_update_grad            = {cos:.9f}")
    print(f"  max|update - (-lr*g)|      = {float((u - exact).abs().max()):.3e}")
    print(f"  float32 ulp at |theta|~{float(theta.abs().median()):.3f}  = {float(np.spacing(np.float32(theta.abs().median()))):.3e}")
    print(f"  median |update coord|      = {float(u.abs().median()):.3e}   <- compare to the ulp above")
