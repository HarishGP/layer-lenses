import numpy as np, torch
from layer_lenses.relu_analysis import (
    run_single_relu_seed_replayable, replay_epoch_from_snapshot,
    regenerate_seed_result_data, flatten_named_tensors, epoch_replay_summary_table,
)

common = dict(dim=20, depth=3, n_train=4000, hidden_dims=[8, 8, 8], total_epochs=2, lr=2e-3,
              batch_size=500, snapshot_epochs=[0, 1, 2], bias=False, show_progress=False,
              device="cpu")

def first_step(optimizer, weight_decay, momentum=0.0):
    sr = run_single_relu_seed_replayable(11, optimizer=optimizer, momentum=momentum,
                                         weight_decay=weight_decay, **common)
    data = regenerate_seed_result_data(sr)
    r = replay_epoch_from_snapshot(sr, 0, data=data)
    s = r["steps"][0]
    g = flatten_named_tensors(s["grad"])
    u = flatten_named_tensors(s["update"])
    cos = float(u @ g) / float(u.norm() * g.norm())
    n = g.numel()
    lr = r["lr"]
    print(f"\n--- optimizer={optimizer} weight_decay={weight_decay} momentum={momentum} ---")
    print(f"  cos_update_grad (measured)          = {cos:.6f}")
    print(f"  ||g||_1 / (sqrt(n) ||g||_2)         = {float(g.abs().sum() / (n**0.5 * g.norm())):.6f}   <- predicted |cos| if update = -lr*sign(g)")
    pred = -lr * g / (g.abs() + 1e-8)
    print(f"  cos(update, -lr*g/(|g|+eps))        = {float(u @ pred) / float(u.norm() * pred.norm()):.9f}")
    print(f"  cos(update, -g)                     = {-cos:.6f} (would be 1.0 if update ~ -lr*g)")
    print(f"  ||update|| = {float(u.norm()):.6f}   lr*sqrt(n) = {lr * n**0.5:.6f}   lr*||g||/||g|| scale = {lr * 1:.6f}")
    print(f"  n params = {n},  ||g||_2 = {float(g.norm()):.6e},  median|g| = {float(g.abs().median()):.3e}, min|g| = {float(g.abs().min()):.3e}")
    frac_below_eps = float((g.abs() < 1e-8).float().mean())
    print(f"  fraction of coords with |g| < eps=1e-8: {frac_below_eps:.3f}")
    return cos

first_step("adam", 0.0)
first_step("adam", 0.01)
first_step("sgd", 0.0, momentum=0.0)
first_step("sgd", 0.0, momentum=0.9)
first_step("sgd", 0.01, momentum=0.0)
