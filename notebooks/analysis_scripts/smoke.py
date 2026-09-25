import os, sys, time
import numpy as np, torch
from layer_lenses.relu_analysis import (
    run_single_relu_seed, run_single_relu_seed_replayable, regenerate_seed_result_data,
    replay_epoch_from_snapshot, epoch_replay_summary_table, save_seed_result_pickle,
    load_seed_result_pickle, flatten_named_tensors, epoch_replay_parameter_series,
)
common = dict(dim=20, depth=3, n_train=4000, hidden_dims=[8, 8, 8], total_epochs=6, lr=2e-3,
              batch_size=500, weight_decay=0.01, snapshot_epochs=range(0, 7), bias=False,
              show_progress=False)

for device in ["cpu", "cuda"]:
    for opt, mom in [("adam", 0.0), ("sgd", 0.9)]:
        print(f"\n=== device={device} optimizer={opt} ===")
        legacy = run_single_relu_seed(11, optimizer=opt, momentum=mom, device=device, **common)
        rep = run_single_relu_seed_replayable(11, optimizer=opt, momentum=mom, device=device, **common)
        # 1. parity with the existing trainer
        for e in legacy["out"]["checkpoint_snapshots"]:
            for k, v in legacy["out"]["checkpoint_snapshots"][e].items():
                assert torch.equal(v, rep["out"]["checkpoint_snapshots"][e][k]), (e, k)
        assert np.allclose(legacy["out"]["epoch_losses"], rep["out"]["epoch_losses"])
        print("parity with run_single_relu_seed: exact")
        assert "data" not in rep
        # 2. data regeneration
        data = regenerate_seed_result_data(rep)
        for k in data:
            assert np.array_equal(data[k], legacy["data"][k]), k
        print("regenerated data: identical")
        # 3. replay
        for e in [0, 3]:
            r = replay_epoch_from_snapshot(rep, e, data=data, record_full_data_gradient=True)
            v = r["verification"]
            print(f"replay epoch {e}: steps={r['num_steps']} lr={r['lr']:.6f} "
                  f"max|param diff|={v['max_abs_param_diff_overall']:.3e} "
                  f"max|opt state diff|={v.get('max_abs_optimizer_state_diff', float('nan')):.3e}")
            assert v["max_abs_param_diff_overall"] == 0.0
            total = sum((flatten_named_tensors(s["update"]) for s in r["steps"]))
            delta = flatten_named_tensors(r["params_after"]) - flatten_named_tensors(r["params_before"])
            print("  sum(updates) vs epoch delta max diff:", float((total - delta).abs().max()))
            if opt == "adam":
                res = max(float((flatten_named_tensors(s["adam_update"]) + flatten_named_tensors(s["weight_decay_update"]) - flatten_named_tensors(s["update"])).abs().max()) for s in r["steps"])
                print("  adam decomposition residual:", res)
                assert res < 1e-6
            else:
                assert "momentum_buffer" in r["steps"][0]["optimizer_state_after"]["hidden_layers.0.weight"]
            df = epoch_replay_summary_table(r)
            print(df.head(3).to_string())
            ser = epoch_replay_parameter_series(r, "grad", "hidden_layers.0.weight")
            assert ser.shape == (r["num_steps"], 8, 20)
        # 4. pickle round trip
        p = f"/tmp/claude-1000/-home-harish-researchcodes-layer-lenses/8b3631ed-dc7e-41c4-af0e-3b349a6ef0e2/scratchpad/rep_{device}_{opt}.pkl"
        save_seed_result_pickle(rep, p)
        rep2 = load_seed_result_pickle(p)
        print("pickle size bytes:", os.path.getsize(p))
        r2 = replay_epoch_from_snapshot(rep2, 2)   # regenerates data itself
        assert r2["verification"]["max_abs_param_diff_overall"] == 0.0
        print("replay from loaded pickle (data regenerated): exact")

# 5. shuffle mode
print("\n=== shuffle_each_epoch=True (cpu, adam) ===")
rep = run_single_relu_seed_replayable(11, optimizer="adam", device="cpu", shuffle_each_epoch=True, **common)
r = replay_epoch_from_snapshot(rep, 4)
print("replay diff:", r["verification"]["max_abs_param_diff_overall"])
assert r["verification"]["max_abs_param_diff_overall"] == 0.0
# 6. error paths
try:
    replay_epoch_from_snapshot(rep, 6)
except ValueError as ex:
    print("final-epoch guard ok:", ex)
print("\nALL SMOKE CHECKS PASSED")
