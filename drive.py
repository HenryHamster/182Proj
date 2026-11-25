"""Driver for the teammate's qtrain.py that keeps the test split sealed: 'test' is aliased to val and test_* fields
are dropped from the record. Modes:
  time  --preset P --methods a,b --steps N    : s/step for each method (short run, no eval)
  run   --preset P --method M --lr X --seed S  : full run, appends a val-only JSON record to --out"""
import argparse, json, os, sys, time
W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
import torch, qtrain as Q
import srq_ext  # registers srq_sr/srq_rot/srqrr@bB (no effect on existing methods)
import team_ext  # registers gpef@eE, adamw_ste, adam_sr@mB (no effect on existing methods)
import team_grp  # registers gpef@e0gG, G in 32/64/128/256 (GPEF below 4 b/w; no effect on existing methods)
import team_rot  # registers gpef_rot@e0, gpef_sr@e0 (format-matched rounding check; no effect on existing methods)
import team_gpx  # registers gpefx@eEgG_<scale>_<rnd>_b<beta>, gpef_frz@e0_<rnd> (rotation x GPEF; no effect on existing methods)
import team_gpf  # registers gpefv@... (teammate GPEF revisions 2026-11-12; no effect on existing methods)

def load(name):
    data, meta = Q.load_data(os.path.join(W, "data", name))
    data["test"] = data["val"]; meta["test"] = meta["val"]          # sealed: never read test.bin
    return data, meta

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["time", "run"]); p.add_argument("--preset", default="pilot")
p.add_argument("--data", default=None); p.add_argument("--methods", default="flip@m4,muon_sr@m4,muon_qat")
p.add_argument("--method", default="flip@m4"); p.add_argument("--lr", type=float, default=0.003)
p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=None)
p.add_argument("--out", default=os.path.join(W, "results", "runs.jsonl")); p.add_argument("--eval_windows", type=int, default=None)
a = p.parse_args()
data, meta = load(a.data or ("wt2_pytorch" if a.preset in ("pilot", "tiny") else "wt103"))
ov = {} if a.steps is None else dict(steps=a.steps)
if a.mode == "time":
    LR = {"flip": 0.003, "muon_sr": 0.01, "muon_qat": 0.01, "adam_qat": 0.03, "dqt": 0.03, "muon_ef4": 0.01, "fp_muon": 0.01}
    for m in a.methods.split(","):
        n = a.steps or 40
        torch.cuda.synchronize(); t = time.time()
        r = Q.run(m, LR[m.split("@")[0]], 0, data, meta, preset=a.preset, device="cuda", log=lambda *x: None, eval_windows=1, steps=n)
        torch.cuda.synchronize()
        print(json.dumps(dict(preset=a.preset, method=m, steps=n, s_per_step=r["train_seconds"] / n, wall=time.time() - t,
                              peak_gb=torch.cuda.max_memory_allocated() / 1e9)), flush=True)
        torch.cuda.reset_peak_memory_stats()
else:
    r = Q.run(a.method, a.lr, a.seed, data, meta, preset=a.preset, device="cuda", eval_windows=a.eval_windows,
              log=lambda *x: print(*x, flush=True), **ov)
    r = {k: v for k, v in r.items() if not k.startswith("test_")}
    r["source_commit"] = "ea4471c"; r["data"] = a.data or ("wt2_pytorch" if a.preset in ("pilot", "tiny") else "wt103")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print("RESULT", json.dumps({k: r[k] for k in ("preset", "method", "lr", "seed", "val_bpb", "train_eval_bpb", "train_seconds", "changed_frac_mean", "on_grid", "diverged")}), flush=True)
