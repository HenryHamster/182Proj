"""CPU checks for gpefr@gG_after_l<lam>: (1) with lam forced to 1 it equals gpefr@g256_after step for step;
(2) flip rate falls with lam; (3) trains on tiny, on grid, 2.25 bits."""
import os, sys, json
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, W)
import torch
torch.set_num_threads(4)
import qtrain as Q, srq_ext, team_ext as T, team_rot, team_grp, team_gpx as X

torch.manual_seed(1)
W0 = torch.randn(96, 256) * .1
ref = X.GPEFRotG(W0, "gpefr@g256_after", torch.Generator().manual_seed(0))
q = X.GPEFRotG(W0, "gpefr@g256_after_l0.5", torch.Generator().manual_seed(0)); q.lam = 1.0; q.useed = ref.useed
for t in range(30):
    g = torch.randn(96, 256) * 1e-3
    ref.w.grad = g.clone(); q.w.grad = g.clone(); ref.step(.3); q.step(.3)
    assert torch.equal(q.k, ref.k) and torch.equal(q.a, ref.a) and torch.equal(q.b, ref.b), t
print("lam=1 == gpefr@g256_after for 30 steps")
for m in ("gpefr@g256_after", "gpefr@g256_after_l0.3", "gpefr@g256_after_l0.1"):
    q = X.GPEFRotG(W0, m, torch.Generator().manual_seed(0))
    for t in range(30):
        q.w.grad = torch.randn(96, 256) * 1e-2; q.step(.03)
    assert q.on_grid() and Q.state_bits_per_weight(m, 256) == 2.25
    print(f"{m}: lam {q.lam}, changed {float(q.chg_sum / q.chg_n):.4f}/step")
data, meta = Q.load_data(os.path.join(W, "data", "wt2_pytorch")); data["test"] = data["val"]
for m in ("gpef@e0g256", "gpefr@g256_after", "gpefr@g256_after_l0.3", "gpefr@g256_after_l0.1"):
    r = Q.run(m, .03, 0, data, meta, preset="tiny", device="cpu", log=lambda *a: None, eval_windows=20, steps=150, log_every=10)
    print(json.dumps(dict(m=m, val_bpb=round(r["val_bpb"], 3), on_grid=r["on_grid"],
                          chg=round(float(r["changed_frac_mean"]), 4), bits=r["state_bits"])), flush=True)
    assert r["on_grid"]
print("ALL OK")
