"""CPU checks for team_rot: (1) with threshold 0.5 gpef_rot/gpef_sr equal gpef@e0 step for step; (2) for a fixed
codebook the rounding is unbiased (SR: mean over draws; rotation: mean over steps); (3) both train on tiny, on grid."""
import os, sys, json
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, W)
import torch
torch.set_num_threads(4)
import qtrain as Q, srq_ext, team_ext as T, team_rot as RR

for kind in ("rot", "sr"):
    torch.manual_seed(1)
    W0 = torch.randn(96, 64) * .1
    ref = T.GPEF(W0, "gpef@e0", torch.Generator().manual_seed(0))
    q = RR.GPEFRound(W0, f"gpef_{kind}@e0", torch.Generator().manual_seed(0))
    q._u = lambda x: torch.full_like(x, .5)
    for t in range(20):
        g = torch.randn(96, 64) * 1e-3
        ref.w.grad = g.clone(); q.w.grad = g.clone()
        ref.step(.3); q.step(.3)
        assert torch.equal(q.k, ref.k) and torch.equal(q.a, ref.a) and torch.equal(q.b, ref.b), (kind, t)
    print(f"gpef_{kind}@e0 with threshold .5 == gpef@e0 for 20 steps (changed {float(ref.chg_sum / ref.chg_n):.3f}/step)")

# unbiasedness of the final rounding on a fixed codebook (bypass the fit: call the rounding block on given a, b)
torch.manual_seed(2)
for kind in ("rot", "sr"):
    q = RR.GPEFRound(torch.randn(64, 32) * .1, f"gpef_{kind}@e0", torch.Generator().manual_seed(0))
    a, b = q.a.clone(), q.b.clone()
    x = (torch.rand(q.k.shape) * 2 - 1) * a[:, None]
    acc = torch.zeros_like(x); n = 400
    for t in range(1, n + 1):
        q.t = t
        aa, bb = a[:, None], b[:, None]
        j = (x >= -bb).long() + (x >= bb).long()
        lo = torch.where(j == 0, -aa, torch.where(j == 1, -bb, bb)); hi = torch.where(j == 0, -bb, torch.where(j == 1, bb, aa))
        kk = (j + (q._u(x) < (x - lo) / (hi - lo)).long()).to(torch.uint8)
        acc += T.decode(kk, a, b)
    err = ((acc / n - x) / a[:, None]).abs()
    print(f"gpef_{kind}: mean decoded - x over {n} steps, |err|/a mean {err.mean():.4f} max {err.max():.4f}")
    assert err.max() < (0.02 if kind == "rot" else 0.2)

data, meta = Q.load_data(os.path.join(W, "data", "wt2_pytorch")); data["test"] = data["val"]
for m in ("gpef_rot@e0", "gpef_sr@e0", "gpef@e0"):
    r = Q.run(m, .03, 0, data, meta, preset="tiny", device="cpu", log=lambda *a: None, eval_windows=20, steps=150, log_every=10)
    c = r["curve"]
    row = dict(m=m, first=c[0][1], last=c[-1][1], val_bpb=r["val_bpb"], on_grid=r["on_grid"], chg=r["changed_frac_mean"], bits=r["state_bits"])
    print(json.dumps(row), flush=True)
    assert r["on_grid"] and row["last"] < row["first"] and row["bits"] == 4.0
print("ALL OK")
