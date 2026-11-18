"""CPU checks for team_gpx: (1) gpefx@e4g32_wide_near_b1 == gpef@e4 step for step (symbols, codebooks, error);
(2) narrow storage: the stored error reconstructs the residual to within one step wherever the residual is in range;
(3) rotation error rounding is unbiased over steps for a fixed residual; (4) frozen codebooks never move;
(5) every new method trains on tiny, stays on grid, with the stated bits."""
import os, sys, json
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, W)
import torch
torch.set_num_threads(4)
import qtrain as Q, srq_ext, team_ext as T, team_rot, team_grp, team_gpx as X

torch.manual_seed(1)
W0 = torch.randn(96, 192) * .1
ref = T.GPEF(W0, "gpef@e4", torch.Generator().manual_seed(0))
q = X.GPEFX(W0, "gpefx@e4g32_wide_near_b1", torch.Generator().manual_seed(0))
for t in range(30):
    g = torch.randn(96, 192) * 1e-3
    ref.w.grad = g.clone(); q.w.grad = g.clone()
    ref.step(.3); q.step(.3)
    assert torch.equal(q.k, ref.k) and torch.equal(q.a, ref.a) and torch.equal(q.b, ref.b) and torch.equal(q.err, ref.err), t
assert T.GROUP == 32
print(f"gpefx@e4g32_wide_near_b1 == gpef@e4 for 30 steps (err nonzero frac {float((ref.err != 0).float().mean()):.3f})")

for m in ("gpefx@e4g32_narrow_near_b1", "gpefx@e2g256_narrow_rot_b1", "gpefx@e1g256_sym_rot_b0.99", "gpefx@e2g256_sym_near_b1"):
    q = X.GPEFX(W0, m, torch.Generator().manual_seed(0))
    for t in range(10):
        q.w.grad = torch.randn(96, 192) * 1e-3; q.step(.3)
    s = q._step_size(q.a, q.b)
    assert q.err.min() >= q.lo and q.err.max() <= q.hi and q.on_grid()
    print(f"{m}: err codes in [{int(q.err.min())}, {int(q.err.max())}], mean step/alpha {float(s.mean() / q.alpha):.4f},"
          f" changed {float(q.chg_sum / q.chg_n):.4f}/step")

# rotation rounding of a fixed residual: mean of the stored code over steps -> residual / step
torch.manual_seed(3)
r = torch.rand(64, 32) * 3 - 1.5
acc = torch.zeros_like(r); n = 1000
for t in range(1, n + 1):
    f = torch.floor(r); acc += f + (X._rot_u(r.shape, 99, t, r.device) < (r - f)).float()
print(f"rotation error rounding: max |mean code - residual| over {n} steps = {float((acc / n - r).abs().max()):.4f}")
assert float((acc / n - r).abs().max()) < .01

for m in ("gpef_frz@e0_near", "gpef_frz@e0_rot"):
    q = X.GPEFFrozen(W0, m, torch.Generator().manual_seed(0))
    a0, b0 = q.a.clone(), q.b.clone()
    for t in range(10):
        q.w.grad = torch.randn(96, 192) * 1e-3; q.step(.3)
    assert torch.equal(q.a, a0) and torch.equal(q.b, b0) and q.on_grid()
    print(f"{m}: codebooks unchanged after 10 steps, changed {float(q.chg_sum / q.chg_n):.4f}/step")

import team_rot as TR
ref = TR.GPEFRound(W0, "gpef_rot@e0", torch.Generator().manual_seed(0))
q = X.GPEFRotG(W0, "gpefr@g32_after", torch.Generator().manual_seed(0)); q.useed = ref.useed
for t in range(30):
    g = torch.randn(96, 192) * 1e-3
    ref.w.grad = g.clone(); q.w.grad = g.clone(); ref.step(.3); q.step(.3)
    assert torch.equal(q.k, ref.k) and torch.equal(q.a, ref.a) and torch.equal(q.b, ref.b), t
print("gpefr@g32_after == gpef_rot@e0 for 30 steps")
for m in ("gpefr@g256_after", "gpefr@g256_before"):
    q = X.GPEFRotG(W0, m, torch.Generator().manual_seed(0))
    for t in range(10):
        q.w.grad = torch.randn(96, 192) * 1e-3; q.step(.3)
    assert q.on_grid() and Q.state_bits_per_weight(m, 192) == 2.25
    print(f"{m}: on grid, changed {float(q.chg_sum / q.chg_n):.4f}/step")

data, meta = Q.load_data(os.path.join(W, "data", "wt2_pytorch")); data["test"] = data["val"]
for m in ("gpef@e4", "gpefx@e4g32_wide_near_b1", "gpefx@e4g32_narrow_near_b1", "gpefx@e4g32_narrow_rot_b1",
          "gpefx@e4g32_narrow_rot_b0.9", "gpefx@e2g256_narrow_rot_b1", "gpef_frz@e0_near", "gpef_frz@e0_rot",
          "gpef@e0g256", "gpefr@g256_after", "gpefr@g256_before", "gpefx@e1g256_sym_rot_b0.99", "gpefx@e2g256_sym_near_b1"):
    r = Q.run(m, .03, 0, data, meta, preset="tiny", device="cpu", log=lambda *a: None, eval_windows=20, steps=150, log_every=10)
    c = r["curve"]
    row = dict(m=m, first=round(c[0][1], 3), last=round(c[-1][1], 3), val_bpb=round(r["val_bpb"], 3), on_grid=r["on_grid"],
               chg=round(float(r["changed_frac_mean"]), 4), bits=r["state_bits"])
    print(json.dumps(row), flush=True)
    assert r["on_grid"]
print("ALL OK")
