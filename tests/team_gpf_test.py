"""CPU checks for team_gpf: (1) gpefv@e0g256_c4 == gpef@e0g256 and gpefv@e8intg32_c4 == gpef@e8 for 40 steps;
(2) es with beta 0 == no momentum (fp8); (3) fp4/fp8 store only representable values; (4) bits; (5) tiny training on
grid with bound stats in the result."""
import os, sys, json
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, W)
import torch
torch.set_num_threads(4)
import qtrain as Q, srq_ext, team_ext as T, team_grp, team_gpf as V


def pair(m1, m2, steps=40, lr=.03, scale=1e-2, same=True):
    torch.manual_seed(1)
    W0 = torch.randn(96, 256) * .1
    p, q = Q.QMat(W0, m1, torch.Generator().manual_seed(0)), Q.QMat(W0, m2, torch.Generator().manual_seed(0))
    nchg = 0
    for t in range(steps):
        g = torch.randn(96, 256) * scale
        p.w.grad = g.clone(); q.w.grad = g.clone(); k0 = p.k.clone(); p.step(lr); q.step(lr)
        nchg += int((p.k != k0).sum())
        if same:
            assert torch.equal(p.k, q.k) and torch.equal(p.a, q.a) and torch.equal(p.b, q.b), (m1, m2, t)
    return p, q, nchg


for m1, m2 in (("gpef@e0g256", "gpefv@e0g256_c4"), ("gpef@e8", "gpefv@e8intg32_c4"),
               ("gpefv@e8fp8g32_c4", "gpefv@e8fp8g32_c4_es0.0")):
    p, q, n = pair(m1, m2)
    if hasattr(p, "err") and p.err is not None and m1 == "gpef@e8":
        assert torch.equal(p.err, q.err)
    print(f"{m1} == {m2} for 40 steps ({n} symbol changes)")
for m in ("gpefv@e8fp8g32_c4", "gpefv@e4fp4g32_c4"):
    _, q, n = pair(m, m, same=False)
    v = q.err.float()
    if "fp4" in m:
        assert torch.isin(v.abs(), V.FP4).all()
    else:
        assert q.err.dtype == V.F8
    print(m, "stored error representable; nonzero share", float((v != 0).float().mean()), "changes", n)
_, q2, _ = pair("gpefv@e0g256_c2", "gpefv@e0g256_cinf", same=False, lr=.3, scale=1.)
print("c2 stats", {k: round(v, 4) if isinstance(v, float) else v for k, v in q2.bound_stats().items() if k != "a_over_s_q"})
B = {"gpefv@e0g256_c2": 2.25, "gpefv@e8intg32_c4": 12, "gpefv@e8fp8g32_c4": 12, "gpefv@e4fp4g32_c4": 8,
     "gpefv@e8fp8g32_c4_es0.9": 20, "gpefv@e8fp8g32_c4_gm0.99": 20, "gpef@e8": 12}
for m, b in B.items():
    assert Q.state_bits_per_weight(m, 256) == b, (m, Q.state_bits_per_weight(m, 256))
print("bits ok")
data, meta = Q.load_data(os.path.join(W, "data", "wt2_pytorch")); data["test"] = data["val"]
for m in ("gpef@e8", "gpefv@e8intg32_c4", "gpefv@e8fp8g32_c4", "gpefv@e4fp4g32_c4", "gpefv@e8fp8g32_c4_es0.9",
          "gpefv@e8fp8g32_c4_gm0.9", "gpefv@e0g256_c16", "gpefv@e0g256_cinf"):
    r = Q.run(m, .03, 0, data, meta, preset="tiny", device="cpu", log=lambda *a: None, eval_windows=20, steps=150, log_every=10)
    gv = r.get("gpefv", {})
    print(json.dumps(dict(m=m, val_bpb=round(r["val_bpb"], 3), on_grid=r["on_grid"], chg=round(float(r["changed_frac_mean"]), 4),
                          bits=r["state_bits"], a_lo=round(gv.get("a_lo_mean", -1), 4), a_hi=round(gv.get("a_hi_mean", -1), 4),
                          b0=round(gv.get("b0_mean", -1), 4), amin=gv.get("a_over_s_min"), amax=gv.get("a_over_s_max"))), flush=True)
    assert r["on_grid"] and (m == "gpef@e8") == ("gpefv" not in r)
    json.dumps(r)
print("ALL OK")
