"""CPU checks for team_ext: (1) gpef@e8/e4/e0 step equals the notebook's GroupDiscrete torch path (quant_core.py,
extracted verbatim) on random gradients; (2) every new method trains on tiny with codes on grid."""
import os, sys, json, types
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, W)
import torch
torch.set_num_threads(4)
import qtrain as Q, srq_ext, team_ext as T, team_ref_quant_core as R

for E in (8, 4, 0):
    torch.manual_seed(E)
    q = T.GPEF(torch.randn(96, 64) * .1, f"gpef@e{E}", torch.Generator().manual_seed(0))
    ref = types.SimpleNamespace(k=q.k.clone(), a=q.a.clone(), b=q.b.clone(), error=q.err.clone(), group=32,
                                span=torch.full_like(q.a, q.span), bound_policy="relative", bound_ratio=4.)
    ref.bounds = lambda r=ref: R.bounds_from_initial(r.span, "relative", 4.)
    model = types.SimpleNamespace(qlayers=lambda: [ref])
    opt = R.GroupDiscrete(model, lr=.05, kind="ef", error_bits=E, passes=2, total_steps=10, floor=1., diag_every=0, cpu_backend="torch")
    worst = 0.
    for t in range(20):
        g = torch.randn(96, 64) * 1e-3
        q.w.grad = g.clone(); ref._w = types.SimpleNamespace(grad=g.clone(), device=torch.device("cpu"))
        q.step(.05); opt.step()
        assert torch.equal(q.k, ref.k), (E, t, (q.k != ref.k).sum())
        assert torch.equal(q.err, ref.error), (E, t)
        worst = max(worst, float((q.a - ref.a).abs().max()), float((q.b - ref.b).abs().max()))
    print(f"gpef@e{E}: 20 steps identical symbols/errors, max codebook diff {worst:.2e}", flush=True)
    assert worst < 1e-6

data, meta = Q.load_data(os.path.join(W, "data", "wt2_pytorch")); data["test"] = data["val"]
for m, lr in (("gpef@e8", .03), ("gpef@e4", .03), ("gpef@e0", .03), ("adamw_ste", .01), ("adam_sr@m8", .03), ("adam_sr@m4", .03), ("adam_sr@m2", .03), ("srq_rot@b2", .03)):
    r = Q.run(m, lr, 0, data, meta, preset="tiny", device="cpu", log=lambda *a: None, eval_windows=20, steps=150, log_every=10)
    c = r["curve"]
    row = dict(m=m, first=c[0][1], last=c[-1][1], val_bpb=r["val_bpb"], on_grid=r["on_grid"], chg=r["changed_frac_mean"], bits=r["state_bits"])
    print(json.dumps(row), flush=True)
    assert r["on_grid"] and row["last"] < row["first"]
print("ALL OK")
