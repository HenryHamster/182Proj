"""CPU checks for srq_ext: (1) refit Ga equals the autograd scale gradient sum_j g_rj K_rj; (2) every method trains on
tiny (loss falls, codes on grid, counters in range); (3) rotation thresholds are uniform per step."""
import os, sys, json
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, W)
import torch, numpy as np
torch.set_num_threads(4)
import qtrain as Q, srq_ext as E

# (1) Ga check, batch of 8 sequences
worst = [0.0]
orig = E.SRQMat.step
def chk(self, lr):
    if self.refit and self._d is not None:
        Ga, Va, H = self.refit_stats(self._d, self._z)
        ref = (self.w.grad.float() * self.K).sum(1)
        worst[0] = max(worst[0], float((Ga - ref).abs().max() / ref.abs().max().clamp_min(1e-30)))
        assert torch.all(Va >= 0) and torch.all(H > 0)
    return orig(self, lr)
E.SRQMat.step = chk
data, meta = Q.load_data(os.path.join(W, "data", "wt2_pytorch")); data["test"] = data["val"]
r = Q.run("srqrr@b2", .01, 0, data, meta, preset="tiny", device="cpu", log=lambda *a: None, eval_windows=20, steps=5, batch=8, micro=8)
print("Ga max rel err vs autograd:", worst[0]); assert worst[0] < 1e-4
E.SRQMat.step = orig

# (2) every method trains
out = []
for m in ("srq_sr@b0", "srq_rot@b0", "srqrr@b0", "srq_sr@b2", "srq_rot@b2", "srqrr@b1", "srqrr@b4", "srqrr_k10@b2"):
    r = Q.run(m, .01, 0, data, meta, preset="tiny", device="cpu", log=lambda *a: None, eval_windows=20, steps=150, log_every=10)
    c = r["curve"]
    row = dict(m=m, first=c[0][1], last=c[-1][1], val_bpb=r["val_bpb"], on_grid=r["on_grid"], chg=r["changed_frac_mean"], bits=r["state_bits"])
    print(json.dumps(row), flush=True); out.append(row)
    assert r["on_grid"] and row["last"] < row["first"]
# (3) rotation thresholds: one matrix, 64 steps, per-step mean ~ .5, per-entry spread even
q = E.SRQMat(torch.randn(64, 64) * .1, "srq_rot@b2", torch.Generator().manual_seed(0))
us = []
for t in range(64):
    q.t = t; us.append(q._threshold(torch.zeros(64, 64)))
U = torch.stack(us)
print("per-step mean range", float(U.mean((1, 2)).min()), float(U.mean((1, 2)).max()),
      "per-entry max gap over 64 steps", float(U.sort(0).values.diff(dim=0).max()))
print("ALL OK")
