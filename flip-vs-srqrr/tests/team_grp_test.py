"""CPU checks for team_grp: (1) gpef@e0g32 == gpef@e0 step for step (symbols and codebooks); (2) larger groups decode
to the same initial weights and hold one codebook per G weights; (3) all train on tiny, on grid, with the stated bits."""
import os, sys, json
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, W)
import torch
torch.set_num_threads(4)
import qtrain as Q, srq_ext, team_ext as T, team_rot, team_grp as TG

torch.manual_seed(1)
W0 = torch.randn(96, 192) * .1
ref = T.GPEF(W0, "gpef@e0", torch.Generator().manual_seed(0))
q = TG.GPEFG(W0, "gpef@e0g32", torch.Generator().manual_seed(0))
for t in range(30):
    g = torch.randn(96, 192) * 1e-3
    ref.w.grad = g.clone(); q.w.grad = g.clone()
    ref.step(.3); q.step(.3)
    assert torch.equal(q.k, ref.k) and torch.equal(q.a, ref.a) and torch.equal(q.b, ref.b), t
assert T.GROUP == 32
print(f"gpef@e0g32 == gpef@e0 for 30 steps (changed {float(ref.chg_sum / ref.chg_n):.4f}/step)")

for G in (64, 128, 256):
    q = TG.GPEFG(W0, f"gpef@e0g{G}", torch.Generator().manual_seed(0))
    assert q.k.shape == (96 * 192 // G, G) and q.a.shape == (96 * 192 // G,)
    assert torch.equal(q.w.detach(), ref.alpha * T.GPEF(W0, "gpef@e0", torch.Generator().manual_seed(0)).K)
    for t in range(5):
        q.w.grad = torch.randn(96, 192) * 1e-3; q.step(.3)
    assert q.on_grid() and T.GROUP == 32
    print(f"g{G}: {q.a.numel()} codebooks, on grid after 5 steps, bits {Q.state_bits_per_weight(f'gpef@e0g{G}', 192)}")

data, meta = Q.load_data(os.path.join(W, "data", "wt2_pytorch")); data["test"] = data["val"]
for m in ("gpef@e0", "gpef@e0g32", "gpef@e0g64", "gpef@e0g128", "gpef@e0g256"):
    r = Q.run(m, .03, 0, data, meta, preset="tiny", device="cpu", log=lambda *a: None, eval_windows=20, steps=150, log_every=10)
    c = r["curve"]
    row = dict(m=m, first=c[0][1], last=c[-1][1], val_bpb=r["val_bpb"], on_grid=r["on_grid"], chg=r["changed_frac_mean"], bits=r["state_bits"])
    print(json.dumps(row), flush=True)
    assert r["on_grid"] and row["last"] < row["first"]
print("ALL OK")
