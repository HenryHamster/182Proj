"""flip-vs-srqrr-001 CPU drift check (2026-11-11): zero-mean random gradients on a 256x512 matrix for 300 steps; an
unbiased method moves symbols up and down equally (up share ~0.50), keeps the mean symbol ~1.5 and the mean stored
error ~0 steps. Compares narrow (asymmetric codes) and sym (half-step codes) error storage."""
import os, sys
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, W)
import torch
torch.set_num_threads(4)
import qtrain as Q, srq_ext, team_ext as T, team_rot, team_grp, team_gpx as X
ME = ["gpef@e0g256"] + [f"gpefx@e{E}g256_{s}_{r}_b{b}" for E in (1, 2, 4) for s in ("narrow", "sym")
                        for r, b in (("near", "1"), ("rot", "0.99"))] + ["gpefx@e4g32_sym_rot_b0.99", "gpefx@e4g32_sym_near_b1"]
for m in ME:
    torch.manual_seed(0); W0 = torch.randn(256, 512) * .05
    q = Q.QMat(W0, m, torch.Generator().manual_seed(0))
    up = dn = 0
    for t in range(300):
        k0 = q.k.clone().long(); q.w.grad = torch.randn(256, 512) * 1e-3; q.step(.03)
        d = q.k.long() - k0.reshape(q.k.shape); up += int((d > 0).sum()); dn += int((d < 0).sum())
    e = (q.err.float() + getattr(q, "off", 0.)).mean() if hasattr(q, "err") and m.startswith("gpefx") else float("nan")
    print(f"{m:34s} up share {up / max(up + dn, 1):.3f}  moves/step {(up + dn) / 300 / W0.numel():.4f}  "
          f"mean symbol {float(q.k.float().mean()):.3f}  mean stored error {float(e):+.3f} steps", flush=True)
