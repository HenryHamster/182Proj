"""Does a GPEF variant keep changing the deployed weights when the update is zero?
200 steps at lr 0.03 with random gradients (to populate the stored error), then 300 steps at lr 0 with zero gradient.
Reports symbol moves per weight per step during the lr-0 phase and the relative change of the decoded matrix."""
import sys
sys.path.insert(0, sys.argv[1])
import torch
torch.set_num_threads(4)
import qtrain as Q, srq_ext, team_ext as T, team_rot, team_grp, team_gpx as X
ME = ["gpef@e0", "gpef@e4", "gpefx@e4g32_wide_near_b1", "gpefx@e4g32_narrow_near_b1", "gpefx@e4g32_narrow_rot_b1",
      "gpefx@e4g32_narrow_rot_b0.99", "gpefx@e4g32_narrow_near_b0.99", "gpefx@e2g32_sym_rot_b0.99", "gpefx@e2g32_sym_rot_b1",
      "gpefx@e2g32_sym_near_b1"]
for m in ME:
    torch.manual_seed(0); W0 = torch.randn(256, 512) * .05
    q = Q.QMat(W0, m, torch.Generator().manual_seed(0))
    for t in range(200):
        q.w.grad = torch.randn(256, 512) * 1e-3; q.step(.03)
    d0 = q._deq().clone(); moves = 0
    for t in range(300):
        k0 = q.k.clone(); q.w.grad = torch.zeros(256, 512); q.step(0.0)
        moves += int((q.k != k0).sum())
    rel = float((q._deq() - d0).norm() / d0.norm())
    print(f"{m:32s} lr=0: moves/weight/step {moves / 300 / W0.numel():.5f}   |dW|/|W| after 300 steps {rel:.4f}", flush=True)
