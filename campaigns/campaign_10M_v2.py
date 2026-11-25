"""flip-vs-srqrr-001, 10M phase revised 2026-11-10 10:20 UTC. The refit is dropped from the main method (pilot: srqrr worse
than srq_rot at every b, ~2x slower), so srqrr at 10M stops at the seed-0 grid already run (b0, b1 full; b2 lr .01/.03).
Rotation-only moves to the front. Same protocol as campaign.py: grid {best/3, best, 3 best} around the pilot best, one edge
extension, seeds 1, 2 at the best lr; results append to results/campaign_10M.jsonl."""
import campaign as C
M = ["srq_rot@b0", "srq_rot@b2", "srq_rot@b4", "flip@m2", "flip@m2g", "flip@m1.58",
     "srq_sr@b0", "srq_sr@b2", "srq_sr@b4", "muon_sr@m2"]
g = {}
for m in M:
    b, _ = C.best("pilot", m)
    g[m] = [round(b / 3, 6), b, round(b * 3, 6)]
print("10M grid v2", g, flush=True)
C.phase("10M", g, ext=1)
print("CAMPAIGN DONE", flush=True)
