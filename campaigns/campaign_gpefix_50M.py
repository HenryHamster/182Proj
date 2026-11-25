"""flip-vs-srqrr-003: GPEF's 8 b/w error storage at 49.2M weights (preset 50M), byte-level WikiText-103, on IconLab.

Question: is gpef@e4's poor 50M result (1.596, behind gpef@e0 1.478) caused by its error storage? Plain e4 stores the
per-weight error in steps of span/7 with nearest rounding, so at lr < ~0.018 an update that flips no symbol is always
rounded to zero (e4 tracks e0 to step 3000 at lr 0.01). The fixed storage scored 1.646 vs 1.758 at 10M.
Methods (team_gpx.py, unchanged):
  gpefx@e4g32_narrow_rot_b0.99   fixed storage, 4-bit error, 8 b/w   (10M: 1.646)
  gpefx@e2g32_sym_rot_b0.99      symmetric 2-bit error, 6 b/w        (never trained before)
Rounds on two GPUs, each finished before the next starts:
  1. e4 fix, seed 0, lr 0.01 and 0.03
  2. e4 fix, seed 1 at its best lr;      e2 sym, seed 0 at that lr
  3. e2 sym, seed 1 at that lr;          e2 sym, seed 0 at the other lr (uses the otherwise idle GPU)
Resumable: skips (method, lr, seed) already recorded.
usage: python campaigns/campaign_gpefix_50M.py GPU_A GPU_B"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_50M as C

C.OUT = os.path.join(C.W, "results", "campaign_50M_gpefix.jsonl")
C.GPUS = sys.argv[1:3] if len(sys.argv) >= 3 else ["0", "3"]
E4, E2, LRS = "gpefx@e4g32_narrow_rot_b0.99", "gpefx@e2g32_sym_rot_b0.99", (0.01, 0.03)

if __name__ == "__main__":
    C.run_round([(E4, lr, 0) for lr in LRS]); print("ROUND 1 DONE", flush=True)
    b = C.best(E4)
    C.run_round([(E4, b, 1), (E2, b, 0)]); print("ROUND 2 DONE", flush=True)
    other = [lr for lr in LRS if lr != b][0]
    C.run_round([(E2, b, 1), (E2, other, 0)]); print("CAMPAIGN DONE", flush=True)
