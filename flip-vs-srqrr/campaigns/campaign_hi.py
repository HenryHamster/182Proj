"""flip-vs-srqrr-001, higher-state points (Henry 2026-11-10: "test 8 bits per weight to expand the trend"):
srq_rot/srq_sr at b6 (8 b/w) and b8 (10 b/w, where ROTATION.md predicts the rotation gain vanishes), flip@m8.
Pilot (same protocol) after campaign_team prints TEAM PILOT DONE; 10M srq_rot@b6, srq_sr@b6 at the pilot-best lr,
seeds 0-2, after campaign_team prints TEAM DONE."""
import os, time
import campaign as C
def wait(log, s):
    while s not in open(os.path.join(C.W, "logs", log)).read():
        time.sleep(120)
SRQ = [0.003, 0.01, 0.03]
PILOT = {"srq_rot@b6": SRQ, "srq_sr@b6": SRQ, "srq_rot@b8": SRQ, "srq_sr@b8": SRQ, "flip@m8": [0.001, 0.003, 0.009]}
wait("campaign_team.log", "TEAM PILOT DONE")
C.PAR["pilot"] = 2
C.phase("pilot", PILOT, ext=2)
print("HI PILOT DONE", flush=True)
wait("campaign_team.log", "TEAM DONE")
C.pool("10M", [(m, C.best("pilot", m)[0], s) for m in ("srq_rot@b6", "srq_sr@b6") for s in (0, 1, 2)])
print("HI DONE", flush=True)
