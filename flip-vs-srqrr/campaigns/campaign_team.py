"""flip-vs-srqrr-001, team extension: teammate methods (gpef@e8/e4/e0, adamw_ste) and Adam references (dqt,
adam_sr@m8/m4/m2) on the same protocol. Pilot now (2 parallel jobs, sharing the GPU with campaign_10M_v2); after
campaign_10M_v2 prints CAMPAIGN DONE: 10M full protocol for gpef@e8/e4/e0, and seeds 0-2 at the pilot-best lr for
adam_sr@m4/m2 (references only)."""
import os, time
import campaign as C
SRQ, ADAM = [0.003, 0.01, 0.03], [0.01, 0.03, 0.09]
PILOT = {"gpef@e8": SRQ, "gpef@e4": SRQ, "gpef@e0": SRQ, "adamw_ste": SRQ,
         "adam_sr@m8": ADAM, "adam_sr@m4": ADAM, "adam_sr@m2": ADAM, "dqt": ADAM}
C.PAR["pilot"] = 2
C.phase("pilot", PILOT, ext=2)
print("TEAM PILOT DONE", flush=True)
LOG = os.path.join(C.W, "logs", "campaign_10M_v2.log")
while "CAMPAIGN DONE" not in open(LOG).read():
    time.sleep(120)
g = {}
for m in ("gpef@e8", "gpef@e4", "gpef@e0"):
    b, _ = C.best("pilot", m)
    if b is not None:
        g[m] = [round(b / 3, 6), b, round(b * 3, 6)]
print("10M team grid", g, flush=True)
C.phase("10M", g, ext=1)
C.pool("10M", [(m, C.best("pilot", m)[0], s) for m in ("adam_sr@m4", "adam_sr@m2") for s in (0, 1, 2)])
print("TEAM DONE", flush=True)
