"""flip-vs-srqrr-001 addendum: srq_rot (rotation rounding without refit) was the best SRQ variant at 2.65M but was not in
the 10M list. Waits for the main campaign to finish (CAMPAIGN DONE), then runs the same 10M protocol for srq_rot@b0/b2/b4."""
import os, time
import campaign as C
LOG = os.path.join(C.W, "logs", "campaign.log")
while "CAMPAIGN DONE" not in open(LOG).read():
    time.sleep(120)
g = {}
for m in ("srq_rot@b0", "srq_rot@b2", "srq_rot@b4"):
    b, _ = C.best("pilot", m)
    g[m] = [round(b / 3, 6), b, round(b * 3, 6)]
print("10M rot grid", g, flush=True)
C.phase("10M", g, ext=1)
print("ROT DONE", flush=True)
