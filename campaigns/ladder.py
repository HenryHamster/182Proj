"""10.6M ladder: seed-0 lr grid per method (3 runs at a time), extend the grid while the best val is at an edge
(max 2 extensions), then seeds 1,2 at the best lr. Resumable: skips (method, lr, seed) already in the results file."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
PRESET, OUT, PAR = "10M", os.path.join(W, "results", "10M.jsonl"), 3
GRID = {"flip@m4": [0.001, 0.003, 0.009], "muon_qat": [0.003, 0.01, 0.03], "muon_sr@m4": [0.003, 0.01, 0.03]}
SEEDS = {"flip@m4": (1, 2), "muon_qat": (1, 2), "muon_sr@m4": ()}

def rows():
    return [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []

def job(m, lr, s):
    if any(r["method"] == m and r["lr"] == lr and r["seed"] == s for r in rows()):
        return
    log = os.path.join(W, "logs", f"10M_{m.replace('@', '-')}_lr{lr:g}_s{s}.log")
    print(time.strftime("%H:%M:%S"), "start", m, lr, s, flush=True)
    with open(log, "w") as f:
        subprocess.run([sys.executable, os.path.join(W, "drive.py"), "run", "--preset", PRESET, "--method", m, "--lr", str(lr),
                        "--seed", str(s), "--out", OUT, "--eval_windows", "2000"], stdout=f, stderr=subprocess.STDOUT, cwd=W)
    r = [x for x in rows() if x["method"] == m and x["lr"] == lr and x["seed"] == s]
    print(time.strftime("%H:%M:%S"), "JOB", m, lr, s, ("val_bpb %.4f" % r[0]["val_bpb"]) if r else "FAILED", flush=True)

def pool(jobs):
    with ThreadPoolExecutor(PAR) as ex:
        list(ex.map(lambda j: job(*j), jobs))

def best(m):
    rs = [r for r in rows() if r["method"] == m and r["seed"] == 0 and r["val_bpb"] == r["val_bpb"]]
    return min(rs, key=lambda r: r["val_bpb"])["lr"], sorted(r["lr"] for r in rs)

pool([(m, lr, 0) for m, lrs in GRID.items() for lr in lrs])
for _ in range(2):
    ext = []
    for m in GRID:
        b, done = best(m)
        if b == done[0]: ext.append((m, round(b / 3, 6), 0))
        elif b == done[-1]: ext.append((m, round(b * 3, 6), 0))
    if not ext: break
    print("extend", ext, flush=True); pool(ext)
pool([(m, best(m)[0], s) for m in GRID for s in SEEDS[m]])
print("LADDER DONE", flush=True)
