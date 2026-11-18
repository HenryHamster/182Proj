"""flip-vs-srqrr-001 campaign: state-budget sweep, 2-bit weights from scratch, byte-level WikiText.
Phase A (pilot, 2.65M): seed-0 lr grid per method (edge extension x3, up to 2), then seeds 1,2 at the best lr.
Phase B (10M, 10.6M): seed-0 grid {best/3, best, 3*best} around the pilot best (edge extension up to 1), then seeds
1,2 at the best lr. Methods run in priority order. Resumable: skips (preset, method, lr, seed) already recorded.
Jobs start only while GPU memory used + RESERVE stays under the 72% shared limit."""
import json, os, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
OUT = {p: os.path.join(W, "results", f"campaign_{p}.jsonl") for p in ("pilot", "10M")}
LIMIT_MB, RESERVE_MB = int(0.72 * 16303), {"pilot": 1500, "10M": 3000}
PAR = {"pilot": 4, "10M": 4}
SRQ = [0.003, 0.01, 0.03]; FLIP = [0.001, 0.003, 0.009]; MUON = [0.003, 0.01, 0.03]
# priority order: headline comparisons first, then ablations and references
PILOT = {**{f"srqrr@b{b}": SRQ for b in (0, 1, 2, 4)},
         **{m: FLIP for m in ("flip@m2", "flip@m2g", "flip@m1.58", "flip@m4")},
         **{f"srq_sr@b{b}": SRQ for b in (0, 1, 2, 4)},
         **{f"srq_rot@b{b}": SRQ for b in (0, 1, 2, 4)},
         "srqrr_k10@b0": SRQ, "srqrr_k10@b2": SRQ,
         "muon_sr@m2": MUON, "muon_sr@m4": MUON, "muon_qat": MUON}
DONE_10M = {"flip@m4", "muon_qat", "muon_sr@m4"}          # already in results/10M.jsonl (flip-optimizer-scale-001)
TENM = ["srqrr@b0", "srqrr@b1", "srqrr@b2", "srqrr@b4", "flip@m2", "flip@m2g", "flip@m1.58",
        "srq_sr@b0", "srq_sr@b2", "srq_sr@b4", "muon_sr@m2"]
LOCK = threading.Lock()
os.makedirs(os.path.join(W, "logs"), exist_ok=True)


def rows(p):
    return [json.loads(l) for l in open(OUT[p])] if os.path.exists(OUT[p]) else []


def have(p, m, lr, s):
    return any(r["method"] == m and r["lr"] == lr and r["seed"] == s for r in rows(p))


def gpu_used():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=30).stdout.split()[0])
    except Exception:
        return LIMIT_MB


def job(p, m, lr, s):
    if have(p, m, lr, s):
        return
    with LOCK:                                         # admission: wait for memory headroom, then let the job allocate
        while gpu_used() + RESERVE_MB[p] > LIMIT_MB:
            time.sleep(30)
        print(time.strftime("%H:%M:%S"), "start", p, m, lr, s, flush=True)
        log = os.path.join(W, "logs", f"camp_{p}_{m.replace('@', '-')}_lr{lr:g}_s{s}.log")
        f = open(log, "w")
        args = [sys.executable, os.path.join(W, "drive.py"), "run", "--preset", p, "--method", m, "--lr", str(lr),
                "--seed", str(s), "--out", OUT[p]] + (["--eval_windows", "2000"] if p == "10M" else [])
        proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, cwd=W)
        time.sleep(12 if p == "pilot" else 40)         # let it reach peak memory before admitting the next job
    proc.wait(); f.close()
    r = [x for x in rows(p) if x["method"] == m and x["lr"] == lr and x["seed"] == s]
    print(time.strftime("%H:%M:%S"), "JOB", p, m, lr, s, ("val_bpb %.4f" % r[0]["val_bpb"]) if r else "FAILED", flush=True)


def pool(p, jobs):
    with ThreadPoolExecutor(PAR[p]) as ex:
        list(ex.map(lambda j: job(p, *j), jobs))


def best(p, m):
    rs = [r for r in rows(p) if r["method"] == m and r["seed"] == 0 and r["val_bpb"] == r["val_bpb"] and not r["diverged"]]
    if not rs:
        return None, []
    return min(rs, key=lambda r: r["val_bpb"])["lr"], sorted(r["lr"] for r in rs)


def phase(p, grid, ext, seeds=(1, 2)):
    pool(p, [(m, lr, 0) for m, lrs in grid.items() for lr in lrs])
    for _ in range(ext):
        e = []
        for m in grid:
            b, done = best(p, m)
            if b is None: continue
            if b == done[0]: e.append((m, round(b / 3, 6), 0))
            elif b == done[-1]: e.append((m, round(b * 3, 6), 0))
        if not e: break
        print("extend", p, e, flush=True); pool(p, e)
    pool(p, [(m, best(p, m)[0], s) for m in grid if best(p, m)[0] is not None for s in seeds])


if __name__ == "__main__":
    phase("pilot", PILOT, ext=2)
    print("PILOT DONE", flush=True)
    g = {}
    for m in TENM:
        b, _ = best("pilot", m)
        if b is not None:
            g[m] = [round(b / 3, 6), b, round(b * 3, 6)]
    print("10M grid", g, flush=True)
    phase("10M", g, ext=1)
    print("CAMPAIGN DONE", flush=True)
