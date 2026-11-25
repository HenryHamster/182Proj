"""flip-vs-srqrr-002: the 10M comparison at 49.2M quantized weights (preset 50M), byte-level WikiText-103, on IconLab.
One worker per GPU (one run saturates an L40S). Rounds, each finished before the next starts:
  1. seed 0 at the method's 10M seed-0 best lr        2. seed 0 at lr/3 (larger models prefer lower lr)
  3. seed 0 at lr/9, only where lr/3 won              4. seed 1 at the best seed-0 lr
Methods run in priority order within a round. Resumable: skips (method, lr, seed) already recorded.
usage: python campaigns/campaign_50M.py GPU_A GPU_B"""
import json, os, queue, subprocess, sys, threading, time
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESET, OUT = "50M", os.path.join(W, "results", "campaign_50M.jsonl")
GPUS = sys.argv[1:3] if len(sys.argv) >= 3 else ["2", "3"]
# method -> 10M base lr (adam_qat: 0.03, its stable 10M setting; 0.09 won seed 0 but one seed blew up)
BASE = {"srq_rot@b0": 0.03, "srq_sr@b0": 0.03, "gpef@e0g256": 0.03, "srq_rot@b2": 0.03, "srq_sr@b2": 0.03,
        "gpef@e0": 0.03, "fp_muon": 0.01, "fp_adam": 0.03, "muon_qat": 0.01, "adamw_ste": 0.03,
        "muon_sr@m2": 0.001, "adam_sr@m2": 0.003333, "srq_rot@b6": 0.03, "srq_sr@b6": 0.03, "gpef@e4": 0.03,
        "adam_qat": 0.03}
LOCK = threading.Lock()
os.makedirs(os.path.join(W, "logs"), exist_ok=True)


def rows():
    return [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []


def rec(m, lr, s):
    r = [x for x in rows() if x["method"] == m and x["lr"] == lr and x["seed"] == s]
    return r[0] if r else None


def run_round(jobs):
    q = queue.Queue()
    for j in jobs:
        if rec(*j) is None: q.put(j)

    def worker(gpu):
        while True:
            try: m, lr, s = q.get_nowait()
            except queue.Empty: return
            with LOCK: print(time.strftime("%m-%d %H:%M:%S"), "start", m, lr, s, "gpu", gpu, flush=True)
            log = os.path.join(W, "logs", f"camp_{PRESET}_{m.replace('@', '-')}_lr{lr:g}_s{s}.log")
            with open(log, "w") as f:
                subprocess.run([sys.executable, os.path.join(W, "drive.py"), "run", "--preset", PRESET, "--method", m,
                                "--lr", str(lr), "--seed", str(s), "--eval_windows", "2000", "--out", OUT],
                               stdout=f, stderr=subprocess.STDOUT, cwd=W, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
            r = rec(m, lr, s)
            with LOCK: print(time.strftime("%m-%d %H:%M:%S"), "JOB", m, lr, s,
                             ("val_bpb %.4f%s" % (r["val_bpb"], " DIVERGED" if r["diverged"] else "")) if r else "FAILED",
                             flush=True)
    ts = [threading.Thread(target=worker, args=(g,)) for g in GPUS]
    for t in ts: t.start()
    for t in ts: t.join()


def best(m):
    rs = [r for r in rows() if r["method"] == m and r["seed"] == 0 and not r["diverged"] and r["val_bpb"] == r["val_bpb"]]
    return min(rs, key=lambda r: r["val_bpb"])["lr"] if rs else None


if __name__ == "__main__":
    lo = lambda m, k: round(BASE[m] / k, 7)
    run_round([(m, BASE[m], 0) for m in BASE]); print("ROUND 1 DONE", flush=True)
    run_round([(m, lo(m, 3), 0) for m in BASE]); print("ROUND 2 DONE", flush=True)
    run_round([(m, lo(m, 9), 0) for m in BASE if best(m) == lo(m, 3)]); print("ROUND 3 DONE", flush=True)
    run_round([(m, best(m), 1) for m in BASE if best(m) is not None]); print("CAMPAIGN DONE", flush=True)
