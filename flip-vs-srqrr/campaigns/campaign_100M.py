"""flip-vs-srqrr-004: the main methods at 99.1M quantized weights (preset 100M), byte-level WikiText-103, on IconLab.
Henry 2026-11-20: "SR RR (2 and 8 bpw is enough first), GPEF @ 0. one of adam/muon" ... "prob use fp muon but otherwise run it".
Methods: srq_sr@b0, srq_rot@b0 (2 b/w), srq_sr@b6, srq_rot@b6 (8 b/w), gpef@e0 (4 b/w), fp_muon (FP reference).
Rounds, each finished before the next starts (larger models have preferred lower lr):
  1. seed 0 at the method's 50M best lr        2. seed 0 at that lr / 3        3. seed 1 at the better seed-0 lr
GPU respect: lihuan may hold at most 2 GPUs on IconLab across all its projects. Before each job a worker waits until
its GPU has no other process and starting would not put lihuan above 2 GPUs (checked every 10 min).
Resumable: skips (method, lr, seed) already recorded.
usage: python campaigns/campaign_100M.py GPU_A GPU_B"""
import os, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_50M as C

C.PRESET = "100M"
C.OUT = os.path.join(C.W, "results", "campaign_100M.jsonl")
C.GPUS = sys.argv[1:3] if len(sys.argv) >= 3 else ["0", "2"]
BASE = {"srq_rot@b6": 0.01, "srq_sr@b6": 0.01, "gpef@e0": 0.03, "srq_rot@b0": 0.03, "srq_sr@b0": 0.01,
        "fp_muon": 0.0033333}  # 50M seed-0 best lr
CAP = 2
_ours, _lock = set(), threading.Lock()


def _gpu_procs():
    """{gpu index: [(user, cmd), ...]} for every compute process."""
    idx = dict(l.split(", ") for l in subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"], capture_output=True, text=True).stdout.split("\n") if l)
    out = {}
    for l in subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.split("\n"):
        if not l: continue
        u, pid = l.split(", ")
        ps = subprocess.run(["ps", "-o", "user=,args=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if ps: out.setdefault(idx[u], []).append(tuple(ps.split(None, 1)) if " " in ps else (ps, ""))
    return out


def _may_start(gpu):
    procs = _gpu_procs()
    if procs.get(gpu): return False                       # someone (anyone) is on this GPU
    other_lihuan = {g for g, ps in procs.items() if any(u == "lihuan" for u, _ in ps)}
    return len(other_lihuan | _ours | {gpu}) <= CAP


def _wait_gpu(gpu):
    first = True
    while True:
        with _lock:
            if _may_start(gpu):
                _ours.add(gpu); return
        if first:
            with C.LOCK: print(time.strftime("%m-%d %H:%M:%S"), "gpu", gpu, "waiting (busy or lihuan cap)", flush=True)
            first = False
        time.sleep(600)


_run = subprocess.run


def _gated_run(cmd, *a, env=None, **k):
    gpu = env["CUDA_VISIBLE_DEVICES"]
    _wait_gpu(gpu)
    try:
        return _run(cmd, *a, env=env, **k)
    finally:
        with _lock: _ours.discard(gpu)


C.subprocess = type("S", (), {"run": staticmethod(_gated_run), "STDOUT": subprocess.STDOUT})

if __name__ == "__main__":
    lo = lambda m: round(BASE[m] / 3, 7)
    C.run_round([(m, BASE[m], 0) for m in BASE]); print("ROUND 1 DONE", flush=True)
    C.run_round([(m, lo(m), 0) for m in BASE]); print("ROUND 2 DONE", flush=True)
    C.run_round([(m, C.best(m), 1) for m in BASE if C.best(m) is not None]); print("CAMPAIGN DONE", flush=True)
