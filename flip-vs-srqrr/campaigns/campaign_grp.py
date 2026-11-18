"""flip-vs-srqrr-001, format-matched check (Henry 2026-11-10 'continue' after the offer): rotation / stochastic rounding
on GPEF's group-32 learned codebooks (team_rot.py), same protocol as the other methods. Pilot: seed-0 grid
.003/.01/.03 (edge extension x3 up to 2), seeds 1,2. 10M: grid {best/3, best, 3 best} around the pilot best (1 extension),
seeds 1,2. Compared with gpef@e0 (same format, nearest rounding) and srq_rot@b2 (same 4 b/w, per-matrix grid)."""
import campaign as C
M = ("gpef_rot@e0", "gpef_sr@e0")
C.PAR["pilot"] = 4
C.phase("pilot", {m: [0.003, 0.01, 0.03] for m in M}, ext=2)
print("GRP PILOT DONE", flush=True)
g = {}
for m in M:
    b, _ = C.best("pilot", m)
    g[m] = [round(b / 3, 6), b, round(b * 3, 6)]
print("10M grp grid", g, flush=True)
C.PAR["10M"] = 4
C.phase("10M", g, ext=1)
print("GRP DONE", flush=True)
