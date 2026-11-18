"""Val bpb vs state bits/weight, one line per optimizer family (10M). Points: 3-seed mean at seed-0 best lr; bars: seed range."""
import json, os, statistics as st, collections as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(D, "..", "results", "campaign_10M.jsonl"))]
g = C.defaultdict(list)
for r in R: g[r["method"]].append(r)

def best(m):
    rs = g[m]
    b = min((r for r in rs if r["seed"] == 0 and not r["diverged"]), key=lambda r: r["val_bpb"])["lr"]
    v = [r["val_bpb"] for r in rs if r["lr"] == b]
    return rs[0]["state_bits"], st.mean(v), min(v), max(v)

F = [("GPEF", "#8c564b", ["gpef@e0g256", "gpef@e0g128", "gpef@e0g64", "gpef@e0", "gpef@e4", "gpef@e8"]),
     ("RR (rotation)", "#2a78d6", ["srq_rot@b0", "srq_rot@b2", "srq_rot@b4", "srq_rot@b6"]),
     ("SR", "#1baf7a", ["srq_sr@b0", "srq_sr@b2", "srq_sr@b4", "srq_sr@b6"]),
     ("Flip", "#eb6834", ["flip@m1.58", "flip@m2", "flip@m2g"]),
     ("Adam (+SR, fp)", "#52514e", ["adam_sr@m2", "adam_sr@m4", "fp_adam"]),
     ("Muon (+SR, fp)", "#c2457a", ["muon_sr@m2", "fp_muon"])]
EXTRA = [("Adam QAT", "#52514e", "adam_qat", "s"), ("AdamW STE", "#52514e", "adamw_ste", "D")]
BG = "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=BG); ax.set_facecolor(BG)
ax.grid(color="#e4e3df", lw=.8); ax.set_axisbelow(True)
for name, c, ms in F:
    pts = sorted(best(m) for m in ms)
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    ax.plot(x, y, color=c, lw=2, marker="o", ms=7, mec=BG, mew=1.5, label=name, zorder=3)
    ax.vlines(x, [p[2] for p in pts], [p[3] for p in pts], color=c, lw=1, alpha=.5, zorder=2)
for name, c, m, mk in EXTRA:
    b, v, lo, hi = best(m)
    ax.plot([b], [v], color=c, marker=mk, ms=7, ls="none", mfc=BG, mew=1.6, label=f"{name}", zorder=3)
ax.set_xscale("log")
T = [2, 3, 4, 6, 8, 12, 20, 32, 64, 100]
ax.set_xticks(T); ax.set_xticklabels([str(t) for t in T]); ax.minorticks_off()
ax.set_xlabel("optimizer + weight state, bits per weight (log scale)")
ax.set_ylabel("validation bits per byte (lower is better)")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Validation bpb vs state bits/weight by optimizer family: 10.6M model, WikiText-103 "
             "(mean of 3 seeds at best lr; bars = seed range)", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(); out = os.path.join(D, "family_bits_10M.png"); fig.savefig(out, dpi=150, facecolor=BG); print(out)
