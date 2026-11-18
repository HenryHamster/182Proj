import json, sys, os, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
D = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for f in ("pilot.jsonl", "10M.jsonl") for l in open(os.path.join(D, "..", "results", f))]
C = {"muon_qat": "#2a78d6", "flip@m4": "#eb6834", "muon_sr@m4": "#1baf7a"}
N = {"muon_qat": "Muon+QAT (66 b/w)", "flip@m4": "flip (6.2 b/w)", "muon_sr@m4": "Muon+SR (6.2 b/w)"}
BEST = {"muon_qat": 0.01, "flip@m4": 0.003, "muon_sr@m4": 0.01}
SZ = {"pilot": 2.65, "10M": 10.6}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="#fcfcfb")
for ax in (a, b): ax.set_facecolor("#fcfcfb"); ax.grid(axis="y", color="#e4e3df", lw=.8); ax.set_axisbelow(True)
for m in C:
    xs, ys = [], []
    for p in ("pilot", "10M"):
        v = [r["val_bpb"] for r in R if r["preset"] == p and r["method"] == m and r["lr"] == BEST[m]]
        a.scatter([SZ[p]] * len(v), v, s=22, color=C[m], alpha=.45, edgecolor="none", zorder=3)
        xs.append(SZ[p]); ys.append(st.mean(v))
    a.plot(xs, ys, color=C[m], lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2, zorder=4)
    a.annotate(N[m], (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", color="#0b0b0b", fontsize=9)
for p in ("pilot", "10M"):
    q = st.mean(r["val_bpb"] for r in R if r["preset"] == p and r["method"] == "muon_qat" and r["lr"] == .01)
    f = st.mean(r["val_bpb"] for r in R if r["preset"] == p and r["method"] == "flip@m4" and r["lr"] == .003)
    a.annotate(f"flip − QAT\n{f - q:.3f}", (SZ[p], (q + f) / 2), xytext=(-12, 0), textcoords="offset points", ha="right", va="center", color="#52514e", fontsize=8.5)
a.set_xscale("log"); a.set_xticks([2.65, 10.6]); a.set_xticklabels(["2.65M\nWikiText-2, 2.5k steps", "10.6M\nWikiText-103, 8k steps"])
a.set_xlim(1.25, 22); a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); a.set_ylabel("validation bits per byte (lower is better)")
a.set_title("Best learning rate per method (dots = seeds)", loc="left", fontsize=10.5, color="#0b0b0b")
for m in C:
    rs = sorted((r for r in R if r["preset"] == "10M" and r["method"] == m and r["seed"] == 0), key=lambda r: r["lr"])
    b.plot([r["lr"] for r in rs], [r["val_bpb"] for r in rs], color=C[m], lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2, label=N[m])
b.set_xscale("log"); b.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); b.set_xticks([.001, .003, .01, .03]); b.set_xticklabels(["0.001", "0.003", "0.01", "0.03"])
b.set_xlabel("learning rate (flip: budget ρ), grid units"); b.set_ylabel("validation bits per byte")
b.set_title("10.6M learning-rate sweep, seed 0", loc="left", fontsize=10.5, color="#0b0b0b")
b.legend(frameon=False, fontsize=9, loc="center right")
fig.suptitle("Flip optimizer vs Muon baselines, 2-bit weights trained from scratch", x=.01, ha="left", fontsize=12, color="#0b0b0b")
fig.tight_layout(); out = os.path.join(D, "flip_scaling.png"); fig.savefig(out, dpi=150, facecolor="#fcfcfb"); print(out)
