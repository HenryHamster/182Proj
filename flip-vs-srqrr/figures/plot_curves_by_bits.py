"""Training curves over steps, families compared at matched state bits/weight (10M). Each line: seed-0 best-val lr,
mean over seeds at that lr of the 100-step mean train loss, in bits/byte. Family colors match family_bits_10M.png."""
import json, os, math, statistics as st, collections as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(D, "..", "results", "campaign_10M.jsonl"))]
g = C.defaultdict(list)
for r in R: g[r["method"]].append(r)
L2 = math.log(2)

def best(m):
    rs = g[m]
    b = min((r for r in rs if r["seed"] == 0 and not r["diverged"]), key=lambda r: r["val_bpb"])["lr"]
    rs = [r for r in rs if r["lr"] == b]
    n = min(len(r["curve"]) for r in rs)
    xs = [rs[0]["curve"][i][0] for i in range(n)]
    ys = [st.mean(r["curve"][i][1] for r in rs) / L2 for i in range(n)]
    return xs, ys, st.mean(r["val_bpb"] for r in rs), rs[0]["state_bits"]

COL = {"GPEF": "#8c564b", "RR": "#2a78d6", "SR": "#1baf7a", "Flip": "#eb6834", "Adam": "#52514e", "Muon": "#c2457a"}
P = [("≈2 bits/weight", [("gpef@e0g256", "GPEF", "-"), ("srq_rot@b0", "RR", "-"), ("srq_sr@b0", "SR", "-")]),
     ("≈4 bits/weight", [("gpef@e0", "GPEF", "-"), ("srq_rot@b2", "RR", "-"), ("srq_sr@b2", "SR", "-"),
                         ("flip@m2", "Flip", "-"), ("muon_sr@m2", "Muon+SR", "-")]),
     ("6–12 bits/weight", [("gpef@e4", "GPEF", "-"), ("gpef@e8", "GPEF", "--"), ("srq_rot@b6", "RR", "-"),
                           ("srq_sr@b6", "SR", "-"), ("adam_sr@m2", "Adam+SR", "-"), ("adam_sr@m4", "Adam+SR", "--")]),
     ("64–102 bits/weight (fp32 state)", [("fp_muon", "fp Muon", "-"), ("fp_adam", "fp Adam", "-"),
                                          ("adamw_ste", "AdamW STE", "--"), ("adam_qat", "Adam QAT", ":")])]
FAM = {"GPEF": "GPEF", "RR": "RR", "SR": "SR", "Flip": "Flip", "Muon+SR": "Muon", "Adam+SR": "Adam", "fp Muon": "Muon",
       "fp Adam": "Adam", "AdamW STE": "Adam", "Adam QAT": "Adam"}
BG = "#fcfcfb"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True, facecolor=BG)
ref = best("fp_muon")
for ax, (title, ms) in zip(axs.flat, P):
    ax.set_facecolor(BG); ax.grid(color="#e4e3df", lw=.8); ax.set_axisbelow(True)
    if not title.startswith("64"):
        ax.plot(ref[0], ref[1], color="#b5b4af", lw=1.3, ls=(0, (2, 2)), zorder=1, label="fp Muon (64)   val 1.460")
    for m, lab, ls in ms:
        xs, ys, v, bits = best(m)
        ax.plot(xs, ys, color=COL[FAM[lab]], lw=2, ls=ls, zorder=3, label=f"{lab} ({bits:.3g})   val {v:.3f}")
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", handlelength=2.2, ncol=2, columnspacing=1.2, borderaxespad=.2)
    ax.set_ylim(1.45, 3.45); ax.set_xlim(0, 8000)
for ax in axs[1]: ax.set_xlabel("training step")
for ax in axs[:, 0]: ax.set_ylabel("train loss, bits/byte (100-step mean)")
fig.suptitle("Training curves at matched state bits/weight: 10.6M model, WikiText-103. Mean of 3 seeds at each method's "
             "best-val lr; (n) = bits/weight", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(); out = os.path.join(D, "curves_by_bits_10M.png"); fig.savefig(out, dpi=150, facecolor=BG); print(out)
