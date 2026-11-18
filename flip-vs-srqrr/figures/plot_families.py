"""Training curves per optimizer family, 10M campaign. Each line: seed-0 best lr, mean over seeds at that lr
(train loss, 100-step means, bits/byte). Within a panel darker = more state bits/weight. Gray dotted = fp Muon."""
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
    return xs, ys, st.mean(r["val_bpb"] for r in rs), len(rs), rs[0]["state_bits"]

# (panel title, ramp light->dark, [(method, label)]) ordered by bits
P = [("SR (stochastic rounding)", ["#9fd9c2", "#4fbf95", "#1b9467", "#0b5e40"],
      [("srq_sr@b0", "2 b/w"), ("srq_sr@b2", "4"), ("srq_sr@b4", "6"), ("srq_sr@b6", "8")]),
     ("RR (rotation rounding)", ["#a9c9f0", "#5d9be3", "#2a6cc4", "#163f7a"],
      [("srq_rot@b0", "2 b/w"), ("srq_rot@b2", "4"), ("srq_rot@b4", "6"), ("srq_rot@b6", "8")]),
     ("GPEF (original)", ["#e2c3b8", "#c48e7b", "#a0644f", "#7a4535", "#55291c", "#2e140c"],
      [("gpef@e0g256", "2.25 b/w"), ("gpef@e0g128", "2.5"), ("gpef@e0g64", "3"), ("gpef@e0", "4"),
       ("gpef@e4", "8"), ("gpef@e8", "12")]),
     ("Adam", ["#cfcfcb", "#9e9e99", "#6e6e69", "#45453f", "#1f1f1b"],
      [("adam_sr@m2", "Adam+SR, 6.1 b/w"), ("adam_sr@m4", "Adam+SR, 10.1"), ("adam_qat", "Adam QAT, 98"),
       ("adamw_ste", "AdamW STE, 102"), ("fp_adam", "fp Adam, 96")]),
     ("Muon", ["#f0b3cb", "#c2457a"],
      [("muon_sr@m2", "Muon+SR, 4.1 b/w"), ("fp_muon", "fp Muon, 64")]),
     ("Flip", ["#f6b99a", "#eb6834", "#a83d12"],
      [("flip@m1.58", "1.58-bit, 3.6 b/w"), ("flip@m2", "2-bit, 4.1"), ("flip@m2g", "2-bit grp, 4.25")])]
BG = "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axs = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True, facecolor=BG)
ref = best("fp_muon")
for ax, (title, ramp, ms) in zip(axs.flat, P):
    ax.set_facecolor(BG); ax.grid(axis="y", color="#e4e3df", lw=.8); ax.set_axisbelow(True)
    if title != "Muon":
        ax.plot(ref[0], ref[1], color="#b5b4af", lw=1.3, ls=(0, (2, 2)), zorder=1)
    for (m, lab), c in zip(ms, ramp):
        xs, ys, v, n, bits = best(m)
        ax.plot(xs, ys, color=c, lw=2, zorder=3,
                label=f"{lab}   val {v:.3f}" + ("" if n >= 3 else f" (n={n})"))
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.6)
    ax.set_ylim(1.45, 3.0); ax.set_xlim(0, 8000)
for ax in axs[1]: ax.set_xlabel("training step")
for ax in axs[:, 0]: ax.set_ylabel("train loss, bits/byte (lower is better)")
fig.suptitle("Training curves by optimizer family: 10.6M model, WikiText-103. Mean of 3 seeds at each method's best lr; "
             "darker = more state bits/weight; gray dotted = fp Muon (1.460)", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(); out = os.path.join(D, "family_curves_10M.png"); fig.savefig(out, dpi=150, facecolor=BG); print(out)
