"""Training curves (train loss, 100-step means, in bits/byte) for the flip-vs-srq campaign.
Per method: seed-0 best lr (same rule as plot_bits.py), mean over the 3 seeds at that lr."""
import json, os, sys, math, statistics as st, collections as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__))
P = sys.argv[1] if len(sys.argv) > 1 else "10M"
FILES = {"pilot": ["campaign_pilot.jsonl"], "10M": ["campaign_10M.jsonl", "10M.jsonl"]}[P]
R = [json.loads(l) for f in FILES for l in open(os.path.join(D, "..", "results", f))]
R = [r for r in R if r.get("preset", P) == P]
g = C.defaultdict(list)
for r in R: g[r["method"]].append(r)
L2 = math.log(2)

def best(m):
    rs = g.get(m)
    if not rs or len({r["seed"] for r in rs}) < 3: return None
    b = min((r for r in rs if r["seed"] == 0 and not r["diverged"]), key=lambda r: r["val_bpb"])["lr"]
    rs = [r for r in rs if r["lr"] == b]
    n = min(len(r["curve"]) for r in rs)
    xs = [rs[0]["curve"][i][0] for i in range(n)]
    ys = [st.mean(r["curve"][i][1] for r in rs) / L2 for i in range(n)]
    return xs, ys, rs

COL = {"srq_rot": "#2a78d6", "srq_sr": "#1baf7a", "flip": "#eb6834", "gpef": "#8c564b",
       "adam_sr": "#7a7a7a", "muon_sr": "#e87ba4"}
NAME = {"srq_rot@b0": "rotation, 2", "srq_sr@b0": "SR, 2", "gpef@e0g256": "GPEF g256, 2.25",
        "gpef@e0g128": "GPEF g128, 2.5", "srq_rot@b1": "rotation, 3", "srq_sr@b1": "SR, 3",
        "gpef@e0g64": "GPEF g64, 3", "flip@m1.58": "flip 1.58-bit, 3.7", "srq_rot@b2": "rotation, 4",
        "srq_sr@b2": "SR, 4", "gpef@e0": "GPEF g32, 4", "flip@m2": "flip 2-bit, 4.1",
        "flip@m2g": "flip 2-bit grp, 4.25", "muon_sr@m2": "Muon+SR, 4.1", "srq_rot@b4": "rotation, 6",
        "srq_sr@b4": "SR, 6", "flip@m4": "flip 4-bit, 6.1", "srq_rot@b6": "rotation, 8",
        "gpef@e4": "GPEF e4, 8", "adam_sr@m4": "Adam+SR, 10.25", "gpef@e8": "GPEF e8, 12"}
PANELS = [("2 – 2.5 bits/weight", ["srq_sr@b0", "srq_rot@b0", "gpef@e0g256", "gpef@e0g128"]),
          ("3 – 4.25 bits/weight", ["srq_sr@b1", "srq_rot@b1", "gpef@e0g64", "flip@m1.58", "srq_sr@b2",
                                    "srq_rot@b2", "gpef@e0", "flip@m2", "flip@m2g", "muon_sr@m2"]),
          ("6 – 12 bits/weight", ["srq_sr@b4", "srq_rot@b4", "flip@m4", "srq_rot@b6", "gpef@e4", "adam_sr@m4", "gpef@e8"])]
DASH = {}  # second member of the same family in a panel gets a dashed line
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axs = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True, facecolor="#fcfcfb")
q = best("muon_qat")
YL = {"pilot": (1.75, 3.2), "10M": (1.5, 3.2)}[P]
for ax, (title, ms) in zip(axs, PANELS):
    ax.set_facecolor("#fcfcfb"); ax.grid(axis="y", color="#e4e3df", lw=.8); ax.set_axisbelow(True)
    if q: ax.plot(q[0], q[1], color="#6250d6", lw=1.3, ls=":", label="Muon+QAT fp32 shadow, 66")
    seen = C.Counter(); ends = []
    for m in ms:
        b = best(m)
        if b is None: continue
        fam = m.split("@")[0]; seen[fam] += 1
        ls = ["-", "--", "-."][seen[fam] - 1]
        ax.plot(b[0], b[1], color=COL[fam], lw=1.8, ls=ls, label=NAME[m])
        ends.append((b[1][-1], NAME[m], COL[fam]))
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xlabel("training step")
    ax.set_ylim(*YL); ax.set_xlim(0, b[0][-1] if b else None)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right", title="method, bits/weight", title_fontsize=7.8)
axs[0].set_ylabel("training loss, bits per byte (100-step mean; lower is better)")
TT = {"pilot": "2.65M model, WikiText-2, 2.5k steps", "10M": "10.6M model, WikiText-103, 8k steps (batch 64×256)"}[P]
fig.suptitle(f"Training curves: {TT}. Mean of 3 seeds at each method's chosen learning rate", x=.01, ha="left", fontsize=12)
fig.tight_layout(); out = os.path.join(D, f"train_curves_{P}.png"); fig.savefig(out, dpi=150, facecolor="#fcfcfb"); print(out)

# steps to reach thresholds (for the text)
for thr in {"pilot": (2.4, 2.2), "10M": (2.4, 2.2, 2.0)}[P]:
    row = []
    for _, ms in PANELS:
        for m in ms:
            b = best(m)
            if b is None: continue
            s = next((x for x, y in zip(*b[:2]) if y <= thr), None)
            row.append(f"{m}:{s}")
    print(thr, " ".join(row))
for m in ["srq_rot@b0", "srq_rot@b2", "gpef@e0", "gpef@e0g256", "flip@m2g", "flip@m4", "muon_qat"]:
    b = best(m)
    if b: print(m, "s/step", round(st.mean(r["train_seconds"] for r in b[2]) / b[0][-1], 3),
                "final train", round(b[1][-1], 3), "val", round(st.mean(r["val_bpb"] for r in b[2]), 3))
