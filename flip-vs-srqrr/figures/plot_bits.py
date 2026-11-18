import json, os, statistics as st, collections as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as T
D = os.path.dirname(os.path.abspath(__file__))
import sys
P = sys.argv[1] if len(sys.argv) > 1 else "pilot"
FILES = {"pilot": ["campaign_pilot.jsonl"], "10M": ["campaign_10M.jsonl", "10M.jsonl"]}[P]
R = [json.loads(l) for f in FILES for l in open(os.path.join(D, "..", "results", f))]
R = [r for r in R if r.get("preset", P) == P]
g = C.defaultdict(list)
for r in R: g[r["method"]].append(r)
def pt(m):
    rs = g.get(m)
    if not rs or len({r["seed"] for r in rs}) < 3: return None   # 3 seeds required
    rs = g[m]; b = min((r for r in rs if r["seed"] == 0 and not r["diverged"]), key=lambda r: r["val_bpb"])["lr"]
    v = [r["val_bpb"] for r in rs if r["lr"] == b]
    return rs[0]["state_bits"], v
FAM = [  # (label, methods in bit order, color, marker)
    ("SRQ rotation rounding (no momentum)", [f"srq_rot@b{b}" for b in (0, 1, 2, 4, 6, 8)], "#2a78d6", "o"),
    ("flip (quantized momentum)", ["flip@m1.58", "flip@m2", "flip@m2g", "flip@m4", "flip@m8"], "#eb6834", "s"),
    ("SRQ stochastic rounding (no momentum)", [f"srq_sr@b{b}" for b in (0, 1, 2, 4, 6, 8)], "#1baf7a", "^"),
    ("GPEF, teammate (learned group codebooks; 2.25–3 b/w: larger groups)", ["gpef@e0g256", "gpef@e0g128", "gpef@e0g64", "gpef@e0", "gpef@e4", "gpef@e8"], "#8c564b", "P"),
    ("Adam + SR, quantized moments (fp32 at 66)", ["adam_sr@m2", "adam_sr@m4", "adam_sr@m8", "dqt"], "#7a7a7a", "X"),
    ("srqrr: rotation + scale refit", [f"srqrr@b{b}" for b in (0, 1, 2, 4)], "#eda100", "D"),
    ("Muon + SR (quantized momentum)", ["muon_sr@m2", "muon_sr@m4"], "#e87ba4", "v"),
    ("GPEF + rotation-rounded error, leak 0.99 (new)", ["gpefx@e2g256_narrow_rot_b0.99", "gpefx@e2g32_narrow_rot_b0.99",
                                                     "gpefx@e4g32_narrow_rot_b0.99"], "#c0262d", "*"),
]
FL = {"flip@m1.58": "1.58-bit", "flip@m2": "2-bit", "flip@m2g": "2-bit\ngrouped", "flip@m4": "4-bit"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})
fig, ax = plt.subplots(figsize=(11.5, 6.2), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.grid(axis="y", color="#e4e3df", lw=.8); ax.set_axisbelow(True)
ax.axvspan(1.8, 4.5, color="#f0efec", zorder=0)
ax.text(1.86, {"pilot": 2.70, "10M": 2.60}[P], "≤ 4.5 bits/weight", color="#52514e", fontsize=8.5, va="top")
q = st.mean(pt("muon_qat")[1])
ax.axhline(q, color="#6250d6", lw=1.5, ls="--", zorder=2)
ax.text(120, q + .01, f"Muon + QAT, FP32 shadow (66 b/w): {q:.3f}", color="#0b0b0b", fontsize=8.5, ha="right", va="bottom")
if pt("fp_muon"):
    f = st.mean(pt("fp_muon")[1]); ax.axhline(f, color="#9a9994", lw=1.2, ls=":", zorder=2)
    ax.text(120, f + .01, f"unquantized Muon, fp32 weights (reference): {f:.3f}", color="#52514e", fontsize=8, ha="right", va="bottom")
for m, lab, c, dy in (("muon_qat", "", "#6250d6", 4), ("adamw_ste", "AdamW-STE (teammate)", "#3d3d3d", 4),
                      ("adam_qat", "Adam + QAT (lr .09, 2 of 3 seeds unstable)", "#3d3d3d", -4)):
    if pt(m) is None: continue
    x, v = pt(m); ax.scatter([x] * len(v), v, s=16, color=c, alpha=.4, edgecolor="none", zorder=3)
    ax.scatter([x], [st.mean(v)], s=90, marker="*", color=c, zorder=5)
    ax.annotate(f"{lab}: {st.mean(v):.3f}" if lab else "", (x, st.mean(v)), xytext=(-9, dy), textcoords="offset points",
                ha="right", va="center", fontsize=7.5, color="#0b0b0b")
for lab, ms, c, mk in FAM:
    xs, ys = [], []
    for m in ms:
        if pt(m) is None: continue
        x, v = pt(m)
        ax.scatter([x] * len(v), v, s=16, color=c, alpha=.4, edgecolor="none", zorder=3)
        xs.append(x); ys.append(st.mean(v))
        if m in FL:
            off = {"flip@m1.58": (-9, 0), "flip@m2": (10, 0), "flip@m2g": (10, 0), "flip@m4": (-8, -12)}[m]
            ax.annotate(FL[m].replace(chr(10), " "), (x, st.mean(v)), xytext=off, textcoords="offset points",
                        ha="right" if off[0] < 0 else "left", va="center", fontsize=7.5, color="#b8471d")
    if not xs: continue
    ax.plot(xs, ys, color=c, lw=2, marker=mk, ms=15 if mk == "*" else 8, mec="#fcfcfb", mew=1.5, zorder=4, label=lab)
for m, lab, c, mk, dy in (("gpef_rot@e0", "rotation on GPEF codebooks", "#2a78d6", "P", -9),
                          ("gpef_sr@e0", "SR on GPEF codebooks", "#1baf7a", "P", 9)):   # format-matched check
    rs = [r for r in g.get(m, []) if not r["diverged"]]
    if not any(r["seed"] == 0 for r in rs): continue
    b = min((r for r in rs if r["seed"] == 0), key=lambda r: r["val_bpb"])["lr"]
    v = [r["val_bpb"] for r in rs if r["lr"] == b]; x = rs[0]["state_bits"]
    ax.scatter([x] * len(v), v, s=16, color=c, alpha=.4, edgecolor="none", zorder=3)
    ax.scatter([x], [st.mean(v)], s=90, marker=mk, facecolor="#fcfcfb", edgecolor=c, lw=2, zorder=5)
    ax.annotate(f"{lab}: {st.mean(v):.3f}" + ("" if len(v) >= 3 else f" ({len(v)} seed{'s' if len(v) > 1 else ''})"),
                (x, st.mean(v)), xytext=(10, dy), textcoords="offset points", fontsize=7.5, color="#0b0b0b", va="center")
for m, lab, c, mk, dy in (("gpefx@e4g32_narrow_near_b1", "error range fitted, nearest", "#c0262d", "o", 0),
                          ("gpefx@e4g32_narrow_rot_b1", "+ rotation, no leak", "#c0262d", "o", 0),
                          ("gpefx@e4g32_narrow_rot_b0.9", "+ rotation, leak 0.9", "#c0262d", "o", 0),
                          ("gpef_frz@e0_near", "GPEF frozen, nearest", "#8c564b", "P", -12),
                          ("gpef_frz@e0_rot", "GPEF frozen, rotation", "#8c564b", "P", 12),
                          ("gpefr@g256_after", "refit, then rotate", "#2a78d6", "D", 0),
                          ("gpefr@g256_before", "rotate, then refit", "#2a78d6", "s", 0),
                          ("gpefr@g32_before", "rotate, then refit", "#2a78d6", "s", 0),
                          ("gpefr@g256_after_l0.3", "refit + rotation of 0.3 x residual", "#2a78d6", "^", -2),
                          ("gpefr@g256_after_l0.1", "refit + rotation of 0.1 x residual", "#2a78d6", "v", 6)):
    if pt(m) is None: continue
    x, v = pt(m); x = x * (1.06 if m.startswith("gpefx") else {"gpefr@g256_after_l0.3": 1.09, "gpefr@g256_after_l0.1": 1.18}.get(m, 1.0))
    ax.scatter([x] * len(v), v, s=16, color=c, alpha=.4, edgecolor="none", zorder=3)
    ax.scatter([x], [st.mean(v)], s=60, marker=mk, facecolor="#fcfcfb", edgecolor=c, lw=1.8, zorder=5)
    L = m == "gpefr@g32_before"
    ax.annotate(f"{lab}: {st.mean(v):.3f}", (x, st.mean(v)), xytext=(-9 if L else 9, dy), textcoords="offset points",
                fontsize=7.5, color="#0b0b0b", va="center", ha="right" if L else "left")
m = "gpefx@e4g32_narrow_rot_b0.99"
if pt(m): ax.annotate(f"leak 0.99: {st.mean(pt(m)[1]):.3f}", (8, st.mean(pt(m)[1])), xytext=(-6, -16),
                      textcoords="offset points", ha="right", fontweight="bold", fontsize=7.5, color="#0b0b0b")
for m, lab in (("gpefx@e2g256_narrow_rot_b0.99", "2-bit error, groups of 256"), ("gpefx@e2g32_narrow_rot_b0.99", "2-bit error, groups of 32")):
    if pt(m): ax.annotate(f"{lab}: {st.mean(pt(m)[1]):.3f}", (pt(m)[0], st.mean(pt(m)[1])), xytext=(10, 0),
                          textcoords="offset points", ha="left", va="center", fontsize=7.5, color="#0b0b0b")
ax.set_xscale("log", base=2); ax.set_xlim(1.8, 128)
ax.set_xticks([2, 3, 4, 6, 8, 12, 16, 32, 66, 102]); ax.xaxis.set_major_formatter(T.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_locator(T.NullLocator())
ax.set_xlabel("training state per weight, bits (2-bit deployed weight + optimizer state; log scale)")
ax.set_ylabel("validation bits per byte (lower is better)")
ax.set_ylim(*{"pilot": (1.70, 2.72), "10M": (1.42, 2.78)}[P])
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
TT = {"pilot": "2.65M model, 2-bit weights from scratch, WikiText-2, 2.5k steps",
      "10M": "10.6M model, 2-bit weights from scratch, WikiText-103, 8k steps"}[P]
ax.set_title(TT + ". Lines: mean of 3 seeds at best lr; dots: seeds",
             loc="left", fontsize=9, color="#52514e")
fig.suptitle("Accuracy vs optimizer state budget", x=.01, ha="left", fontsize=12.5, color="#0b0b0b")
fig.tight_layout(); out = os.path.join(D, f"bits_vs_bpb_{P}.png"); fig.savefig(out, dpi=150, facecolor="#fcfcfb"); print(out)
