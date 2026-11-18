import json, os, statistics as st, collections as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(D, "..", "results", "campaign_10M.jsonl"))]
g = C.defaultdict(list)
for r in R: g[r["method"]].append(r)
def best(m):
    rs = g[m]; b = min((r for r in rs if r["seed"] == 0), key=lambda r: r["val_bpb"])["lr"]
    return b, [r["val_bpb"] for r in rs if r["lr"] == b]
BASE, NEW = "#8a8a86", "#2a78d6"
ROWS = [("1  codebook divisor c", None, None),
        ("c = 4, current  (2.25 b/w)", "gpef@e0g256", BASE), ("c = 2  (2.25 b/w)", "gpefv@e0g256_c2", NEW),
        ("c = 4, current  (12 b/w)", "gpef@e8", BASE), ("c = 2  (12 b/w)", "gpefv@e8intg32_c2", NEW),
        ("2  error storage format", None, None),
        ("int8 error, current  (12 b/w)", "gpef@e8", BASE), ("fp8 e4m3 error  (12 b/w)", "gpefv@e8fp8g32_c4", NEW),
        ("int4 error, current  (8 b/w)", "gpef@e4", BASE), ("fp4 e2m1 error  (8 b/w)", "gpefv@e4fp4g32_c4", NEW),
        ("3  fp8 momentum on fp8 error  (20 b/w)", None, None),
        ("no momentum, fp8 error  (12 b/w)", "gpefv@e8fp8g32_c4", BASE),
        ("smoothed error  β 0.5", "gpefv@e8fp8g32_c4_es0.5", NEW), ("smoothed error  β 0.9", "gpefv@e8fp8g32_c4_es0.9", NEW),
        ("gradient momentum  β 0.9", "gpefv@e8fp8g32_c4_gm0.9", NEW), ("gradient momentum  β 0.99", "gpefv@e8fp8g32_c4_gm0.99", NEW)]
fig, ax = plt.subplots(figsize=(10, 6.2), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#e4e3df", lw=.8); ax.set_axisbelow(True)
ax.axvline(1.607, color="#c9c8c3", lw=1, ls=(0, (3, 3)), zorder=1)
ax.text(1.607, len(ROWS) - .35, "current best at 12 b/w, 1.607", fontsize=7.5, color="#6b6b68", ha="left")
labels = []
for i, (lab, m, c) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    if m is None:
        labels.append((y, lab, True)); continue
    labels.append((y, lab, False))
    lr, v = best(m)
    ax.scatter(v, [y] * len(v), s=22, color=c, alpha=.45, edgecolor="none", zorder=3)
    ax.scatter([st.mean(v)], [y], s=80, color=c, edgecolor="#fcfcfb", lw=1.5, zorder=4)
    ax.text(2.33, y, f"{st.mean(v):.3f}   lr {lr:g}", va="center", fontsize=8.5, color="#3d3d3d")
ax.set_yticks([y for y, _, _ in labels]); ax.set_yticklabels([l for _, l, _ in labels], fontsize=9)
for t, (_, _, h) in zip(ax.get_yticklabels(), labels):
    if h: t.set_fontweight("bold"); t.set_color("#0b0b0b")
ax.tick_params(axis="y", length=0); ax.set_xlim(1.56, 2.48)
ax.set_xlabel("validation bits per byte (lower is better); large dot: mean of 3 seeds at best lr, small dots: seeds")
fig.suptitle("Teammate's GPEF revisions: gray = current setting, blue = revision (10.6M model)",
             x=.01, ha="left", fontsize=11, color="#0b0b0b")
fig.tight_layout(); out = os.path.join(D, "gpef_revisions_10M.png"); fig.savefig(out, dpi=150, facecolor="#fcfcfb"); print(out)
