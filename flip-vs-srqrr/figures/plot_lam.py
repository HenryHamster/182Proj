import json, os, statistics as st, collections as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(D, "..", "results", "campaign_10M.jsonl"))]
g = C.defaultdict(list)
for r in R: g[r["method"]].append(r)
def best(m):
    rs = g[m]; b = min((r for r in rs if r["seed"] == 0), key=lambda r: r["val_bpb"])["lr"]
    v = [r["val_bpb"] for r in rs if r["lr"] == b]; c = [r["changed_frac_mean"] for r in rs if r["lr"] == b]
    return b, v, st.mean(c)
ROWS = [("GPEF, nearest + refit  (λ = 0)", "gpef@e0g256", "#8c564b"),
        ("refit + rotation of 0.1 × residual", "gpefr@g256_after_l0.1", "#2a78d6"),
        ("refit + rotation of 0.3 × residual", "gpefr@g256_after_l0.3", "#2a78d6"),
        ("refit + rotation of full residual  (λ = 1)", "gpefr@g256_after", "#2a78d6"),
        ("rotation, codebooks frozen (2 b/w)", "gpef_frz@e0_rot", "#7a7a7a"),
        ("rotation, fixed grid, SRQ (2 b/w)", "srq_rot@b0", "#7a7a7a")]
fig, ax = plt.subplots(figsize=(9, 3.6), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#e4e3df", lw=.8); ax.set_axisbelow(True)
for i, (lab, m, c) in enumerate(ROWS):
    lr, v, ch = best(m); y = len(ROWS) - 1 - i
    ax.scatter(v, [y] * len(v), s=22, color=c, alpha=.45, edgecolor="none", zorder=3)
    ax.scatter([st.mean(v)], [y], s=80, color=c, edgecolor="#fcfcfb", lw=1.5, zorder=4)
    ax.text(2.42, y, f"{st.mean(v):.3f}   lr {lr:g}, {100 * ch:.1f}% symbols changed/step", va="center", fontsize=8.5, color="#3d3d3d")
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS][::-1], fontsize=9)
ax.tick_params(axis="y", length=0); ax.set_xlim(2.18, 2.72)
ax.set_xlabel("validation bits per byte (lower is better); dots: 3 seeds at best lr")
fig.suptitle("Splitting the step: codebook at full lr, rotation on λ × residual (2.25 b/w, 10.6M model)",
             x=.01, ha="left", fontsize=11, color="#0b0b0b")
fig.tight_layout(); out = os.path.join(D, "lambda_split_10M.png"); fig.savefig(out, dpi=150, facecolor="#fcfcfb"); print(out)
