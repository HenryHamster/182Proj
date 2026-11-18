"""Training state per quantized weight, broken down by component, with 10.6M val bpb (mean of 3 seeds, best lr)."""
import json, os, statistics as st, collections as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__))
def load(P, files):
    R = [json.loads(l) for f in files for l in open(os.path.join(D, "..", "results", f))]
    g = C.defaultdict(list)
    for r in R:
        if r.get("preset", P) == P: g[r["method"]].append(r)
    return g
G = {"10M": load("10M", ["campaign_10M.jsonl", "10M.jsonl"]), "pilot": load("pilot", ["campaign_pilot.jsonl"])}
def bpb(m):
    for P in ("10M", "pilot"):
        rs = G[P].get(m)
        if rs and len({r["seed"] for r in rs}) >= 3:
            b = min((r for r in rs if r["seed"] == 0 and not r["diverged"]), key=lambda r: r["val_bpb"])["lr"]
            return st.mean(r["val_bpb"] for r in rs if r["lr"] == b), rs[0]["state_bits"], P
NQ = 10616832
# (method, label, {component: bits}); "scales" absorbs the remainder to the recorded total
ROWS = [("srq_rot@b0", "SRQ rotation, b0", {"code": 2}),
        ("gpef@e0g256", "GPEF, groups of 256", {"code": 2, "codebook": .25}),
        ("gpef@e0g128", "GPEF, groups of 128", {"code": 2, "codebook": .5}),
        ("gpef@e0g64", "GPEF, groups of 64", {"code": 2, "codebook": 1}),
        ("flip@m1.58", "flip, 1.58-bit momentum", {"code": 2, "momentum": 1.58}),
        ("srq_rot@b2", "SRQ rotation, b2", {"code": 2, "remainder": 2}),
        ("gpef@e0", "GPEF, groups of 32", {"code": 2, "codebook": 2}),
        ("flip@m2", "flip, 2-bit momentum", {"code": 2, "momentum": 2}),
        ("muon_sr@m2", "Muon + SR, 2-bit momentum", {"code": 2, "momentum": 2}),
        ("flip@m2g", "flip, 2-bit grouped momentum", {"code": 2, "momentum": 2}),
        ("srq_rot@b4", "SRQ rotation, b4", {"code": 2, "remainder": 4}),
        ("flip@m4", "flip, 4-bit momentum", {"code": 2, "momentum": 4}),
        ("adam_sr@m2", "Adam + SR, 2-bit moments", {"code": 2, "momentum": 4}),
        ("srq_rot@b6", "SRQ rotation, b6", {"code": 2, "remainder": 6}),
        ("gpef@e4", "GPEF + 4-bit error feedback", {"code": 2, "codebook": 2, "error feedback": 4}),
        ("adam_sr@m4", "Adam + SR, 4-bit moments", {"code": 2, "momentum": 8}),
        ("gpef@e8", "GPEF + 8-bit error feedback", {"code": 2, "codebook": 2, "error feedback": 8}),
        ("muon_qat", "Muon + QAT (fp32 shadow)", {"code": 2, "fp32 shadow": 32, "momentum": 32}),
        ("adamw_ste", "AdamW-STE (fp32 shadow)", {"code": 2, "fp32 shadow": 32, "momentum": 64})]
COMP = [("code", "#3d3d3d", "deployed 2-bit code"), ("remainder", "#2a78d6", "SRQ sub-grid remainder"),
        ("codebook", "#8c564b", "GPEF group codebook (fp32)"), ("error feedback", "#c49c94", "GPEF error feedback"),
        ("fp32 shadow", "#6250d6", "fp32 shadow weights"),
        ("momentum", "#eb6834", "momentum / moments"), ("scales", "#b9b8b3", "scales")]
XMAX = 16
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#0b0b0b"})
fig, ax = plt.subplots(figsize=(11, 7.6), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.grid(axis="x", color="#e4e3df", lw=.8); ax.set_axisbelow(True)
used = set()
for i, (m, lab, parts) in enumerate(ROWS):
    v, tot, P = bpb(m)
    parts = dict(parts); rest = round(tot - sum(parts.values()), 3)
    if rest > 0: parts["scales"] = rest
    x = 0
    for k, c, _ in COMP:
        w = parts.get(k, 0)
        if not w: continue
        ax.barh(i, min(w, XMAX + 1 - x) if x < XMAX + 1 else 0, left=x, height=.68, color=c, edgecolor="#fcfcfb", lw=1.2)
        x += w; used.add(k)
    end = min(tot, XMAX + 1)
    note = f"{round(tot, 2):g} b/w · {tot * NQ / 8e6:.1f} MB · {v:.3f} bpb" + (" (2.65M)" if P == "pilot" else "")
    if tot > XMAX:
        ax.text(XMAX + 1 - .15, i, "⟩⟩", ha="right", va="center", color="#fcfcfb", fontsize=9, fontweight="bold")
    ax.text(end + .2, i, note, va="center", fontsize=8.3, color="#0b0b0b")
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[1] for r in ROWS]); ax.invert_yaxis()
ax.set_xlim(0, XMAX + 9); ax.set_xticks(range(0, XMAX + 1, 2))
ax.set_xlabel(f"training state per quantized weight, bits (bars over {XMAX} are cut: ⟩⟩)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for k, c, _ in COMP if k in used],
          labels=[l for k, _, l in COMP if k in used], frameon=False, fontsize=8.5, loc="upper right", title="component", title_fontsize=8.5)
fig.suptitle("Memory footprint by method", x=.01, ha="left", fontsize=12.5)
ax.set_title("Per quantized weight, packed. MB = 10.6M quantized weights. bpb = 10.6M validation, mean of 3 seeds "
             "(lower is better). Excludes fp32 embeddings/head/norms, gradient and activations (same for all).",
             loc="left", fontsize=8.3, color="#52514e", wrap=True)
fig.tight_layout(); out = os.path.join(D, "memory_footprint.png"); fig.savefig(out, dpi=150, facecolor="#fcfcfb"); print(out)
