"""Paper figures 1-3, drawn at print size (5.5 in text width, 8 pt text) so nothing is rescaled in LaTeX.

Selection rule everywhere: best lr = lowest seed-0 val bpb among non-diverged runs; plotted value = mean over all
seeds at that lr; error bar / band = min-max over those seeds (3 seeds at 10M, 2 at 50M and 100M).
Val split only (test aliased to val). Writes fig1_frontier.pdf/png, fig2_curves.pdf/png, fig3_scale.pdf/png.
"""
import json, math, statistics as st, collections as C
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
from matplotlib.lines import Line2D

HERE = Path(__file__).parent; RES = HERE.parent / "results"
W = 5.5
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#3d3c39", "xtick.color": "#1f1f1d",
                     "ytick.color": "#1f1f1d", "axes.linewidth": .8, "pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix"})
INK, MUTED, GRID = "#1f1f1d", "#6b6a66", "#e6e5e1"
# One color + marker per family, identical in all figures.
FAM = {"GPEF": ("#8c564b", "o"), "RR": ("#2a78d6", "s"), "SR": ("#1baf7a", "^"), "Flip": ("#eb6834", "D"),
       "Adam": ("#52514e", "v"), "Muon": ("#c2457a", "P")}


def load(*names, preset=None):
    g = C.defaultdict(list)
    for n in names:
        for l in open(RES / n):
            r = json.loads(l)
            if preset and r.get("preset", preset) != preset:
                continue
            g[r["method"]].append(r)
    return g


def best(g, m):
    rs = g[m]
    lr = min((r for r in rs if r["seed"] == 0 and not r["diverged"]), key=lambda r: r["val_bpb"])["lr"]
    rs = [r for r in rs if r["lr"] == lr]
    v = [r["val_bpb"] for r in rs]
    return dict(bits=rs[0]["state_bits"], mean=st.mean(v), lo=min(v), hi=max(v), runs=rs)


def style(ax):
    ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)


def logx(ax, ticks, lim):
    ax.set_xscale("log"); ax.set_xlim(*lim)
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())


def errpts(ax, pts, c, mk, ms=3.8, **kw):
    x = [p["bits"] for p in pts]; y = [p["mean"] for p in pts]
    yerr = [[p["mean"] - p["lo"] for p in pts], [p["hi"] - p["mean"] for p in pts]]
    kw = {"lw": 1.1, **kw}
    ax.plot(x, y, color=c, marker=mk, ms=ms, mec="white", mew=.6, zorder=3, **kw)
    # Seed min-max bars drawn above line and marker so short ranges stay visible.
    ax.errorbar(x, y, yerr=yerr, color=c, ls="none", elinewidth=.8, capsize=2.2, capthick=.8, zorder=4)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=.02)
    print("wrote", name)


g10 = load("campaign_10M.jsonl"); g10q = load("campaign_10M.jsonl", "10M.jsonl", preset="10M")
g50 = load("campaign_50M.jsonl")

# ---- Figure 1: 10M quality-memory frontier ---------------------------------------------------------------
LINES = [("GPEF", ["gpef@e0g256", "gpef@e0g128", "gpef@e0g64", "gpef@e0", "gpef@e4", "gpef@e8"]),
         ("RR", ["srq_rot@b0", "srq_rot@b2", "srq_rot@b4", "srq_rot@b6"]),
         ("SR", ["srq_sr@b0", "srq_sr@b2", "srq_sr@b4", "srq_sr@b6"]),
         ("Adam", ["adam_sr@m2", "adam_sr@m4", "fp_adam"]),
         ("Muon", ["muon_sr@m2", "fp_muon"])]
LAB1 = {"GPEF": "GPEF (ours)", "RR": "RR (ours)", "SR": "SR", "Flip": "Flip",
        "Adam": "Adam + SR → FP Adam", "Muon": "Muon + SR → FP Muon"}
fig, ax = plt.subplots(figsize=(W, 2.6)); style(ax)
for fam, ms in LINES:
    c, mk = FAM[fam]
    kw = dict(ls=(0, (1.5, 1.5)), lw=1.1) if fam in ("Adam", "Muon") else {}
    errpts(ax, sorted((best(g10, m) for m in ms), key=lambda p: p["bits"]), c, mk, label=LAB1[fam], **kw)
for m, lab, mk in (("adam_qat", "Adam QAT", "s"), ("adamw_ste", "AdamW STE", "D")):
    p = best(g10, m)
    ax.errorbar([p["bits"]], [p["mean"]], yerr=[[p["mean"] - p["lo"]], [p["hi"] - p["mean"]]], color=FAM["Adam"][0],
                marker=mk, ms=3.8, mfc="white", mew=1, ls="none", elinewidth=.9, capsize=2, zorder=3,
                label=f"{lab} (shadow)")
logx(ax, [2, 3, 4, 6, 8, 12, 16, 32, 64, 100], (1.8, 115))
ax.set_xlabel("Optimizer + weight state (bits per weight, log scale)")
ax.set_ylabel("Validation BPB (lower is better)"); ax.set_ylim(1.4, 2.95)
ax.legend(frameon=False, loc="upper right", handlelength=2.2, borderaxespad=0, ncol=2, columnspacing=1.2, labelspacing=.3)
save(fig, "fig1_frontier")

# ---- Figure 2: training curves at matched state ---------------------------------------------------------
L2 = math.log(2)


def curve(g, m):
    rs = best(g, m)["runs"]; n = min(len(r["curve"]) for r in rs)
    xs = [rs[0]["curve"][i][0] for i in range(n)]
    Y = [[r["curve"][i][1] / L2 for r in rs] for i in range(n)]
    return xs, [st.mean(y) for y in Y], [min(y) for y in Y], [max(y) for y in Y], best(g, m)["bits"]


P = [("≈2 bits/weight", [("gpef@e0g256", "GPEF", "-"), ("srq_rot@b0", "RR", "-"), ("srq_sr@b0", "SR", "-")]),
     ("≈4 bits/weight", [("gpef@e0", "GPEF", "-"), ("srq_rot@b2", "RR", "-"), ("srq_sr@b2", "SR", "-"),
                         ("muon_sr@m2", "Muon+SR", "-")]),
     ("6–12 bits/weight", [("gpef@e4", "GPEF", "-"), ("gpef@e8", "GPEF", "--"), ("srq_rot@b6", "RR", "-"),
                           ("srq_sr@b6", "SR", "-"), ("adam_sr@m2", "Adam+SR", "-"), ("adam_sr@m4", "Adam+SR", "--")]),
     ("64–102 bits/weight (FP32 state)", [("fp_muon", "FP Muon", "-"), ("fp_adam", "FP Adam", "-"),
                                          ("adamw_ste", "AdamW STE", "--"), ("adam_qat", "Adam QAT", ":")])]
FAMOF = {"Muon+SR": "Muon", "Adam+SR": "Adam", "FP Muon": "Muon", "FP Adam": "Adam", "AdamW STE": "Adam",
         "Adam QAT": "Adam"}
fig, axs = plt.subplots(2, 2, figsize=(W, 3.7), sharex=True, sharey=True)
ref = curve(g10, "fp_muon")
for i, (ax, (title, ms)) in enumerate(zip(axs.flat, P)):
    style(ax)
    if i < 3:
        ax.plot(ref[0], ref[1], color="#a3a29d", lw=1, ls=(0, (1.5, 1.5)), zorder=1)
        ax.text(7950, 1.45, "FP Muon (64)", ha="right", va="top", fontsize=7, color=MUTED)
    for m, lab, ls in ms:
        xs, y, lo, hi, bits = curve(g10, m)
        c = FAM[FAMOF.get(lab, lab)][0]
        ax.fill_between(xs, lo, hi, color=c, alpha=.1 if m == "adam_qat" else .18, lw=0, zorder=2)
        ax.plot(xs, y, color=c, lw=1.1, ls=ls, zorder=3, label=f"{lab} ({bits:.3g})")
    ax.set_title(title, loc="left", color=INK, pad=3)
    lg = ax.legend(loc="upper right", fontsize=7, handlelength=1.8, labelspacing=.25, ncol=2 if len(ms) > 3 else 1,
                   columnspacing=.8, borderaxespad=.2, borderpad=.3, frameon=True, fancybox=False)
    lg.get_frame().set(facecolor="white", edgecolor="none", alpha=.9); lg.set_zorder(5)
    ax.set_ylim(1.3, 3.5); ax.set_xlim(0, 8000)
    ax.set_xticks([0, 2000, 4000, 6000, 8000]); ax.set_xticklabels(["0", "2k", "4k", "6k", "8k"])
for ax in axs[1]: ax.set_xlabel("Training step")
for ax in axs[:, 0]: ax.set_ylabel("Train loss (bits/byte)")
fig.tight_layout(h_pad=.8, w_pad=.6)
save(fig, "fig2_curves")

# ---- Figure 3: 10M / 50M / 100M, low-state methods; FP and shadow-weight baselines as reference lines ------
g100 = load("campaign_100M.jsonl")
LINES3 = [("GPEF", ["gpef@e0g256", "gpef@e0", "gpef@e4"]), ("RR", ["srq_rot@b0", "srq_rot@b2", "srq_rot@b6"]),
          ("SR", ["srq_sr@b0", "srq_sr@b2", "srq_sr@b6"])]
SHADOW = {"muon_qat": "Muon QAT", "adamw_ste": "AdamW STE", "adam_qat": "Adam QAT"}
fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), sharey=True)
for ax, g, title in ((axs[0], g10q, "10.6M, 8k steps, 3 seeds"), (axs[1], g50, "49.2M, 30k steps, 2 seeds"),
                     (axs[2], g100, "99.1M, 60k steps, 2 seeds")):
    style(ax)
    for fam, ms in LINES3:
        errpts(ax, [best(g, m) for m in ms if m in g], *FAM[fam])
    for m, fam in (("muon_sr@m2", "Muon"), ("adam_sr@m2", "Adam")):
        if m in g:
            errpts(ax, [best(g, m)], *FAM[fam], ms=5)
    refs = [(best(g, "fp_muon"), "FP Muon", (0, (1.5, 1.5)), "top")]
    sh = [best(g, m) | {"m": m} for m in SHADOW if m in g]
    if sh:
        q = min(sh, key=lambda p: p["mean"]); refs.append((q, SHADOW[q["m"]], (0, (5, 2)), "bottom"))
    for p, lab, ls, va in refs:
        ax.axhline(p["mean"], color=MUTED, lw=.9, ls=ls, zorder=1)
        ax.text(1.92, p["mean"] + (-.025 if va == "top" else .025), f"{lab} ({p['bits']:.0f} b/w)",
                ha="left", va=va, fontsize=7, color=MUTED)
    logx(ax, [2, 3, 4, 6, 8], (1.85, 9.5)); ax.set_ylim(0.88, 2.65)
    ax.set_title(title, loc="left", color=INK, pad=3, fontsize=8.5)
axs[0].set_ylabel("Validation BPB"); axs[1].set_xlabel("State (bits per weight, log scale)")
h = [Line2D([], [], color=FAM[f][0], marker=FAM[f][1], mec="white", mew=.6, ms=3.8, lw=1.4, label=l)
     for f, l in (("GPEF", "GPEF (ours)"), ("RR", "RR (ours)"), ("SR", "SR"))]
h += [Line2D([], [], color=FAM[f][0], marker=FAM[f][1], mec="white", mew=.6, ms=5, ls="none", label=l)
      for f, l in (("Muon", "Muon + SR"), ("Adam", "Adam + SR"))]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, 0), handlelength=2,
           columnspacing=1.6)
fig.tight_layout(rect=(0, .08, 1, 1), w_pad=.8)
save(fig, "fig3_scale")
