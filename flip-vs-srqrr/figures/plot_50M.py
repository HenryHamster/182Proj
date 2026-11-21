"""Val bpb vs state bits/weight at 50M and 10M (flip-vs-srqrr-002). Writes bits_vs_bpb_50M.png with --fig.

Rule (same at both sizes): best lr = lowest seed-0 val bpb among on-grid,
non-diverged runs; reported value = mean over all seeds at that lr.
Inputs: campaign_50M.jsonl, campaign_10M.jsonl (val split only; test aliased to val).
"""
import json, sys, statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE.parent / "results"
METHODS = ["fp_muon", "fp_adam", "muon_qat", "adamw_ste", "adam_qat",
           "muon_sr@m2", "adam_sr@m2",
           "srq_sr@b0", "srq_sr@b2", "srq_sr@b6",
           "srq_rot@b0", "srq_rot@b2", "srq_rot@b6",
           "gpef@e0g256", "gpef@e0", "gpef@e4"]
FAMILY = {"srq_sr": "SR", "srq_rot": "RR", "gpef": "GPEF",
          "muon_sr": "Muon/Adam SR", "adam_sr": "Muon/Adam SR",
          "muon_qat": "QAT", "adamw_ste": "QAT", "adam_qat": "QAT",
          "fp_muon": "FP", "fp_adam": "FP"}
FAM_ORDER = ["SR", "RR", "GPEF", "Muon/Adam SR", "QAT", "FP"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def load(name, fallback=None):
    runs = defaultdict(lambda: defaultdict(dict))  # method -> lr -> seed -> rec
    for line in open(RES / name):
        r = json.loads(line)
        runs[r["method"]][r["lr"]][r["seed"]] = r
    if fallback:  # methods absent from the campaign file (muon_qat at 10M)
        missing = set(METHODS) - set(runs)
        for line in open(RES / fallback):
            r = json.loads(line)
            if r["method"] in missing and r.get("preset") == "10M":
                runs.setdefault(r["method"], defaultdict(dict))[r["lr"]][r["seed"]] = r
    return runs


def summarize(runs, m):
    by_lr = runs.get(m)
    if not by_lr:
        return None
    s0 = {lr: d[0]["val_bpb"] for lr, d in by_lr.items()
          if 0 in d and not d[0]["diverged"]}
    lr = min(s0, key=s0.get)
    lrs = sorted(by_lr)
    vals = [by_lr[lr][s]["val_bpb"] for s in sorted(by_lr[lr])]
    edge = lr in (lrs[0], lrs[-1]) and len(lrs) > 1
    return dict(lr=lr, vals=vals, mean=st.mean(vals), edge=edge,
                bits=by_lr[lr][0]["state_bits"], n_lr=len(lrs))


def main():
    r50, r10 = load("campaign_50M.jsonl"), load("campaign_10M.jsonl", "10M.jsonl")
    rows = []
    for m in METHODS:
        a, b = summarize(r50, m), summarize(r10, m)
        rows.append((m, a, b))
    out = ["| method | state b/w | 50M lr | 50M seeds | 50M mean | 10M lr | 10M mean (3 seeds) | 50M − 10M |",
           "|---|---|---|---|---|---|---|---|"]
    for m, a, b in sorted(rows, key=lambda t: t[1]["mean"]):
        seeds = " / ".join(f"{v:.4f}" for v in a["vals"])
        flag = " (grid edge)" if a["edge"] else ""
        out.append(f"| {m} | {a['bits']:g} | {a['lr']:g}{flag} | {seeds} | **{a['mean']:.4f}** | "
                   f"{b['lr']:g} | {b['mean']:.4f} | {a['mean'] - b['mean']:+.4f} |")
    table = "\n".join(out)
    print(table)
    incomplete = [m for m, a, _ in rows if len(a["vals"]) < 2]
    print("\nmethods with <2 seeds at best lr:", incomplete or "none")

    # Key checks at matched state bits (means).
    get = {m: a["mean"] for m, a, _ in rows}
    get10 = {m: b["mean"] for m, _, b in rows}
    print("\nRR - SR (negative = rotation better):")
    for b in ("b0", "b2", "b6"):
        print(f"  {b}: 50M {get['srq_rot@'+b]-get['srq_sr@'+b]:+.4f}   10M {get10['srq_rot@'+b]-get10['srq_sr@'+b]:+.4f}")
    print("GPEF vs RR at matched bits:")
    for g, r in (("gpef@e0", "srq_rot@b2"), ("gpef@e4", "srq_rot@b6")):
        print(f"  {g} - {r}: 50M {get[g]-get[r]:+.4f}   10M {get10[g]-get10[r]:+.4f}")
    if "--fig" not in sys.argv:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    R = {m: (a, b) for m, a, b in rows}
    # Methods that share a quantizer at several state widths are joined by a line.
    lines = [("SR", ["srq_sr@b0", "srq_sr@b2", "srq_sr@b6"]),
             ("RR", ["srq_rot@b0", "srq_rot@b2", "srq_rot@b6"]),
             ("GPEF", ["gpef@e0g256", "gpef@e0", "gpef@e4"])]
    # Single-width methods: own marker, identified in the legend.
    singles = [("muon_sr@m2", "Muon/Adam SR", "v"), ("adam_sr@m2", "Muon/Adam SR", "^"),
               ("muon_qat", "QAT", "o"), ("adamw_ste", "QAT", "s"), ("adam_qat", "QAT", "D"),
               ("fp_muon", "FP", "o"), ("fp_adam", "FP", "s")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=150, sharey=True)
    for ax, idx, title in ((axes[0], 1, "10M (10.6M weights, 3 seeds)"),
                           (axes[1], 0, "50M (49.2M weights, 2 seeds)")):
        for fam, ms in lines:
            c = COLORS[FAM_ORDER.index(fam)]
            xs = [R[m][idx]["bits"] for m in ms]
            ys = [R[m][idx]["mean"] for m in ms]
            ax.plot(xs, ys, "-o", color=c, lw=2, ms=6, mec="white", mew=1, zorder=3)
            if idx == 0:
                for m in ms:
                    v = R[m][0]["vals"]
                    ax.vlines(R[m][0]["bits"], min(v), max(v), color=c, lw=1.5, alpha=0.6, zorder=2)
        for m, fam, mk in singles:
            c = COLORS[FAM_ORDER.index(fam)]
            p = R[m][idx]
            ax.scatter(p["bits"], p["mean"], marker=mk, s=60 if mk == "D" else 46, color=c, edgecolors="white",
                       lw=0.8, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(1.6, 150)
        ax.set_xlabel("state bits per quantized weight (log scale)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, color="#e6e6e6", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # The five 64-102 b/w references sit within 0.09 bpb at 50M: zoom on them.
    ins = axes[1].inset_axes([0.60, 0.50, 0.26, 0.20])
    lab = {m: (5, 0, "left") for m in ("fp_muon", "muon_qat", "adam_qat", "fp_adam", "adamw_ste")}
    for m, fam, mk in singles:
        if m not in lab:
            continue
        p = R[m][0]
        ins.scatter(p["bits"], p["mean"], marker=mk, s=30 if mk == "D" else 24,
                    color=COLORS[FAM_ORDER.index(fam)], edgecolors="white", lw=0.8, zorder=3)
        dx, dy, ha = lab[m]
        ins.annotate(m, (p["bits"], p["mean"]), xytext=(dx, dy),
                     textcoords="offset points", ha=ha, va="center", fontsize=6, color="#555")
    ins.set_xscale("log", base=2)
    ins.set_xlim(56, 260); ins.set_ylim(1.115, 1.235)
    ins.set_xticks([]); ins.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ins.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ins.set_yticks([1.13, 1.18, 1.23]); ins.tick_params(labelsize=6, length=2, pad=1)
    ins.grid(True, axis="y", color="#eeeeee", lw=0.5)
    for sp in ins.spines.values():
        sp.set_edgecolor("#bbbbbb")
    axes[1].indicate_inset([56, 1.115, 74, 0.12], ins, edgecolor="#999")
    axes[0].set_ylabel("val bits per byte (lower is better)")
    h = [Line2D([], [], marker="o", color=COLORS[FAM_ORDER.index(f)], lw=2, mec="white",
                label={"SR": "SR (b0, b2, b6)", "RR": "RR (b0, b2, b6)",
                       "GPEF": "GPEF (e0g256, e0, e4)"}[f]) for f, _ in lines]
    h += [Line2D([], [], marker=mk, ls="", color=COLORS[FAM_ORDER.index(fam)], mec="white", label=m)
         for m, fam, mk in singles]
    fig.legend(handles=h, fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Val bpb vs optimizer + weight state, mean over seeds at best lr", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(HERE / "bits_vs_bpb_50M.png", bbox_inches="tight")
    print("wrote", HERE / "bits_vs_bpb_50M.png")

main()
