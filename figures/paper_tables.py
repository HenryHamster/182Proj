"""Appendix tables of the report, written next to this script.
lr_table.tex (Table 3): grid tested (seed 0) and selected lr per method and model size; a selected lr at the end of
its grid is marked with a dagger.
ablation_table.tex (Table 4): GPEF ablations at 10.6M, each changing one setting from the default.
Selection rule as in the figures: lowest seed-0 val BPB among non-diverged runs, then the mean over all seeds at
that lr.
usage: python paper_tables.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
SIZES = [("10M", ["campaign_10M.jsonl", "10M.jsonl"]), ("50M", ["campaign_50M.jsonl"]), ("100M", ["campaign_100M.jsonl"])]
ROWS = [("GPEF, $S{=}256$, no residual", "gpef@e0g256", 2.25), ("GPEF, no residual", "gpef@e0", 4),
        ("GPEF, 4-bit residual", "gpef@e4", 8), ("GPEF, 8-bit residual", "gpef@e8", 12),
        ("RR", "srq_rot@b0", 2), ("RR, 2-bit residual", "srq_rot@b2", 4), ("RR, 6-bit residual", "srq_rot@b6", 8),
        ("SR", "srq_sr@b0", 2), ("SR, 2-bit residual", "srq_sr@b2", 4), ("SR, 6-bit residual", "srq_sr@b6", 8),
        ("Muon+SR, 2-bit momentum", "muon_sr@m2", 4.06), ("Adam+SR, 2-bit moments", "adam_sr@m2", 6.12),
        ("Adam+SR, 4-bit moments", "adam_sr@m4", 10.1), ("Muon (FP)", "fp_muon", 64), ("Adam (FP)", "fp_adam", 96),
        ("Muon QAT", "muon_qat", 66), ("Adam QAT", "adam_qat", 98), ("AdamW STE", "adamw_ste", 102)]


def load(files, preset):
    out = []
    for f in files:
        out += [r for r in map(json.loads, open(RES / f, encoding="utf-8")) if r["preset"] == preset]
    return out


def fmt(x):
    s = f"{x:.2g}" if x >= 0.001 else f"{x:.1e}".replace("e-0", "e-")
    return {"0.0033": ".0033", "0.0011": ".0011", "0.00033": "3.3e-4"}.get(s, s.lstrip("0") if s.startswith("0.") else s)


def best_lr(rs, m):
    ok = [r for r in rs if r["method"] == m and r["seed"] == 0 and not r["diverged"]]
    return min(ok, key=lambda r: r["val_bpb"])["lr"] if ok else None


def mean_bpb(rs, m):
    lr = best_lr(rs, m)
    v = [r["val_bpb"] for r in rs if r["method"] == m and r["lr"] == lr]
    return sum(v) / len(v)


def cell(rs, m):
    s0 = [r for r in rs if r["method"] == m and r["seed"] == 0]
    if not s0:
        return "--"
    grid = sorted({r["lr"] for r in s0})
    ok = [r for r in s0 if not r["diverged"]]
    best = min(ok, key=lambda r: r["val_bpb"])["lr"]
    edge = len(grid) > 1 and best in (grid[0], grid[-1])
    return f"\\textbf{{{fmt(best)}}}{'$^\\dagger$' if edge else ''} \\{{{', '.join(fmt(g) for g in grid)}\\}}"


data = {p: load(fs, p) for p, fs in SIZES}
lines = [r"\begin{tabular}{lr" + "l" * len(SIZES) + "}", r"\toprule",
         r"Method & b/w & 10.6M & 49.2M & 99.1M \\", r"\midrule"]
for name, m, bits in ROWS:
    lines.append(f"{name} & {bits:g} & " + " & ".join(cell(data[p], m) for p, _ in SIZES) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE / "lr_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

# Table 4. Each row: setting, (label, method) for the default, then the alternatives.
ABL = [("Clip, 2.25 b/w", ("$c = 4$", "gpef@e0g256"), [("$c = 2$", "gpefv@e0g256_c2")]),
       ("Clip, 12 b/w", ("$c = 4$", "gpef@e8"), [("$c = 2$", "gpefv@e8intg32_c2")]),
       ("8-bit residual", ("INT8", "gpef@e8"), [("FP8", "gpefv@e8fp8g32_c4")]),
       ("4-bit residual", ("INT4", "gpef@e4"), [("FP4", "gpefv@e4fp4g32_c4")]),
       ("Momentum (FP8 residual)", ("none", "gpefv@e8fp8g32_c4"),
        [(r"$\beta = 0.9$", "gpefv@e8fp8g32_c4_gm0.9"), (r"$\beta = 0.99$", "gpefv@e8fp8g32_c4_gm0.99")]),
       ("Group size", ("32", "gpef@e0"),
        [("64", "gpef@e0g64"), ("128", "gpef@e0g128"), ("256", "gpef@e0g256")])]
rs = data["10M"]
lines = [r"\begin{tabular}{lll}", r"\toprule", r"Setting & Default & Alternative \\", r"\midrule"]
for name, (dl, dm), alts in ABL:
    alt = "; ".join(f"{l}: {mean_bpb(rs, m):.3f}" for l, m in alts)
    lines.append(f"{name} & {dl}: {mean_bpb(rs, dm):.3f} & {alt} " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE / "ablation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
