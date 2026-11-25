"""Learning-rate table for the report appendix: grid tested (seed 0) and selected lr per method and model size.
Selection rule as in the figures: lowest seed-0 val BPB among non-diverged runs. A selected lr at the end of its grid
is marked with a dagger. Writes lr_table.tex next to this script.
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
