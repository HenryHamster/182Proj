# flip-vs-srqrr-004: main methods at 100M

## Question
Do the 50M rankings and the RR-vs-SR margins hold at 99.1M quantized weights? Henry chose the method set on
2026-11-20: "SR RR (2 and 8 bpw is enough first), GPEF @ 0. one of adam/muon".

## Design
- Preset `100M` in `qtrain.py`: d=768, 14 layers, 12 heads, ctx 512, batch 64, 60k steps. That is 1.97B training
  bytes, about 20 bytes per weight, the same ratio as 50M. Byte-level WikiText-103, 2000 val windows. The test split
  was never scored.
- Methods: srq_sr@b0, srq_rot@b0 (2 b/w), srq_sr@b6, srq_rot@b6 (8 b/w), gpef@e0 (4 b/w), fp_muon (FP reference).
- lr: seed 0 at the method's 50M best and at that value / 3, then seed 1 at the better one. The protocol is in
  `campaigns/campaign_100M.py`.

## Execution
- IconLab, at most 2 L40S GPUs, one job per GPU. Each worker waited for a free GPU before starting.
- Time per step: 0.25 s for SR/RR, 0.30 s for gpef@e0, 0.26 s for fp_muon.
- 18 runs, finished 2026-11-23 12:32 PST. Output is `results/campaign_100M.jsonl`.

## Result (val bpb, mean of 2 seeds at the selected lr; 50M in brackets)

| b/w | method | 100M | 50M |
|---|---|---|---|
| 2 | SR | 1.961 | 2.026 |
| 2 | RR | **1.872** | 1.933 |
| 4 | GPEF e0 | 1.321 | 1.478 |
| 8 | SR | 1.159 | 1.260 |
| 8 | RR | **1.143** | 1.243 |
| 64 | fp Muon | 1.053 | 1.131 |

- Both RR seeds beat both SR seeds at both budgets. RR − SR is −0.089 at 2 b/w (−0.093 at 50M) and −0.016 at 8 b/w
  (−0.017 at 50M).
- The rankings match 50M. RR@b6 is 0.090 behind fp Muon (0.112 at 50M).
- Seed ranges are ≤ 0.034 (RR b0: 1.888 / 1.856).

**Limit.** For every method except srq_rot@b0, the best lr is at the top of its 2-point grid, so the margins are not
fully tuned.

Figure: `figures/fig3_scale.png` (`figures/paper_figs.py`).
