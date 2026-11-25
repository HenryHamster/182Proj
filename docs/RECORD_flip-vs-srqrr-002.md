# flip-vs-srqrr-002: the 10M comparison at 50M

## Question
Does the ordering from flip-vs-srqrr-001 (10.6M quantized weights) still hold at 49.2M weights? The comparison covers SR, rotation rounding (RR), GPEF and the low-bit Muon/Adam SR references, measured against the full-precision and QAT references.

## Design
- **Model and data.** Preset `50M` in `qtrain.py`: d=640, 10 layers, 10 heads, ctx 512, batch 64, 30k steps. That is 49,152,000 quantized weights and about 0.98B training bytes, or about 1.8 passes over byte-level WikiText-103 (about 20 bytes per weight). Evaluation uses 2000 val windows. The test split is aliased to val (`data/wt103/test.bin` is a symlink to `val.bin`), so the real test split was never read.
- **Methods (16).**
  - srq_rot@b0/b2/b6 and srq_sr@b0/b2/b6
  - gpef@e0g256, gpef@e0, gpef@e4
  - fp_muon, fp_adam
  - muon_qat, adamw_ste, adam_qat
  - muon_sr@m2, adam_sr@m2
- **Learning rate.**
  - Round 1 starts at each method's 10M seed-0 best lr. adam_qat uses 0.03, because at 0.09 one 10M seed diverged.
  - Round 2 tries lr/3, and round 3 tries lr/9 only where lr/3 won. Larger models usually prefer a lower lr.
  - Round 4 adds seed 1 at the best seed-0 lr.
  - The reported value is the mean of the seeds at the seed-0 best lr, the same rule as at 10M.
- **Comparison.** val bpb against state bits per weight, on the same axes as the 10M figure. The main checks are:
  - the rotation-vs-SR gap at 2 and 4 b/w;
  - GPEF's lead below 4 b/w;
  - the gap to the QAT and FP references.

## Execution
- **Hardware.** Two L40S GPUs, one run per GPU at a time.
- **Launch.** `python campaigns/campaign_50M.py 2 3` (the arguments are GPU ids). It is resumable and appends to `results/campaign_50M.jsonl`.
- **Environment.** Python 3.12, torch 2.9.1+cu128.
- **Throughput.** Most runs took 70-84 minutes. gpef@e0/e0g256 took about 92 minutes, gpef@e4 about 124 and fp_muon about 131. Peak GPU memory was 9-11 GB per run.
- **Dates.** 2026-11-18 11:12 to 2026-11-20 10:26 PST, 57 runs. No checkpoints were saved.

## Result
**Tested:** 16 methods at 49.2M quantized weights (preset `50M`), against the same 16 at 10.6M (preset `10M`, flip-vs-srqrr-001).
- **Replication unit:** an independent training seed. 50M has 2 seeds at the tuned lr; 10M has 3.
- **Metric:** val bits per byte on 2000 windows. The test split was never scored.
- **Analysis:** `figures/plot_50M.py` prints the table below and, with `--fig`, draws `figures/bits_vs_bpb_50M.png`. Its inputs are `results/campaign_50M.jsonl`, `results/campaign_10M.jsonl`, and `results/10M.jsonl` for 10M muon_qat.

The table is sorted by the 50M mean. Each mean is taken at the best seed-0 lr, and "grid edge" marks a best lr at the end of the tested grid.

| method | state b/w | 50M lr | 50M seeds | 50M mean | 10M lr | 10M mean (3 seeds) | 50M − 10M |
|---|---|---|---|---|---|---|---|
| fp_muon | 64 | 0.0033333 | 1.1304 / 1.1319 | **1.1312** | 0.01 | 1.4600 | -0.3288 |
| muon_qat | 66 | 0.0033333 | 1.1739 / 1.1720 | **1.1729** | 0.01 | 1.5303 | -0.3574 |
| adam_qat | 98 | 0.03 (grid edge) | 1.1875 / 1.1838 | **1.1857** | 0.09 | 1.8071 | -0.6215 |
| fp_adam | 96 | 0.01 | 1.2006 / 1.2060 | **1.2033** | 0.03 | 1.5482 | -0.3449 |
| adamw_ste | 102 | 0.03 (grid edge) | 1.2180 / 1.2230 | **1.2205** | 0.03 | 1.5993 | -0.3787 |
| srq_rot@b6 | 8 | 0.01 | 1.2435 / 1.2430 | **1.2432** | 0.03 | 1.6345 | -0.3913 |
| srq_sr@b6 | 8 | 0.01 | 1.2612 / 1.2596 | **1.2604** | 0.03 | 1.6365 | -0.3761 |
| gpef@e0 | 4 | 0.03 (grid edge) | 1.4731 / 1.4825 | **1.4778** | 0.03 | 1.8885 | -0.4107 |
| gpef@e4 | 8 | 0.01 | 1.5930 / 1.5989 | **1.5959** | 0.03 | 1.7580 | -0.1620 |
| srq_rot@b2 | 4 | 0.03 (grid edge) | 1.5934 / 1.6760 | **1.6347** | 0.03 | 1.9408 | -0.3061 |
| srq_sr@b2 | 4 | 0.01 | 1.6881 / 1.6903 | **1.6892** | 0.03 | 2.0547 | -0.3655 |
| gpef@e0g256 | 2.25 | 0.03 (grid edge) | 1.8990 / 1.9056 | **1.9023** | 0.03 | 2.2212 | -0.3189 |
| srq_rot@b0 | 2 | 0.03 (grid edge) | 1.9336 / 1.9328 | **1.9332** | 0.03 | 2.3344 | -0.4012 |
| srq_sr@b0 | 2 | 0.01 | 2.0330 / 2.0194 | **2.0262** | 0.03 | 2.3907 | -0.3645 |
| muon_sr@m2 | 4.0375 | 0.0001111 (grid edge) | 2.2848 / 2.2699 | **2.2774** | 0.001 | 2.5452 | -0.2678 |
| adam_sr@m2 | 6.075 | 0.003333 (grid edge) | 2.2960 / 2.3489 | **2.3225** | 0.003333 | 2.5243 | -0.2018 |

**What the 50M runs show:**
- **Rotation beats SR at every bit width, at both sizes.** At 50M, each rotation seed beats each SR seed at the same width.

  | width | RR − SR, 50M | RR − SR, 10M |
  |---|---|---|
  | 2 b/w (b0) | −0.093 | −0.056 |
  | 4 b/w (b2) | −0.055 | −0.114 |
  | 8 b/w (b6) | −0.017 | −0.002 |

  The 4 b/w gap at 50M is the least certain, because srq_rot@b2's two seeds differ by 0.083 (1.593 vs 1.676).
- **GPEF's lead below 4 b/w held and grew.**
  - At 4 b/w, gpef@e0 beats srq_rot@b2 by 0.157 bpb, up from 0.052 at 10M. Both gpef@e0 seeds beat both srq_rot@b2 seeds.
  - At 2-2.25 b/w, gpef@e0g256 beats srq_rot@b0 by 0.031, down from 0.113. gpef@e0g256 also uses 0.25 b/w more state.
- **GPEF at 8 b/w fell further behind.** gpef@e4 improved least of all 16 methods from 10M to 50M (−0.16 bpb), so it now trails srq_rot@b6 by 0.353, up from 0.124.
- **8 b/w closes most of the gap to the references.**
  - srq_rot@b6 is 0.070 behind muon_qat, down from 0.104 at 10M.
  - It is 0.112 behind fp_muon, down from 0.175.
  - It uses 8 b/w of state, against 66 for muon_qat and 64 for fp_muon.
- **The ordering is otherwise stable.** All methods improve by 0.16-0.62 bpb from 10M to 50M.
  - Muon remains the best FP and QAT optimizer.
  - muon_sr@m2 and adam_sr@m2 stay last. At 4-6 b/w they are worse than SR at 2 b/w.
  - adam_qat's large improvement (−0.62) partly reflects its 10M value: that value was taken at lr 0.09, where one 10M seed was poor (1.9966).

**Limits:**
- **Grid-edge learning rates.** The lr search went downward only, from each method's 10M best lr. The 7 methods whose round-1 lr won never had a higher lr tested: adam_qat, adamw_ste, gpef@e0, gpef@e0g256, srq_rot@b2, srq_rot@b0 and adam_sr@m2. These include both rotation entries at 2 and 4 b/w, while the matching SR entries were tuned in the interior of the grid. Tuning those rotation entries further could therefore only widen the rotation-vs-SR gaps above. muon_sr@m2 won at the smallest lr tested (lr/9).
- **What "scale" changes.** The `50M` preset changes several things together: width, depth, context (256 to 512), step count and data passes. The 50M − 10M column measures that whole change, not parameter count alone.
- **Data.** There is one dataset (byte-level WikiText-103) and one validation split. The sealed test split was not used.

**Next decision.** The supported conclusions are that rotation beats SR, GPEF@e0 leads at 4 b/w, and GPEF@e4 trails at 8 b/w. If a write-up depends on the exact 4 b/w rotation-vs-SR gap, the smallest useful follow-up is a third seed of srq_rot@b2 and srq_sr@b2, plus one ×3 lr run of srq_rot@b2. That is about 4 runs, or about 3 hours on two GPUs.

### Learning-rate search (seed 0)
Round 1 used each method's 10M best lr. Round 2 tried lr/3. Round 3 tried lr/9 for the methods where lr/3 won.

| method | round-1 lr | round-1 bpb | lr/3 bpb | lr/9 bpb |
|---|---|---|---|---|
| fp_muon | 0.01 | 1.1325 | **1.1304** | 1.1622 |
| muon_qat | 0.01 | 1.1957 | **1.1739** | 1.2203 |
| fp_adam | 0.03 | 2.3931 | **1.2006** | 1.2765 |
| adam_qat | 0.03 | **1.1875** | 1.2119 | |
| adamw_ste | 0.03 | **1.2180** | 1.2217 | |
| srq_rot@b6 | 0.03 | 1.3451 | **1.2435** | 1.3619 |
| srq_sr@b6 | 0.03 | 1.3779 | **1.2612** | 1.3728 |
| gpef@e0 | 0.03 | **1.4731** | 1.5472 | |
| gpef@e4 | 0.03 | 3.3076 | **1.5930** | 1.6817 |
| srq_rot@b2 | 0.03 | **1.5934** | 1.6073 | |
| srq_sr@b2 | 0.03 | 1.7319 | **1.6881** | 1.7598 |
| gpef@e0g256 | 0.03 | **1.8990** | 1.9017 | |
| srq_rot@b0 | 0.03 | **1.9336** | 1.9733 | |
| srq_sr@b0 | 0.03 | 2.1066 | **2.0330** | 2.1473 |
| adam_sr@m2 | 0.003333 | **2.2960** | 2.3333 | |
| muon_sr@m2 | 0.001 | 2.3421 | 2.3419 | **2.2848** |

At the round-1 lr, fp_adam and gpef@e4 were unstable:
- fp_adam's train loss spiked from 0.91 to 2.53 at step 25k.
- gpef@e4 stayed near 2.5 nats and never trained.

Both trained normally at lr/3.
