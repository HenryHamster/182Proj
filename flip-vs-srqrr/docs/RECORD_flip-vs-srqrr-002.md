# flip-vs-srqrr-002: the 10M comparison at 50M

## Question
Does the ordering from flip-vs-srqrr-001 (10.6M quantized weights) still hold at 49.2M weights? The comparison covers SR, rotation rounding (RR), GPEF and the low-bit Muon/Adam SR references, measured against the full-precision and QAT references.

Henry asked on 2026-11-18 for the main methods on a larger model and dataset, due Friday 2026-11-20 at noon PST. His words were "id still like to keep some space open but you can use both GPUs. so 50 M ?".

## Design
- **Model and data.** Preset `50M` in `qtrain.py`: d=640, 10 layers, 10 heads, ctx 512, batch 64, 30k steps. That is 49,152,000 quantized weights and about 0.98B training bytes, or about 1.8 passes over byte-level WikiText-103 (about 20 bytes per weight). Evaluation uses 2000 val windows. The test split is aliased to val: on IconLab `data/wt103/test.bin` is a symlink to `val.bin`, and the real test split was never copied there.
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
- **Host and GPUs.** IconLab (`icon.ist.berkeley.edu`), user lihuan, GPUs 2 and 3 (L40S), one worker per GPU. This uses lihuan's full 2-GPU allowance.
- **Launch.**
  - Command: `setsid nohup .venv/bin/python campaigns/campaign_50M.py 2 3 > logs/campaign_50M.out`, run in `~/projects/flip-vs-srqrr`.
  - Started 2026-11-18 11:12 PST. Campaign PID 1981769.
- **Source.** Branch flip-vs-srqrr with two uncommitted edits: the `50M` preset in `qtrain.py`, and the new `campaigns/campaign_50M.py`.
- **Environment.** Python 3.12, torch 2.9.1+cu128.
- **Throughput.** 0.144 s per step, about 72 minutes per run, about 9.3 GB peak per run.
- **Expected size.** 48–54 runs, about 31–35 hours, ending Thursday 2026-11-19 in the evening or late night PST.
- **Outputs.** Results go to `results/campaign_50M.jsonl`, logs to `logs/camp_50M_*.log` and `logs/campaign_50M.out`.
- **Owner.** Claude session fde41eae (Henry's Windows desktop).

## Status
Running. Round 1 was in progress when last verified at 2026-11-18 11:30 PST.

## Result
Pending.
