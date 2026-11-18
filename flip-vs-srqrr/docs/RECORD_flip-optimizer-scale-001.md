# flip-optimizer-scale-001

**Question.** Does the teammate flip optimizer (EthelynHuang/flip-optimizer @ ea4471c; 2-bit weights, 4-bit momentum,
top-k one-level flips with row/col caps, 6.2 bits/weight) keep up with Muon+QAT (66 b) and Muon+SR@m4 (6.2 b) as the
model grows from 2.65M to 10.6M to 25.2M quantized weights? CPU result at 2.65M: flip 1.873 vs QAT 1.739 bpb.

**Setup.** src/ = unmodified teammate code. drive.py aliases test->val (test.bin never read) and drops test_* fields.
Data: data/wt2_pytorch (pilot), data/wt103 (10M/25M), byte level, from prep.py.

**Timing (2026-11-09, RTX 5080 shared with knife-skills at 100% util, 30 steps):** s/step flip / muon_sr / muon_qat:
pilot 0.23/0.20/0.23, 10M 0.33/0.31/0.31, 25M 0.71/0.60/0.65 -> full runs ~9 min / ~44 min / ~2.9 h. Flip costs
+7-17%/step. Timing used ~150 GPU-s.

**Speed fix.** Profile (prof.py): the quantized-matrix step cost ~214 ms/step at every size, from per-matrix host
syncs (.item(), host-side top-k count, boolean-mask index) that each wait a time slice on the shared GPU. src_fast/
(patch.py) removes them; equiv.py shows src and src_fast give identical losses, codes and momentum for 60 CPU steps
(flip rho .003/.03, muon_qat, muon_sr). Step: pilot 242 -> 49 ms, 10M 333 -> 158 ms. drive.py uses src_fast.

**Pilot (2.65M, wt2_pytorch, 2500 steps, seed 0, GPU, 3 runs concurrent, 134 s training each), val bpb:**
flip@m4 rho .003 1.878 (teammate CPU 1.886); muon_qat lr .01 1.746 (CPU 1.759); muon_sr@m4 lr .01 2.254 (untuned).
GPU reproduces CPU within .013; flip - QAT gap .132 (CPU .127). results/pilot.jsonl, logs/pilot_*.

**10M ladder (10.6M quantized weights, wt103 bytes, 8000 steps, batch 64 x 256; ladder.py, 3 concurrent,
2026-11-09 23:33 -> 11-10 01:59 UTC; 23,754 run-seconds; val on 2000 windows), val bpb:**
- seed-0 grids: flip rho .001/.003/.009 = 1.716/1.675/1.699; muon_qat .003/.01/.03 = 1.536/1.528/1.616;
  muon_sr@m4 .003/.01/.03 = 2.126/2.075/2.101. All optima interior; no edge extension needed.
- at best lr, seeds 0,1,2: flip 1.6753/1.6755/1.6788 (mean 1.677 +- .002); muon_qat 1.5278/1.5243/1.5390 (1.530 +- .008).
- flip - QAT per seed .148/.151/.140, mean .146 (pilot seed 0: .132). flip - muon_sr (seed 0) -.400 (pilot -.376).
- Flip's best rho (.003) is the same as at 2.65M; Muon's best lr (.01) also unchanged. Flip changes .22% of codes/step.

**Interpretation.** From 2.65M to 10.6M flip keeps its large lead over the same-memory baseline (Muon+SR) and its gap to
Muon+QAT does not shrink (.13 -> .15 bpb). The rungs also differ in data (wt2 vs wt103) and steps, so the small widening
is not a clean size effect. Flip at 10.6M beats QAT at 2.65M on a different dataset; no cross-rung claim from that.

**Status (2026-11-10).** 10M done. Next decision (Henry): 25M rung, one seed per method at the 10M lrs.
