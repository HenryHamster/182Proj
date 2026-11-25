# flip-vs-srqrr-001

**Question.** With 2-bit weights trained from scratch (flip's harness), how does val bpb depend on the optimizer state
budget for flip (momentum quantized to 4 / 2 / 2g / 1.58 bits) versus srqrr (b = 0, 1, 2, 4 sub-grid bits, no
momentum), and does srqrr's edge over plain SR from the 4-bit Pythia work hold here and at 10.6M?

**Rationale.** Henry (2026-11-10): "more extensive experiments on both flip and srqrr, the latter particularly on lower
than 4 bit (where AFAIK flip fails)", ~12 h, results feed a project draft. The flip repo makes no low-bit-momentum
claim, so "flip fails below 4 bits" is tested here (flip@m2, @m2g, @m1.58), not assumed.

**Methods (srq_ext.py, registered into qtrain via drive.py; teammate code unchanged).**
- `srq_sr@bB`: x~ = clip(K + f - lr (alpha0/alpha_r) g/rms(g), -1.5, 1.5); SR onto the 2^-B sub-grid; K nearest code,
  f remainder. 2 + B bits/weight. Per-matrix alpha (fixed), same format as flip.
- `srq_rot@bB`: same, rotation threshold frac(u0 + t phi), u0 regenerated from a per-matrix seed. 2 + B bits/weight.
- `srqrr@bB`: rotation + shrunk Newton refit of a per-row scale alpha_r (srqrr.py constants kappa .003, alpha 3,
  cap .05; per-sequence stats from captured layer outputs). 2 + B + 32/n_in bits/weight. `srqrr_k10` = kappa x10.
- Port differences from the Pythia srqrr: per-row scale instead of per-32-group (format here has one alpha per
  matrix); no FP embedding/head refit (they are FP AdamW in this harness); from scratch, not continued training.
- References from the harness: flip@m4/m2/m2g/m1.58, muon_sr@m4/m2, muon_qat.

**Checks (srq_test.py, CPU).** Refit Ga equals the autograd scale gradient sum_j g_rj K_rj (rel err 4e-7); all eight
tested configs train on `tiny` with codes on grid; rotation thresholds have per-step mean .496-.505 and per-entry max
gap .021 over 64 steps. GPU timing (200 steps, shared card): srqrr@b2 .020 / .102 s/step pilot / 10M; srq_sr@b2 .012 / .052.

**Protocol (campaign.py).** Selection on val only; test.bin never read (drive.py aliases test->val).
- Phase A, pilot (2.65M, wt2_pytorch, 2500 steps): seed-0 grid per method (srq .003/.01/.03, flip .001/.003/.009,
  muon .003/.01/.03), edge extension x3 up to 2, then seeds 1,2 at best lr. results/campaign_pilot.jsonl.
- Phase B, 10M (10.6M, wt103, 8000 steps, 2000 val windows): srqrr@b0/1/2/4, flip@m2/m2g/m1.58, srq_sr@b0/2/4,
  muon_sr@m2; grid {best/3, best, 3 best} around the pilot best, 1 extension, seeds 1,2. results/campaign_10M.jsonl.
  flip@m4, muon_qat, muon_sr@m4 at 10M come from flip-optimizer-scale-001 (results/10M.jsonl, same protocol).
- Independent unit: training seed (init + data order + rounding draws); 3 seeds at the chosen lr.
- Admission: a job starts only while GPU memory used + reserve < 72% of 16 GB.

**Execution.** Launched 2026-11-10 08:11 UTC on hamsterwheel, `nohup .venv/bin/python campaign.py`, log
logs/campaign.log, per-run logs logs/camp_*. Owner: Claude session fde41eae (Henry's Windows desktop). Estimated
~1 h pilot + ~8-9 h 10M.

**Addendum (2026-11-10 08:58 UTC).** srq_rot was the best SRQ variant at 2.65M but was missing from the 10M list;
`campaign_rot.py` (log logs/campaign_rot.log) waits for CAMPAIGN DONE, then runs srq_rot@b0/b2/b4 at 10M with the same
protocol, printing ROT DONE.

**Pilot outcome (2.65M, mean of 3 seeds at the seed-0 best lr; val bpb).** Completed 08:57 UTC, no failures or divergences.

| state b/w | flip | srq_sr | srq_rot | srqrr | reference |
|---|---|---|---|---|---|
| 2.0-2.1 | n/a | 2.570 | **2.519** | 2.541 (k10 2.521) | |
| 3.0-3.7 | m1.58 2.416 | 2.383 | **2.332** | 2.394 | |
| 4.0-4.25 | m2 2.327, m2g 2.242 | 2.228 | **2.168** | 2.255 (k10 2.395) | muon_sr@m2 2.641 |
| 6.0-6.1 | **m4 1.873** | 1.976 | 1.959 | 2.096 (sd .058) | muon_sr@m4 2.255 |
| 66 | | | | | muon_qat 1.745 |

Reading: flip does not fail below 4-bit momentum but degrades sharply (1.873 -> 2.24-2.42); at <=4.25 b/w the
momentum-free SRQ methods match or beat it, rotation best. At 6 b/w flip@m4 wins by .09. The per-row refit hurts
(srqrr worse than srq_rot at every b; k10 worse still), so the port's gain comes from rotation rounding, not the refit.
Seed sd mostly .004-.03. 10M confirms or overturns this.

**Revision (2026-11-10 10:20 UTC, Henry agreed to drop the refit).** The 10M phase ran at ~4.5 jobs/h, which would
finish ~02:00 UTC. campaign.py and campaign_rot.py were stopped; `campaign_10M_v2.py` (log logs/campaign_10M_v2.log) runs
srq_rot@b0/b2/b4, flip@m2/m2g/m1.58, srq_sr@b0/b2/b4, muon_sr@m2 with the same protocol. srqrr at 10M stops at the seed-0
grid: b0 and b1 full, b2 at lr .01 and .03 (two jobs left running at the switch); no srqrr seeds 1-2, no srqrr@b4.

**Extension (2026-11-10 10:26 UTC; Henry: put the teammate's methods on the same graph, a few low-bit Adam/Muon
references, and points near 8 bits/weight).**
- `team_ext.py` (registered in drive.py) ports the Stage43b notebook's GPEF (`gpef@eE`, E-bit error feedback,
  group-32 learned 4-level codebooks; `gpef@e8` = the notebook's gpef8; 2 + E + 2 b/w) and AdamW-STE (`adamw_ste`,
  102 b/w), plus `adam_sr@mB` (DQT with both Adam moments quantized to B bits; 2 + 2B + 64/n_in b/w).
  Same initial deployed weights as every harness method; the harness LR schedule replaces the notebook's constant LR.
  GPEF's group codebooks are a richer deployed format (+2 b/w) than the fixed per-matrix grid used by the others.
- Check (team_test.py, CPU): gpef@e8/e4/e0 give identical symbols and error codes to the notebook's GroupDiscrete
  torch path (quant_core.py extracted verbatim) over 20 random-gradient steps (codebook diff <= 1.3e-7); all new
  methods train on `tiny` with codes on grid.
- `campaign_team.py` (log logs/campaign_team.log): pilot, same protocol, for gpef@e8/e4/e0, adamw_ste, dqt,
  adam_sr@m8/m4/m2; after CAMPAIGN DONE, 10M full protocol for gpef@e8/e4/e0 and seeds 0-2 at the pilot-best lr for
  adam_sr@m4/m2. `campaign_hi.py` (logs/campaign_hi.log): pilot srq_rot/srq_sr at b6 (8 b/w) and b8 (10 b/w, where
  ROTATION.md predicts the rotation gain vanishes) and flip@m8, then 10M srq_rot@b6/srq_sr@b6 at the pilot-best lr,
  seeds 0-2.

**Extension pilot outcome (2.65M, 3 seeds at the seed-0 best lr, all lr optima interior; completed 12:18 UTC).**

| method | state b/w | lr | val bpb (sd) |
|---|---|---|---|
| gpef@e0 | 4 | .03 | 2.135 (.015) |
| gpef@e4 | 8 | .03 | 2.054 (.011) |
| gpef@e8 | 12 | .01 | 1.879 (.004) |
| srq_rot@b6 / srq_sr@b6 | 8 | .03 | 1.934 (.006) / 1.938 (.013) |
| srq_rot@b8 / srq_sr@b8 | 10 | .03 | 1.906 (.005) / 1.918 (.015) |
| flip@m8 | 10.1 | .003 | 1.853 (.008) |
| adam_sr@m2 / m4 / m8 | 6.25 / 10.25 / 18.25 | .0033 / .09 / .09 | 2.606 / 2.111 / 2.074 |
| dqt (Adam + SR, fp32 moments) | 66 | .09 | 2.067 (.040) |
| adamw_ste | 102 | .03 | 1.826 (.004) |

Reading: GPEF without error feedback (4 b/w) edges srq_rot@b2 (2.135 vs 2.168) but uses group-32 learned codebooks, a
richer deployed format; with error feedback it does not use bits as well as SRQ (e4 at 8 b/w 2.054 vs 1.934) until e8
(12 b/w, 1.879, between flip@m8 and SRQ@b8). The rotation gain shrinks to noise at 8-10 b/w, as ROTATION.md predicts.
Adam + SR is ~2.07 regardless of moment precision (fp32 66 b/w included) and collapses at 2-bit moments, so the
Adam direction with SR onto 2-bit codes, not moment quantization, limits it; flip and SRQ beat it by .2-.4 at a
fraction of the state. Figure: results/bits_vs_bpb_pilot.png (results/plot_bits.py).

**10M main outcome (10.6M, mean of 3 seeds at the seed-0 best lr; val bpb; completed 15:13 UTC, no failures).**
flip@m4, muon_qat from flip-optimizer-scale-001 (results/10M.jsonl). All lr optima interior except muon_sr@m2
(edge-extended once: .000333 gives 2.545 vs 2.541 at .001).

| state b/w | flip | srq_sr | srq_rot | reference |
|---|---|---|---|---|
| 2.0 | n/a | 2.391 (.004) | **2.334** (.013) | |
| 3.7 | m1.58 2.204 (.006) | | | |
| 4.0-4.25 | m2 2.124 (.003), m2g 2.058 (.022) | 2.055 (.008) | **1.941** (.034) | muon_sr@m2 2.545 |
| 6.0-6.1 | **m4 1.676** (.002) | 1.721 (.005) | 1.689 (.003) | |
| 66 | | | | muon_qat 1.530 |

Reading: the pilot ordering holds at 4x the parameters and 3x the steps. At 4 b/w rotation beats flip's best low-bit
variant (m2g, 4.25 b/w) by .117 and SR by .114; at 2 b/w rotation is .13 worse than flip@m1.58 at 3.7 b/w, so rotation
is the best method per bit below ~4.5 b/w here. At 6 b/w flip@m4 still wins, but its margin over srq_rot@b4 shrank from
.09 (pilot) to .013. srqrr at 10M has seed 0 only (stopped by the 10:20 revision) and is not on the figure.
Figure: results/bits_vs_bpb_10M.png (`plot_bits.py 10M`; methods need 3 seeds to appear).

**10M extension outcome (10.6M, 3 seeds; completed 17:02 UTC, no failures; GPEF lr optima interior; adam_sr and b6
at the pilot-best lr).**

| method | state b/w | lr | val bpb (sd) |
|---|---|---|---|
| gpef@e0 | 4 | .03 | 1.889 (.002) |
| gpef@e4 | 8 | .03 | 1.758 (.008) |
| gpef@e8 | 12 | .01 | 1.607 (.004) |
| srq_rot@b6 / srq_sr@b6 | 8 | .03 | 1.635 (.004) / 1.636 (.006) |
| adam_sr@m2 / m4 | 6.25 / 10.25 | .0033 / .09 | 2.524 (.018) / 1.894 (.007) |

Reading: GPEF without error feedback beats srq_rot@b2 at 4 b/w by .052 (pilot .033). It has neither momentum nor
error feedback, so the margin comes from the group-32 learned codebooks (a 4 b/w deployed format); the discriminating
test is srq_rot on the same group-codebook format. With error feedback GPEF uses bits worse than SRQ (e4 1.758 vs
srq_rot@b6 1.635 at 8 b/w). Rotation and SR tie at 8 b/w. Adam + SR stays .2-.9 behind at every budget.
Draft: outputs/flip-vs-srq/flip_vs_srq_draft.{tex,pdf} on Henry's desktop; copy in drafts/ here.

**Format-matched check (2026-11-10 20:05 UTC; Henry: "continue" after the offer).** gpef@e0 beat srq_rot@b2 at 4 b/w
but differs in format (group-32 learned codebooks) and rounding (nearest). `team_rot.py` (imported in drive.py after
team_ext; drive.py backup logs/drive.py.bak-20261110) adds `gpef_rot@e0` / `gpef_sr@e0`: GPEF@e0 unchanged except the
final symbols, rotation / stochastic rounding of x between the bracketing codebook levels (4 b/w). gpef@e0 vs gpef_rot@e0
isolates rounding on the codebook format; gpef_rot@e0 vs srq_rot@b2 isolates the format at equal state.
- Check (team_rot_test.py, CPU): with threshold .5 both equal gpef@e0 (symbols and codebooks) for 20 steps; over 400
  steps the mean decoded value matches x to max .004 a (rotation) / .058 a (SR, independent draws); both train on tiny,
  on grid, 4.0 b/w. On tiny gpef@e0 changes .01% of codes/step vs 2.2-2.5% for the new variants.
- `campaign_grp.py` (logs/campaign_grp.log): pilot grid .003/.01/.03 + seeds 1,2; 10M grid around the pilot best,
  seeds 1,2. PAR 4. Estimated done ~21:15 UTC.

**Pilot result, format-matched check (3 seeds, val bpb).** gpef_rot@e0 2.505 (lr .01), gpef_sr@e0 2.555 (lr .003,
extended to .001, interior), vs gpef@e0 2.135 and srq_rot@b2 2.168 (both 4 b/w), srq_rot@b0 2.519 (2 b/w). Rotation /
SR on the group codebooks lose .37-.42 to nearest rounding and land at the 2 b/w SRQ level: GPEF@e0 learns through the
fp32 codebook refit with stable symbols (.01% codes changed/step on tiny), which stochastic symbol redraws with no
remainder destroy. GPEF's 4 b/w advantage is nearest rounding + refit codebooks, not a format rotation can reuse.

**GPEF below 4 b/w (2026-11-10 20:25 UTC; Henry: "add it").** All GPEF runs so far have >= 4 b/w (2-bit symbols + fp32
a, b per 32 weights). `team_grp.py` (drive.py import after team_rot; backup logs/drive.py.bak-20261110b) adds
`gpef@e0gG`: gpef@e0 with one codebook per G contiguous weights of the flattened matrix (a group may span two rows at
d=192/384), G = 64 / 128 / 256 -> 3 / 2.5 / 2.25 b/w. team_grp_test.py: g32 equals gpef@e0 (symbols, codebooks) for 30
steps; larger groups decode to the same initial weights, stay on grid, train on tiny. `campaign_gsz.py`
(logs/campaign_gsz.log) waits for campaign_grp, then pilot grid .01/.03/.09 + seeds 1,2 and 10M around the pilot best.

**10M result, format-matched check (3 seeds).** gpef_rot@e0 2.261 (.042, lr .03), gpef_sr@e0 2.316 (.007, lr .027;
lr .081 seed 0 gave 2.538, so .027 is interior) vs gpef@e0 1.889. Same reading as the pilot: rotation beats SR by .05
on the codebook format, and both lose ~.4 to nearest rounding.

**GPEF below 4 b/w, result (campaign_gsz, GSZ DONE 22:26 UTC; 3 seeds at the seed-0 best lr, val bpb (sd)).**

| method | b/w | pilot lr | pilot | 10M lr | 10M |
|---|---|---|---|---|---|
| gpef@e0g256 | 2.25 | .09 | 2.450 (.060) | .03 | 2.221 (.018) |
| gpef@e0g128 | 2.5 | .03 | 2.377 (.018) | .03 | 2.133 (.014) |
| gpef@e0g64 | 3 | .03 | 2.260 (.017) | .03 | 2.019 (.003) |
| gpef@e0 (g32) | 4 | .03 | 2.135 | .03 | 1.889 |

All best lrs interior (pilot g256 extended to .27: 3.408; 10M g256 grid .03/.09/.27 extended to .01: 2.344).
Reading: GPEF is the best method at every budget from 2.25 to 4 b/w. 10M: g256 at 2.25 b/w beats srq_rot@b0 (2 b/w,
2.334) by .11; g64 at 3 b/w beats flip@m1.58 (3.7 b/w, 2.204) by .19; pilot g64 beats srq_rot@b1 (3 b/w, 2.332) by
.07. Each doubling of group size costs ~.1 bpb. Rotation rounding remains best on the fixed per-matrix format.
Open: GPEF's nearest-level + codebook-refit update on the per-matrix grid (separates format from update rule); not run.
Figures results/bits_vs_bpb_{pilot,10M}.png re-rendered (GPEF legend "learned group codebooks; 2.25-3 b/w: larger
groups", 1.58-bit label moved). Draft drafts/flip_vs_srq_draft.{tex,pdf} updated (table, GPEF findings, limitations).
Method note for teammates: drafts/rotation_rounding.{tex,pdf} (+ rotation_fig.pdf; CPU simulation in the note's fig.py).

**Status.** COMPLETE 2026-11-10 22:40 UTC. Next decision (Henry): review draft; 25.2M rung unapproved.

**Rotation x GPEF batch (2026-11-11 03:06 UTC; Henry: "thats fun run it").** Teammate asked for (1a) forgetting in
GPEF's error feedback, (1c) AdamW/Muon STE references, (2) codebook adaptation for rotation. Diagnosis from team_ext.py:
gpef@eE stores the post-step residual rounded to nearest, with step span/(2^(E-1)-1) over +-span (span = 1.5 alpha),
while a nearest-level symbol leaves at most R = max((a-b)/2, b); so most error codes are unused and sub-step residual is
lost (same stall as nearest rounding, one level finer). Hypothesis: this is why gpef@e4 (1.758) spends bits worse than
srq_rot@b6 (1.635) at 8 b/w. The error is the SRQ remainder by another name, so 1a and 2 become one method.
`team_gpx.py` (drive.py import after team_grp; backup logs/drive.py.bak-20261110c) adds
`gpefx@eEgG_<wide|narrow>_<near|rot>_b<beta>` (narrow: step R/2^(E-1), codes -2^(E-1)..2^(E-1)-1; rot: rotation
rounding of the stored error; beta: target uses beta*e) and `gpef_frz@e0_<near|rot>` (codebooks frozen at init, which is
the per-matrix grid; 2 b/w). team_gpx_test.py: gpefx@e4g32_wide_near_b1 == gpef@e4 for 30 steps (symbols, codebooks,
error); rotation error rounding unbiased (max |mean - residual| .002 over 1000 steps); frozen codebooks fixed; all train on
tiny on grid; gpef_frz@e0_near changed 0 symbols on tiny. GPU smoke (200 steps, logs/smoke_gpx.jsonl) ok, ~.044 s/step.
`campaign_gpx.py` (logs/campaign_gpx.log), 10M, campaign.py protocol, grid .01/.03/.09: stage 1 ladder at e4g32
(narrow_near_b1, narrow_rot_b1, narrow_rot_b0.9, narrow_rot_b0.99) + gpef_frz@e0_{near,rot}; stage 2 best ladder variant
at e2g256 (4.25 b/w) and e2g32 (6 b/w) + adamw_ste, adam_qat, fp_adam (grid .01/.03/.09), fp_muon (.003/.01/.03).
Predictions: frozen near barely trains; frozen rot near srq_rot@b0 (2.334); narrow/rot beats gpef@e4; e2g256 vs gpef@e0
1.889 / srq_rot@b2 1.941 decides whether the merged method is the low-bit candidate to scale.
**Status.** RUNNING since 03:06 UTC 2026-11-11, campaign_gpx.py PID 2529621 on hamsterwheel; owner Claude session fde41eae.

### 2026-11-11 05:50 UTC: rotation on GPEF's refit codebooks (queued; Henry: 'we should prob run rotation on the refit codebooks')
Stage-1 interim (3-seed means, 8 b/w, vs gpef@e4 1.758): narrow_near 1.709, narrow_rot_b1 1.761 (seed 2 at 1.911, seeds 0/1 1.68-1.69),
narrow_rot_b0.9 1.841, narrow_rot_b0.99 1.646 (selected for stage 2). Frozen codebooks (2 b/w): near 2.320 (lr .09 only; .01/.03 never move, 2.566), rot 2.328 (~ srq_rot@b0 2.334).
Question: does rotation help GPEF at <=4 b/w with no stored error? Only prior point: gpef_rot@e0 (g32, 4 b/w) 2.261 vs gpef@e0 1.889.
gpef_rot fits the codebook for the nearest symbols and then rotates, so the stored codebook does not match the stored symbols.
New in team_gpx.py (additive; GPEFX/GPEFFrozen unchanged): gpefr@gG_after (= gpef_rot@e0 at G=32, checked step-exact for 30 steps)
and gpefr@gG_before (rotate with the current codebook, then least-squares refit for the rotated symbols). 2 + 64/G b/w.
Tiny check: all on grid; gpef@e0g256 changed 0.0 symbols/step at lr .03 (learns only through a, b); gpefr g256 after .022, before .074.
Runs: campaign_gpr.py (gpefr@g256_after, gpefr@g256_before at 2.25 b/w; gpefr@g32_before at 4 b/w), ~15 runs, starts when campaign_gpx.py exits; log logs/campaign_gpr.log, PID 2945005.
Predictions: g256 'before' beats 'after'; whether either beats gpef@e0g256 2.220 is open.

## 2026-11-11 07:50 UTC: rotation x GPEF stage 2 and references (campaign_gpx.py done)
Protocol as before: 10M, seed-0 lr grid {.01,.03,.09} + one edge extension, seeds 1,2 at the seed-0 best; unit of replication = seed.
- Stage 1 ladder (8 b/w, e4g32): narrow_near_b1 1.709, narrow_rot_b1 1.761, narrow_rot_b0.9 1.841, narrow_rot_b0.99 1.646 (gpef@e4 1.758). Frozen codebooks: near 2.320 (lr .09), rot 2.328 (lr .03).
- Stage 2, winner narrow_rot_b0.99 with a 2-bit error: e2g256 (4.25 b/w) 2.719 (2.724/2.717/2.717), e2g32 (6 b/w) 2.437 (2.425/2.442/2.445), best lr .03 interior for both.
  Both are worse than the same codebooks with no error at all (gpef@e0g256 2.220, gpef@e0 1.889): a 2-bit error in this format hurts.
- Mechanism check (CPU, 256x512 random matrix, 200 steps, grads N(0,1e-3), lr .03): 2-bit codes {-2,-1,0,1} occur 6/22/37/35%, i.e. 35% saturated at +1 vs 6% at -2;
  e4 codes are near-symmetric, 3% at +7. The narrow 2-bit range clips positive residuals at one step: hypothesis for the loss, not yet tested by a training run.
  A symmetric half-step code set ({-1.5,-.5,.5,1.5} steps) would test it.
- References (3 seeds): fp_muon (64 b/w, unquantized) 1.460 at lr .01; fp_adam (96) 1.548 at .03; adamw_ste (102, teammate format) 1.599 at .03;
  adam_qat (98) 1.807 at .09 (1.561/1.997/1.863: .09 is at the stability edge; seed 0 at .03 gave 1.566). muon_qat 1.530 from earlier.
- Figure: flipfig/bits_vs_bpb_10M.png (local scratch, plot_bits.py).

## 2026-11-11: why the 2-bit-error variants lose (CPU drift check)
Check: 256x512 random matrix, 300 steps, zero-mean random gradients, lr .03; count upward vs downward symbol moves (an unbiased method gives ~50/50 and keeps mean symbol ~1.5).
- gpef@e0 (no error): up share 0.49, mean symbol 1.498.
- gpefx@e2g32_narrow_rot_b0.99: up share 0.40, mean symbol 0.352 (weights migrate to the lowest level -a).
- gpefx@e2g32_narrow_near_b1: up share 0.50, mean symbol 1.500.
- gpefx@e4g32_narrow_rot_b0.99: 0.48, 1.380; gpefx@e4g32_narrow_near_b1: 0.49, 1.364.
Conclusion: the 2-bit rotation-rounded error drifts weights downward; nearest rounding of the same 2-bit error does not. Consistent with the asymmetric code range {-2,-1,0,1}
(only the top is clipped, and rotation sends residuals above +1 step into the clipped code; 35% of codes sit at +1). Stage-2 losses (2.719, 2.437) are attributed to this bias,
not to the idea of a 2-bit error; a symmetric half-step code set ({-1.5,-.5,.5,1.5} steps) is the proposed fix, untested.

## 2026-11-11 09:00 UTC: rotation on refit GPEF codebooks (campaign_gpr.py done)
Same protocol (seed = unit of replication). 3-seed means at the seed-0 best lr:
- gpefr@g256_after (2.25 b/w): 2.321 at lr .01 (2.334/2.310/2.320); symbols changed 1.8%/step.
- gpefr@g256_before (2.25 b/w): 2.443 at lr .03 (2.434/2.462/2.431); 10%/step.
- gpefr@g32_before (4 b/w): 2.411 at lr .01 (2.427/2.417/2.391); 8%/step, and 7.5%/step even at lr .0033.
Comparisons: gpef@e0g256 2.220 and gpef@e0 1.889 (nearest, refit), srq_rot@b0 2.334, gpef_frz@e0_rot 2.328, gpef_rot@e0 2.261.
Outcome: negative. Rotation on GPEF's codebooks loses to GPEF's nearest rounding at both budgets, and my prediction ('before' beats 'after') was wrong:
refitting the codebook to the rotated symbols is worse (2.443 vs 2.321). Drift check (CPU, zero-mean random gradients): all gpefr variants are unbiased (up share 0.50, mean symbol 1.49-1.50), so this is not the 2-bit-error bias.
Interpretation: every memoryless rotation variant at ~2 b/w lands at 2.32-2.33 (srq_rot@b0 per-matrix grid, frozen GPEF codebooks, GPEF refit-then-rotate), so a refit codebook adds nothing once symbols are rotated;
GPEF's nearest-rounding variant learns through a,b with few symbol moves and is 0.10 better. The lr-independent ~8% symbol churn of 'before' is observed; its mechanism is not established.
Next decision: rotation helps GPEF only with per-weight memory (gpefx e4, 1.646 at 8 b/w). Below 3 b/w the best remains gpef@e0g256 (2.220). Proposed, not run: symmetric 2-bit error codes for gpefx (4.25/6 b/w).

## 2026-11-11: symmetric-error ladder (campaign_sym.py launched)
Question: once GPEF's stored per-weight error has symmetric codes, does rotation rounding of that error beat nearest
rounding at every error width (Henry 2026-11-11 'Run the ladder')? Rationale: rotation helped only with per-weight memory
(e4g32 narrow rot b0.99 1.646 vs near 1.709); at E=2 the asymmetric narrow codes drifted weights down.
Change: team_gpx.py adds scale 'sym' (code c stores (c+1/2)*step, initial codes -1/0 at random); narrow/wide unchanged
(team_gpx_test.py: gpefx@e4g32_wide_near_b1 == gpef@e4 still exact; sym methods train on tiny, on grid, 3.25/4.25 b/w; ALL OK).
CPU drift check (drift_sym.py, 256x512, 300 steps, zero-mean gradients, lr .03; up share / mean symbol, unbiased = 0.50 / 1.5):
  narrow rot b0.99: e1 0.315/0.125, e2 0.380/0.365, e4 0.468/1.310; narrow near: e4 0.479/1.272 (E=4 narrow also drifts mildly)
  sym near and sym rot, e1/e2/e4, g256 and e4g32: 0.499-0.500 / 1.487-1.499; mean stored error within 0.01 step of 0.
Runs: gpefx@e{1,2,4}g256_sym_{near_b1,rot_b0.99} (3.25/4.25/6.25 b/w), lr grid .01/.03/.09 + one x3 extension on seed 0,
seeds 1,2 at the seed-0 best; ~30 runs at 10M. References gpef@e0g256 2.220, gpef@e0 1.889, gpef@e4 1.758.
Execution: launched 2026-11-11 on hamsterwheel, nohup .venv/bin/python campaign_sym.py, log logs/campaign_sym.out, PID 724190.
Jobs wait for the 72% GPU admission guard (another project's job held 6.8 GB at launch).

## 2026-11-11: λ-split, GPEF refit + scaled rotation (campaign_lam.py launched; symmetric ladder stopped)

- Question: can codebook refit (GPEF) and rotation rounding be combined without per-weight memory at 2.25 b/w? Every rotation variant at ~2 b/w scores 2.32-2.33 (srq_rot@b0 2.334, gpef_frz rot 2.328, gpefr@g256_after 2.321) vs gpef@e0g256 2.220: rotation's per-weight noise sets the floor and scrambles class membership the refit relies on.
- Change: team_gpx.py `gpefr@gG_after_l<lam>`: refit + nearest twice at full lr (codebook takes the full step), then rotation of x_rot = decode(k) + lam*(x - decode(k)). Flip probability, per-weight expected update and rounding-noise variance all scale by lam. lam=1 is gpefr@g256_after (lam_test.py: identical for 30 steps); flips/step on the CPU check 0.0284 / 0.0083 / 0.0022 at lam 1 / 0.3 / 0.1. team_gpx_test.py still ALL OK. Remote md5 team_gpx.py a5eb4e30dc3085d6163f4f540bd2b7a7.
- Run: 10M, gpefr@g256_after_l{0.1,0.3}, seed-0 lr {0.01,0.03,0.09} (edge extension 1), seeds 1,2 at best; PAR 4; logs/campaign_lam.out. Comparison: gpef@e0g256 2.220 (lam->0 end) and gpefr@g256_after 2.321 (lam=1).
- Decision rule: an interior lam below 2.220 means the combination works without memory; otherwise the conflict is real and stored error (symmetric ladder, campaign_sym.py, relaunchable unchanged) is the remaining route.
- Henry 2026-11-11 'atp just run this one': the symmetric ladder (PID 724190) was stopped with 0 runs started (blocked on the GPU guard throughout).

### 2026-11-11 result: λ-split (campaign_lam.py, logs/campaign_lam.out, results/campaign_10M.jsonl)

| method (2.25 b/w, g256) | best lr | λ·lr | val bpb, 3 seeds | mean | changed/step |
|---|---|---|---|---|---|
| gpef@e0g256 (λ=0) | 0.03 | 0 | 2.241 2.217 2.206 | 2.221 | 1.5% |
| gpefr@g256_after_l0.1 | 0.09 | 0.009 | 2.324 2.325 2.338 | 2.329 | 7.4% |
| gpefr@g256_after_l0.3 | 0.03 | 0.009 | 2.330 2.281 2.339 | 2.316 | 5.3% |
| gpefr@g256_after (λ=1) | 0.01 | 0.01 | 2.334 2.310 2.320 | 2.321 | 1.8% |

- Outcome: negative. No λ beats GPEF; every λ>0 lands at the rotation floor 2.32-2.33 shared by gpef_frz rot (2.328) and srq_rot@b0 (2.334). Unit of replication: training seed (3 per method).
- Integrity: λ->0 (lam=1e-12) reproduces gpef@e0g256 exactly for 200 CPU steps with flips (0.22%/step), so the gap is not a code difference.
- Interpretation: the lr search picks the same per-weight rate λ·lr ≈ 0.01 for every λ, so λ acts only as a per-weight learning rate. Unbiased rounding on a fixed lattice has noise variance linear in the step (r·(gap−r)), so scaling the step scales signal and noise together; the steady-state spread is set by the level gap, not the step size. Without a carried residual (memory) there is no λ that trades noise for signal; the codebook lr (0.09 vs 0.01) barely matters once rotation is on.
- Next decision (Henry): relaunch the symmetric-error ladder (adds the missing memory at 3.25/4.25 b/w), or stop pursuing rotation below 3 b/w.
- Figures: flipfig/lambda_split_10M.png, flipfig/bits_vs_bpb_10M.png (local scratchpad; plot_lam.py).

## 2026-11-12: teammate's GPEF revisions (campaign_gpf.py launched 22:28 UTC; Henry "sounds good run them")
Question: do the teammate's three revisions to GPEF change its accuracy at matched state bits? (1) the codebook-constraint
divisor c in s/c <= a <= c s (c = 4 so far; s = 1.5 alpha per matrix in this implementation, not a per-group statistic);
(2) fp8 instead of int8 error storage; (3) momentum, stored fp8, to smooth the error feedback.
Code: team_gpf.py (gpefv@e<E><fmt>g<G>_c<c>[_<mom><beta>]), registered in drive.py (backup /tmp/drive.py.bak_gpf);
checks team_gpf_test.py ALL OK on CPU: gpefv@e0g256_c4 == gpef@e0g256 and gpefv@e8intg32_c4 == gpef@e8 step for step
(40 steps, with symbol changes), es beta 0 == no momentum, stored fp8/fp4 values representable, bits, tiny training on grid.
Results carry a 'gpefv' entry: share of groups at the lower/upper a bound and b = 0 (step mean and end), a/s range.
Design (10M, campaign protocol, 5 runs per arm):
- c: first one seed of c=4 at the known best lr with bound logging (must reproduce 2.241 / 1.607); then c=2 only if
  a/s left [0.5, 2], c=16 and inf only if >= 1% of groups sat at a c=4 bound. Arms: e0g256 (2.25 b/w), e8 int g32 (12).
- error format at equal bits: fp8 e4m3 vs int8 (gpef@e8 1.607, 12 b/w); fp4 e2m1 vs int4 (gpef@e4 1.758, 8 b/w).
- momentum (fp8 state, +8 b/w, on fp8 error, 20 b/w): the teammate's "error momentum" has two readings, both run:
  gm = gradient momentum u <- beta u + (1-beta) clamp(g/rms), step from u, beta .9/.99;
  es = smoothed error m <- beta m + (1-beta) e fed back instead of e, beta .5/.9. es accumulates only (1-beta) of each
  step between flips (tiny CPU check: 0 symbol changes in 150 steps at beta .9), so beta .99 was not run.
Expected cost ~62 runs worst case, ~4 GPU-h at 4 parallel. Log logs/campaign_gpf.out, results results/campaign_10M.jsonl.
Owner: this Claude session (fde41eae). Status verified running 22:29 UTC 2026-11-12.

### result: teammate's GPEF revisions (campaign_gpf.py DONE 02:10 UTC 2026-11-13; 47 runs)
Unit of replication: training seed (0, 1, 2) at the seed-0 best lr; val bpb, 10.6M, WikiText-103. Baselines from the
existing campaign rows (gpef@e8, gpef@e4, gpef@e0g256, 3 seeds). Integrity: gpefv@e8intg32_c4 1.6073 and gpefv@e0g256_c4
2.2412 reproduce the gpef@e8 / gpef@e0g256 seed-0 runs to 4 decimals.

| arm | b/w | best lr | mean val bpb (3 seeds) | sd | baseline at same bits |
|---|---|---|---|---|---|
| e0g256 c2 | 2.25 | .03 | 2.251 | .018 | c4 (gpef@e0g256) 2.221 |
| e8 int g32 c2 | 12 | .01 | 1.611 | .003 | c4 (gpef@e8) 1.607 |
| e8 fp8 g32 c4 | 12 | .01 | 1.639 | .004 | int8 (gpef@e8) 1.607 |
| e4 fp4 g32 c4 | 8 | .03 | 1.866 | .036 | int4 (gpef@e4) 1.758 |
| e8 fp8 + gm .9 | 20 | .03 | 1.898 | .279 (2.219, 1.744, 1.729) | fp8 no momentum 1.639 |
| e8 fp8 + gm .99 | 20 | .03 | 2.132 | .014 | fp8 no momentum 1.639 |
| e8 fp8 + es .5 | 20 | .03 | 1.616 | .008 | fp8 no momentum 1.639; int8 1.607 at 12 b/w |
| e8 fp8 + es .9 | 20 | .03 | 1.968 | .018 | fp8 no momentum 1.639 |

lr .09 extensions for gm .9/.99 and es .5 all diverged (3.49-3.61), so .03 is the interior best.
Interpretation:
- (1) divisor: at c = 4 the bound is almost inactive in the baselines (0.08% of groups at e0g256, 0.15% step mean at e8),
  so c = 16 / inf were not run (gate: >= 1%). Tightening to c = 2 costs 0.030 bpb at 2.25 b/w and nothing resolvable at
  12 b/w. c = 4 stays. Not tested: in the higher-lr arms (fp4 lr .03, es .5, gm .9, e8 c2) 13-18% of groups sit at the
  UPPER bound a = c s on step average and 30-50% at the end; at the divergent lrs 50-90%. Whether a looser c helps
  those arms (or only delays divergence) is open; the gate measured only the two baselines.
- (2) float error storage loses at equal bits: fp8 +0.032 (8 sd), fp4 +0.108. The error lives in +-s; e4m3 has 3
  mantissa bits, so its spacing is s/16 on [s/2, s) against s/127 for int8 everywhere, and e2m1 is coarser still. The
  compute argument does not apply here: the error is read and written once per step elementwise, not used in a matmul.
- (3) momentum: neither reading beats the no-momentum int8 GPEF. Gradient momentum is worse (best 1.73 on two seeds,
  2.22 on seed 0) at +8 b/w. Smoothed error at beta .5 recovers what fp8 lost (1.616 vs 1.639) but only ties int8 at
  12 b/w while costing 20; beta .9 is far worse. es carries (1-beta) of each residual, so it acts as a reduced lr on the
  error feedback (symbol change rate 0.04% at .5, ~0 at .9 vs 0.24% without), not as noise smoothing; the best lr rises to
  compensate. The teammate's expectation of insensitivity to beta does not hold for either reading.
- Best GPEF per bit budget is unchanged: gpef@e0g256 2.221 (2.25 b/w), gpef@e4 1.758 (8), gpef@e8 1.607 (12).
Next decision (Henry / teammate): accept c = 4 + int error and no momentum; or one more test of a looser c (16, inf) on
the arms whose upper bound binds (fp4, es .5); no run launched.
Footprint: results rows + logs only (results 2.5M, logs 3.4M total); no checkpoints kept. Workers exited (verified 04:19Z 2026-11-13).
