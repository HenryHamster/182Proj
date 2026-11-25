# flip-vs-srqrr-003: GPEF error storage at 50M

## Question
Is gpef@e4's poor 50M result (1.596, behind gpef@e0 at 1.478) caused by how it stores the per-weight error? Plain e4
stores the error in steps of span/7 with nearest rounding. At lr below about 0.018, an update that flips no symbol is
always rounded to zero. The fixed storage (`team_gpx.py`) scored 1.646 vs 1.758 at 10M.

## Design
- Preset `50M` (49.2M quantized weights), byte-level WikiText-103, 2000 val windows. The test split is aliased to val
  and was never scored.
- Methods, unchanged from `team_gpx.py`:
  - `gpefx@e4g32_narrow_rot_b0.99`: fixed storage, 4-bit error, 8 b/w.
  - `gpefx@e2g32_sym_rot_b0.99`: symmetric 2-bit error, 6 b/w, never trained before.
- lr grid {0.01, 0.03}, seed 1 at the better value. The protocol is in the header of `campaigns/campaign_gpefix_50M.py`.

## Execution
IconLab, 2 L40S GPUs. 6 runs, 2026-11-20 16:48-21:55 PST. Output is `results/campaign_50M_gpefix.jsonl`.

## Result (val bpb)

| method | b/w | lr 0.01 (s0) | lr 0.03 (s0, s1) | mean |
|---|---|---|---|---|
| gpefx e4 fixed | 8 | 1.756 | 1.477, 1.397 | **1.437** |
| gpefx e2 sym | 6 | 2.821 | 2.378, 2.361 | 2.370 |

- The fixed e4 storage beats gpef@e4 (1.596) and gpef@e0 (1.478). It misses the prediction (srq_rot@b6, 1.243) by
  0.19. The best lr is at the grid edge.
- e2 sym is worse than e0, as the 10M e2 narrow runs had already shown.

**Cause found.** GPEF with stored error has no fixed point at zero update. During the decay phase (last 30%, lr → 0)
its weights keep moving and train loss rises: fixed e4 goes from a minimum of 1.003/0.954 nats to an end of
1.046/0.993, and e2 sym from 1.18 to 1.58. The CPU check `tests/zero_lr.py` trains 200 steps, then runs 300 steps at
lr 0 with zero gradient:

- gpef@e0 makes 0 moves.
- gpef@e4 moves 3.6% of symbols per step, and its codebook inflates 2.5×.
- Fixed e4 moves 0.012% of symbols per step, and |ΔW|/|W| = 9%.
- e2 sym moves 0.59% of symbols per step, and |ΔW|/|W| = 41%.

The decay-phase rise accounts for about 0.06 bpb of the fixed e4 gap. gpef@e0 results are unaffected.

## Next decision
Henry decides. The recommendation is to use gpef@e0 as the GPEF entry at 100M and srq_rot@b6 for 8 b/w, unless a
stationary GPEF error store is built first.
