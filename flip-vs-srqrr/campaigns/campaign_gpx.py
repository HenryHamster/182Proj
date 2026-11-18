"""flip-vs-srqrr-001, rotation x GPEF batch (Henry 2026-11-10 'run it'). 10M only, campaign.py protocol: seed-0 grid of 3,
one x3 edge extension, seeds 1,2 at the seed-0 best.
Stage 1: GPEF error-storage ladder at E=4, groups of 32 (8 b/w; baseline gpef@e4 1.758) and the frozen-codebook ablation.
Stage 2: the best ladder variant (3-seed mean) at e2g256 (4.25 b/w) and e2g32 (6 b/w), plus the references the teammate
asked for: adamw_ste (theirs, group-32 format), adam_qat (harness AdamW-STE, per-matrix format, pairs with muon_qat),
fp_adam and fp_muon (no quantization)."""
import statistics as st
import campaign as C

C.PAR['10M'] = 4
G3 = [0.01, 0.03, 0.09]
LADDER = ['gpefx@e4g32_narrow_near_b1', 'gpefx@e4g32_narrow_rot_b1', 'gpefx@e4g32_narrow_rot_b0.9',
          'gpefx@e4g32_narrow_rot_b0.99']
C.phase('10M', {**{m: G3 for m in LADDER}, 'gpef_frz@e0_near': G3, 'gpef_frz@e0_rot': G3}, ext=1)


def mean3(m):
    b = C.best('10M', m)[0]
    return st.mean(r['val_bpb'] for r in C.rows('10M') if r['method'] == m and r['lr'] == b)


sc = {m: round(mean3(m), 4) for m in LADDER}
print('LADDER', sc, flush=True)
v = min(sc, key=sc.get).split('_', 1)[1]
S2 = {f'gpefx@e2g256_{v}': G3, f'gpefx@e2g32_{v}': G3}
print('STAGE2', list(S2), flush=True)
C.phase('10M', {**S2, 'adamw_ste': G3, 'adam_qat': G3, 'fp_adam': G3, 'fp_muon': [0.003, 0.01, 0.03]}, ext=1)
print('GPX DONE', flush=True)
