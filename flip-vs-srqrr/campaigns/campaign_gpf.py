"""flip-vs-srqrr-001, teammate's GPEF revisions (Henry 2026-11-12 'run them'); methods in team_gpf.py. 10M, campaign.py
protocol (seed-0 grid of 3, one x3 edge extension, seeds 1,2 at the seed-0 best).
  1 constraint divisor c (a in [s/c, c s]): first one seed of c=4 at the known best lr with bound logging
    (gpefv@e0g256_c4 lr .03 and gpefv@e8intg32_c4 lr .01, which must reproduce gpef@e0g256 2.241 and gpef@e8 1.607);
    then c=2 only if some a/s left [0.5, 2], and c=16, inf only if >= 1% of groups sat at a c=4 bound.
  2 error storage at the same bits: fp8 e4m3 vs int8 (gpef@e8, 12 b/w); fp4 e2m1 vs int4 (gpef@e4 1.758, 8 b/w).
  3 momentum, fp8 state, on fp8 error (20 b/w): gm (gradient momentum) beta .9/.99; es (smoothed error fed back)
    beta .5/.9 -- es accumulates only (1-beta) of each step, so .99 would barely move (team_gpf_test.py, tiny).
"""
import campaign as C

C.PAR['10M'] = 4
LO, MID = [0.0033333, 0.01, 0.03], [0.01, 0.03, 0.09]
DIAG = {'gpefv@e0g256_c4': 0.03, 'gpefv@e8intg32_c4': 0.01}
FMT = {'gpefv@e8fp8g32_c4': LO, 'gpefv@e4fp4g32_c4': MID}
MOM = {'gpefv@e8fp8g32_c4_gm0.9': LO, 'gpefv@e8fp8g32_c4_gm0.99': LO,
       'gpefv@e8fp8g32_c4_es0.5': LO, 'gpefv@e8fp8g32_c4_es0.9': MID}
C.pool('10M', [(m, lr, 0) for m, lr in DIAG.items()] + [(m, lr, 0) for g in (FMT, MOM) for m, lrs in g.items() for lr in lrs])

SWEEP = {}
for m, lr in DIAG.items():
    r = [x for x in C.rows('10M') if x['method'] == m and x['lr'] == lr and x['seed'] == 0][0]
    v = r['gpefv']
    hit = max(v['a_lo_mean'], v['a_hi_mean'], v['a_lo_end'], v['a_hi_end'])
    print('BOUNDS', m, 'val_bpb %.4f' % r['val_bpb'], 'bound share %.4f' % hit,
          'a/s range %.3f..%.3f' % (v['a_over_s_min'], v['a_over_s_max']), flush=True)
    g = LO if 'e8' in m else MID
    if v['a_over_s_min'] < 0.5 or v['a_over_s_max'] > 2.0:
        SWEEP[m.replace('_c4', '_c2')] = g
    if hit >= 0.01:
        SWEEP[m.replace('_c4', '_c16')] = g
        SWEEP[m.replace('_c4', '_cinf')] = g
print('SWEEP', sorted(SWEEP), flush=True)
C.phase('10M', {**FMT, **MOM, **SWEEP}, ext=1)
print('GPF DONE', flush=True)
