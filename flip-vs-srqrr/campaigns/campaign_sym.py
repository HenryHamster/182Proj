"""flip-vs-srqrr-001, symmetric-error ladder (Henry 2026-11-11 'Run the ladder'). Does rotation rounding of GPEF's stored
per-weight error beat nearest rounding at every error width once the codes are symmetric (sym: levels +-1/2, +-3/2 ..
steps)? drift_sym.py: sym is unbiased (up share 0.50, mean symbol 1.49-1.50) where narrow drifts down at E=1,2 (and
mildly at E=4). 10M, campaign.py protocol (seed-0 grid of 3, one x3 edge extension, seeds 1,2 at the seed-0 best).
Groups of 256: E=1/2/4 -> 3.25/4.25/6.25 b/w; references gpef@e0g256 2.220 (2.25), gpef@e0 1.889 (4), gpef@e4 1.758 (8)."""
import campaign as C

C.PAR['10M'] = 4
G3 = [0.01, 0.03, 0.09]
C.phase('10M', {f'gpefx@e{E}g256_sym_{r}': G3 for E in (1, 2, 4) for r in ('near_b1', 'rot_b0.99')}, ext=1)
print('SYM DONE', flush=True)
