"""Format-matched check for flip-vs-srqrr-001: rotation / stochastic rounding on GPEF's group-32 learned codebooks.

Import after team_ext; registers, without changing existing methods:
  gpef_rot@e0   GPEF@e0 (same rms-normalized step, same two least-squares codebook/symbol passes, same bounds) except the
                final symbols: rotation rounding of x between the two bracketing codebook levels, threshold
                frac(u0 + t*phi) with u0 regenerated per entry from a per-matrix seed (no state).   2 + 0 + 2 bits/weight
  gpef_sr@e0    same, stochastic rounding (independent uniform thresholds).                          2 + 0 + 2 bits/weight
With threshold 0.5 everywhere both reduce exactly to gpef@e0 (nearest level), which team_rot_test.py checks.
gpef@e0 vs gpef_rot@e0 isolates the rounding rule on the group-codebook format; gpef_rot@e0 vs srq_rot@b2 (both
4 b/w of training state) isolates the format.
"""
import re
import torch
import qtrain as Q
import team_ext as T

PHI = (5 ** .5 - 1) / 2
_PAT = re.compile(r"^gpef_(rot|sr)@e0$")
_COUNT = [0]


class GPEFRound(T.GPEF):
    def __init__(self, W0, method, gen, bits="2"):
        super().__init__(W0, "gpef@e0", gen, bits)
        self.method, self.kind = method, _PAT.match(method).group(1)
        self.useed = 2000003 + _COUNT[0]; _COUNT[0] += 1

    def _u(self, x):
        if self.kind == "sr":
            return torch.rand(x.shape, generator=self.gen, device=x.device)
        g = torch.Generator(device=x.device); g.manual_seed(self.useed)
        return torch.frac(torch.rand(x.shape, generator=g, device=x.device) + (self.t * PHI) % 1.0)

    @torch.no_grad()
    def step(self, lr):
        g = self.w.grad.float().reshape(-1, T.GROUP)
        self.w.grad = None
        self.t += 1
        rms = (g.square().mean(1, keepdim=True) + 1e-12).sqrt()
        x = T.decode(self.k, self.a, self.b) - (lr * self.a[:, None]) * (g / rms).clamp(-4, 4)
        a, b, kk = self.a, self.b, self.k
        for p in range(2):                                         # identical to GPEF: codebook fit + nearest
            a, b = T.fit_for_symbols(x, kk, a, b, self.bounds)
            kk = T.nearest(x, a, b)
        aa, bb = a[:, None], b[:, None]
        xc = torch.maximum(torch.minimum(x, aa), -aa)
        j = (xc >= -bb).long() + (xc >= bb).long()                 # interval 0: [-a,-b), 1: [-b,b), 2: [b,a]
        lo = torch.where(j == 0, -aa, torch.where(j == 1, -bb, bb))
        hi = torch.where(j == 0, -bb, torch.where(j == 1, bb, aa))
        p = ((xc - lo) / (hi - lo).clamp_min(1e-20)).clamp(0, 1)
        kk = (j + (self._u(x) < p).long()).to(torch.uint8)
        self.a, self.b = a, b
        self._set(kk)


def _factory(W0, method, gen, bits="2"):
    if _PAT.match(method):
        return GPEFRound(W0, method, gen, bits)
    return _prev_factory(W0, method, gen, bits)


def _state_bits(method, n_in, bits="2"):
    if _PAT.match(method):
        return 2.0 + 0 + 64.0 / T.GROUP
    return _prev_bits(method, n_in, bits)


_prev_factory, _prev_bits = Q.QMat, Q.state_bits_per_weight
Q.QMat = _factory
Q.state_bits_per_weight = _state_bits
