"""flip-vs-srqrr-001, rotation x GPEF (Henry 2026-11-10 'run it'): GPEF's error feedback stored as an SRQ-style remainder,
plus the frozen-codebook ablation.

Import after team_ext, team_rot, team_grp; registers, without changing existing methods:
  gpefx@eEgG_<scale>_<rnd>_b<beta>
      GPEF with codebook groups of G (as gpef@e0gG) and an E-bit per-weight error e (int8), one change at a time:
        scale  wide    error step span/(2^(E-1)-1), range +-span (= 1.5 alpha): GPEF's original storage
               narrow  error step R/2^(E-1), codes -2^(E-1) .. 2^(E-1)-1, R = max((a-b)/2, b) per group: the largest
                       residual a nearest-level symbol can leave inside [-a, a] (residual beyond +-a is clipped)
               sym     as narrow, but code c stores (c + 1/2) * step, i.e. levels +-1/2, +-3/2, .. steps: symmetric
                       about zero, so rotation rounding has no preferred direction (narrow's range -2^(E-1)..2^(E-1)-1
                       drifts weights down at E=2, RECORD 2026-11-11). Initial codes -1/0 at random (stored +-1/2 step).
        rnd    near    error rounded to the nearest step (GPEF's original)
               rot     rotation rounding: up when frac exceeds frac(u0 + t*phi), u0 per entry from a per-matrix seed
        beta   leak on the stored error: target x = decode(k) - step + beta * e * step_size (beta=1: no forgetting)
      gpefx@e4g32_wide_near_b1 reproduces gpef@e4 exactly (team_gpx_test.py).       2 + E + 64/G bits/weight
  gpef_frz@e0_<rnd>
      GPEF@e0's step with the codebooks frozen at their initial values (a = 1.5 alpha, b = 0.5 alpha, i.e. the
      per-matrix grid alpha*{+-.5, +-1.5}); symbols set to the nearest level (near) or by rotation between the two
      bracketing levels (rot). No refit, so no codebook storage.                         2 bits/weight
  gpefr@gG_<order>
      GPEF@e0gG (refit codebooks, no error) with rotation rounding of the symbols between the two bracketing levels.
        after   fit + nearest twice (as GPEF), then rotate x between levels of that codebook; the stored codebook was
                fit for the nearest symbols, not the rotated ones. G=32 reproduces gpef_rot@e0 (team_gpx_test.py).
        before  rotate x between levels of the current codebook, then refit the codebook by least squares for those
                rotated symbols (one fit_for_symbols pass), so the stored codebook matches the stored symbols.
                                                                                         2 + 64/G bits/weight
  gpefr@gG_after_l<lam>
      as after, but the codebook takes the full step (refit at full lr) and rotation sees only lam times each weight's
      residual after the refit: x_rot = decode(k) + lam * (x - decode(k)). Flip probability and per-weight expected
      update scale by lam, as does the rounding-noise variance. lam -> 1 is after (Henry 2026-11-11 'just run this one').
"""
import re
import torch
import qtrain as Q
import team_ext as T
import team_grp as TG

PHI = (5 ** .5 - 1) / 2
_PX = re.compile(r"^gpefx@e([1-8])g(32|64|128|256)_(wide|narrow|sym)_(near|rot)_b(1|0\.\d+)$")
_PF = re.compile(r"^gpef_frz@e0_(near|rot)$")
_PR = re.compile(r"^gpefr@g(32|64|128|256)_(after|before)(?:_l(0\.\d+))?$")
_COUNT = [0]


def _useed():
    _COUNT[0] += 1
    return 3000017 + _COUNT[0]


def _rot_u(shape, seed, t, device):
    g = torch.Generator(device=device); g.manual_seed(seed)
    return torch.frac(torch.rand(shape, generator=g, device=device) + (t * PHI) % 1.0)


def round_between(x, a, b, u):
    """Unbiased rounding of x (clipped to [-a, a]) between its two bracketing levels of {-a,-b,b,a}: up if u < p."""
    aa, bb = a[:, None], b[:, None]
    xc = torch.maximum(torch.minimum(x, aa), -aa)
    j = (xc >= -bb).long() + (xc >= bb).long()
    lo = torch.where(j == 0, -aa, torch.where(j == 1, -bb, bb))
    hi = torch.where(j == 0, -bb, torch.where(j == 1, bb, aa))
    p = ((xc - lo) / (hi - lo).clamp_min(1e-20)).clamp(0, 1)
    return (j + (u < p).long()).clamp(0, 3).to(torch.uint8)


class GPEFX(TG.GPEFG):
    def __init__(self, W0, method, gen, bits="2"):
        E, G, scale, rnd, beta = _PX.match(method).groups()
        super().__init__(W0, f"gpef@e0g{G}", gen, bits)           # gpef@e0gG init: symbols, codebooks, bounds
        self.method, self.scale, self.rnd, self.beta = method, scale, rnd, float(beta)
        self.E = int(E)
        if scale == "wide":
            self.levels, self.lo, self.hi = (1 << (self.E - 1)) - 1, -((1 << (self.E - 1)) - 1), (1 << (self.E - 1)) - 1
        else:
            self.levels, self.lo, self.hi = 1 << (self.E - 1), -(1 << (self.E - 1)), (1 << (self.E - 1)) - 1
        self.err = torch.zeros(self.k.shape, dtype=torch.int8, device=W0.device)
        self.useed = _useed()
        self.off = .5 if scale == "sym" else 0.
        if scale == "sym":
            g0 = torch.Generator(device=W0.device); g0.manual_seed(self.useed + 7)
            self.err = -(torch.rand(self.k.shape, generator=g0, device=W0.device) < .5).to(torch.int8)

    def _step_size(self, a, b):
        if self.scale == "wide":
            return self.span / self.levels
        return (torch.maximum((a - b) * .5, b) / self.levels)[:, None]

    @torch.no_grad()
    def step(self, lr):
        with TG._Group(self.G):
            g = self.w.grad.float().reshape(-1, self.G)
            self.w.grad = None
            self.t += 1
            rms = (g.square().mean(1, keepdim=True) + 1e-12).sqrt()
            x = T.decode(self.k, self.a, self.b) - (lr * self.a[:, None]) * (g / rms).clamp(-4, 4)
            e = self.err.float() + self.off
            e = e if self.beta == 1.0 else self.beta * e
            x = x + e * self._step_size(self.a, self.b)
            a, b, kk = self.a, self.b, self.k
            for p in range(2):
                a, b = T.fit_for_symbols(x, kk, a, b, self.bounds)
                kk = T.nearest(x, a, b)
            res = x - T.decode(kk, a, b)
            r = res / self.span * self.levels if self.scale == "wide" else res / self._step_size(a, b) - self.off
            if self.rnd == "near":
                q = torch.round(r)
            else:
                q = torch.floor(r)
                q = q + (_rot_u(r.shape, self.useed, self.t, r.device) < (r - q)).float()
            self.err = q.clamp(self.lo, self.hi).to(torch.int8)
            self.a, self.b = a, b
            self._set(kk)


class GPEFFrozen(T.GPEF):
    def __init__(self, W0, method, gen, bits="2"):
        super().__init__(W0, "gpef@e0", gen, bits)
        self.method, self.rnd = method, _PF.match(method).group(1)
        self.useed = _useed()

    @torch.no_grad()
    def step(self, lr):
        g = self.w.grad.float().reshape(-1, T.GROUP)
        self.w.grad = None
        self.t += 1
        rms = (g.square().mean(1, keepdim=True) + 1e-12).sqrt()
        x = T.decode(self.k, self.a, self.b) - (lr * self.a[:, None]) * (g / rms).clamp(-4, 4)
        if self.rnd == "near":
            kk = T.nearest(x, self.a, self.b)
        else:
            kk = round_between(x, self.a, self.b, _rot_u(x.shape, self.useed, self.t, x.device))
        self._set(kk)                                             # a, b never change


class GPEFRotG(TG.GPEFG):
    def __init__(self, W0, method, gen, bits="2"):
        G, order, lam = _PR.match(method).groups()
        super().__init__(W0, f"gpef@e0g{G}", gen, bits)
        self.method, self.order = method, order
        self.lam = 1.0 if lam is None else float(lam)
        assert order == "after" or lam is None
        self.useed = _useed()

    @torch.no_grad()
    def step(self, lr):
        with TG._Group(self.G):
            g = self.w.grad.float().reshape(-1, self.G)
            self.w.grad = None
            self.t += 1
            rms = (g.square().mean(1, keepdim=True) + 1e-12).sqrt()
            x = T.decode(self.k, self.a, self.b) - (lr * self.a[:, None]) * (g / rms).clamp(-4, 4)
            u = _rot_u(x.shape, self.useed, self.t, x.device)
            a, b, kk = self.a, self.b, self.k
            if self.order == "after":
                for p in range(2):
                    a, b = T.fit_for_symbols(x, kk, a, b, self.bounds)
                    kk = T.nearest(x, a, b)
                if self.lam != 1.0:                               # rotate only lam * (residual after the refit)
                    d = T.decode(kk, a, b)
                    x = d + self.lam * (x - d)
                kk = round_between(x, a, b, u)
            else:
                kk = round_between(x, a, b, u)
                a, b = T.fit_for_symbols(x, kk, a, b, self.bounds)
            self.a, self.b = a, b
            self._set(kk)


def _factory(W0, method, gen, bits="2"):
    if _PR.match(method):
        return GPEFRotG(W0, method, gen, bits)
    if _PX.match(method):
        return GPEFX(W0, method, gen, bits)
    if _PF.match(method):
        return GPEFFrozen(W0, method, gen, bits)
    return _prev_factory(W0, method, gen, bits)


def _state_bits(method, n_in, bits="2"):
    m = _PX.match(method)
    if m:
        return 2.0 + int(m.group(1)) + 64.0 / int(m.group(2))
    if _PF.match(method):
        return 2.0
    if _PR.match(method):
        return 2.0 + 64.0 / int(_PR.match(method).group(1))
    return _prev_bits(method, n_in, bits)


_prev_factory, _prev_bits = Q.QMat, Q.state_bits_per_weight
Q.QMat = _factory
Q.state_bits_per_weight = _state_bits
