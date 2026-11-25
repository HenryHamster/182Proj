"""flip-vs-srqrr-001, teammate's GPEF revisions (Henry 2026-11-12 'run them'): codebook-constraint divisor, float error
storage, and momentum, one change at a time.

Import after team_ext, team_grp; registers, without changing existing methods:
  gpefv@e<E><fmt>g<G>_c<c>[_<mom><beta>]
      GPEF (team_ext.GPEF step: rms-normalized clamped step in units of a, two fit/nearest passes) with
        E, fmt  per-weight error e = x - decode(k), stored relative to span s (= 1.5 alpha, per matrix) as
                (none)  E=0: no error feedback
                int     E-bit integer on +-s, step s/(2^(E-1)-1), nearest (team_ext's storage; e8int = gpef@e8)
                fp8     float8 e4m3 of e/s, clipped to +-1, round-to-nearest cast                       8 bits
                fp4     e2m1 values {0,.5,1,1.5,2,3,4,6} of 6 e/s, clipped to +-s, nearest             4 bits
        G       codebook group size (fp32 a, b per group: 64/G bits/weight)
        c       codebook constraint s/c <= a <= c s, b >= 0, a - b >= 1e-6 s; c=4 is team_ext's; inf drops the a bounds
        mom     es  error smoothing: m <- beta m + (1-beta) e, the stored m (fp8, relative to s) is fed back instead of e
                gm  gradient momentum: u <- beta u + (1-beta) clamp(g/rms, +-4), step -lr a u; u stored fp8 (|u| <= 4,
                    no scale); error feedback unchanged
                    (the teammate's 'error momentum' admits both readings; both are run.)        +8 bits
      gpefv@e0g256_c4 == gpef@e0g256 and gpefv@e8intg32_c4 == gpef@e8 step for step (team_gpf_test.py).
      Bits/weight 2 + E + 64/G (+ 8 with mom).
Each instance also records codebook-bound activity (share of groups with a at its lower / upper bound, b at 0) averaged
over steps and at the end, and quantiles of a/s; Q.run results gain a 'gpefv' entry with the weight-averaged figures.
"""
import re
import torch
import qtrain as Q
import team_ext as T
import team_grp as TG

_P = re.compile(r"^gpefv@e(0|4|8)(int|fp8|fp4)?g(32|64|128|256)_c(2|4|16|inf)(?:_(es|gm)(0\.\d+))?$")
FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
F8 = torch.float8_e4m3fn
_REG = []


def _fp4(r):
    """Nearest e2m1 value of r (|r| <= 6 after clipping); returns the dequantized value."""
    t = FP4.to(r.device)
    a = r.abs().clamp(max=6.)
    i = (a[..., None] - t).abs().argmin(-1)
    return torch.sign(r) * t[i]


def _f8(r):
    return r.clamp(-448., 448.).to(F8)


class GPEFV(T.GPEF):
    def __init__(self, W0, method, gen, bits="2"):
        E, fmt, G, c, mom, beta = _P.match(method).groups()
        assert (E == "0") == (fmt is None), method
        assert mom != "es" or E != "0", method
        with TG._Group(32):
            super().__init__(W0, "gpef@e0", gen, bits)            # gpef@e0 init, then regroup (as team_grp)
        self.E, self.fmt, self.G = int(E), fmt, int(G)
        self.method, self.mom, self.beta = method, mom, None if beta is None else float(beta)
        ng = self.k.numel() // self.G
        assert self.k.numel() % self.G == 0
        self.k = self.k.reshape(ng, self.G)
        self.a = torch.full((ng,), 1.5 * self.alpha, device=W0.device)
        self.b = torch.full((ng,), 0.5 * self.alpha, device=W0.device)
        s = torch.full_like(self.a, self.span)
        lo, hi = (torch.zeros_like(s), torch.full_like(s, 1e30)) if c == "inf" else (s / float(c), s * float(c))
        self.bounds = (lo, torch.zeros_like(s), s * 1e-6, hi)
        self.levels = (1 << (self.E - 1)) - 1 if self.E else 0
        z = torch.zeros(self.k.shape, device=W0.device)
        self.err = z.to(torch.int8) if fmt == "int" else (z.to(F8) if fmt == "fp8" else z.clone())  # fp4: dequantized
        self.mstate = z.to(F8) if mom else None
        self.bs = torch.zeros(3, device=W0.device); self.bn = 0
        with torch.no_grad():
            self.w.copy_(self._deq())
        _REG.append(self)

    def state(self):
        return {k: getattr(self, k) for k in ("k", "K", "a", "b", "err", "mstate", "bs", "bn", "t", "chg_sum", "chg_n")}

    def _err_value(self):
        """Stored error (or its smoothed version), in weight units."""
        if self.mom == "es":
            return self.mstate.float() * self.span
        if self.fmt == "int":
            return self.err.float() * (self.span / self.levels)
        return self.err.float() * self.span if self.fmt == "fp8" else self.err * (self.span / 6.)

    def _store_err(self, res):
        r = (res / self.span).clamp(-1., 1.)
        if self.fmt == "int":
            q = torch.round(r * self.levels).clamp(-self.levels, self.levels)
            self.err = q.to(torch.int8)
            e = q * (self.span / self.levels)
        elif self.fmt == "fp8":
            self.err = _f8(r)
            e = self.err.float() * self.span
        else:
            self.err = _fp4(r * 6.)
            e = self.err * (self.span / 6.)
        if self.mom == "es":                                       # smooth the quantized error, store m in fp8
            self.mstate = _f8(self.beta * self.mstate.float() + (1 - self.beta) * e / self.span)

    @torch.no_grad()
    def step(self, lr):
        with TG._Group(self.G):
            g = self.w.grad.float().reshape(-1, self.G)
            self.w.grad = None
            self.t += 1
            rms = (g.square().mean(1, keepdim=True) + 1e-12).sqrt()
            d = (g / rms).clamp(-4, 4)
            if self.mom == "gm":
                self.mstate = _f8(self.beta * self.mstate.float() + (1 - self.beta) * d)
                d = self.mstate.float()
            x = T.decode(self.k, self.a, self.b) - (lr * self.a[:, None]) * d
            if self.E:
                x = x + self._err_value()
            a, b, kk = self.a, self.b, self.k
            for p in range(2):
                a, b = T.fit_for_symbols(x, kk, a, b, self.bounds)
                kk = T.nearest(x, a, b)
            if self.E:
                self._store_err(x - T.decode(kk, a, b))
            self.a, self.b = a, b
            self._set(kk)
            lo, _, _, hi = self.bounds
            self.bs += torch.stack(((a <= lo * (1 + 1e-5)).float().mean(), (a >= hi * (1 - 1e-5)).float().mean(),
                                    (b <= 0).float().mean()))
            self.bn += 1

    def bound_stats(self):
        lo, _, _, hi = self.bounds
        q = torch.quantile((self.a / self.span).float().cpu(), torch.tensor([0., .01, .5, .99, 1.]))
        m = (self.bs / max(self.bn, 1)).tolist()
        return dict(n=self.a.numel() * self.G, a_lo_mean=m[0], a_hi_mean=m[1], b0_mean=m[2],
                    a_lo_end=float((self.a <= lo * (1 + 1e-5)).float().mean()),
                    a_hi_end=float((self.a >= hi * (1 - 1e-5)).float().mean()),
                    b0_end=float((self.b <= 0).float().mean()), a_over_s_q=[round(float(v), 4) for v in q])


def _factory(W0, method, gen, bits="2"):
    if _P.match(method):
        return GPEFV(W0, method, gen, bits)
    return _prev_factory(W0, method, gen, bits)


def _state_bits(method, n_in, bits="2"):
    m = _P.match(method)
    if m:
        return 2.0 + int(m.group(1)) + 64.0 / int(m.group(3)) + (8.0 if m.group(5) else 0.0)
    return _prev_bits(method, n_in, bits)


def _run(*a, **k):
    _REG.clear()
    r = _prev_run(*a, **k)
    if isinstance(r, dict) and _REG:
        st = [q.bound_stats() for q in _REG]
        n = sum(s["n"] for s in st)
        agg = {key: sum(s[key] * s["n"] for s in st) / n for key in st[0] if key not in ("n", "a_over_s_q")}
        agg["a_over_s_min"] = min(s["a_over_s_q"][0] for s in st)
        agg["a_over_s_max"] = max(s["a_over_s_q"][4] for s in st)
        agg["per_matrix_a_over_s_q"] = [s["a_over_s_q"] for s in st]
        r["gpefv"] = agg
    _REG.clear()
    return r


_prev_factory, _prev_bits, _prev_run = Q.QMat, Q.state_bits_per_weight, Q.run
Q.QMat = _factory
Q.state_bits_per_weight = _state_bits
Q.run = _run
