"""Teammate methods (Stage43b notebook) and low-bit Adam references for the flip harness (qtrain.py).

Import after srq_ext; registers, without changing existing methods:
  gpef@eE      teammate's GPEF (GroupDiscrete kind='ef', passes=2): 4-level group-32 learned codebook {-a,-b,b,a},
               rms-normalized step (clamp +-4) in units of a, two least-squares codebook/symbol passes, E-bit
               deterministic error feedback (E in 0, 4, 8; gpef@e8 = the notebook's gpef8).     2 + E + 2 bits/weight
  adamw_ste    teammate's AdamW-STE baseline: FP32 shadow, symbols from shadow thresholds, Adam on the shadow and on
               the group codebooks, codebooks projected to the GPEF bounds after each step.         32 + 64 + 2 + 4
  adam_sr@mB   Adam direction + stochastic rounding (DQT) with both Adam moments quantized to B bits per entry:
               m signed absmax per row (harness quantize_state), v zero-excluded {1..2^B}*rowmax/2^B.  2 + 2B + 64/n_in

Port notes (gpef, adamw_ste): same initial deployed weights as every harness method (symbols from K, a = 1.5 alpha,
b = 0.5 alpha in every group, so span = a0 is one scalar per matrix); lr in units of the initial a (gpef) or of alpha
(adamw_ste, both shadow and codebooks share lr, weight decay 0); the harness LR schedule replaces the notebook's
constant LR. Group-32 codebooks change the deployed format (2 extra bits/weight), unlike the fixed per-matrix grid of
the other methods. Code for fit_project/decode follows the notebook's quant_core.py.
"""
import re
import torch
import qtrain as Q
import srq_ext as _S

Q._QMatOrig = _S._QMat                                            # the harness's own QMat class

GROUP = 32
_PAT = re.compile(r"^(gpef@e(0|4|8)|adamw_ste|adam_sr@m(8|4|2))$")


def decode(k, a, b):
    aa, bb = a[:, None], b[:, None]
    return torch.where(k == 0, -aa, torch.where(k == 1, -bb, torch.where(k == 2, bb, aa)))


def nearest(x, a, b):
    th = ((a + b) * .5)[:, None]
    return (x >= -th).to(torch.uint8) + (x >= 0).to(torch.uint8) + (x >= th).to(torch.uint8)


def fit_project(ua, ub, ha, hb, bounds):
    amin, bmin, gap, amax = bounds
    amin = torch.maximum(amin, bmin + gap)
    ha, hb = ha.clamp_min(1e-20), hb.clamp_min(1e-20)
    valid = (ua >= amin) & (ua <= amax) & (ub >= bmin) & (ub <= ua - gap)
    a1 = amin; b1 = torch.maximum(torch.minimum(ub, amin - gap), bmin)
    a2 = amax; b2 = torch.maximum(torch.minimum(ub, amax - gap), bmin)
    b3 = bmin; a3 = torch.maximum(torch.minimum(ua, amax), amin)
    b4 = ((ha * (ua - gap) + hb * ub) / (ha + hb)).clamp(min=torch.maximum(bmin, amin - gap), max=amax - gap)
    a4 = b4 + gap
    aa = torch.stack((ua, a1, a2, a3, a4)); bb = torch.stack((ub, b1, b2, b3, b4))
    cost = ha[None] * (aa - ua).square() + hb[None] * (bb - ub).square()
    cost[0] = torch.where(valid, cost[0], torch.inf)
    ch = cost.argmin(0, keepdim=True)
    return aa.gather(0, ch)[0], bb.gather(0, ch)[0]


def fit_for_symbols(x, k, a, b, bounds):
    sa = (k == 3).float() - (k == 0).float()
    sb = (k == 2).float() - (k == 1).float()
    na, nb = sa.square().sum(1), sb.square().sum(1)
    aa = torch.where(na > 0, (sa * x).sum(1) / na.clamp_min(1e-8), a)
    bb = torch.where(nb > 0, (sb * x).sum(1) / nb.clamp_min(1e-8), b)
    return fit_project(aa, bb, na.clamp_min(1e-8), nb.clamp_min(1e-8), bounds)


class GroupMat:
    """Shared base for gpef / adamw_ste: symbols k (n_groups x 32, uint8), codebooks a, b per group."""

    def __init__(self, W0, method, gen, bits="2"):
        base = Q._QMatOrig(W0, "muon_sr", gen, bits)             # harness init: alpha, K = clip(round(W0/alpha))
        self.method, self.gen, self.alpha = method, gen, base.alpha
        self.m, self.n = base.K.shape
        assert self.n % GROUP == 0
        self.K = base.K                                           # kept for the harness (n_quant_weights, n_in)
        self.k = (base.K + 1.5).round().to(torch.uint8).reshape(-1, GROUP)
        ng = self.k.shape[0]
        self.a = torch.full((ng,), 1.5 * self.alpha, device=W0.device)
        self.b = torch.full((ng,), 0.5 * self.alpha, device=W0.device)
        self.span = 1.5 * self.alpha                              # = a0, one scalar per matrix here
        s = torch.full_like(self.a, self.span)
        self.bounds = (s / 4, torch.zeros_like(s), s * 1e-6, s * 4)
        self.t, self.chg_sum, self.chg_n = 0, 0.0, 0
        self.w = self._deq().requires_grad_(True)

    def _deq(self):
        return decode(self.k, self.a, self.b).reshape(self.m, self.n)

    def on_grid(self):
        return bool((self.k <= 3).all() and (self.a > self.b).all() and (self.b >= 0).all()
                    and torch.equal(self.w.detach(), self._deq()))

    def _set(self, knew):
        self.chg_sum = self.chg_sum + (knew != self.k).float().mean()
        self.chg_n += 1
        self.k = knew
        self.K = (knew.float() - 1.5).reshape(self.m, self.n)
        self.w.copy_(self._deq())


class GPEF(GroupMat):
    def __init__(self, W0, method, gen, bits="2"):
        super().__init__(W0, method, gen, bits)
        self.E = int(method.split("@e")[1])
        self.levels = (1 << (self.E - 1)) - 1 if self.E else 0
        self.err = torch.zeros(self.k.shape, dtype=torch.int8, device=W0.device)

    def state(self):
        return {k: getattr(self, k) for k in ("k", "K", "a", "b", "err", "t", "chg_sum", "chg_n")}

    def load(self, s):
        for k, v in s.items():
            setattr(self, k, v)
        with torch.no_grad():
            self.w.copy_(self._deq())

    @torch.no_grad()
    def step(self, lr):
        g = self.w.grad.float().reshape(-1, GROUP)
        self.w.grad = None
        self.t += 1
        rms = (g.square().mean(1, keepdim=True) + 1e-12).sqrt()               # notebook eps
        x = decode(self.k, self.a, self.b) - (lr * self.a[:, None]) * (g / rms).clamp(-4, 4)
        if self.E:
            x = x + self.err.float() * (self.span / self.levels)
        a, b, kk = self.a, self.b, self.k
        for p in range(2):
            a, b = fit_for_symbols(x, kk, a, b, self.bounds)
            kk = nearest(x, a, b)
        if self.E:
            self.err = torch.round(((x - decode(kk, a, b)) / self.span * self.levels)
                                   .clamp(-self.levels, self.levels)).to(torch.int8)
        self.a, self.b = a, b
        self._set(kk)


class AdamWSTE(GroupMat):
    def __init__(self, W0, method, gen, bits="2"):
        super().__init__(W0, method, gen, bits)
        self.S = self._deq().reshape(-1, GROUP).clone()          # shadow = initial deployed weight
        z = torch.zeros_like
        self.mS, self.vS, self.ma, self.va, self.mb, self.vb = z(self.S), z(self.S), z(self.a), z(self.a), z(self.b), z(self.b)

    def state(self):
        return {k: getattr(self, k) for k in ("k", "K", "a", "b", "S", "mS", "vS", "ma", "va", "mb", "vb", "t", "chg_sum", "chg_n")}

    load = GPEF.load

    def _adam(self, p, g, m, v, lr):
        b1, b2 = .9, .999
        m.mul_(b1).add_(g, alpha=1 - b1); v.mul_(b2).addcmul_(g, g, value=1 - b2)
        p.add_(-lr * (m / (1 - b1 ** self.t)) / ((v / (1 - b2 ** self.t)).sqrt() + 1e-8 * self.alpha))

    @torch.no_grad()
    def step(self, lr):
        g = self.w.grad.float().reshape(-1, GROUP)
        self.w.grad = None
        self.t += 1
        k = self.k
        ga = (g * ((k == 3).float() - (k == 0).float())).sum(1)
        gb = (g * ((k == 2).float() - (k == 1).float())).sum(1)
        L = lr * self.alpha
        self._adam(self.S, g, self.mS, self.vS, L)                # STE: dL/dshadow = dL/dw
        self._adam(self.a, ga, self.ma, self.va, L)
        self._adam(self.b, gb, self.mb, self.vb, L)
        one = torch.ones_like(self.a)
        self.a, self.b = fit_project(self.a, self.b, one, one, self.bounds)
        self._set(nearest(self.S, self.a, self.b))


class AdamSR(Q._QMatOrig):
    def __init__(self, W0, method, gen, bits="2"):
        super().__init__(W0, "dqt", gen, bits)
        self.method, self.B = method, method.split("@m")[1]

    @torch.no_grad()
    def step(self, lr):
        G = self.w.grad.float() * self.alpha
        self.w.grad = None
        self.t += 1
        b1, b2 = .9, .999
        M = Q.quantize_state(self.M.mul(b1).add(G, alpha=1 - b1), self.B, self.gen)
        V = self.V.mul(b2).addcmul(G, G, value=1 - b2)
        L = 2 ** int(self.B)
        s = V.amax(1, keepdim=True).clamp_min(1e-30) / L
        V = torch.clamp(torch.floor(V / s + torch.rand(V.shape, generator=self.gen, device=V.device)), 1, L) * s
        self.M, self.V = M, V
        D = (M / (1 - b1 ** self.t)) / ((V / (1 - b2 ** self.t)).sqrt() + 1e-8 * self.alpha)
        Kn = self._clip(self._sr(self.K - lr * D))
        self.chg_sum = self.chg_sum + (Kn != self.K).float().mean()
        self.chg_n += 1
        self.K = Kn
        self.w.copy_(self.alpha * self.K)


def _factory(W0, method, gen, bits="2"):
    if _PAT.match(method):
        cls = GPEF if method.startswith("gpef") else AdamWSTE if method == "adamw_ste" else AdamSR
        return cls(W0, method, gen, bits)
    return _prev_factory(W0, method, gen, bits)


def _state_bits(method, n_in, bits="2"):
    if not _PAT.match(method):
        return _prev_bits(method, n_in, bits)
    if method.startswith("gpef"):
        return 2.0 + int(method.split("@e")[1]) + 64.0 / GROUP
    if method == "adamw_ste":
        return 32.0 + 64.0 + 64.0 / GROUP + 128.0 / GROUP
    return 2.0 + 2 * int(method.split("@m")[1]) + 64.0 / n_in


_prev_factory, _prev_bits = Q.QMat, Q.state_bits_per_weight
Q.QMat = _factory
Q.state_bits_per_weight = _state_bits
