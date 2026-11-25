"""srqrr family for the flip harness (qtrain.py): 2-bit weights trained from scratch with b extra bits per weight.

Import after qtrain; it registers three method families without changing existing methods:
  srq_sr@bB     normalized gradient step + stochastic rounding onto the 2^-B sub-grid      2 + B bits/weight
  srq_rot@bB    same step, rotation rounding (threshold frac(u0 + t*phi), u0 regenerated)  2 + B bits/weight
  srqrr@bB      rotation rounding + shrunk Newton refit of a per-row scale alpha_r          2 + B + 32/n_in bits/weight
Suffix "_kX" (e.g. srqrr_k10@b2) multiplies the refit constant kappa by X.

Port of the 4-bit Pythia srqrr (PR HenryHamster/182Proj#2, srqrr.py) to this format:
  - position x = K + f, K on the 2-bit grid {-1.5,-0.5,0.5,1.5}, f on 2^-b Z in [-1/2, 1/2 - 2^-b]; only K is deployed.
  - step in levels: x~ = clip(K + f - lr * (alpha0/alpha_r) * g/rms(g), -1.5, 1.5), g = dL/dW (no momentum).
  - refit: theta = alpha_r (one scale per row, replacing group32 scales). Per-sequence statistics come from the
    layer output z = x K^T and its gradient, captured in F.linear: a_b,r = B sum_t d_btr z_btr, H_r = mean_n z^2 * row_r.
    Shrinkage and constants (kappa .003, alpha 3, cap .05) are unchanged from srqrr.py.
"""
import math, re, types
import torch
import torch.nn.functional as TF
import qtrain as Q

PHI = (5 ** .5 - 1) / 2
_REG = {}                      # id(leaf w) -> SRQMat, for output capture
_COUNT = [0]
_PAT = re.compile(r"^(srq_sr|srq_rot|srqrr)(?:_k([0-9.]+))?@b([0-9]+)$")


def parse(method):
    m = _PAT.match(method)
    return None if m is None else dict(kind=m.group(1), kmul=float(m.group(2) or 1), bits=int(m.group(3)))


def _linear(x, w, b=None):
    out = TF.linear(x, w, b)
    q = _REG.get(id(w))
    if q is not None and q.refit and torch.is_grad_enabled() and out.requires_grad:
        q._z = out.detach()
        out.register_hook(q._grab)
    return out


class _FProxy(types.ModuleType):
    def __getattr__(self, k):
        return getattr(TF, k)


class SRQMat(Q.QMat):
    def __init__(self, W0, method, gen, bits="2"):
        super().__init__(W0, "muon_sr", gen, bits)       # same init as every other method: K = clip(round(W0/alpha))
        c = parse(method)
        self.method, self.kind, self.b = method, c["kind"], c["bits"]
        self.refit = self.kind == "srqrr"
        self.q = 2.0 ** -self.b
        self.M = None                                     # no momentum
        self.f = torch.zeros(self.K.shape, dtype=torch.int8, device=self.K.device)
        self.alpha0 = self.alpha
        self.ar = torch.full((self.K.shape[0], 1), self.alpha, device=self.K.device)
        self.kappa, self.calpha, self.cap = .003 * c["kmul"], 3.0, .05
        self.useed = 1000003 + _COUNT[0]; _COUNT[0] += 1
        self.base_lr, self._z, self._d = None, None, None
        with torch.no_grad():
            self.w = (self.ar * self.K).requires_grad_(True)
        _REG[id(self.w)] = self

    def _grab(self, g):
        self._d = g.detach()

    def state(self):
        return {k: getattr(self, k) for k in ("K", "f", "ar", "t", "chg_sum", "chg_n")}

    def load(self, s):
        for k, v in s.items():
            setattr(self, k, v)
        with torch.no_grad():
            self.w.copy_(self.ar * self.K)

    def _threshold(self, x):
        if self.kind == "srq_sr":
            return torch.rand(x.shape, generator=self.gen, device=x.device)
        g = torch.Generator(device=x.device); g.manual_seed(self.useed)
        return torch.frac(torch.rand(x.shape, generator=g, device=x.device) + (self.t * PHI) % 1.0)

    def refit_stats(self, d, z):
        """d = dL/dout (B,T,R) for the batch-mean loss, z = x K^T (B,T,R). Returns Ga, Va, H per row (R,)."""
        d, z = d.float(), z.float()
        B, T, R = d.shape
        z = z / self.ar.view(1, 1, R)                     # out = alpha_r * (x K^T)
        ab = B * (d * z).sum(1)                           # (B, R): per-sequence d l_b / d alpha_r
        Ga = ab.mean(0)
        S2a = (ab / B).square().sum(0)
        Va = ((B * S2a - Ga ** 2) / (B - 1)).clamp_min(0)
        hz = z.square().mean((0, 1))
        row = (B * d).square().sum((0, 1)) * (T / B)
        return Ga, Va, hz * row

    @torch.no_grad()
    def step(self, lr):
        gw = self.w.grad.float()
        self.w.grad = None
        if self.base_lr is None:
            self.base_lr = lr
        sp = self.ar / self.alpha0                         # level spacing relative to the initial scale
        xn = (self.K + self.f.float() * self.q - lr * gw / gw.square().mean().sqrt().clamp_min(1e-30) / sp)
        xn = xn.clamp(self.qmin, self.qmax)
        y = (xn - self.off) / self.q
        lo = torch.floor(y)
        xq = ((lo + (self._threshold(xn) < (y - lo)).float()) * self.q + self.off).clamp(self.qmin, self.qmax)
        Kn = self._clip(torch.floor(xq - self.off + .5) + self.off)
        if self.b:
            self.f = torch.round((xq - Kn) / self.q).clamp(-2 ** (self.b - 1), 2 ** (self.b - 1) - 1).to(torch.int8)
        self.chg_sum = self.chg_sum + (Kn != self.K).float().mean()
        self.chg_n += 1
        if self.refit and self._d is not None:
            Ga, Va, H = self.refit_stats(self._d, self._z)
            sig2 = (Ga ** 2 - Va).clamp_min(0).mean()
            mu = Ga * sig2 / (sig2 + Va).clamp_min(1e-30)
            dth = -self.kappa * (lr / self.base_lr) * mu / (self.calpha * H).clamp_min(1e-30)
            th = self.ar[:, 0]
            lim = self.cap * th.abs()
            self.ar = (th + torch.maximum(torch.minimum(dth, lim), -lim)).clamp_min(1e-8)[:, None]
        self._z = self._d = None
        self.K = Kn
        self.t += 1
        self.w.copy_(self.ar * self.K)


_QMat, _sbits = Q.QMat, Q.state_bits_per_weight


def _factory(W0, method, gen, bits="2"):
    return SRQMat(W0, method, gen, bits) if parse(method) else _QMat(W0, method, gen, bits)


def _state_bits(method, n_in, bits="2"):
    c = parse(method)
    if c is None:
        return _sbits(method, n_in, bits)
    return float(bits) + c["bits"] + (32.0 / n_in if c["kind"] == "srqrr" else 0.0)


Q.QMat = _factory
Q.state_bits_per_weight = _state_bits
_F = _FProxy("F_proxy"); _F.linear = _linear
Q.F = _F
