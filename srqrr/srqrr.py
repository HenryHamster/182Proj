"""srqrr: rotation rounding + level refit for training 4-bit group-quantized weights with b extra bits per weight.

Format (group32x4). Every 2D weight matrix W (R x C) is stored as 4-bit codes c (R x C, values 0..15) and
one FP32 scale per row and 32-column group, theta (R x C/32). The deployed weight is
    W[r, j] = theta[r, j // 32] * (c[r, j] - 7.5).
Here the scales are kept as a level table L = theta[..., None] * (k - 7.5), k = 0..15, of shape (R, C/32, 16).

Optimizer memory. A b-bit counter f per weight (b = 0 stores nothing). The optimizer's position in level units
is x = c + f with f on the grid 2^-b Z in [-1/2, 1/2 - 2^-b]; only c is deployed.

Optimizers (all share the same normalized step  x~ = clip(c + f - lr * g / (rms(g) * spacing), 0, 15)):
    SRCounter        baseline: stochastic rounding of x~ onto the 2^-b grid (fresh uniform threshold every step)
    RotationCounter  rotation rounding: threshold u = frac(u0 + t * phi), u0 regenerated from a fixed seed (no memory)
    SRQRR            rotation rounding + a stateless refit of the group scales theta from the same batch
See WRITEUP.md for the derivation (Definition 3 lists the srqrr step).
"""
import math
import torch
import torch.nn.functional as F
from torch.func import functional_call

EMB, HEAD = 'gpt_neox.embed_in.weight', 'embed_out.weight'   # Pythia (GPT-NeoX) embedding and output head
GROUP, BITS = 32, 4
K = 2 ** BITS


def schedule(t, steps, floor=.2):
    """lr multiplier: cosine half-wave from 1 down to `floor`."""
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * min(t, steps - 1) / max(1, steps - 1)))


# ----------------------------------------------------------------------------- format
def expand(x, C):
    """(R, ng) -> (R, C) by repeating each group value over its columns."""
    R, ng = x.shape
    return x[:, :, None].expand(R, ng, C // ng).reshape(R, C)


def decode(c, L):
    """codes (R, C) and level table (R, ng, K) -> deployed weight (R, C)."""
    R, C = c.shape
    ng = L.shape[1]
    return L.gather(2, c.reshape(R, ng, C // ng).long()).reshape(R, C)


def nearest(w, L):
    """Nearest-level codes of w under the level table L."""
    R, C = w.shape
    ng = L.shape[1]
    mids = (L[..., 1:] + L[..., :-1]) / 2
    return (w.reshape(R, ng, C // ng, 1) >= mids[:, :, None, :]).sum(-1).reshape(R, C).to(torch.uint8)


def group_levels(s):
    """Group scales s (R, ng) -> level table s * (k - 7.5)."""
    return s[..., None] * (torch.arange(K, device=s.device, dtype=s.dtype) - (K - 1) / 2)


def group_initial_scale(w):
    """Uniform 4-bit grid fit per group: start at 0.335 * rms (Gaussian-optimal step), then 12 Lloyd steps."""
    half = (K - 1) / 2
    x = w.reshape(w.shape[0], -1, GROUP)
    s = x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6) * .335
    for _ in range(12):
        q = torch.floor(x / s + half + .5).clamp(0, K - 1) - half
        s = ((x * q).sum(-1, keepdim=True) / q.square().sum(-1, keepdim=True)).clamp(min=1e-6)
    return s[..., 0]


class GroupGrid(torch.autograd.Function):
    """QAT fake-quantizer used only to build the 4-bit starting point: nearest level, clipped STE, LSQ scale grad."""
    @staticmethod
    def forward(ctx, w, s):
        half = (K - 1) / 2
        shape = w.shape
        u = w.reshape(shape[0], -1, GROUP) / s
        code = torch.floor(u + half + .5).clamp(0, K - 1) - half
        ctx.save_for_backward(u, code); ctx.half = half
        return (code * s).reshape(shape)

    @staticmethod
    def backward(ctx, g):
        u, code = ctx.saved_tensors
        gg = g.reshape_as(u)
        inside = (u >= -ctx.half) & (u <= ctx.half)
        ds = (gg * torch.where(inside, code - u, code)).sum(-1, keepdim=True) / math.sqrt(GROUP * ctx.half)
        return (gg * inside).reshape_as(g), ds


# ----------------------------------------------------------------------------- model access
def logits(model, W, x):
    """Forward pass with the 2D weights replaced by W (dict name -> tensor)."""
    return functional_call(model, W, (), dict(input_ids=x, use_cache=False, return_dict=True), strict=False).logits


def loss_fn(model, W, x, y):
    return F.cross_entropy(logits(model, W, x).flatten(0, 1), y.flatten())


class Capture:
    """Forward hooks on the modules owning each 2D weight; records their inputs and outputs while `on`."""
    def __init__(self, model, keys):
        self.inp, self.out, self.on = {}, {}, False
        mods = dict(model.named_modules())
        for k in keys:
            mods[k[:-len('.weight')]].register_forward_hook(self._hook(k))

    @staticmethod
    def of(model, keys):
        if getattr(model, '_capture', None) is None:
            model._capture = Capture(model, keys)
        return model._capture

    def clear(self):
        self.inp.clear(); self.out.clear()

    def _hook(self, k):
        def f(m, i, o):
            if self.on:
                self.inp[k] = i[0].detach() if i[0].is_floating_point() else i[0]
                self.out[k] = o
        return f


@torch.no_grad()
def evaluate(model, W, tokens, T=128, bs=32):
    """Token-weighted mean NLL over all non-overlapping T-token windows."""
    n = (len(tokens) - 1) // T
    total, count = 0., 0
    dev = next(iter(W.values())).device
    tok = tokens.to(dev)
    for i in range(0, n, bs):
        j = torch.arange(i, min(n, i + bs))
        ix = (j[:, None] * T + torch.arange(T)).to(dev)
        lo = logits(model, W, tok[ix])
        total += float(F.cross_entropy(lo.flatten(0, 1), tok[ix + 1].flatten(), reduction='sum'))
        count += ix.numel()
    return total / count


def _head_ggn(lg, af, qg):
    """Exact Gauss-Newton diagonal of the output head along each group direction, summed over tokens:
    h[r, g] = sum_n p_nr (1 - p_nr) (a_{n,g} . q_{r,g})^2, p = softmax(lg). lg (N, V), af (N, ng, gs), qg (V, ng, gs)."""
    p = lg.softmax(-1); w = p - p * p
    return torch.stack([((af[:, g] @ qg[:, g].T).square() * w).sum(0) for g in range(qg.shape[1])], 1)


_head_ggn_fused = None


def head_ggn(lg, af, qg):
    """_head_ggn fused by torch.compile (~3.6x faster on GPU); falls back to eager if compile is unavailable."""
    global _head_ggn_fused
    if _head_ggn_fused is None:
        try:
            _head_ggn_fused = torch.compile(_head_ggn, dynamic=False)
            return _head_ggn_fused(lg, af, qg)
        except Exception:
            _head_ggn_fused = _head_ggn
    return _head_ggn_fused(lg, af, qg)


def uniform_spacing(L, C):
    """Level spacing theta of each group, expanded to (R, C)."""
    return expand(L[..., 1] - L[..., 0], C).clamp_min(1e-12)


def sr_fine(x, q, gen):
    """Unbiased stochastic rounding of x onto the grid q*Z."""
    lo = torch.floor(x / q)
    return (lo + (torch.rand(x.shape, device=x.device, generator=gen) < (x / q - lo))) * q


# ----------------------------------------------------------------------------- optimizers
class SRCounter:
    """Baseline: b-bit SR master weight. x~ = c + f - lr*g/rms(g)/spacing, SR onto the 2^-b grid,
    c = nearest level, f = remainder (b = 0: f = 0, i.e. plain SR of the codes).

    Q: dict name -> dict(c=uint8 codes (R, C), L=level table (R, C/32, 16)); gen: torch.Generator for the SR draws."""
    def __init__(self, model, Q, bits, gen):
        self.model, self.gen = model, gen
        self.c = {k: q['c'].clone() for k, q in Q.items()}
        self.L = {k: q['L'].clone() for k, q in Q.items()}
        self.bits = int(bits); self.q = 2. ** -self.bits
        self.f = {k: torch.zeros(c.shape, device=c.device, dtype=torch.int8) for k, c in self.c.items()}  # f = e * q
        self.state_bytes = self.bits / 8 * sum(c.numel() for c in self.c.values())
        self.t = 0

    def deployed(self):
        return {k: decode(self.c[k], self.L[k]) for k in self.c}

    def leaves(self):
        return {k: w.requires_grad_(True) for k, w in self.deployed().items()}

    def frac(self, k):
        return self.f[k].float() * self.q

    def threshold(self, i, x):
        """Rounding threshold u ~ U[0, 1) for matrix i; SR draws it fresh every step."""
        return torch.rand(x.shape, device=x.device, generator=self.gen)

    def move_codes(self, G, lr):
        """Steps 2-3: normalized step in level units, round onto the 2^-b grid, split into code c and counter f."""
        for i, (k, g) in enumerate(G.items()):
            c = self.c[k]
            sp = uniform_spacing(self.L[k], c.shape[1])
            xn = (c.float() + self.frac(k) - lr * g / g.square().mean().sqrt().clamp_min(1e-30) / sp).clamp(0, K - 1)
            lo = torch.floor(xn / self.q)
            xq = ((lo + (self.threshold(i, xn) < (xn / self.q - lo))) * self.q).clamp(0, K - 1)
            cn = torch.floor(xq + .5).clamp(0, K - 1)
            self.c[k] = cn.to(torch.uint8)
            if self.bits:
                self.f[k] = torch.round((xq - cn) / self.q).clamp(-2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1).to(torch.int8)
        self.t += 1

    def step(self, x, y, lr):
        W = self.leaves()
        loss = loss_fn(self.model, W, x, y)
        G = dict(zip(W, torch.autograd.grad(loss, list(W.values()))))
        with torch.no_grad():
            self.move_codes(G, lr)
        return loss.detach()


class RotationCounter(SRCounter):
    """Rotation rounding: weight i's threshold at step t is u = frac(u0_i + t*phi), phi = golden-ratio conjugate.
    u0 is regenerated every step from a fixed per-matrix seed, so it is never stored: memory is SR's b bits/weight.
    Each u is marginally uniform (every step unbiased, as SR); one weight's thresholds are spread evenly over [0, 1),
    so its rounding errors cancel over time instead of adding up as a random walk (WRITEUP.md, Proposition 1)."""
    PHI = (5 ** .5 - 1) / 2

    def __init__(self, model, Q, bits, gen, qr_seed=0):
        super().__init__(model, Q, bits, gen)
        self.useed = 1000003 * (int(qr_seed) + 1)

    def threshold(self, i, x):
        gq = torch.Generator(device=x.device); gq.manual_seed(self.useed + i)
        u0 = torch.rand(x.shape, device=x.device, generator=gq)
        return torch.frac(u0 + (self.t * self.PHI) % 1.0)


class SRQRR(RotationCounter):
    """srqrr: rotation rounding plus a stateless refit of the group scales theta (step 4 of WRITEUP.md, Definition 3).

    From the same batch as the code move, per row r and group g, with M_rg = (c - 7.5) on the group's columns:
        a_b = <g_b, M_rg> per sequence b,  Ga = mean_b a_b,  V = Var_b(a_b)/B,
        sig2 = mean_r max(0, Ga^2 - V),    mu = Ga * sig2 / (sig2 + V)           (empirical-Bayes shrinkage)
        H    = curvature of the mean loss along M_rg (factored Fisher hidden, exact GGN head, occurrence Fisher embedding)
        dtheta = clip(-kappa * (lr/base_lr) * mu / (alpha * H), +-cap*|theta|)
    theta is part of the deployed weight, so the refit adds no optimizer memory. Codes move against the old scales.
    Per-sequence statistics come from one batched forward/backward (captured layer inputs and output gradients)."""
    def __init__(self, model, Q, bits, gen, base_lr, kappa=.003, alpha=3., cap=.05, qr_seed=0):
        super().__init__(model, Q, bits, gen, qr_seed)
        self.base_lr, self.kappa, self.alpha, self.rcap = base_lr, kappa, alpha, cap
        self.cap = Capture.of(model, list(Q))

    def theta(self, k):
        return self.L[k][..., -1] / ((K - 1) / 2)   # (R, ng)

    def set_theta(self, k, th):
        self.L[k] = group_levels(th.clamp_min(1e-6))

    def _stats(self, W, x, y):
        """loss, G (batch-mean gradient), and per (r, g): Ga = mean_b a_b, S2a = sum_b (a_b/B)^2,
        Ha = curvature along M_rg (hidden matrices: before the row factor), row (hidden: row factor)."""
        B, T = x.shape; N = B * T
        keys = list(W); hid = [k for k in keys if k not in (EMB, HEAD)]; cap = self.cap
        cap.clear(); cap.on = True
        out = logits(self.model, W, x)
        cap.on = False
        lb = F.cross_entropy(out.flatten(0, 1), y.flatten(), reduction='none').view(B, T).mean(1)   # per-sequence loss
        loss = lb.mean()
        outs = hid + [EMB, HEAD]
        gr = torch.autograd.grad(loss, [W[k] for k in keys] + [cap.out[k] for k in outs])
        G = dict(zip(keys, gr[:len(keys)]))
        D = {k: d * B for k, d in zip(outs, gr[len(keys):])}   # d l_b / d out for the sequence b owning each token
        Ga, S2a, Ha, row = {}, {}, {}, {}

        def qgroups(k):   # M_rg restricted to its own columns: (R, ng, gs)
            c = self.c[k]; R, C = c.shape; ng = self.L[k].shape[1]
            return (c.float() - (K - 1) / 2).view(R, ng, C // ng)

        def put(k, ab):   # ab: (B, R, ng) per-sequence group gradients a_b
            Ga[k] = ab.mean(0); S2a[k] = (ab / B).square().sum(0)

        with torch.no_grad():
            # embedding: rows are vocabulary entries, gradients arrive through the token occurrences
            qg = qgroups(EMB); R, ng, gs = qg.shape; ids = x.reshape(-1)
            z = (D[EMB].reshape(N, ng, gs) * qg[ids]).sum(-1)              # (N, ng): d_t . M_{x_t, g}
            ab = torch.zeros(B * R, ng, device=z.device, dtype=z.dtype)
            ab.index_add_(0, (torch.arange(B, device=z.device)[:, None] * R + x).reshape(-1), z)
            put(EMB, ab.view(B, R, ng))
            Ha[EMB] = torch.zeros(R, ng, device=z.device, dtype=z.dtype).index_add_(0, ids, z.square_().mul_(T / B))
            # head: exact GGN diagonal along M_rg
            qg = qgroups(HEAD); R, ng, gs = qg.shape; a = cap.inp[HEAD]
            put(HEAD, (torch.einsum('btr,btc->brc', D[HEAD], a).view(B, R, ng, gs) * qg).sum(-1))
            Ha[HEAD] = head_ggn(out.detach().flatten(0, 1), a.reshape(N, ng, gs), qg) / N
            # hidden matrices, stacked across layers by shape
            by_shape = {}
            for k in hid:
                by_shape.setdefault(tuple(W[k].shape), []).append(k)
            for ks in by_shape.values():
                qg = torch.stack([qgroups(k) for k in ks]); Lk, R, ng, gs = qg.shape
                a = torch.stack([cap.inp[k] for k in ks]).reshape(Lk, B, T, ng * gs)
                d = torch.stack([D[k] for k in ks])                                # (Lk, B, T, R)
                ab = (torch.einsum('lbtr,lbtc->lbrc', d, a).view(Lk, B, R, ng, gs) * qg[:, None]).sum(-1)
                hz = torch.einsum('lnpc,lrpc->lnpr', a.view(Lk, N, ng, gs), qg).square_().sum(1).transpose(1, 2).div_(N)
                rw = d.square().sum((1, 2)).mul_(T / B)
                for i, k in enumerate(ks):
                    put(k, ab[i]); Ha[k] = hz[i]; row[k] = rw[i]
        cap.clear()
        return loss.detach(), G, Ga, S2a, Ha, row

    def step(self, x, y, lr):
        B = x.shape[0]
        W = self.leaves()
        loss, G, Ga, S2a, Ha, row = self._stats(W, x, y)
        with torch.no_grad():
            self.move_codes(G, lr)                         # steps 2-3: rotation-rounded code move
            sched = lr / self.base_lr
            for k in self.c:                               # step 4: level refit
                H = Ha[k] * row[k][:, None] if k in row else Ha[k]
                Va = ((B * S2a[k] - Ga[k] ** 2) / (B - 1)).clamp_min(0)
                sig2 = (Ga[k] ** 2 - Va).clamp_min(0).mean(0, keepdim=True)
                mu = Ga[k] * sig2 / (sig2 + Va).clamp_min(1e-30)
                th = self.theta(k)
                dth = -self.kappa * sched * mu / (self.alpha * H).clamp_min(1e-30)
                lim = self.rcap * th.abs()
                self.set_theta(k, th + torch.maximum(torch.minimum(dth, lim), -lim))
        return loss


def make_optimizer(name, model, Q, bits, gen, base_lr):
    """name: 'sr' (baseline), 'rotation' (rotation rounding only) or 'srqrr' (rotation + level refit)."""
    if name == 'sr':
        return SRCounter(model, Q, bits, gen)
    if name == 'rotation':
        return RotationCounter(model, Q, bits, gen)
    if name == 'srqrr':
        return SRQRR(model, Q, bits, gen, base_lr)
    raise ValueError(name)
