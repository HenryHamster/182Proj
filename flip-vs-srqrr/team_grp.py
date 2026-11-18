"""GPEF below 4 bits/weight (Henry 2026-11-10 'add it'): gpef@e0 with larger codebook groups.

Import after team_ext (and team_rot); registers, without changing existing methods:
  gpef@e0gG   GPEF@e0 (no error feedback) with one learned codebook {-a,-b,b,a} per G contiguous weights of the row-major
              flattened matrix, G in {32, 64, 128, 256}. With G=32 this is gpef@e0 exactly (team_grp_test.py checks).
              State 2 + 64/G bits/weight: G=64 -> 3, 128 -> 2.5, 256 -> 2.25.
Groups are contiguous blocks of the flattened matrix, as in team_ext (reshape(-1, G)); team_ext asserts n_in % 32 == 0,
here only m*n % G == 0 is required, so at d=192/384 a 128- or 256-group can span two rows. Same codebook storage
(fp32 a, b), same step, same two fit/nearest passes, same bounds.
"""
import re
import torch
import qtrain as Q
import team_ext as T

_PAT = re.compile(r"^gpef@e0g(32|64|128|256)$")


class _Group:
    """Temporarily set team_ext.GROUP (read at call time by GroupMat.__init__ and GPEF.step)."""
    def __init__(self, G): self.G = G
    def __enter__(self): self.old, T.GROUP = T.GROUP, self.G
    def __exit__(self, *a): T.GROUP = self.old


class GPEFG(T.GPEF):
    def __init__(self, W0, method, gen, bits="2"):
        G = int(_PAT.match(method).group(1))
        assert W0.numel() % G == 0, (tuple(W0.shape), G)
        with _Group(32):                       # build as gpef@e0 (all groups start at a=1.5 alpha, b=.5 alpha), regroup
            super().__init__(W0, "gpef@e0", gen, bits)
        self.method, self.G = method, G
        ng = self.k.numel() // G
        self.k = self.k.reshape(ng, G)
        self.a = torch.full((ng,), 1.5 * self.alpha, device=W0.device)
        self.b = torch.full((ng,), 0.5 * self.alpha, device=W0.device)
        s = torch.full_like(self.a, self.span)
        self.bounds = (s / 4, torch.zeros_like(s), s * 1e-6, s * 4)
        self.err = torch.zeros(self.k.shape, dtype=torch.int8, device=W0.device)
        with torch.no_grad():
            self.w.copy_(self._deq())

    @torch.no_grad()
    def step(self, lr):
        with _Group(self.G):
            super().step(lr)


def _factory(W0, method, gen, bits="2"):
    if _PAT.match(method):
        return GPEFG(W0, method, gen, bits)
    return _prev_factory(W0, method, gen, bits)


def _state_bits(method, n_in, bits="2"):
    if _PAT.match(method):
        return 2.0 + 64.0 / int(_PAT.match(method).group(1))
    return _prev_bits(method, n_in, bits)


_prev_factory, _prev_bits = Q.QMat, Q.state_bits_per_weight
Q.QMat = _factory
Q.state_bits_per_weight = _state_bits
