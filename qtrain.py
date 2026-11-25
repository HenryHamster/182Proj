# Harness: EthelynHuang/flip-optimizer @ ea4471c, src/qtrain.py, plus a speed patch that removes per-matrix host syncs
# (.item(), host-side top-k count, boolean-mask index). Losses, codes and momentum are identical to the original
# for 60 CPU steps (docs/RECORD_flip-optimizer-scale-001.md, "Speed fix").
"""
qtrain.py -- train a Transformer language model FROM SCRATCH with 2-bit weights, comparing the flip optimizer
(no shadow weights, 4-bit momentum) against QAT and other baselines. GPU-ready version of the CS182 experiments
(Exp 04b-08), for running at larger scale.

Model: GPT-style decoder (pre-LN, GELU MLP, learned positions, untied output head). Every weight matrix inside the
blocks (attn qkv, attn out, mlp in, mlp out) is quantized: W = alpha * K, K on the 2-bit SEQ grid
{-1.5, -0.5, 0.5, 1.5}, alpha = mean|W0| per matrix, fixed at init. Token/position embeddings, LayerNorms and the
output head stay full precision and are trained with AdamW (identical for every method).

All matrix steps are in grid units: lr = 1 moves the RMS entry by one codebook level.

Methods (append "@m8", "@m4", "@m2", "@m2g" to quantize the momentum of Muon-family / flip methods):
  fp_muon     Muon, no quantization                         (reference)
  fp_adam     Adam, no quantization                         (reference)
  muon_qat    Muon on an FP32 shadow copy, forward at round(shadow)   (STE QAT)        66 bits/weight
  adam_qat    Adam on an FP32 shadow copy                                               98 bits/weight
  muon_ef4    Muon + 4-bit residual buffer with threshold alignment (Bonin-style)       38 bits (10.2 @m4)
  dqt         Adam step + stochastic rounding, no shadow (DQT)                          66 bits/weight
  muon_sr     Muon step + stochastic rounding, no shadow                                34 bits (6.2 @m4)
  flip        spectral flip optimizer: flip the top-k |momentum| entries by one level, with per-row and
              per-column caps (Schur bound on ||Z||_2, slack 2), momentum reset on flips. lr = flip budget rho.
              Use flip@m4 (6.2 bits/weight) as the main method.
  flip_s1     same, slack 1 (tighter spreading; its best rho is ~3x larger)
  flip_frob   exact Frobenius solution: plain top-k |momentum| (no caps)                [geometry control]
  flip_rand   random feasible entries with the same caps, moved along -sign(momentum)  [mechanism control]

Differences from the CPU experiments: Muon's msign uses 5 Newton-Schulz iterations instead of an exact SVD; AMP
(bf16, or fp16 with loss scaling); the FP AdamW parameters follow the same warmup/WSD multiplier as the matrices.

Reproducibility: a run is deterministic on CPU with one thread, and resuming from a checkpoint is then exact. On
GPUs (and multi-threaded CPU) floating-point reductions are not bit-reproducible, and rounding to a 2-bit grid
amplifies tiny differences, so repeated or resumed runs differ slightly -- this is seed-level noise, not a bug.

Usage (single run):   python qtrain.py --preset 25M --method flip@m4 --lr 0.003 --seed 0 --data_dir data/wt103_bytes
Usage (sweep):        see the notebook, or python qtrain.py --sweep --preset 25M --data_dir ...
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------------ quantization
GRIDS = {"2": (-1.5, 1.5, 0.5), "1.58": (-1.0, 1.0, 0.0), "4": (-8.0, 7.0, 0.0)}
MOM_QMAX = {"8": 127, "4": 7, "1.58": 1}

BASE = {
    "fp_muon":   dict(dir="muon", rnd="none"),
    "fp_adam":   dict(dir="adam", rnd="none"),
    "muon_qat":  dict(dir="muon", rnd="shadow"),
    "adam_qat":  dict(dir="adam", rnd="shadow"),
    "muon_ef4":  dict(dir="muon", rnd="ef4"),
    "dqt":       dict(dir="adam", rnd="sr"),
    "muon_sr":   dict(dir="muon", rnd="sr"),
    "flip":      dict(dir="flip", sel="schur", slack=2.0),
    "flip_s1":   dict(dir="flip", sel="schur", slack=1.0),
    "flip_frob": dict(dir="flip", sel="frob"),
    "flip_rand": dict(dir="flip", sel="rschur", slack=2.0),
}


def parse_method(method):
    base, mom = (method.split("@m") + [None])[:2]
    c = dict(BASE[base])
    if mom is not None:
        assert c["dir"] != "adam", "Adam-state quantization is not implemented"
        c["mom_bits"] = mom
    return c


def state_bits_per_weight(method, n_in, bits="2"):
    """Training-state bits per quantized weight: weights + shadow/buffer + optimizer state (+ momentum scales)."""
    c = parse_method(method)
    w = 32.0 if c.get("rnd") == "none" else (1.58 if bits == "1.58" else float(bits))
    if c["dir"] == "adam":
        opt = 64.0
    else:
        mb = c.get("mom_bits")
        if mb is None:
            opt = 32.0
        elif mb.endswith("g"):
            opt = float(mb[:-1]) + 0.25            # FP16 scale per 64 entries
        else:
            opt = float(mb) + 32.0 / n_in          # FP32 scale per row
    extra = {"shadow": 32.0, "ef4": 4.0}.get(c.get("rnd"), 0.0)
    return w + opt + extra


def quantize_state(X, bits, gen):
    """Momentum quantizer: absmax scale per row (or per 64 entries with suffix 'g') + unbiased stochastic rounding.
    Grids: 8 {-127..127}, 4 {-7..7}, 1.58 {-1,0,1}, 2 {-1.5,-0.5,0.5,1.5}. Returns the dequantized tensor."""
    if bits is None:
        return X
    group = bits.endswith("g")
    b = bits[:-1] if group else bits
    qmax, off = (1.5, 0.5) if b == "2" else (MOM_QMAX[b], 0.0)
    m, n = X.shape
    Y = X.reshape(m, n // 64, 64) if group else X
    s = Y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / qmax
    q = torch.floor(Y / s - off + torch.rand(Y.shape, generator=gen, device=X.device)) + off
    return (torch.clamp(q, -qmax, qmax) * s).reshape(m, n)


def msign(G, steps=5):
    """U V^T via quintic Newton-Schulz (standard Muon coefficients)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    use_bf16 = G.is_cuda and torch.cuda.is_bf16_supported()
    X = G.bfloat16() if use_bf16 else G.float()
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tr:
        X = X.T
    return X.float()


def unit_rms(D):
    return D / (D.pow(2).mean().sqrt() + 1e-12)


class QMat:
    """One quantized weight matrix (out x in) plus its optimizer. self.w is the FP32 leaf used in the forward
    pass; it always equals alpha * K (the dequantized codes), so no information is stored in it."""

    def __init__(self, W0, method, gen, bits="2"):
        c = parse_method(method)
        self.c, self.method, self.gen = c, method, gen
        self.dir, self.rnd, self.sel = c["dir"], c.get("rnd"), c.get("sel")
        self.slack, self.mom_bits = c.get("slack", 2.0), c.get("mom_bits")
        self.qmin, self.qmax, self.off = GRIDS[bits]
        self.quant = self.rnd != "none"
        self.alpha = W0.abs().mean().item() if bits in ("2", "1.58") else W0.abs().max().item() / self.qmax
        X = W0.float() / self.alpha
        self.K = X.clone() if not self.quant else self._clip(self._round(X))
        self.e = torch.zeros_like(X) if self.rnd in ("shadow", "ef4") else None
        self.M = torch.zeros_like(X)
        self.V = torch.zeros_like(X) if self.dir == "adam" else None
        self.w = (self.alpha * self.K).requires_grad_(True)
        self.t, self.chg_sum, self.chg_n = 0, 0.0, 0

    # --- state for checkpointing
    def state(self):
        return {k: getattr(self, k) for k in ("K", "e", "M", "V", "t", "chg_sum", "chg_n")}

    def load(self, s):
        for k, v in s.items():
            setattr(self, k, v)
        with torch.no_grad():
            self.w.copy_(self.alpha * self.K)

    def _round(self, x):
        return torch.round(x - self.off) + self.off

    def _clip(self, x):
        return x.clamp(self.qmin, self.qmax)

    def _sr(self, x, off=None):
        o = self.off if off is None else off
        return torch.floor(x - o + torch.rand(x.shape, generator=self.gen, device=x.device)) + o

    @torch.no_grad()
    def step(self, lr):
        G = self.w.grad.float() * self.alpha          # gradient w.r.t. the codes
        self.w.grad = None
        self.t += 1
        if self.dir == "adam":
            b1, b2, eps = 0.9, 0.999, 1e-8 * self.alpha
            self.M.mul_(b1).add_(G, alpha=1 - b1)
            self.V.mul_(b2).addcmul_(G, G, value=1 - b2)
            D = (self.M / (1 - b1 ** self.t)) / ((self.V / (1 - b2 ** self.t)).sqrt() + eps)
        else:
            self.M.mul_(0.95).add_(G)
            if self.mom_bits:
                self.M = quantize_state(self.M, self.mom_bits, self.gen)
            Dn = G + 0.95 * self.M                       # Nesterov
            if self.dir == "flip":
                self._flip(Dn, lr)
                self.w.copy_(self.alpha * self.K)
                return
            D = unit_rms(msign(Dn))
        S = -lr * D
        if self.rnd == "none":
            self.K.add_(S)
        else:
            Kold = self.K
            if self.rnd == "shadow":
                L = (self.K + self.e + S).clamp(self.qmin - 0.5, self.qmax + 0.5)
                Kn = self._clip(self._round(L))
                self.e = L - Kn
            elif self.rnd == "sr":
                Kn = self._clip(self._sr(self.K + S))
            elif self.rnd == "ef4":   # 4-bit residual buffer {-8..7}/16, flip threshold aligned at +-0.5
                t = self._sr((self.e + S) * 16, off=0.0) / 16
                n = torch.floor(t + 0.5)
                Kn = self._clip(self.K + n)
                self.e = torch.clamp(t - (Kn - self.K), -0.5, 7 / 16)
            else:
                raise ValueError(self.rnd)
            self.chg_sum = self.chg_sum + (Kn != Kold).float().mean()
            self.chg_n += 1
            self.K = Kn
        self.w.copy_(self.alpha * self.K)

    def _flip(self, D, rho):
        m, n = D.shape
        k = int(round(rho * m * n))
        self.chg_n += 1
        if k == 0:
            return
        direction = -torch.sign(D)
        Kn = self.K + direction
        feas = (Kn >= self.qmin) & (Kn <= self.qmax) & (D != 0)
        mag = torch.where(feas, D.abs(), torch.zeros_like(D))
        if self.sel == "rschur":
            mag = torch.where(mag > 0, torch.rand(mag.shape, generator=self.gen, device=mag.device) + 1e-6,
                              torch.zeros_like(mag))
        if self.sel in ("schur", "rschur"):
            # per-row cap a and per-column cap b with a*b = slack^2 (k/m)(k/n)  =>  ||Z||_2 <= slack * k / sqrt(mn)
            a = max(1, min(n, math.ceil(self.slack * k / m)))
            b = max(1, min(m, math.ceil(self.slack * k / n)))
            rt = torch.topk(mag, a, dim=1).values[:, -1:]
            ct = torch.topk(mag, b, dim=0).values[-1:, :]
            mag = torch.where((mag >= rt) & (mag >= ct) & (mag > 0), mag, torch.zeros_like(mag))
        flat = mag.flatten()
        v, idx = torch.topk(flat, min(k, flat.numel()))
        Z = torch.zeros_like(flat)
        Z[idx] = direction.flatten()[idx] * (v > 0)
        Z = Z.view(m, n)
        self.K.add_(Z)
        self.M.masked_fill_(Z != 0, 0.0)                  # momentum reset on flipped entries
        self.chg_sum = self.chg_sum + (v > 0).sum() / (m * n)

    def on_grid(self):
        if not self.quant:
            return True
        return bool(torch.all(self.K == self._round(self.K)) and self.K.min() >= self.qmin and self.K.max() <= self.qmax)


# ------------------------------------------------------------------------------------------------------- model
PRESETS = {   # quantized weights = 12 * d^2 * layers
    "tiny":  dict(d=64,  layers=2,  heads=4,  ctx=64,  batch=16, micro=16, steps=200,   fp_lr=2e-3),   # CPU smoke test
    "pilot": dict(d=192, layers=6,  heads=6,  ctx=128, batch=32, micro=32, steps=2500,  fp_lr=2e-3),   # = Exp 07 (2.65M)
    "10M":   dict(d=384, layers=6,  heads=6,  ctx=256, batch=64, micro=64, steps=8000,  fp_lr=1e-3),   # 10.6M
    "25M":   dict(d=512, layers=8,  heads=8,  ctx=512, batch=64, micro=32, steps=15000, fp_lr=1e-3),   # 25.2M
    "50M":   dict(d=640, layers=10, heads=10, ctx=512, batch=64, micro=64, steps=30000, fp_lr=1e-3),   # 49.2M (added; ~20 bytes/weight)
    "100M":  dict(d=768, layers=14, heads=12, ctx=512, batch=64, micro=64, steps=60000, fp_lr=1e-3),   # 99.1M (flip-vs-srqrr-004; ~20 bytes/weight as 50M)
    "85M":   dict(d=768, layers=12, heads=12, ctx=512, batch=128, micro=32, steps=30000, fp_lr=6e-4),  # 84.9M
}


class QGPT:
    def __init__(self, V, d, layers, heads, ctx, method, gen, device):
        self.V, self.d, self.L, self.H, self.ctx = V, d, layers, heads, ctx
        P = {"tok": torch.randn(V, d) * 0.02, "pos": torch.randn(ctx, d) * 0.02,
             "lnf_w": torch.ones(d), "lnf_b": torch.zeros(d), "head": torch.randn(V, d) / math.sqrt(d)}
        self.Q = {}
        for l in range(layers):
            P[f"ln1w{l}"], P[f"ln1b{l}"] = torch.ones(d), torch.zeros(d)
            P[f"ln2w{l}"], P[f"ln2b{l}"] = torch.ones(d), torch.zeros(d)
            for name, m, n, scale in ((f"qkv{l}", 3 * d, d, 1.0), (f"o{l}", d, d, 1 / math.sqrt(2 * layers)),
                                      (f"fc{l}", 4 * d, d, 1.0), (f"pr{l}", d, 4 * d, 1 / math.sqrt(2 * layers))):
                W0 = torch.randn(m, n) * (scale / math.sqrt(n))
                self.Q[name] = QMat(W0.to(device), method, gen)
        self.P = {k: v.to(device).requires_grad_(True) for k, v in P.items()}

    def fp_params(self):
        return list(self.P.values())

    def forward(self, idx):
        P, Q, d, H = self.P, self.Q, self.d, self.H
        B, T = idx.shape
        x = P["tok"][idx] + P["pos"][:T]
        for l in range(self.L):
            h = F.layer_norm(x, (d,), P[f"ln1w{l}"], P[f"ln1b{l}"])
            q, k, v = F.linear(h, Q[f"qkv{l}"].w).view(B, T, 3, H, d // H).permute(2, 0, 3, 1, 4)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + F.linear(y.transpose(1, 2).reshape(B, T, d), Q[f"o{l}"].w)
            h = F.layer_norm(x, (d,), P[f"ln2w{l}"], P[f"ln2b{l}"])
            x = x + F.linear(F.gelu(F.linear(h, Q[f"fc{l}"].w)), Q[f"pr{l}"].w)
        x = F.layer_norm(x, (d,), P["lnf_w"], P["lnf_b"])
        return F.linear(x, P["head"])


# -------------------------------------------------------------------------------------------------------- data
def load_data(data_dir):
    """data_dir holds train.bin / val.bin / test.bin (uint8 for bytes, uint16 for GPT-2 tokens) and meta.json."""
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    dt = np.uint8 if meta["vocab"] <= 256 else np.uint16
    data = {s: np.memmap(os.path.join(data_dir, f"{s}.bin"), dtype=dt, mode="r") for s in ("train", "val", "test")}
    return data, meta


def write_split(data_dir, name, text, tokenizer=None):
    if tokenizer is None:
        arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    else:
        ids = []
        step = 1 << 20
        for i in range(0, len(text), step):        # 1M-character chunks
            ids.extend(tokenizer(text[i:i + step])["input_ids"])
        arr = np.array(ids, dtype=np.uint16)
    arr.tofile(os.path.join(data_dir, f"{name}.bin"))
    return dict(tokens=int(len(arr)), bytes=len(text.encode("utf-8")))


def prepare_from_texts(data_dir, texts, tokenizer_name=None):
    """texts: dict train/val/test -> str. Byte-level if tokenizer_name is None, else a Hugging Face tokenizer."""
    os.makedirs(data_dir, exist_ok=True)
    tok = None
    if tokenizer_name:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        tok.model_max_length = 10 ** 12
    meta = {"vocab": 256 if tok is None else len(tok), "tokenizer": tokenizer_name or "bytes"}
    for s in ("val", "test", "train"):
        meta[s] = write_split(data_dir, s, texts[s], tok)
        print(f"{s}: {meta[s]['tokens']:,} tokens, {meta[s]['bytes']:,} bytes")
    json.dump(meta, open(os.path.join(data_dir, "meta.json"), "w"))
    return meta


def get_batch(data, batch, ctx, rng, device):
    ix = rng.integers(0, len(data) - ctx - 1, size=batch)
    x = torch.from_numpy(np.stack([data[i:i + ctx] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + ctx + 1] for i in ix]).astype(np.int64))
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def amp_dtype(device):
    if not device.startswith("cuda"):
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def evaluate(model, data, ctx, device, bs=32, max_windows=None, starts=None):
    """Mean NLL in nats/token over non-overlapping windows (or the given window starts)."""
    if starts is None:
        nwin = (len(data) - 1) // ctx
        if max_windows:
            nwin = min(nwin, max_windows)
        starts = np.arange(nwin) * ctx
    dt = amp_dtype(device)
    tot, cnt = 0.0, 0
    for i in range(0, len(starts), bs):
        s = starts[i:i + bs]
        x = torch.from_numpy(np.stack([data[j:j + ctx] for j in s]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([data[j + 1:j + ctx + 1] for j in s]).astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
            logits = model.forward(x)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum").item()
        cnt += y.numel()
    return tot / cnt


def to_bpb(nats_per_token, m):
    return nats_per_token * m["tokens"] / m["bytes"] / math.log(2)


# --------------------------------------------------------------------------------------------------------- run
def run(method, lr, seed, data, meta, preset="pilot", device="cuda", warmup_frac=0.0, decay_frac=0.3, log_every=100,
        eval_every=0, eval_windows=None, mid_eval_windows=100, ckpt_dir=None, ckpt_every=1000, stop_after=None,
        log=print, **overrides):
    cfg = dict(PRESETS[preset], **overrides)
    d, layers, heads, ctx, batch, micro, steps, fp_lr = (cfg[k] for k in ("d", "layers", "heads", "ctx", "batch", "micro", "steps", "fp_lr"))
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 12345)
    rng = np.random.default_rng(seed)
    model = QGPT(meta["vocab"], d, layers, heads, ctx, method, gen, device)
    opt = torch.optim.AdamW(model.fp_params(), lr=fp_lr, betas=(0.9, 0.95), weight_decay=0.0)
    dt = amp_dtype(device)
    scale = 2.0 ** 16 if dt == torch.float16 else 1.0
    accum = max(1, batch // micro)
    warm = int(warmup_frac * steps)
    decay_start = int((1 - decay_frac) * steps)
    leaves = model.fp_params() + [q.w for q in model.Q.values()]
    key = f"{preset}_{method}_lr{lr:g}_s{seed}".replace("@", "-").replace(":", "-")
    ck = os.path.join(ckpt_dir, key + ".pt") if ckpt_dir else None
    step, curve, val_curve, losses, diverged, skipped, good, elapsed0 = 0, [], [], [], False, 0, 0, 0.0
    if ck and os.path.exists(ck):                       # resume
        s = torch.load(ck, map_location=device, weights_only=False)
        with torch.no_grad():
            for k, v in s["P"].items():
                model.P[k].copy_(v)
        opt.load_state_dict(s["opt"])
        for k, q in model.Q.items():
            q.load(s["Q"][k])
        gen.set_state(s["gen"].cpu())
        rng.bit_generator.state = s["rng"]
        step, curve, val_curve, losses, scale, good, skipped, elapsed0 = (s[k] for k in ("step", "curve", "val_curve", "losses", "scale", "good", "skipped", "elapsed"))
        log(f"  resumed {key} at step {step}")
    t0 = time.time() - elapsed0
    while step < steps:
        tot = 0.0
        for _ in range(accum):
            xb, yb = get_batch(data["train"], batch // accum, ctx, rng, device)
            with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
                logits = model.forward(xb)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), yb.view(-1)) / accum
            (loss * scale).backward()
            tot += loss.item()
        if not math.isfinite(tot):
            diverged = True
            break
        if scale != 1.0:                                 # fp16: skip steps with overflowing gradients
            if not all(bool(torch.isfinite(p.grad).all()) for p in leaves if p.grad is not None):
                for p in leaves:
                    p.grad = None
                scale /= 2.0
                skipped += 1
                continue
            for p in leaves:
                if p.grad is not None:
                    p.grad.div_(scale)
            good += 1
            if good % 1000 == 0:
                scale *= 2.0
        mult = (step + 1) / warm if step < warm else (1.0 if step < decay_start else (steps - step) / (steps - decay_start))
        for q in model.Q.values():
            q.step(lr * mult)
        for g in opt.param_groups:
            g["lr"] = fp_lr * mult
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(tot)
        step += 1
        if step % log_every == 0:
            curve.append((step, sum(losses[-log_every:]) / log_every))
            el = time.time() - t0
            log(f"  [{key}] step {step}/{steps} train {curve[-1][1]:.4f} nats  {el / step:.3f} s/step  "
                f"ETA {el / step * (steps - step) / 60:.1f} min")
        if eval_every and step % eval_every == 0:
            val_curve.append((step, evaluate(model, data["val"], ctx, device, max_windows=mid_eval_windows)))
        if ck and (step % ckpt_every == 0 or (stop_after and step >= stop_after)):
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(dict(P={k: v.detach() for k, v in model.P.items()}, opt=opt.state_dict(),
                            Q={k: q.state() for k, q in model.Q.items()}, gen=gen.get_state(), rng=rng.bit_generator.state,
                            step=step, curve=curve, val_curve=val_curve, losses=losses[-log_every:], scale=scale,
                            good=good, skipped=skipped, elapsed=time.time() - t0), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
        if stop_after and step >= stop_after:
            return None                                  # interrupted on purpose (used to test resuming)
    train_secs = time.time() - t0
    if diverged:
        val = test = tr = float("inf")
    else:
        val = evaluate(model, data["val"], ctx, device, max_windows=eval_windows)
        test = evaluate(model, data["test"], ctx, device, max_windows=eval_windows)
        starts = np.random.default_rng(555).integers(0, len(data["train"]) - ctx - 1, size=200)
        tr = evaluate(model, data["train"], ctx, device, starts=starts)
    qs = list(model.Q.values())
    nq = sum(q.K.numel() for q in qs)
    chg = [float(q.chg_sum) / q.chg_n for q in qs if q.chg_n]
    if ck and os.path.exists(ck):
        os.remove(ck)
    return dict(key=key, preset=preset, method=method, lr=lr, seed=seed, **{k: cfg[k] for k in ("d", "layers", "heads", "ctx", "batch", "steps", "fp_lr")},
                vocab=meta["vocab"], tokenizer=meta.get("tokenizer"), val_nats=val, test_nats=test, train_eval_nats=tr,
                val_bpb=to_bpb(val, meta["val"]), test_bpb=to_bpb(test, meta["test"]), train_eval_bpb=to_bpb(tr, meta["train"]),
                diverged=diverged, fp16_skipped=skipped, on_grid=all(q.on_grid() for q in qs),
                changed_frac_mean=sum(chg) / len(chg) if chg else None,
                state_bits=sum(state_bits_per_weight(method, q.K.shape[1]) * q.K.numel() for q in qs) / nq,
                n_quant_weights=nq, tokens_seen=steps * batch * ctx, curve=curve, val_curve=val_curve,
                train_seconds=train_secs, device=torch.cuda.get_device_name() if device.startswith("cuda") else "cpu")


# ------------------------------------------------------------------------------------------------------- sweep
DEFAULT_GRID = {            # learning rates in grid units (flip methods: flip budget rho); centred on the CPU optimum
    "muon_qat":   [0.003, 0.01, 0.03],
    "adam_qat":   [0.01, 0.03, 0.09],
    "flip@m4":    [0.001, 0.003, 0.009],
    "dqt":        [0.01, 0.03, 0.09],
    "fp_muon":    [0.01],
}


def read_results(out):
    return [json.loads(l) for l in open(out)] if os.path.exists(out) else []


def sweep(grid, data, meta, out, preset, seeds=(1, 2), max_edge_ext=2, seed_methods=None, **kw):
    """Phase 1: seed 0 over each method's grid. Edge: if the best (by val) is at a grid edge, extend by x3 / /3
    (up to max_edge_ext times). Phase 2: extra seeds at the best lr. Resumable: finished runs are skipped."""
    def done():
        return {(r["method"], r["lr"], r["seed"]) for r in read_results(out) if r["preset"] == preset}

    def do(method, lr, seed):
        if (method, lr, seed) in done():
            return
        r = run(method, lr, seed, data, meta, preset=preset, **kw)
        if r is not None:
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")

    def best(method):
        rs = [r for r in read_results(out) if r["preset"] == preset and r["method"] == method and r["seed"] == 0]
        return min(rs, key=lambda r: r["val_nats"]), sorted(r["lr"] for r in rs)

    for m, lrs in grid.items():
        for lr in lrs:
            do(m, lr, 0)
        for _ in range(max_edge_ext):
            b, lrs_done = best(m)
            if len(lrs_done) < 2 or lrs_done[0] < b["lr"] < lrs_done[-1]:
                break
            do(m, round(b["lr"] * 3, 6) if b["lr"] == lrs_done[-1] else round(b["lr"] / 3, 6), 0)
    for m in (seed_methods or grid):
        b, _ = best(m)
        for s in seeds:
            do(m, b["lr"], s)


def summarize(out, preset=None):
    """Markdown table: per method, lr chosen on val (seed 0), test bpb mean +- std over seeds at that lr."""
    import statistics as st
    rows = [r for r in read_results(out) if preset is None or r["preset"] == preset]
    lines = ["| Preset | Method | Bits/weight | LR | Seeds | Test bpb | Test − train bpb | Code change rate/step | s/run |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted({r["preset"] for r in rows}):
        for m in sorted({r["method"] for r in rows if r["preset"] == p}):
            rs0 = [r for r in rows if r["preset"] == p and r["method"] == m and r["seed"] == 0]
            if not rs0:
                continue
            b = min(rs0, key=lambda r: r["val_nats"])
            at = [r for r in rows if r["preset"] == p and r["method"] == m and r["lr"] == b["lr"]]
            t = [r["test_bpb"] for r in at]
            sd = st.stdev(t) if len(t) > 1 else float("nan")
            gap = st.mean(r["test_bpb"] - r["train_eval_bpb"] for r in at)
            chg = b["changed_frac_mean"]
            lines.append(f"| {p} | {m} | {b['state_bits']:.1f} | {b['lr']:g} | {len(t)} | {st.mean(t):.4f} ± {sd:.4f} | "
                         f"{gap:+.3f} | {('%.3f%%' % (100 * chg)) if chg is not None else '—'} | {st.mean(r['train_seconds'] for r in at):.0f} |")
    return "\n".join(lines)


def plot_curves(out, preset, path=None):
    import matplotlib.pyplot as plt
    rows = [r for r in read_results(out) if r["preset"] == preset]
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in sorted({r["method"] for r in rows}):
        rs0 = [r for r in rows if r["method"] == m and r["seed"] == 0]
        b = min(rs0, key=lambda r: r["val_nats"])
        xs, ys = zip(*b["curve"])
        f = rows[0]["test_bpb"] / rows[0]["test_nats"]          # nats/token -> bits/byte factor
        ax.plot(xs, [y * f for y in ys], label=f"{m} (lr {b['lr']:g}, {b['state_bits']:.1f} b)")
    ax.set_xlabel("step")
    ax.set_ylabel("training loss (bits per byte)")
    ax.set_title(f"Training loss, preset {preset}, best lr per method (seed 0)")
    ax.grid(alpha=0.3)
    ax.legend()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--preset", default="pilot", choices=list(PRESETS))
    ap.add_argument("--method", default="flip@m4")
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--ckpt_dir", default="ckpt")
    ap.add_argument("--sweep", action="store_true", help="run DEFAULT_GRID with edge extension + seeds 1,2")
    a = ap.parse_args()
    data, meta = load_data(a.data_dir)
    if a.sweep:
        sweep(DEFAULT_GRID, data, meta, a.out, a.preset, device=a.device, ckpt_dir=a.ckpt_dir)
        print(summarize(a.out, a.preset))
    else:
        r = run(a.method, a.lr, a.seed, data, meta, preset=a.preset, device=a.device, ckpt_dir=a.ckpt_dir)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print({k: v for k, v in r.items() if k not in ("curve", "val_curve")})
