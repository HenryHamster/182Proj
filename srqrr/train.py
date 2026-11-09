"""Reproduce the srqrr results on Pythia-14m (group32x4: 4-bit codes, one FP32 scale per 32 weights).

  python train.py source --seed 9702 --out runs/source-9702.pt
      4-bit QAT starting point (2000 Adam steps on source-train), saved as codes + level tables.
  python train.py run --source runs/source-9702.pt --method srqrr --bits 2 --lr 3e-4 --seed 2 --task wikitext
      600 steps of a quantized-state optimizer on the task; prints and optionally saves dev NLL at steps 300 and 600.

Methods: sr (baseline), rotation (rotation rounding only), srqrr (rotation + level refit). --bits is the optimizer
memory b per weight (0, 1, 2, 4, 8). --lr is in units of the source's median level spacing.
Tuned lrs (wikitext, seed 2): b0 1e-4, b1/b2 3e-4, b4/b8 3e-3, the same for all methods.
Data: --data points to a dict of 1-D token tensors (default data/tokens.pt next to this file)."""
import argparse, json, math, os, time
import torch
from transformers import AutoModelForCausalLM
import srqrr as S

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = {'wikitext': ('wikitext-train', 'wikitext-dev'), 'fineweb': ('fineweb-train', 'fineweb-dev')}


def load_data(path):
    d = torch.load(path, weights_only=True)
    return {k: v.long() for k, v in d.items()}


def load_model(name, device):
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, [k for k, p in m.named_parameters() if p.ndim == 2]


def windows(tokens, steps, B, T, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, len(tokens) - T - 1, (steps, B), generator=g)


def batch(tok, starts, T):
    ix = starts.to(tok.device)[:, None] + torch.arange(T, device=tok.device)
    return tok[ix], tok[ix + 1]


def spacing_ref(Q):
    """Median level spacing over all groups: the unit of --lr."""
    return float(torch.cat([((q['L'][..., -1] - q['L'][..., 0]) / (S.K - 1)).flatten() for q in Q.values()]).median())


def source(a):
    """4-bit QAT from the pretrained weights: Adam, weight lr 3e-5, scale lr 1e-4, both decayed linearly to 10%."""
    d = load_data(a.data); dev = a.device
    model, keys = load_model(a.model, dev)
    params = dict(model.named_parameters())
    w = {k: torch.nn.Parameter(params[k].detach().clone()) for k in keys}
    s = {k: torch.nn.Parameter(S.group_initial_scale(v.detach())[..., None]) for k, v in w.items()}
    levels = lambda: {k: S.group_levels(s[k].detach()[..., 0]) for k in w}
    wlr, slr = 3e-5, 1e-4
    opt = torch.optim.Adam([dict(params=list(w.values()), lr=wlr), dict(params=list(s.values()), lr=slr)], betas=(.9, .999), eps=1e-8)
    train = d['source-train'].to(dev)
    starts = windows(d['source-train'], a.steps, 4, 128, a.seed + 20000)
    for t in range(a.steps):
        x, y = batch(train, starts[t], 128)
        loss = S.loss_fn(model, {k: S.GroupGrid.apply(v, s[k]) for k, v in w.items()}, x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        for gr, lr in zip(opt.param_groups, [wlr, slr]):
            gr['lr'] = lr * (1 - .9 * t / max(1, a.steps - 1))
        opt.step()
        with torch.no_grad():
            for v in s.values():
                v.clamp_(min=1e-6)
        if (t + 1) % max(1, a.steps // 4) == 0:
            print(f'source step {t + 1} train loss {float(loss):.4f}', flush=True)
    L = levels()
    Q = {k: dict(c=S.nearest(v.detach(), L[k]).cpu(), L=L[k].cpu()) for k, v in w.items()}
    dev_nll = S.evaluate(model, {k: S.decode(q['c'].to(dev), q['L'].to(dev)) for k, q in Q.items()}, d['source-dev'])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(dict(Q=Q, spacing_ref=spacing_ref(Q), seed=a.seed, steps=a.steps, source_dev=dev_nll), a.out)
    print(f'saved {a.out}  source-dev NLL {dev_nll:.4f}  spacing_ref {spacing_ref(Q):.3e}')


def run(a):
    d = load_data(a.data); dev = a.device
    model, _ = load_model(a.model, dev)
    src = torch.load(a.source, weights_only=True)
    Q = {k: dict(c=q['c'].to(dev), L=q['L'].to(dev).float()) for k, q in src['Q'].items()}
    tr, dv = TASKS[a.task]
    tok = d[tr].to(dev)
    starts = windows(d[tr], a.steps, a.B, a.T, a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
    base = a.lr * src['spacing_ref']
    opt = S.make_optimizer(a.method, model, Q, a.bits, gen, base)
    evals = sorted({a.steps // 2, a.steps})
    curve, t0 = [], time.monotonic()
    for t in range(a.steps):
        x, y = batch(tok, starts[t], a.T)
        loss = opt.step(x, y, base * S.schedule(t, a.steps))
        if t + 1 in evals:
            nll = S.evaluate(model, opt.deployed(), d[dv])
            curve.append(dict(step=t + 1, dev_nll=nll, train_loss=float(loss), seconds=time.monotonic() - t0))
            print(f'{a.method} b{a.bits} {a.task} seed {a.seed}  step {t + 1}  {dv} NLL {nll:.4f}', flush=True)
            if not math.isfinite(nll):
                break
    res = dict(method=a.method, bits=a.bits, lr=a.lr, seed=a.seed, task=a.task, steps=a.steps, source=os.path.basename(a.source),
               curve=curve, final_dev_nll=curve[-1]['dev_nll'], optimizer_state_bytes=opt.state_bytes)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='EleutherAI/pythia-14m', help='Hugging Face id or local path')
    p.add_argument('--data', default=os.path.join(HERE, 'data', 'tokens.pt'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('source'); s.add_argument('--seed', type=int, default=9702); s.add_argument('--steps', type=int, default=2000)
    s.add_argument('--out', required=True)
    r = sub.add_parser('run'); r.add_argument('--source', required=True)
    r.add_argument('--method', choices=['sr', 'rotation', 'srqrr'], default='srqrr')
    r.add_argument('--bits', type=int, default=2); r.add_argument('--lr', type=float, default=3e-4)
    r.add_argument('--seed', type=int, default=2); r.add_argument('--task', choices=list(TASKS), default='wikitext')
    r.add_argument('--steps', type=int, default=600); r.add_argument('--B', type=int, default=8); r.add_argument('--T', type=int, default=128)
    r.add_argument('--out', default=None, help='optional JSON result path')
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # results were run in full FP32
    {'source': source, 'run': run}[a.cmd](a)


if __name__ == '__main__':
    main()
