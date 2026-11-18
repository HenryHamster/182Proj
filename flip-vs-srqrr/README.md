# flip-vs-srqrr: 2-bit training at a fixed optimizer-state budget

Every method here trains a byte-level transformer from scratch with **2-bit deployed weights**. The question is how
validation bits per byte (bpb) depends on how much optimizer + weight state each weight carries, counted in
bits/weight. The families compared are:

- **flip**: the teammate's flip optimizer, with quantized momentum.
- **SR / RR**: a normalized gradient step that stores a B-bit sub-grid remainder. It is rounded by stochastic rounding
  (SR) or rotation rounding (RR, threshold frac(u0 + tφ)).
- **GPEF**: the teammate's group-32 learned codebook with error feedback.
- **Low-bit and fp Adam / Muon** references.

The setup is the teammate's harness ([EthelynHuang/flip-optimizer](https://github.com/EthelynHuang/flip-optimizer)
@ ea4471c). The pilot uses 2.65M quantized weights on WikiText-2. The main rung uses 10.6M on WikiText-103: 8000 steps,
batch 64×256, and 2000 val windows. Learning rates are chosen on val only, and the test split is never read.

## Results (10.6M, val bpb, lower is better)

Each cell is the mean of 3 seeds at the learning rate that scored best on val for seed 0.

| state bits/weight | SR | RR | GPEF | other |
|---|---|---|---|---|
| ≈2 | 2.391 | 2.334 | **2.221** (2.25 b/w) | |
| ≈4 | 2.055 | 1.941 | **1.889** | flip@m2 2.124, Muon+SR 2.545 |
| 6–8 | 1.636 | **1.635** | 1.758 | flip@m4 1.677 (6.1 b/w) |
| 10–12 | | | **1.607** (12 b/w) | Adam+SR@m4 1.894 (10.1 b/w) |
| 64–102 | | | | fp Muon **1.460**, Muon QAT 1.530, fp Adam 1.548, AdamW STE 1.599, Adam QAT 1.807 |

- At 2–4 bits/weight, GPEF is best and RR beats SR.
- At 8 bits/weight, RR and SR tie and beat GPEF@e4 at the same budget. They also beat flip@m4, which has 6.1 bits/weight.
- Quantized Muon and Adam momentum (Muon+SR, Adam+SR) is far behind at the same budget.

The full record is in [docs/RECORD_flip-vs-srqrr-001.md](docs/RECORD_flip-vs-srqrr-001.md). It includes the pilot,
the ablations (srqrr refit, rotation × GPEF, symmetric-error ladder, λ-split, and the teammate's GPEF revisions), the
checks, and the interpretation.

![val bpb vs state bits/weight](figures/family_bits_10M.png)
![training curves at matched state](figures/curves_by_bits_10M.png)

## Layout

| path | contents |
|---|---|
| `qtrain.py` | Harness: the teammate's `src/qtrain.py` plus a host-sync speed patch. It gives identical results; see the header. |
| `drive.py` | Runs one (method, lr, seed) and appends a JSON record. `test` is aliased to `val`, so test.bin is never read. |
| `prep.py` | Builds byte-level WikiText-2 / WikiText-103 into `data/`. |
| `srq_ext.py` | `srq_sr@bB`, `srq_rot@bB`, and `srqrr@bB` (rotation + per-row scale refit). |
| `team_ext.py` | `gpef@eE`, `adamw_ste`, and `adam_sr@mB`. |
| `team_rot.py`, `team_grp.py`, `team_gpx.py`, `team_gpf.py` | GPEF variants: RR/SR rounding on GPEF codebooks, larger groups, rotation × GPEF error storage, and the teammate's revisions. |
| `campaigns/` | The executed sweep scripts. `campaign.py` defines the protocol; each `campaign_*.py` is one recorded phase. |
| `tests/` | CPU checks for each extension (equivalence to the notebook, on-grid codes, unbiasedness, drift). |
| `results/` | Run records (`campaign_10M.jsonl`, `campaign_pilot.jsonl`, plus `10M.jsonl` / `pilot.jsonl` from the scale study). Each record has `val_bpb`, `train_eval_bpb`, `state_bits`, and `curve` (train loss in nats every 100 steps). |
| `figures/` | Every figure and the script that draws it from `results/`. |
| `docs/` | Experiment records, the rotation-rounding derivation (`rotation_rounding.tex`), and the draft. |

Each extension registers its methods into `qtrain` on import and leaves the existing methods unchanged. `drive.py`
imports all of them.

## Running

```bash
python prep.py                                                # data/wt2_pytorch, data/wt103
python drive.py run --preset 10M --method srq_rot@b2 --lr 0.03 --seed 0 --eval_windows 2000 --out results/runs.jsonl
python tests/srq_test.py                                      # CPU, needs data/wt2_pytorch
python campaigns/campaign.py                                  # full protocol; resumable; GPU admission at 72% of 16 GB
uv run --with matplotlib python figures/plot_family_bits.py
```

`plot_bits.py` and `plot_curves.py` take `10M` or `pilot` as an argument.
