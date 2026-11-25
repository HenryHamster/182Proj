# Reproducing the CS182 report

Every figure and table in the report comes from the run records in `results/*.jsonl`. These are committed, so the
figures and tables can be rebuilt in seconds **without retraining**. Retraining is only needed to regenerate those
records.

All paths are relative to `flip-vs-srqrr/`.

## 1. Figures and tables from the committed results (CPU, under a minute)

```bash
pip install matplotlib                 # only dependency for this step
python figures/paper_figs.py           # Fig. 1-3 -> figures/fig{1,2,3}_*.pdf/.png
python figures/paper_tables.py         # learning-rate table -> figures/lr_table.tex
```

| report item | script | inputs (`results/`) |
|---|---|---|
| Fig. 1, quality vs state bits/weight (10.6M) | `figures/paper_figs.py` | `campaign_10M.jsonl` |
| Fig. 2, training curves at matched state (10.6M) | `figures/paper_figs.py` | `campaign_10M.jsonl` |
| Fig. 3, scaling to 49.2M and 99.1M | `figures/paper_figs.py` | `campaign_10M.jsonl`, `10M.jsonl`, `campaign_50M.jsonl`, `campaign_100M.jsonl` |
| Learning-rate table (appendix) | `figures/paper_tables.py` | same as Fig. 3 |
| GPEF ablations (appendix) | `docs/RECORD_flip-vs-srqrr-001.md` | `campaign_10M.jsonl` |
| GPEF error storage at 50M | `docs/RECORD_flip-vs-srqrr-003.md` | `campaign_50M_gpefix.jsonl` |

**Selection rule**, used by every script:

- For each method and model size, the selected learning rate is the one with the lowest **seed-0** validation
  bits per byte among non-diverged runs.
- The reported value is the mean over all seeds at that learning rate.
- Error bars and bands show the min-max over those seeds: 3 seeds at 10.6M, 2 at 49.2M and 99.1M.

**Test split.** The test split is sealed. `drive.py` aliases `test` to `val` and drops every `test_*` field, so
`test.bin` is never read and all reported numbers are validation numbers.

## 2. Environment and data (for retraining)

- **Software.** Python 3.12, PyTorch 2.9.1 with CUDA 12.8, numpy. Add `datasets` for data preparation and
  `matplotlib` for plots.
- **Data.** The datasets are public and are not included in the repo. `python prep.py` downloads them and builds byte-level WikiText-2 (pilot) and WikiText-103 (all other sizes) in `data/`.
  WikiText-103 comes from the Hugging Face hub.
- **Hardware used.** The pilot and 10.6M runs used an RTX 5080 (16 GB), several runs at a time. The 49.2M and
  99.1M runs used an L40S (48 GB), one run per GPU, with 9-11 GB peak memory per run at 49.2M.

One run is a single command:

```bash
python drive.py run --preset 10M --method srq_rot@b2 --lr 0.03 --seed 0 --eval_windows 2000 --out results/runs.jsonl
```

## 3. Retraining the result files

Each campaign is resumable: it skips any `(method, lr, seed)` already present in its output. Its learning-rate
protocol is in the script's header. To re-run a campaign from scratch, move its output file aside first.

| size (preset) | command | writes (`results/`) | runs | GPU-hours |
|---|---|---|---|---|
| 2.65M pilot + 10.6M main methods | `python campaigns/campaign.py`, then `campaign_10M_v2.py`, `campaign_rot.py`, `campaign_team.py`, `campaign_hi.py` | `campaign_pilot.jsonl`, `campaign_10M.jsonl` | 225 pilot | 5 (pilot) |
| 10.6M GPEF group sizes, format checks, references | `campaigns/campaign_gsz.py`, `campaign_grp.py`, `campaign_grp2.py`, `campaign_gpx.py` | `campaign_pilot.jsonl`, `campaign_10M.jsonl` | | |
| 10.6M GPEF ablations | `campaigns/campaign_gpf.py`, `campaign_sym.py`, `campaign_gpr.py`, `campaign_lam.py` | `campaign_10M.jsonl` | | |
| 10.6M Muon QAT, Flip m4, Muon+SR m4 | `campaigns/ladder.py` | `10M.jsonl` | 13 | 7 |
| 49.2M (`50M`) | `python campaigns/campaign_50M.py GPU_A GPU_B` | `campaign_50M.jsonl` | 57 | 92 |
| 49.2M GPEF error storage | `python campaigns/campaign_gpefix_50M.py GPU_A GPU_B` | `campaign_50M_gpefix.jsonl` | 6 | 10 |
| 99.1M (`100M`) | `python campaigns/campaign_100M.py GPU_A GPU_B` | `campaign_100M.jsonl` | 18 | 79 |

All the 10.6M campaigns together are 251 runs and 60 GPU-hours. GPU-hours are the sum of the recorded
`train_seconds` in each file.

- **Ordering.** Several 10.6M scripts wait for an earlier one to finish (for example `campaign_rot.py` waits for
  `campaign.py`) and read the pilot's best learning rate from `campaign_pilot.jsonl`, so run them in the order listed.
- **Run time per size.** A 10.6M run takes under 0.6 h. A 49.2M run takes 1.2-2.5 h on an L40S, and a 99.1M run takes
  4.2-5.2 h.

Experiment records with design, outcomes and limits are in `docs/RECORD_flip-vs-srqrr-00{1,2,3,4}.md`.
