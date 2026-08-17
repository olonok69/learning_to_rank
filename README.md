# Learning-to-Rank Demo

> Documentación en español: [README.es.md](README.es.md)

A runnable, self-contained demo of learning-to-rank (LTR) covering four modeling
strategies on one shared synthetic query-grouped dataset:

1. **Pointwise baseline** — calibrated `XGBClassifier` ranked by predicted probability
2. **Pairwise/listwise (tree-based)** — `XGBRanker` (`rank:ndcg`)
3. **Pairwise/listwise (tree-based)** — `LGBMRanker` (`lambdarank`)
4. **Listwise (neural)** — [allRank](https://github.com/allegro/allRank), a PyTorch LTR framework

All four approaches train and evaluate against the same generated dataset, so their
`NDCG@5` / `NDCG@10` (and, for the baseline, `MAP@k` / `MRR@k`) results are directly comparable.

## Why this exists

Learning-to-rank problems (search, marketplace matching, recommendations) are about
ordering candidates *within a query/slate*, not predicting a single global score. This
repo is a compact, working reference for:

- how query grouping (`qid` / `group`) is represented in each library's API
- how graded relevance labels feed listwise objectives like NDCG/LambdaMART
- how a simple pointwise baseline compares against dedicated ranking objectives
- how to get allRank running locally, including on Windows

For a full theory-to-code walkthrough, see [docs/THEORY_TO_PRACTICE_GUIDE.md](docs/THEORY_TO_PRACTICE_GUIDE.md).
For a condensed command-only cheat sheet, see [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md).

## Project structure

```text
.
├─ pyproject.toml
├─ data/
│  └─ demo/
│     ├─ dataset.npz            # tree-ranker + baseline format
│     └─ allrank/                # LibSVM format (train.txt, vali.txt)
├─ docs/
│  ├─ DEMO_RUNBOOK.md            # quick command sequence + troubleshooting
│  ├─ DEMO_RUNBOOK.es.md         # Spanish runbook
│  ├─ THEORY_TO_PRACTICE_GUIDE.md# LTR theory mapped to this codebase
│  └─ THEORY_TO_PRACTICE_GUIDE.es.md
├─ outputs/
│  └─ allrank/                   # generated config + training results/logs
├─ scripts/
│  ├─ generate_dataset.py        # builds the synthetic dataset (NPZ + LibSVM)
│  ├─ run_pointwise_baseline.py  # calibrated classifier baseline + NDCG/MAP/MRR
│  ├─ run_tree_rankers.py        # XGBRanker + LGBMRanker
│  └─ run_allrank.py             # builds allRank config and launches training
└─ src/ltr_demo/
   ├─ data_utils.py              # synthetic data generation + NPZ/LibSVM export
   └─ metrics.py                 # grouped NDCG@k, MAP@k, MRR@k
```

## 1) Environment setup (uv)

This repository targets Python 3.10 for smooth allRank compatibility.

```powershell
cd D:\repos\learning_to_rank

# Create and activate a virtual env
uv venv .venv --python 3.10
.\.venv\Scripts\Activate.ps1

# Install base deps (XGBoost, LightGBM, scikit-learn, numpy, scipy)
uv sync
```

allRank has extra, heavier dependencies (PyTorch), so it's installed separately:

```powershell
uv pip install --python .venv\Scripts\python.exe torch torchvision
uv pip install --python .venv\Scripts\python.exe git+https://github.com/allegro/allRank.git --no-deps
uv pip install --python .venv\Scripts\python.exe attrs flatten-dict tensorboardX gcsfs google-auth pandas
```

## 2) Generate the demo dataset

```powershell
uv run --python .venv\Scripts\python.exe python scripts/generate_dataset.py
```

This builds one synthetic ranking dataset — fixed-size query "slates" with graded
relevance labels (`0`–`4`) — and writes it in two formats:

- `data/demo/dataset.npz` — used by the tree-ranker and pointwise-baseline scripts
- `data/demo/allrank/train.txt` and `data/demo/allrank/vali.txt` — LibSVM format with `qid`, used by allRank

See `generate_synthetic_dataset` in [src/ltr_demo/data_utils.py](src/ltr_demo/data_utils.py) for the generation logic
(feature signal, non-linear terms, quantization into graded labels).

## 3) Run the pointwise baseline

```powershell
uv run --python .venv\Scripts\python.exe python scripts/run_pointwise_baseline.py
```

Trains a calibrated `XGBClassifier` (`binary:logistic` + `CalibratedClassifierCV`) on a
binarized version of the graded labels, ranks items by calibrated positive-class
probability, and reports:

- `NDCG@5`, `NDCG@10` (graded relevance)
- `MAP@5`, `MAP@10` (binary relevance)
- `MRR@5`, `MRR@10` (binary relevance)

This is the simplest possible baseline — a good reference point before comparing
against dedicated ranking objectives below.

## 4) Run the tree-based rankers

```powershell
uv run --python .venv\Scripts\python.exe python scripts/run_tree_rankers.py
```

Trains and evaluates, side by side, on the same train/validation split:

- `XGBRanker(objective="rank:ndcg", eval_metric=["ndcg@5", "ndcg@10"])`
- `LGBMRanker(objective="lambdarank", metric="ndcg", label_gain=[0, 1, 3, 7, 15])`

Both use the `group`/`qid` vectors to define query boundaries, and both are scored
with the same grouped-NDCG implementation for a fair comparison.

## 5) Run the allRank neural demo

```powershell
uv run --python .venv\Scripts\python.exe python scripts/run_allrank.py
```

This script:

1. Builds an allRank JSON config in memory (small FC network, `lambdaLoss` with
   `lambdaRank_scheme`, NDCG@5/@10 metrics) and writes it to `outputs/allrank/allrank_config.json`
2. Launches `python -m allrank.main` as a subprocess using the LibSVM data
3. On Windows, installs lightweight `cp`/`rm` shims (see `prepare_windows_shims` in
   [scripts/run_allrank.py](scripts/run_allrank.py)) so allRank's internal shell calls work without patching the package
4. Writes results (model checkpoints, TensorBoard event logs, predictions) to
   `outputs/allrank/results/demo_allrank/`

## Evaluation metrics

All grouped metrics live in [src/ltr_demo/metrics.py](src/ltr_demo/metrics.py) and are computed **per query**, then
averaged — never mixed across query boundaries:

| Metric | Function | Relevance type | Used by |
|---|---|---|---|
| `NDCG@k` | `grouped_ndcg_at_k` / `summarize_ndcg` | graded (0–4) | all four approaches |
| `MAP@k` | `grouped_map_at_k` | binary | pointwise baseline |
| `MRR@k` | `grouped_mrr_at_k` | binary | pointwise baseline |

## Inspecting allRank training with TensorBoard

Training curves (loss, learning rate, NDCG@5/@10 for train/val) are logged as
TensorBoard event files under `outputs/allrank/tb_evals/`:

```powershell
uv run --python .venv\Scripts\python.exe tensorboard --logdir outputs/allrank/tb_evals
```

## Notes and known limitations

- Labels are synthetic graded relevance (`0`–`4`); this repo is meant to compare
  *modeling approaches*, not to represent a real product dataset.
- allRank has older, narrower transitive dependency constraints and can be
  environment-sensitive on Windows. If its install fails, keep the tree-ranker and
  pointwise-baseline demos working first, then retry allRank in a clean/isolated env
  (see the troubleshooting section in [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md)).
- Offline NDCG/MAP/MRR are necessary but not sufficient — they validate ranking
  quality on held-out data, not real user/business impact. Online experimentation is
  the natural next step when integrating any of these approaches into a live product.
