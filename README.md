# Learning-to-Rank Demo (TaskRabbit Interview Prep)

This demo provides a runnable example for the three ranking stacks you requested:

1. XGBoost with `XGBRanker`
2. LightGBM with `LGBMRanker` (`objective="lambdarank"`)
3. allRank (PyTorch neural learning-to-rank framework)

The project uses one synthetic query-grouped ranking dataset and runs each library on top of it.

## 1) Environment setup with uv

This repository is configured for Python 3.10 (required for smooth allRank compatibility).

```powershell
cd D:\repos\learning_to_rank

# Create virtual env
uv venv .venv --python 3.10

# Activate
.\.venv\Scripts\Activate.ps1

# Install base deps (XGBoost + LightGBM + utilities)
uv sync
```

For allRank, install separately after `uv sync`:

```powershell
uv pip install --python .venv\Scripts\python.exe torch torchvision
uv pip install --python .venv\Scripts\python.exe git+https://github.com/allegro/allRank.git --no-deps
uv pip install --python .venv\Scripts\python.exe attrs flatten-dict tensorboardX gcsfs google-auth pandas
```

## 2) Generate demo dataset

```powershell
uv run --python .venv\Scripts\python.exe python scripts/generate_dataset.py
```

This writes:

- `data/demo/dataset.npz` for tree rankers
- `data/demo/allrank/train.txt` and `data/demo/allrank/vali.txt` for allRank

## 3) Run XGBoost + LightGBM demo

```powershell
uv run --python .venv\Scripts\python.exe python scripts/run_tree_rankers.py
```

Expected output includes `NDCG@5` and `NDCG@10` for both:

- `XGBRanker(objective="rank:ndcg")`
- `LGBMRanker(objective="lambdarank")`

## 3b) Run pointwise baseline + ranking metrics

This mirrors the TaskRabbit reference pointwise baseline:

- model: calibrated `XGBClassifier` (pointwise probability model)
- ranking score: calibrated `P(positive outcome)`
- metrics: `NDCG@k`, `MAP@k`, `MRR@k`

```powershell
uv run --python .venv\Scripts\python.exe python scripts/run_pointwise_baseline.py
```

## 4) Run allRank demo

```powershell
uv run --python .venv\Scripts\python.exe python scripts/run_allrank.py
```

This command:

1. Writes an allRank JSON config to `outputs/allrank/allrank_config.json`
2. Launches `python -m allrank.main`
3. Stores results in `outputs/allrank/results/demo_allrank`

## Project structure

```text
.
├─ pyproject.toml
├─ scripts/
│  ├─ generate_dataset.py
│  ├─ run_tree_rankers.py
│  └─ run_allrank.py
└─ src/ltr_demo/
   ├─ data_utils.py
   └─ metrics.py
```

## Notes

- The synthetic labels are graded relevance levels (`0..4`) to match listwise ranking metrics like NDCG.
- allRank has older transitive constraints and can be environment-sensitive on Windows. If install fails, keep tree demos working first, then test allRank in an isolated env.

## Deep technical guide

For a full theory-to-implementation walkthrough with direct code mapping, see:

- `docs/THEORY_TO_PRACTICE_GUIDE.md`
