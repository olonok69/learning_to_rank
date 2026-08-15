# Demo Runbook

For the full technical theory-to-code walkthrough, read:

- `docs/THEORY_TO_PRACTICE_GUIDE.md`

## Quick command sequence

```powershell
uv venv .venv --python 3.10
.\.venv\Scripts\Activate.ps1
uv sync
uv pip install --python .venv\Scripts\python.exe torch torchvision
uv pip install --python .venv\Scripts\python.exe git+https://github.com/allegro/allRank.git --no-deps
uv pip install --python .venv\Scripts\python.exe attrs flatten-dict tensorboardX gcsfs google-auth pandas
uv run --python .venv\Scripts\python.exe python scripts/generate_dataset.py
uv run --python .venv\Scripts\python.exe python scripts/run_tree_rankers.py
uv run --python .venv\Scripts\python.exe python scripts/run_allrank.py
```

## Expected artifacts

- `data/demo/dataset.npz`
- `data/demo/allrank/train.txt`
- `data/demo/allrank/vali.txt`
- `outputs/allrank/allrank_config.json`
- `outputs/allrank/results/demo_allrank/`

## Troubleshooting

1. If `python 3.10` is missing, install it first, then rerun `uv venv .venv --python 3.10`.
2. If allRank install fails due to wheel/build constraints, retry with a clean env:

```powershell
Remove-Item -Recurse -Force .venv
uv venv .venv --python 3.10
.\.venv\Scripts\Activate.ps1
uv sync
uv pip install --python .venv\Scripts\python.exe torch torchvision
uv pip install --python .venv\Scripts\python.exe git+https://github.com/allegro/allRank.git --no-deps
uv pip install --python .venv\Scripts\python.exe attrs flatten-dict tensorboardX gcsfs google-auth pandas
```

3. If allRank command errors with missing dataset files, regenerate data:

```powershell
uv run python scripts/generate_dataset.py
```
