"""Train and evaluate XGBRanker and LGBMRanker on shared synthetic data.

Theory -> practice mapping:
- Objective choices represent listwise/pairwise ranking optimization behavior.
- Group vectors define query boundaries required by LTR algorithms.
- NDCG@k is computed per query and then averaged for offline evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lightgbm import LGBMRanker
from xgboost import XGBRanker

from ltr_demo.metrics import summarize_ndcg


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    """Load compressed dataset produced by scripts/generate_dataset.py."""

    # np.load returns a lazy object; convert to plain dict for simpler downstream use.
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def main() -> None:
    # Parse dataset location so users can swap in alternate generated data.
    parser = argparse.ArgumentParser(description="Train XGBRanker and LGBMRanker")
    parser.add_argument("--dataset", type=Path, default=Path("data/demo/dataset.npz"))
    args = parser.parse_args()

    # Read train/validation features, labels, and group vectors.
    data = load_npz(args.dataset)
    X_train = data["train_X"]
    y_train = data["train_y"]
    group_train = data["train_group"]
    X_valid = data["valid_X"]
    y_valid = data["valid_y"]
    group_valid = data["valid_group"]

    # XGBoost ranker configured for direct NDCG optimization.
    # `group` is mandatory and indicates rows per query in order.
    xgb_ranker = XGBRanker(
        objective="rank:ndcg",
        eval_metric=["ndcg@5", "ndcg@10"],
        learning_rate=0.07,
        n_estimators=200,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    xgb_ranker.fit(X_train, y_train, group=group_train)

    # Predict relevance scores for validation rows and evaluate by grouped NDCG.
    xgb_preds = xgb_ranker.predict(X_valid)
    xgb_metrics = summarize_ndcg(y_valid, xgb_preds, group_valid)

    # LightGBM LambdaMART-style ranker.
    # `label_gain` maps graded labels to ranking gains used by NDCG internals.
    lgbm_ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        learning_rate=0.07,
        num_leaves=31,
        label_gain=[0, 1, 3, 7, 15],
        random_state=42,
        verbose=-1,
    )
    lgbm_ranker.fit(X_train, y_train, group=group_train)

    # Evaluate with exactly the same grouped metric logic for a fair comparison.
    lgbm_preds = lgbm_ranker.predict(X_valid)
    lgbm_metrics = summarize_ndcg(y_valid, lgbm_preds, group_valid)

    # Print side-by-side metrics for quick baseline benchmarking.
    print("XGBoost (XGBRanker)")
    print(f"  NDCG@5:  {xgb_metrics['ndcg@5']:.4f}")
    print(f"  NDCG@10: {xgb_metrics['ndcg@10']:.4f}")
    print()
    print("LightGBM (LGBMRanker / lambdarank)")
    print(f"  NDCG@5:  {lgbm_metrics['ndcg@5']:.4f}")
    print(f"  NDCG@10: {lgbm_metrics['ndcg@10']:.4f}")


if __name__ == "__main__":
    main()
