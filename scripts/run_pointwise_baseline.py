"""Run a pointwise baseline aligned with the TaskRabbit reference document.

Pointwise baseline strategy:
1) Convert graded labels into a binary target (e.g., booking vs non-booking).
2) Train a probabilistic classifier per row.
3) Calibrate predicted probabilities.
4) Rank items within each query by calibrated probability.
5) Evaluate ranking quality with grouped NDCG/MAP/MRR.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from ltr_demo.metrics import grouped_map_at_k, grouped_mrr_at_k, summarize_ndcg


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pointwise calibrated XGBoost baseline")
    parser.add_argument("--dataset", type=Path, default=Path("data/demo/dataset.npz"))
    parser.add_argument(
        "--positive-threshold",
        type=int,
        default=2,
        help="Rows with graded relevance >= threshold are treated as positive outcomes.",
    )
    args = parser.parse_args()

    data = load_npz(args.dataset)
    X_train = data["train_X"]
    y_train_graded = data["train_y"]
    X_valid = data["valid_X"]
    y_valid_graded = data["valid_y"]
    group_valid = data["valid_group"]

    # Convert graded labels to binary outcomes for a pointwise classifier.
    y_train_binary = (y_train_graded >= args.positive_threshold).astype(np.int32)
    y_valid_binary = (y_valid_graded >= args.positive_threshold).astype(np.int32)

    # Base pointwise model: independent per-row prediction.
    base_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.07,
        n_estimators=250,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    # Probability calibration improves ranking score interpretability.
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train_binary)

    # Use calibrated positive-class probability as ranking score.
    prob_valid = calibrated.predict_proba(X_valid)[:, 1]

    # NDCG uses graded labels; MAP/MRR use binary labels.
    ndcg = summarize_ndcg(y_valid_graded, prob_valid, group_valid)
    map5 = grouped_map_at_k(y_valid_binary, prob_valid, group_valid, k=5)
    map10 = grouped_map_at_k(y_valid_binary, prob_valid, group_valid, k=10)
    mrr5 = grouped_mrr_at_k(y_valid_binary, prob_valid, group_valid, k=5)
    mrr10 = grouped_mrr_at_k(y_valid_binary, prob_valid, group_valid, k=10)

    print("Pointwise Baseline (Calibrated XGBoost Classifier)")
    print(f"  Positive threshold (graded -> binary): >= {args.positive_threshold}")
    print(f"  NDCG@5:  {ndcg['ndcg@5']:.4f}")
    print(f"  NDCG@10: {ndcg['ndcg@10']:.4f}")
    print(f"  MAP@5:   {map5:.4f}")
    print(f"  MAP@10:  {map10:.4f}")
    print(f"  MRR@5:   {mrr5:.4f}")
    print(f"  MRR@10:  {mrr10:.4f}")


if __name__ == "__main__":
    main()
