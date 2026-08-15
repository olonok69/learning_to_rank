"""Ranking metric helpers used by demo scripts.

This module computes NDCG per query group and then averages across groups.
That mirrors typical LTR offline evaluation, where each query contributes one
metric value rather than mixing all rows globally.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import ndcg_score


def grouped_ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, group: np.ndarray, k: int) -> float:
    """Compute average NDCG@k across query groups.

    Args:
        y_true: Flat true relevance labels in row order.
        y_score: Flat model scores in row order.
        group: Number of rows per query, in the same row order.
        k: Cutoff for NDCG@k.

    Returns:
        Mean NDCG@k over all query groups.
    """

    # `start` and `end` walk over contiguous query blocks.
    start = 0
    values: list[float] = []
    for g in group:
        end = start + int(g)

        # sklearn.ndcg_score expects 2D arrays shaped [n_queries, slate_len],
        # so each single query is wrapped in a one-element list.
        values.append(ndcg_score([y_true[start:end]], [y_score[start:end]], k=k))
        start = end

    # Average across queries to avoid over-weighting larger datasets by row count.
    return float(np.mean(values))


def summarize_ndcg(y_true: np.ndarray, y_score: np.ndarray, group: np.ndarray) -> dict[str, float]:
    """Return the two core ranking metrics used in this repository."""

    return {
        "ndcg@5": grouped_ndcg_at_k(y_true, y_score, group, 5),
        "ndcg@10": grouped_ndcg_at_k(y_true, y_score, group, 10),
    }


def grouped_map_at_k(y_true_binary: np.ndarray, y_score: np.ndarray, group: np.ndarray, k: int) -> float:
    """Compute mean average precision at k across query groups.

    Args:
        y_true_binary: Binary relevance labels in row order.
        y_score: Predicted scores/probabilities in row order.
        group: Number of rows per query.
        k: Cutoff for AP@k.

    Returns:
        MAP@k aggregated over all query groups.
    """

    start = 0
    ap_values: list[float] = []
    for g in group:
        end = start + int(g)
        y_q = y_true_binary[start:end]
        s_q = y_score[start:end]

        # Sort descending by predicted score and keep top-k items.
        order = np.argsort(-s_q)[:k]
        y_top = y_q[order]

        total_relevant = int(np.sum(y_q))
        if total_relevant == 0:
            ap_values.append(0.0)
            start = end
            continue

        hits = 0
        precision_sum = 0.0
        for rank, rel in enumerate(y_top, start=1):
            if rel == 1:
                hits += 1
                precision_sum += hits / rank

        ap_values.append(precision_sum / min(total_relevant, k))
        start = end

    return float(np.mean(ap_values))


def grouped_mrr_at_k(y_true_binary: np.ndarray, y_score: np.ndarray, group: np.ndarray, k: int) -> float:
    """Compute mean reciprocal rank at k across query groups."""

    start = 0
    rr_values: list[float] = []
    for g in group:
        end = start + int(g)
        y_q = y_true_binary[start:end]
        s_q = y_score[start:end]

        order = np.argsort(-s_q)[:k]
        y_top = y_q[order]

        reciprocal_rank = 0.0
        for rank, rel in enumerate(y_top, start=1):
            if rel == 1:
                reciprocal_rank = 1.0 / rank
                break

        rr_values.append(reciprocal_rank)
        start = end

    return float(np.mean(rr_values))
