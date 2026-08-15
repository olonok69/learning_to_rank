"""Data generation and export utilities for the ranking demos.

This module creates a synthetic learning-to-rank dataset that mimics query-grouped
candidate lists ("slates"), then exports the data in two formats:

1) NPZ for tree rankers (XGBoost and LightGBM scripts).
2) LibSVM text files for allRank.

The generated labels are graded relevance levels, which makes them suitable for
NDCG-oriented objectives and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import dump_svmlight_file


@dataclass
class RankingSplit:
    """One split of ranking data (train or validation).

    Attributes:
        X: Feature matrix with shape [num_rows, num_features].
        y: Graded relevance labels with shape [num_rows].
        qid: Query id per row; rows with same qid belong to one slate.
        group: Group-size vector used by tree rankers (one value per query).
    """

    X: np.ndarray
    y: np.ndarray
    qid: np.ndarray
    group: np.ndarray


@dataclass
class RankingDataset:
    """Container for complete train/validation ranking data and schema metadata."""

    train: RankingSplit
    valid: RankingSplit
    n_features: int
    slate_size: int
    relevance_levels: int


def _build_split(
    n_queries: int,
    slate_size: int,
    n_features: int,
    relevance_levels: int,
    rng: np.random.Generator,
) -> RankingSplit:
    """Build a synthetic split where each query has a fixed slate size.

    Theory connection:
    - In learning-to-rank, each query has multiple documents/items to order.
    - Here we model that by repeating each query id across a fixed number of rows.
    """

    # Assign query ids so each block of `slate_size` rows belongs to one query.
    qid = np.repeat(np.arange(n_queries), slate_size)

    # Draw continuous features from a normal distribution.
    X = rng.normal(0.0, 1.0, size=(n_queries * slate_size, n_features)).astype(np.float32)

    # Create a latent quality signal from the strongest feature dimensions.
    # The decaying weights make early features more informative than later ones.
    weights = np.linspace(1.4, 0.3, num=min(6, n_features), dtype=np.float32)
    signal = X[:, : len(weights)] @ weights

    # Add non-linear interactions to avoid a purely linear target.
    signal += 0.4 * np.sin(X[:, 0]) + 0.2 * np.cos(X[:, 1])

    # Normalize to [0, 1], then quantize into graded relevance levels.
    # Example with 5 levels: labels become integers in [0, 4].
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    y = np.floor(signal * relevance_levels).astype(np.int32)
    y = np.clip(y, 0, relevance_levels - 1)

    # Tree rankers expect group sizes per query in row order.
    group = np.full(shape=(n_queries,), fill_value=slate_size, dtype=np.int32)
    return RankingSplit(X=X, y=y, qid=qid, group=group)


def generate_synthetic_dataset(
    train_queries: int = 200,
    valid_queries: int = 80,
    slate_size: int = 15,
    n_features: int = 20,
    relevance_levels: int = 5,
    seed: int = 42,
) -> RankingDataset:
    """Create train/validation synthetic ranking dataset with reproducible randomness."""

    # Use NumPy Generator for modern, deterministic random state handling.
    rng = np.random.default_rng(seed)
    train = _build_split(train_queries, slate_size, n_features, relevance_levels, rng)
    valid = _build_split(valid_queries, slate_size, n_features, relevance_levels, rng)
    return RankingDataset(
        train=train,
        valid=valid,
        n_features=n_features,
        slate_size=slate_size,
        relevance_levels=relevance_levels,
    )


def save_npz(dataset: RankingDataset, output_file: Path) -> None:
    """Save dataset as compressed NPZ for fast loading in tree-ranker scripts."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        train_X=dataset.train.X,
        train_y=dataset.train.y,
        train_qid=dataset.train.qid,
        train_group=dataset.train.group,
        valid_X=dataset.valid.X,
        valid_y=dataset.valid.y,
        valid_qid=dataset.valid.qid,
        valid_group=dataset.valid.group,
        n_features=dataset.n_features,
        slate_size=dataset.slate_size,
        relevance_levels=dataset.relevance_levels,
    )


def save_libsvm(dataset: RankingDataset, output_dir: Path) -> None:
    """Save dataset in LibSVM format expected by allRank loaders.

    allRank convention:
    - training file name: train.txt
    - validation file name: vali.txt
    - each row carries a query id (qid) to define slate membership
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_svmlight_file(
        dataset.train.X,
        dataset.train.y,
        str(output_dir / "train.txt"),
        query_id=dataset.train.qid,
    )
    dump_svmlight_file(
        dataset.valid.X,
        dataset.valid.y,
        str(output_dir / "vali.txt"),
        query_id=dataset.valid.qid,
    )
