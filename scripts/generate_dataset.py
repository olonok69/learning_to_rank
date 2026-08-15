"""Generate synthetic data used by all demo pipelines.

This script is the first practical step in the repository workflow:
1) Build synthetic LTR train/validation data.
2) Save NPZ for tree rankers.
3) Save LibSVM files for allRank.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ltr_demo.data_utils import generate_synthetic_dataset, save_libsvm, save_npz


def main() -> None:
    # CLI options are intentionally minimal to keep onboarding simple.
    parser = argparse.ArgumentParser(description="Generate synthetic ranking data for all demos")
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Create one consistent dataset used by all libraries.
    dataset = generate_synthetic_dataset(seed=args.seed)

    # NPZ path is consumed by scripts/run_tree_rankers.py.
    npz_file = args.output_dir / "dataset.npz"

    # LibSVM directory is consumed by scripts/run_allrank.py.
    libsvm_dir = args.output_dir / "allrank"

    # Persist both representations from the same in-memory dataset.
    save_npz(dataset, npz_file)
    save_libsvm(dataset, libsvm_dir)

    # Keep terminal output explicit so users can inspect generated artifacts quickly.
    print(f"Wrote tree-ranker dataset to: {npz_file}")
    print(f"Wrote allRank LibSVM files to: {libsvm_dir}")


if __name__ == "__main__":
    main()
