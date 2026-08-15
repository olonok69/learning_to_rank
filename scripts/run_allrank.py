"""Run allRank training on the generated LibSVM dataset.

This script bridges repository data artifacts and the allRank CLI.
It also contains a Windows compatibility layer because allRank uses Unix-like
shell commands (`cp`, `rm`) internally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def build_config(data_path: Path) -> dict:
    """Construct allRank JSON config tuned for a lightweight local demo run.

    Key choices:
    - Small FC model only (no transformer) for fast CPU execution.
    - LambdaLoss with lambdaRank scheme to align with ranking objectives.
    - NDCG metrics at 5 and 10.
    """

    return {
        "model": {
            "fc_model": {
                "sizes": [64, 32],
                "input_norm": True,
                "activation": "ReLU",
                "dropout": 0.1,
            },
            "transformer": None,
            "post_model": {"d_output": 1, "output_activation": None},
        },
        "data": {
            "path": str(data_path),
            "num_workers": 0,
            "batch_size": 16,
            "slate_length": 15,
            "validation_ds_role": "vali",
        },
        "optimizer": {"name": "Adam", "args": {"lr": 1e-3}},
        "training": {
            "epochs": 5,
            "gradient_clipping_norm": 1.0,
            "early_stopping_patience": 2,
        },
        "loss": {
            "name": "lambdaLoss",
            "args": {"weighing_scheme": "lambdaRank_scheme", "k": 10},
        },
        "metrics": ["ndcg_5", "ndcg_10"],
        "lr_scheduler": {"name": "", "args": {}},
        "val_metric": "ndcg_10",
        "expected_metrics": {},
        "detect_anomaly": False,
    }


def prepare_windows_shims(base_dir: Path) -> Path | None:
    """Create minimal command shims so allRank shell calls work on Windows.

    allRank executes shell commands such as:
    - cp <src> <dst>
    - rm -rf <path>

    On Windows, these commands are missing by default in cmd/PowerShell.
    We prepend a shim directory to PATH containing compatible `.cmd` wrappers.
    """

    if os.name != "nt":
        return None

    shim_dir = base_dir / "_allrank_shims"
    shim_dir.mkdir(parents=True, exist_ok=True)

    # Shim for Unix `cp` command used by allRank to copy the active config.
    cp_cmd = shim_dir / "cp.cmd"
    cp_cmd.write_text(
        "@echo off\n"
        "copy /Y \"%~1\" \"%~2\" >nul\n",
        encoding="utf-8",
    )

    # Shim for Unix `rm` command used by allRank cleanup helpers.
    rm_cmd = shim_dir / "rm.cmd"
    rm_cmd.write_text(
        "@echo off\n"
        "if /I \"%~1\"==\"-rf\" (\n"
        "  if exist \"%~2\" rmdir /S /Q \"%~2\"\n"
        ") else (\n"
        "  if exist \"%~1\" del /F /Q \"%~1\"\n"
        ")\n",
        encoding="utf-8",
    )

    return shim_dir


def main() -> None:
    # Parse configurable runtime paths so this script can be reused in CI pipelines.
    parser = argparse.ArgumentParser(description="Run allRank on generated LibSVM data")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/demo/allrank"))
    parser.add_argument("--job-dir", type=Path, default=Path("outputs/allrank"))
    parser.add_argument("--run-id", type=str, default="demo_allrank")
    args = parser.parse_args()

    # Write allRank config next to outputs for full reproducibility.
    args.job_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.job_dir / "allrank_config.json"
    config = build_config(args.dataset_dir)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Use current interpreter to guarantee command runs inside the selected venv.
    command = [
        sys.executable,
        "-m",
        "allrank.main",
        "--job-dir",
        str(args.job_dir),
        "--run-id",
        args.run_id,
        "--config-file-name",
        str(config_path),
    ]

    print("Running allRank training...")
    print("Command:")
    print(" ".join(command))

    # Prepare process environment and inject Windows shims if needed.
    env = os.environ.copy()
    shim_dir = prepare_windows_shims(args.job_dir)
    if shim_dir:
        env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"

    # Execute allRank training and fail fast on non-zero exit.
    subprocess.run(command, check=True, env=env)

    # Surface deterministic output location for downstream inspection.
    print("allRank run complete.")
    print(f"Results directory: {args.job_dir / 'results' / args.run_id}")


if __name__ == "__main__":
    main()
