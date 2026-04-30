"""Top-level orchestrator: runs the full experiment matrix.

For each (model, condition, seed) in configs.yaml:matrix:
    1. Train (calls train.py via subprocess so each run gets a clean process / GPU state).
    2. Evaluate every checkpoint (calls evaluate.py via subprocess).

Use --only to filter, --skip_train to re-evaluate existing runs without retraining.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def run_id_for(model: str, condition: str, seed: int) -> str:
    return f"{model}__{condition}__seed{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs.yaml")
    ap.add_argument("--only", default=None,
                    help="comma-separated substring filter on run_id")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    matrix = cfg["matrix"]

    filters = args.only.split(",") if args.only else None

    for entry in matrix:
        rid = run_id_for(entry["model"], entry["condition"], entry["seed"])
        if filters and not any(f in rid for f in filters):
            continue
        print(f"\n############ {rid} ############")

        results_dir = Path(cfg["output"]["results_dir"]) / rid

        if not args.skip_train:
            cmd = [
                sys.executable, "train.py",
                "--config", args.config,
                "--model", entry["model"],
                "--condition", entry["condition"],
                "--seed", str(entry["seed"]),
            ]
            print("  $", " ".join(cmd))
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"  TRAIN FAILED for {rid}; skipping eval.")
                continue

        if not args.skip_eval:
            manifest = results_dir / "manifest.json"
            if not manifest.exists():
                print(f"  No manifest at {manifest}; skipping eval.")
                continue
            cmd = [
                sys.executable, "evaluate.py",
                "--results_dir", str(results_dir),
            ]
            print("  $", " ".join(cmd))
            subprocess.run(cmd)


if __name__ == "__main__":
    main()
