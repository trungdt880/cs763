"""Fan out per-checkpoint evaluation across all available GPUs.

Usage:
    python scripts/eval_fast.py --results_dir results/qwen3_0p6b__cot__seed0
    python scripts/eval_fast.py --results_dir results/qwen3_0p6b__cot__seed0 --gpus 0,1,2,3

What it does:
  1. Reads the run's manifest to discover every checkpoint (milestones + final).
  2. Launches one `python evaluate.py --checkpoint <label>` subprocess per
     checkpoint, round-robin across the listed GPUs, capped at --max_parallel
     concurrent processes.
  3. Each subprocess writes its result to `results/<run>/_shards/<label>.json`.
  4. When all are done, merges the shards into `results/<run>/metrics.json`
     (the same format `evaluate.py` would have produced in sequential mode),
     then deletes the shard files.

With 8 GPUs and 11 checkpoints this drops per-run eval wall-clock from ~1 hr
to ~5 min: every checkpoint loads its own model copy onto its own GPU in
parallel, so the only serialization is the merge at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# Make sure the project root is importable regardless of where we're launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def discover_checkpoints(results_dir: Path) -> List[str]:
    """Return the list of checkpoint labels present on disk for this run."""
    manifest = json.loads((results_dir / "manifest.json").read_text())
    ckpt_root = Path(manifest["checkpoints_dir"])
    labels: List[str] = []
    for ms in manifest["milestone_steps"]:
        if (ckpt_root / f"checkpoint-step-{ms}").exists():
            labels.append(f"step-{ms}")
    if (ckpt_root / "checkpoint-final").exists():
        labels.append("final")
    return labels


def parse_gpus(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="results/<run_id> directory")
    ap.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU indices to use (default: 0..7).",
    )
    ap.add_argument(
        "--max_parallel",
        type=int,
        default=None,
        help="Maximum concurrent eval processes. Defaults to len(gpus).",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (defaults to the current one).",
    )
    ap.add_argument(
        "--keep_shards",
        action="store_true",
        help="Don't delete _shards/ after merging.",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not (results_dir / "manifest.json").exists():
        sys.exit(f"no manifest.json at {results_dir}")

    gpus = parse_gpus(args.gpus)
    max_parallel = args.max_parallel or len(gpus)
    labels = discover_checkpoints(results_dir)
    if not labels:
        sys.exit(f"no checkpoints found for {results_dir}")

    print(f"Run:          {results_dir.name}")
    print(f"Checkpoints:  {len(labels)}")
    print(f"GPUs:         {gpus}")
    print(f"Max parallel: {max_parallel}")
    print()

    manifest = json.loads((results_dir / "manifest.json").read_text())
    shard_dir = results_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    log_dir = results_dir / "_eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    running: List[tuple] = []  # (label, proc, log_file_handle, gpu)
    pending = list(labels)

    def launch(label: str, gpu: int) -> tuple:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TOKENIZERS_PARALLELISM"] = "false"  # avoid noisy warnings
        log_path = log_dir / f"{label}.log"
        log_f = open(log_path, "w")
        cmd = [
            args.python,
            str(PROJECT_ROOT / "evaluate.py"),
            "--results_dir",
            str(results_dir),
            "--checkpoint",
            label,
        ]
        print(f"  [gpu {gpu}] launching {label}  (log: {log_path.name})")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        return (label, proc, log_f, gpu)

    gpu_cursor = 0
    while pending or running:
        # Launch up to max_parallel.
        while pending and len(running) < max_parallel:
            gpu = gpus[gpu_cursor % len(gpus)]
            gpu_cursor += 1
            running.append(launch(pending.pop(0), gpu))

        # Poll for any finished child.
        still_running: List[tuple] = []
        for label, proc, log_f, gpu in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((label, proc, log_f, gpu))
                continue
            log_f.close()
            if rc != 0:
                tail = (log_dir / f"{label}.log").read_text().splitlines()[-20:]
                print(f"  [gpu {gpu}] {label} FAILED rc={rc}")
                for line in tail:
                    print(f"    | {line}")
            else:
                print(f"  [gpu {gpu}] {label} done")
        running = still_running
        if running:
            time.sleep(0.5)

    elapsed = time.time() - t0
    print(f"\nAll eval subprocesses finished in {elapsed:.1f}s")

    # ----- Merge shards into metrics.json -----
    checkpoints_metrics: dict = {}
    missing = []
    for label in labels:
        shard = shard_dir / f"{label}.json"
        if not shard.exists():
            missing.append(label)
            continue
        checkpoints_metrics[label] = json.loads(shard.read_text())

    if missing:
        print(f"WARNING: {len(missing)} checkpoints missing from shards: {missing}")

    # Preserve the milestone_steps ordering + final at the end.
    ordered = {}
    for ms in manifest["milestone_steps"]:
        k = f"step-{ms}"
        if k in checkpoints_metrics:
            ordered[k] = checkpoints_metrics[k]
    if "final" in checkpoints_metrics:
        ordered["final"] = checkpoints_metrics["final"]

    out = {"manifest": manifest, "checkpoints": ordered}
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Merged {len(ordered)} checkpoints -> {results_dir / 'metrics.json'}")

    if not args.keep_shards and not missing:
        for p in shard_dir.glob("*.json"):
            p.unlink()
        try:
            shard_dir.rmdir()
        except OSError:
            pass  # not empty; leave it


if __name__ == "__main__":
    main()
