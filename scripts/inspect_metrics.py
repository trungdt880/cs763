"""Pretty-print the generation samples saved inside a run's metrics.json.

Usage:
    python scripts/inspect_metrics.py results/qwen3_0p6b__cot__seed0
    python scripts/inspect_metrics.py results/qwen3_0p6b__cot__seed0 --checkpoint final
    python scripts/inspect_metrics.py results/qwen3_0p6b__cot__seed0 --checkpoint step-1250 --n 5

Shows per checkpoint:
  * summary metrics
  * N task-accuracy samples (prompt, gold, generation, correct)
  * a few canary-extraction samples at k=0 and k=3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def truncate(s: str, n: int = 300) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + " …"


def show_checkpoint(label: str, m: dict, n_task: int = 5, n_canary: int = 3) -> None:
    print(f"\n{'=' * 72}\n=== {label}\n{'=' * 72}")
    ta = m.get("task_accuracy", {})
    print(f"task accuracy: {ta.get('accuracy', 'N/A'):.3f}  (n={ta.get('n', 0)})")

    ve = m.get("verbatim_extraction", {})
    for k in ("k=0", "k=1", "k=2", "k=3"):
        kv = ve.get(k, {})
        print(
            f"  extraction {k}: overall={kv.get('overall_em', 0):.3f}  by_dup={kv.get('by_dup', {})}"
        )

    mia = m.get("mia", {})
    print(
        f"  nll_train={mia.get('mean_nll_train', 0):.3f}  nll_holdout={mia.get('mean_nll_holdout', 0):.3f}  "
        f"auc_nll={mia.get('auc_nll', 0):.3f}  auc_mink20={mia.get('auc_mink20', 0):.3f}"
    )
    if "nll_by_dup" in mia:
        print(f"  nll_by_dup: {mia['nll_by_dup']}")

    # -- Task accuracy samples --
    samples = ta.get("samples", [])
    if samples:
        print(f"\n-- task samples ({min(n_task, len(samples))} of {len(samples)}) --")
        for s in samples[:n_task]:
            tick = "✓" if s.get("correct") else "✗"
            print(
                f"  {tick} [{s.get('template')}] gold={s.get('gold')}  pred={s.get('predicted')}"
            )
            print(f"    gen: {truncate(s.get('generation', ''), 240)}")

    # -- Canary extraction samples --
    for k in ("k=0", "k=3"):
        samples = ve.get(k, {}).get("samples", [])
        if not samples:
            continue
        print(
            f"\n-- canary extraction {k} samples ({min(n_canary, len(samples))} of {len(samples)}) --"
        )
        for s in samples[:n_canary]:
            tick = "✓" if s.get("exact_match") else "✗"
            print(f"  {tick} dup={s.get('dup'):>3}  expected={s.get('expected')}")
            print(f"    prefix:    {s.get('prefix')}")
            print(f"    generated: {truncate(s.get('generated', ''), 160)}")

    # -- Prompt-based extraction --
    pe = m.get("prompt_extraction", [])
    if pe:
        print(f"\n-- prompt-based probes --")
        for r in pe:
            print(f"  probe: {r.get('probe')!r}")
            print(f"    gen: {truncate(r.get('generation', ''), 180)}")
            print(f"    leaked: {r.get('leaked_canary')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="path to results/<run_id>")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="specific checkpoint label, e.g. 'step-1250' or 'final'. "
        "default: show all in order.",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=5,
        help="how many task-accuracy samples to show per checkpoint",
    )
    ap.add_argument(
        "--n_canary",
        type=int,
        default=3,
        help="how many canary-extraction samples to show per k-group",
    )
    args = ap.parse_args()

    path = Path(args.results_dir) / "metrics.json"
    if not path.exists():
        sys.exit(f"no metrics.json at {path}")
    data = json.loads(path.read_text())
    man = data.get("manifest", {})
    print(
        f"run: {man.get('run_id')}  condition: {man.get('condition')}  seed: {man.get('seed')}"
    )

    ckpts = data.get("checkpoints", {})
    if args.checkpoint is not None:
        if args.checkpoint not in ckpts:
            sys.exit(
                f"checkpoint {args.checkpoint!r} not in metrics. Available: {list(ckpts.keys())}"
            )
        show_checkpoint(
            args.checkpoint,
            ckpts[args.checkpoint],
            n_task=args.n,
            n_canary=args.n_canary,
        )
    else:
        for label, m in ckpts.items():
            show_checkpoint(label, m, n_task=args.n, n_canary=args.n_canary)


if __name__ == "__main__":
    main()
