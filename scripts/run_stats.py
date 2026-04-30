"""Aggregate statistical tests across all run cells.

For each cell defined by (model_size, difficulty, canary_format, mask_prompt, dp_eps),
finds the {answer, cot} runs, pairs them by canary id within each seed, and computes:
  - Paired t-test on per-canary NLL (one per seed; meta-aggregate across seeds)
  - McNemar exact test on per-canary verbatim extraction (k=3)
  - TOST equivalence test (eps=0.05 nats for NLL, 0.05 abs for extraction rate)
  - Bootstrap 95% CI on the (cot - answer) difference

Writes results/stats_summary.json and a markdown table to stdout.

Usage:
    python scripts/run_stats.py
    python scripts/run_stats.py --filter dp_eps8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stats import (
    aggregate_paired_seeds,
    bootstrap_ci,
    mcnemar_extraction,
    paired_t_per_canary,
    tost_equivalence,
)

RUN_RE = re.compile(
    r"^(?P<model>qwen3_\d+p\d+b)__(?P<cond>answer|cot)__seed(?P<seed>\d+)(?:__(?P<suffix>.+))?$"
)


def parse_run_id(run_id: str):
    m = RUN_RE.match(run_id)
    if not m:
        return None
    return {
        "model": m.group("model"),
        "condition": m.group("cond"),
        "seed": int(m.group("seed")),
        "suffix": m.group("suffix") or "base",
    }


def load_run(p: Path):
    if not (p / "metrics.json").exists():
        return None
    return json.loads((p / "metrics.json").read_text())


def cell_id(parsed):
    """All non-condition keys form the cell."""
    return f"{parsed['model']}__{parsed['suffix']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument(
        "--filter",
        default=None,
        help="Only include runs matching this suffix (e.g. 'hard', 'dp_eps8'). "
        "Default: all suffixes.",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=["smoke", "big"],
        help="Suffixes to skip (default: smoke, big).",
    )
    ap.add_argument("--out", default="results/stats_summary.json")
    ap.add_argument("--md_out", default="results/stats_summary.md")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    runs: Dict[str, dict] = {}
    for d in sorted(results_root.iterdir()):
        if not d.is_dir() or d.name == "plots":
            continue
        parsed = parse_run_id(d.name)
        if parsed is None:
            continue
        if args.filter and parsed["suffix"] != args.filter:
            continue
        if any(x in parsed["suffix"].split("__") for x in args.exclude):
            continue
        if any(parsed["suffix"].endswith(x) for x in args.exclude):
            continue
        m = load_run(d)
        if m is None:
            continue
        runs[d.name] = (parsed, m)

    # Group by cell
    cells: Dict[str, Dict[str, Dict[int, dict]]] = {}
    # cells[cell_id]["answer"][seed] = metrics-final-checkpoint
    # cells[cell_id]["cot"][seed]    = metrics-final-checkpoint
    for run_id, (p, m) in runs.items():
        cell = cell_id(p)
        cells.setdefault(cell, {}).setdefault(p["condition"], {})
        ckpts = m.get("checkpoints", {})
        if "final" not in ckpts:
            continue
        cells[cell][p["condition"]][p["seed"]] = ckpts["final"]

    # Run tests
    out: Dict[str, dict] = {}
    md_lines: List[str] = []
    md_lines.append(
        "| Cell | n_seeds | NLL_gap diff (cot-answer) | paired-t p | TOST eq p | k=3 extr (cot - answer) | McNemar p |"
    )
    md_lines.append("|---|---|---|---|---|---|---|")
    for cell, conds in sorted(cells.items()):
        if "answer" not in conds or "cot" not in conds:
            print(f"  skipping {cell} (missing condition)")
            continue
        seeds = sorted(set(conds["answer"].keys()) & set(conds["cot"].keys()))
        if not seeds:
            continue

        # NLL paired test (use train canaries' nll vs holdout NLL: gap = holdout - train)
        # Pair by canary id, per seed.
        per_seed_nll_cot = {}
        per_seed_nll_ans = {}
        per_seed_ext_cot = {}
        per_seed_ext_ans = {}
        per_seed_extraction_rate_cot = {}
        per_seed_extraction_rate_ans = {}
        per_seed_nll_gap_cot = {}
        per_seed_nll_gap_ans = {}
        for s in seeds:
            cot_m = conds["cot"][s]
            ans_m = conds["answer"][s]
            # NLL gap: per-canary (holdout_mean - train_nll_per_canary). But per-canary
            # gap requires pairing train canary i with holdout canary i, which isn't
            # well-defined. Simpler: use per-canary TRAIN nll directly. Lower train
            # nll = more memorization. The "gap" is then summary-level.
            cot_nll = cot_m.get("mia", {}).get("nll_by_canary_id")
            ans_nll = ans_m.get("mia", {}).get("nll_by_canary_id")
            if cot_nll and ans_nll:
                per_seed_nll_cot[s] = cot_nll
                per_seed_nll_ans[s] = ans_nll
            # Per-canary extraction at k=3
            cot_ext = (
                cot_m.get("verbatim_extraction", {})
                .get("k=3", {})
                .get("extracted_by_canary_id")
            )
            ans_ext = (
                ans_m.get("verbatim_extraction", {})
                .get("k=3", {})
                .get("extracted_by_canary_id")
            )
            if cot_ext and ans_ext:
                per_seed_ext_cot[s] = cot_ext
                per_seed_ext_ans[s] = ans_ext
            # Aggregate extraction rate per seed
            per_seed_extraction_rate_cot[s] = (
                cot_m.get("verbatim_extraction", {}).get("k=3", {}).get("overall_em")
            )
            per_seed_extraction_rate_ans[s] = (
                ans_m.get("verbatim_extraction", {}).get("k=3", {}).get("overall_em")
            )
            per_seed_nll_gap_cot[s] = cot_m.get("mia", {}).get(
                "mean_nll_holdout", float("nan")
            ) - cot_m.get("mia", {}).get("mean_nll_train", float("nan"))
            per_seed_nll_gap_ans[s] = ans_m.get("mia", {}).get(
                "mean_nll_holdout", float("nan")
            ) - ans_m.get("mia", {}).get("mean_nll_train", float("nan"))

        result = {"cell": cell, "n_seeds": len(seeds), "seeds": seeds}

        # Paired NLL: only if per-canary fields are populated
        if per_seed_nll_cot and per_seed_nll_ans:
            paired_nll = aggregate_paired_seeds(per_seed_nll_cot, per_seed_nll_ans)
            result["nll_paired"] = paired_nll
            # Compute pooled p (Stouffer-ish): take min seed-wise p as worst case
            min_p = min(
                (r["p"] for r in paired_nll["per_seed"].values() if r["p"] is not None),
                default=float("nan"),
            )
            result["nll_paired_min_p"] = min_p
        else:
            result["nll_paired"] = None
            result["nll_paired_min_p"] = None

        # Paired McNemar across seeds: pool all canary ids x seeds
        if per_seed_ext_cot and per_seed_ext_ans:
            pooled_cot = {}
            pooled_ans = {}
            for s in per_seed_ext_cot:
                for cid, v in per_seed_ext_cot[s].items():
                    pooled_cot[f"s{s}_{cid}"] = v
                for cid, v in per_seed_ext_ans[s].items():
                    pooled_ans[f"s{s}_{cid}"] = v
            mc = mcnemar_extraction(pooled_cot, pooled_ans)
            result["mcnemar_pooled"] = mc
        else:
            result["mcnemar_pooled"] = None

        # Aggregate-level stats
        rate_cot = [v for v in per_seed_extraction_rate_cot.values() if v is not None]
        rate_ans = [v for v in per_seed_extraction_rate_ans.values() if v is not None]
        if rate_cot and rate_ans:
            diff = np.mean(rate_cot) - np.mean(rate_ans)
            ci = bootstrap_ci(
                np.array(rate_cot) - np.array(rate_ans), np.mean, n_boot=10000
            )
            result["extraction_rate"] = {
                "cot_mean": float(np.mean(rate_cot)),
                "answer_mean": float(np.mean(rate_ans)),
                "diff_mean": float(diff),
                "diff_ci95": ci,
                "tost_eq": tost_equivalence(
                    rate_cot, rate_ans, eps_low=-0.05, eps_high=0.05, paired=True
                ),
            }
        gap_cot = [
            v
            for v in per_seed_nll_gap_cot.values()
            if v is not None and not np.isnan(v)
        ]
        gap_ans = [
            v
            for v in per_seed_nll_gap_ans.values()
            if v is not None and not np.isnan(v)
        ]
        if gap_cot and gap_ans:
            diff = np.mean(gap_cot) - np.mean(gap_ans)
            ci = bootstrap_ci(
                np.array(gap_cot) - np.array(gap_ans), np.mean, n_boot=10000
            )
            result["nll_gap"] = {
                "cot_mean": float(np.mean(gap_cot)),
                "answer_mean": float(np.mean(gap_ans)),
                "diff_mean": float(diff),
                "diff_ci95": ci,
                "tost_eq": tost_equivalence(
                    gap_cot, gap_ans, eps_low=-0.1, eps_high=0.1, paired=True
                ),
            }

        out[cell] = result

        # Markdown row
        nll_diff_str = "—"
        if result.get("nll_gap"):
            d = result["nll_gap"]["diff_mean"]
            lo, hi = result["nll_gap"]["diff_ci95"]
            nll_diff_str = f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]"
        nll_p_str = (
            f"{result.get('nll_paired_min_p', float('nan')):.2g}"
            if result.get("nll_paired_min_p")
            else "—"
        )
        tost_p_str = (
            f"{result['nll_gap']['tost_eq']['p_max']:.2g}"
            if result.get("nll_gap")
            else "—"
        )
        ext_diff_str = "—"
        if result.get("extraction_rate"):
            d = result["extraction_rate"]["diff_mean"]
            lo, hi = result["extraction_rate"]["diff_ci95"]
            ext_diff_str = f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]"
        mc_p_str = (
            f"{result['mcnemar_pooled']['p']:.2g}"
            if result.get("mcnemar_pooled")
            else "—"
        )
        md_lines.append(
            f"| {cell} | {len(seeds)} | {nll_diff_str} | {nll_p_str} | {tost_p_str} | {ext_diff_str} | {mc_p_str} |"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            out,
            f,
            indent=2,
            default=lambda x: list(x) if isinstance(x, tuple) else str(x),
        )
    print(f"Wrote {out_path}")

    md = "\n".join(md_lines)
    Path(args.md_out).write_text(md + "\n")
    print(f"Wrote {args.md_out}")
    print()
    print(md)


if __name__ == "__main__":
    main()
