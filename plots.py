"""Generate publication-quality plots from results/<run_id>/metrics.json files.

Phase 2 plots (under results/plots/):
  1. extraction_trajectory.png   — k=3 verbatim extraction rate over training steps
  2. extraction_by_dup.png       — extraction rate by duplication factor (k=1 & k=3, final)
  3. nll_trajectory.png          — train-canary NLL vs holdout NLL over training
  4. nll_by_dup.png              — NLL by duplication factor (final checkpoint)
  5. mia_trajectory.png          — MIA AUC (NLL & Min-k%) over training
  6. accuracy_trajectory.png     — task accuracy over training
  7. accuracy_vs_leakage.png     — the headline tradeoff plot

Phase 3 additions:
  8. difficulty_panel.png        — easy vs hard task family
  9. canary_format_panel.png     — ZK vs PII canary format
  10. mask_ablation.png          — extraction with/without --mask_prompt
  11. dp_tradeoff.png            — privacy/utility tradeoff (eps=8 vs no DP)
  12. dup_trajectory.png         — extraction vs step, faceted by duplication bucket
  13. roc_curves.png             — final-checkpoint MIA ROC, faceted by cell
  14. forest_gap.png             — forest plot of (cot - answer) NLL gap with bootstrap CIs

All trajectory plots use bootstrap 95% CI (replacing the Phase 2 ±1σ heuristic).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import roc_curve
except ImportError:
    roc_curve = None

try:
    from stats import bootstrap_ci
except ImportError:
    bootstrap_ci = None

COND_STYLE = {
    "answer": {"color": "#1f77b4", "label": "Answer-only", "ls": "-"},
    "cot":    {"color": "#d62728", "label": "CoT",         "ls": "--"},
}

RUN_RE = re.compile(
    r"^(?P<model>qwen3_\d+p\d+b)__(?P<cond>answer|cot)__seed(?P<seed>\d+)(?:__(?P<suffix>.+))?$"
)


def parse_run_id(run_id: str) -> Optional[dict]:
    m = RUN_RE.match(run_id)
    if not m:
        return None
    return {
        "model": m.group("model"),
        "condition": m.group("cond"),
        "seed": int(m.group("seed")),
        "suffix": m.group("suffix") or "",
    }


def _ci(values, conf=0.95):
    if bootstrap_ci is not None and len(values) >= 2:
        lo, hi = bootstrap_ci(values, np.mean, n_boot=2000, alpha=1 - conf)
        return lo, hi
    # Fallback: ±1 std
    m = np.mean(values)
    s = np.std(values)
    return m - s, m + s


def load_runs(results_dir: Path, suffix_filter: Optional[str] = "",
              model_filter: Optional[str] = None) -> List[dict]:
    """Load runs, filtering by suffix and (optionally) model.

    suffix_filter="" (default) -> only base Phase 2 runs (no suffix)
    suffix_filter="hard"       -> only __hard runs
    suffix_filter=None         -> all runs (incl. all suffixes)
    model_filter="qwen3_0p6b"  -> restrict to one model size
    """
    runs = []
    for p in sorted(results_dir.glob("*/metrics.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        # Use directory name (NOT manifest's run_id) for filtering — directories
        # like `qwen3_0p6b__cot__seed0_big` carry the suffix; the manifest's
        # run_id may not.
        dir_name = p.parent.name
        if any(x in dir_name for x in ("__smoke", "_big")):
            continue
        parsed = parse_run_id(dir_name)
        if parsed is None:
            continue
        if suffix_filter is not None and parsed["suffix"] != suffix_filter:
            continue
        if model_filter is not None and parsed["model"] != model_filter:
            continue
        d["_parsed"] = parsed
        runs.append(d)
    return runs


def resolve_step(label: str, total_steps: int) -> int:
    if label == "final":
        return total_steps
    if label.startswith("step-"):
        return int(label.split("-")[1])
    return -1


def gather_trajectories(runs: List[dict], extractor) -> Dict[str, Dict[int, List[float]]]:
    """Returns {condition: {step: [value_seed0, value_seed1, ...]}}."""
    out: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        man = r["manifest"]
        cond = man["condition"]
        total = man["total_steps"]
        for label, m in r["checkpoints"].items():
            step = resolve_step(label, total)
            if step < 0:
                continue
            v = extractor(m)
            if v is not None:
                out[cond][step].append(v)
    return dict(out)


def plot_trajectory_with_band(
    runs: List[dict],
    extractor,
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple = None,
) -> None:
    data = gather_trajectories(runs, extractor)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in ["answer", "cot"]:
        if cond not in data:
            continue
        style = COND_STYLE[cond]
        steps = sorted(data[cond].keys())
        means = [np.mean(data[cond][s]) for s in steps]
        stds  = [np.std(data[cond][s])  for s in steps]
        ax.plot(steps, means, marker="o", markersize=4, color=style["color"],
                ls=style["ls"], label=style["label"], linewidth=2)
        ax.fill_between(steps,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=style["color"], alpha=0.15)
    ax.set_xlabel("Optimizer step", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_extraction_by_dup(runs: List[dict], out_path: Path) -> None:
    """Grouped bar chart: extraction rate by duplication, k=1 and k=3."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, k_label, k_key in zip(axes, ["k=1 (3 groups to complete)", "k=3 (1 group to complete)"], ["k=1", "k=3"]):
        by_cond_dup: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in runs:
            cond = r["manifest"]["condition"]
            ckpts = r["checkpoints"]
            m = ckpts.get("final", list(ckpts.values())[-1])
            ext = m.get("verbatim_extraction", {}).get(k_key, {})
            for d, v in ext.get("by_dup", {}).items():
                by_cond_dup[cond][int(d)].append(v)
        dups = sorted({d for v in by_cond_dup.values() for d in v})
        x = np.arange(len(dups))
        width = 0.35
        for i, cond in enumerate(["answer", "cot"]):
            if cond not in by_cond_dup:
                continue
            means = [np.mean(by_cond_dup[cond].get(d, [0])) for d in dups]
            stds  = [np.std(by_cond_dup[cond].get(d, [0]))  for d in dups]
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                   color=COND_STYLE[cond]["color"], label=COND_STYLE[cond]["label"],
                   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}x" for d in dups])
        ax.set_xlabel("Canary duplication factor", fontsize=11)
        ax.set_title(f"Extraction rate ({k_label})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Verbatim extraction rate", fontsize=11)
    fig.suptitle("Extraction rate by canary duplication (final checkpoint)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_nll_trajectory(runs: List[dict], out_path: Path) -> None:
    """Train vs holdout NLL over training, per condition."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in ["answer", "cot"]:
        style = COND_STYLE[cond]
        for nll_key, nll_label, alpha in [("mean_nll_train", "train", 1.0), ("mean_nll_holdout", "holdout", 0.4)]:
            data = gather_trajectories(runs, lambda m, k=nll_key: m.get("mia", {}).get(k))
            if cond not in data:
                continue
            steps = sorted(data[cond].keys())
            means = [np.mean(data[cond][s]) for s in steps]
            stds  = [np.std(data[cond][s])  for s in steps]
            label_str = f"{style['label']} ({nll_label})"
            ax.plot(steps, means, marker="o" if nll_label == "train" else "s",
                    markersize=3, color=style["color"], ls=style["ls"],
                    label=label_str, linewidth=1.8, alpha=alpha)
            ax.fill_between(steps,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=style["color"], alpha=0.08)
    ax.set_xlabel("Optimizer step", fontsize=11)
    ax.set_ylabel("Mean NLL (nats/token)", fontsize=11)
    ax.set_title("Per-canary NLL: train canaries vs holdout", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_nll_by_dup(runs: List[dict], out_path: Path) -> None:
    """Bar chart: NLL by duplication factor at final checkpoint."""
    by_cond_dup: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        cond = r["manifest"]["condition"]
        ckpts = r["checkpoints"]
        m = ckpts.get("final", list(ckpts.values())[-1])
        nll_by_dup = m.get("mia", {}).get("nll_by_dup", {})
        for d, v in nll_by_dup.items():
            by_cond_dup[cond][int(d)].append(v)
    # Also add holdout mean
    holdout_vals: Dict[str, List[float]] = defaultdict(list)
    for r in runs:
        cond = r["manifest"]["condition"]
        ckpts = r["checkpoints"]
        m = ckpts.get("final", list(ckpts.values())[-1])
        holdout_vals[cond].append(m.get("mia", {}).get("mean_nll_holdout", 0))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    dups = sorted({d for v in by_cond_dup.values() for d in v})
    x = np.arange(len(dups) + 1)  # +1 for holdout
    width = 0.35
    for i, cond in enumerate(["answer", "cot"]):
        means = [np.mean(by_cond_dup[cond].get(d, [0])) for d in dups]
        stds  = [np.std(by_cond_dup[cond].get(d, [0]))  for d in dups]
        means.append(np.mean(holdout_vals[cond]))
        stds.append(np.std(holdout_vals[cond]))
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               color=COND_STYLE[cond]["color"], label=COND_STYLE[cond]["label"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}x" for d in dups] + ["holdout"])
    ax.set_xlabel("Canary duplication factor", fontsize=11)
    ax.set_ylabel("Mean NLL (nats/token)", fontsize=11)
    ax.set_title("Per-canary NLL by duplication (final checkpoint)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_accuracy_vs_leakage(runs: List[dict], out_path: Path) -> None:
    """Accuracy vs k=3 extraction rate, one point per checkpoint, per run."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in runs:
        cond = r["manifest"]["condition"]
        style = COND_STYLE[cond]
        total = r["manifest"]["total_steps"]
        xs, ys = [], []
        for label, m in r["checkpoints"].items():
            acc = m.get("task_accuracy", {}).get("accuracy")
            ext = m.get("verbatim_extraction", {}).get("k=3", {}).get("overall_em")
            if acc is None or ext is None:
                continue
            xs.append(ext)
            ys.append(acc)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=4, alpha=0.5,
                    color=style["color"], label=style["label"], linewidth=1.5)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=10)
    ax.set_xlabel("Verbatim canary extraction rate (k=3)", fontsize=11)
    ax.set_ylabel("Task accuracy", fontsize=11)
    ax.set_title("Accuracy vs. memorization leakage", fontsize=13)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# Phase 3 additions
# ============================================================================

def _final_metric(r, key_path):
    ckpts = r.get("checkpoints", {})
    m = ckpts.get("final") or list(ckpts.values())[-1] if ckpts else {}
    cur = m
    for k in key_path:
        if cur is None:
            return None
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def _by_condition_means(runs, key_path):
    """Returns dict cond -> list of per-seed final values."""
    out = defaultdict(list)
    for r in runs:
        cond = r["_parsed"]["condition"]
        v = _final_metric(r, key_path)
        if v is not None:
            out[cond].append(v)
    return out


def plot_panel_compare(
    base_runs, alt_runs, alt_label, out_path, title,
    metric_path=("verbatim_extraction", "k=3", "overall_em"),
    metric_label="Verbatim extraction (k=3)",
):
    """Compare two run groups (e.g. easy vs hard, ZK vs PII) on a final-checkpoint metric.

    Shows side-by-side bars with bootstrap 95% CI error bars per condition.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    metric2_path = ("task_accuracy", "accuracy")
    for ax, mpath, mlabel, mtitle in zip(
        axes,
        [metric_path, metric2_path],
        [metric_label, "Task accuracy"],
        [metric_label, "Task accuracy"],
    ):
        base_data = _by_condition_means(base_runs, mpath)
        alt_data = _by_condition_means(alt_runs, mpath)
        groups = [("Phase 2 (base)", base_data), (alt_label, alt_data)]
        x = np.arange(len(groups))
        width = 0.35
        for i, cond in enumerate(["answer", "cot"]):
            means, lows, highs = [], [], []
            for label, data in groups:
                v = data.get(cond, [])
                if v:
                    m = float(np.mean(v))
                    lo, hi = _ci(v) if len(v) > 1 else (m, m)
                    means.append(m); lows.append(m - lo); highs.append(hi - m)
                else:
                    means.append(0); lows.append(0); highs.append(0)
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=[lows, highs], capsize=4,
                   color=COND_STYLE[cond]["color"], label=COND_STYLE[cond]["label"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups])
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(mtitle, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_dp_tradeoff(base_runs, dp_runs_by_eps, out_path, title="Privacy/utility tradeoff under DP-SGD"):
    """Plot extraction & accuracy vs eps, with non-DP baseline as eps=infty.

    dp_runs_by_eps: dict eps_value -> list of runs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    items = sorted(dp_runs_by_eps.items())
    eps_vals = [e for e, _ in items] + [np.inf]
    eps_labels = [str(e) for e, _ in items] + ["∞ (no DP)"]
    x = np.arange(len(eps_vals))
    for ax, mpath, mlabel in zip(
        axes,
        [("verbatim_extraction", "k=3", "overall_em"), ("task_accuracy", "accuracy")],
        ["k=3 extraction rate", "Task accuracy"],
    ):
        for cond in ["answer", "cot"]:
            means, lows, highs = [], [], []
            for eps, runs in items:
                vals = [v for v in [_final_metric(r, mpath) for r in runs
                                    if r["_parsed"]["condition"] == cond] if v is not None]
                if vals:
                    m = float(np.mean(vals)); lo, hi = _ci(vals) if len(vals) > 1 else (m, m)
                else:
                    m, lo, hi = float("nan"), float("nan"), float("nan")
                means.append(m); lows.append(m - lo); highs.append(hi - m)
            base_vals = [v for v in [_final_metric(r, mpath) for r in base_runs
                                     if r["_parsed"]["condition"] == cond] if v is not None]
            if base_vals:
                bm = float(np.mean(base_vals))
                blo, bhi = _ci(base_vals) if len(base_vals) > 1 else (bm, bm)
            else:
                bm, blo, bhi = float("nan"), float("nan"), float("nan")
            means.append(bm); lows.append(bm - blo); highs.append(bhi - bm)
            ax.errorbar(x, means, yerr=[lows, highs], marker="o", capsize=4,
                        color=COND_STYLE[cond]["color"], ls=COND_STYLE[cond]["ls"],
                        label=COND_STYLE[cond]["label"], linewidth=2, markersize=6)
        ax.set_xticks(x); ax.set_xticklabels(eps_labels)
        ax.set_xlabel("Privacy budget ε", fontsize=11)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_dup_trajectory(runs, out_path, title="Extraction over training, by canary duplication"):
    """4-panel facet by duplication bucket {1, 4, 16, 64}; extraction (k=3) vs step."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    dups = [1, 4, 16, 64]
    for ax, dup in zip(axes.flat, dups):
        for cond in ["answer", "cot"]:
            traj = defaultdict(list)
            for r in runs:
                if r["_parsed"]["condition"] != cond:
                    continue
                total = r["manifest"]["total_steps"]
                for label, m in r["checkpoints"].items():
                    step = resolve_step(label, total)
                    if step < 0:
                        continue
                    by_dup = m.get("verbatim_extraction", {}).get("k=3", {}).get("by_dup", {})
                    v = by_dup.get(str(dup))
                    if v is not None:
                        traj[step].append(v)
            steps = sorted(traj.keys())
            if not steps:
                continue
            means = [np.mean(traj[s]) for s in steps]
            cis = [_ci(traj[s]) if len(traj[s]) > 1 else (m_, m_) for s, m_ in zip(steps, means)]
            lows = [c[0] for c in cis]; highs = [c[1] for c in cis]
            ax.plot(steps, means, marker="o", markersize=4, color=COND_STYLE[cond]["color"],
                    ls=COND_STYLE[cond]["ls"], label=COND_STYLE[cond]["label"], linewidth=2)
            ax.fill_between(steps, lows, highs, color=COND_STYLE[cond]["color"], alpha=0.18)
        ax.set_title(f"{dup}× duplication", fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Optimizer step", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("k=3 extraction rate", fontsize=11)
    axes[0, 0].legend(fontsize=10, loc="lower right")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_roc_curves(runs, out_path, title="MIA ROC curves (final checkpoint)"):
    """Final-checkpoint MIA ROC, faceted by condition. Needs per-canary score arrays."""
    if roc_curve is None:
        print(f"  skip {out_path} (sklearn not available)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, cond in zip(axes, ["answer", "cot"]):
        for r in runs:
            if r["_parsed"]["condition"] != cond:
                continue
            seed = r["_parsed"]["seed"]
            mia = (r.get("checkpoints", {}).get("final") or {}).get("mia", {})
            train_d = mia.get("nll_by_canary_id", {})
            hold_d = mia.get("nll_holdout_by_id", {})
            if not train_d or not hold_d:
                continue
            train_nll = list(train_d.values())
            hold_nll = list(hold_d.values())
            y_true = [1] * len(train_nll) + [0] * len(hold_nll)
            # Lower NLL = more likely member -> use -nll as score
            y_score = [-x for x in train_nll] + [-x for x in hold_nll]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, color=COND_STYLE[cond]["color"], alpha=0.6,
                    linewidth=1.5, label=f"seed {seed}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("False positive rate", fontsize=11)
        ax.set_title(f"{COND_STYLE[cond]['label']}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("True positive rate", fontsize=11)
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_forest_gap(stats_summary_path, out_path, title="(CoT − answer) NLL gap by experimental cell"):
    """Forest plot of the (cot - answer) NLL gap per cell, with bootstrap CIs."""
    if not Path(stats_summary_path).exists():
        print(f"  skip {out_path} ({stats_summary_path} not found)")
        return
    summary = json.loads(Path(stats_summary_path).read_text())
    cells = []
    for cell, r in summary.items():
        gap = r.get("nll_gap")
        if not gap:
            continue
        cells.append((cell, gap["diff_mean"], gap["diff_ci95"][0], gap["diff_ci95"][1]))
    if not cells:
        print(f"  skip {out_path} (no nll_gap entries)")
        return
    cells.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(cells) + 1)))
    y = np.arange(len(cells))
    means = [c[1] for c in cells]
    los = [c[1] - c[2] for c in cells]
    his = [c[3] - c[1] for c in cells]
    ax.errorbar(means, y, xerr=[los, his], fmt="o", capsize=5, color="#444444", ecolor="#999999", linewidth=2)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.axvspan(-0.1, 0.1, color="#cccccc", alpha=0.3, label="TOST equivalence band (±0.1)")
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in cells])
    ax.set_xlabel("(cot − answer) NLL gap (nats)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="results/plots")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2 plots use the 0.6B base runs only — the 1.7B runs would otherwise
    # pool into the same condition curves and mask the headline finding.
    runs = load_runs(results_dir, suffix_filter="", model_filter="qwen3_0p6b")
    if not runs:
        print(f"No runs found in {results_dir}")
        return
    print(f"Loaded {len(runs)} 0.6B base runs (no suffix)")

    # 1. Extraction trajectory (k=3)
    plot_trajectory_with_band(
        runs,
        extractor=lambda m: m.get("verbatim_extraction", {}).get("k=3", {}).get("overall_em"),
        title="Verbatim canary extraction over training (k=3 prefix)",
        ylabel="Exact-match extraction rate",
        out_path=out_dir / "extraction_trajectory.png",
        ylim=(-0.02, 1.02),
    )

    # 2. Extraction by duplication (k=1 and k=3)
    plot_extraction_by_dup(runs, out_dir / "extraction_by_dup.png")

    # 3. NLL trajectory (train vs holdout)
    plot_nll_trajectory(runs, out_dir / "nll_trajectory.png")

    # 4. NLL by duplication factor
    plot_nll_by_dup(runs, out_dir / "nll_by_dup.png")

    # 5. MIA AUC trajectory
    plot_trajectory_with_band(
        runs,
        extractor=lambda m: m.get("mia", {}).get("auc_mink20"),
        title="Membership inference AUC (Min-k%, k=20%)",
        ylabel="MIA AUC",
        out_path=out_dir / "mia_trajectory.png",
        ylim=(0.4, 1.02),
    )

    # 6. Accuracy trajectory
    plot_trajectory_with_band(
        runs,
        extractor=lambda m: m.get("task_accuracy", {}).get("accuracy"),
        title="Held-out task accuracy",
        ylabel="Accuracy",
        out_path=out_dir / "accuracy_trajectory.png",
        ylim=(0.6, 1.02),
    )

    # 7. Accuracy vs leakage (k=3)
    plot_accuracy_vs_leakage(runs, out_dir / "accuracy_vs_leakage.png")

    # ----- Phase 3 additions -----
    # 8. Per-duplication trajectory (existing data, new visualization)
    plot_dup_trajectory(runs, out_dir / "dup_trajectory.png")

    # 9. ROC curves (only when per-canary data is present)
    plot_roc_curves(runs, out_dir / "roc_curves.png")

    # 10. Hard vs easy panel
    hard_runs = load_runs(results_dir, suffix_filter="hard")
    if hard_runs:
        plot_panel_compare(
            base_runs=runs, alt_runs=hard_runs,
            alt_label="Hard (Phase 3)",
            out_path=out_dir / "difficulty_panel.png",
            title="Effect of task difficulty on extraction & accuracy",
        )

    # 11. ZK vs PII canary format
    pii_runs = load_runs(results_dir, suffix_filter="pii")
    if pii_runs:
        plot_panel_compare(
            base_runs=runs, alt_runs=pii_runs,
            alt_label="PII canaries",
            out_path=out_dir / "canary_format_panel.png",
            title="Effect of canary format (high-entropy ZK vs human-readable PII)",
        )

    # 12. Mask-prompt ablation
    mask_runs = load_runs(results_dir, suffix_filter="maskprompt")
    if mask_runs:
        plot_panel_compare(
            base_runs=runs, alt_runs=mask_runs,
            alt_label="Mask prompt (ablation)",
            out_path=out_dir / "mask_ablation.png",
            title="Prompt-masking ablation: extraction collapses without prompt loss",
        )

    # 13. DP-SGD privacy/utility tradeoff
    dp_eps_runs = {}
    for eps in [2, 4, 8, 16, 32]:
        r = load_runs(results_dir, suffix_filter=f"dp_eps{eps}")
        if r:
            dp_eps_runs[eps] = r
    if dp_eps_runs:
        plot_dp_tradeoff(runs, dp_eps_runs, out_dir / "dp_tradeoff.png")

    # 14. Forest plot of CoT-vs-answer NLL gap across cells (uses stats_summary.json)
    plot_forest_gap(results_dir / "stats_summary.json", out_dir / "forest_gap.png")


if __name__ == "__main__":
    main()
