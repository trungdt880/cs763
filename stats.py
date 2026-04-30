"""Statistical helpers for the CoT-vs-leakage final report.

Replaces "overlapping +/- 1 sigma bands" rhetoric with formal tests:

  - paired_t_per_canary: paired t-test on per-canary NLL across conditions.
    With n=200 canaries paired by id within a seed, this has real power.
  - mcnemar_extraction: paired binary outcome (extracted yes/no) per canary.
  - bootstrap_ci: nonparametric 95% CI for any aggregate statistic.
  - tost_equivalence: two one-sided tests for the equivalence claim
    "|mean_a - mean_b| < epsilon", which is the right tool to FORMALLY support
    a null finding (a non-significant difference test does NOT prove equivalence).
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from scipy import stats as scistats


def paired_t_per_canary(
    a_by_id: Dict[str, float],
    b_by_id: Dict[str, float],
) -> dict:
    """Paired t-test on per-canary scalar scores (e.g. NLL) matched by id.

    Returns a dict with t, p, df, mean_diff, ci95, cohen_dz, n_pairs.
    """
    common = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
    a = np.array([a_by_id[k] for k in common], dtype=float)
    b = np.array([b_by_id[k] for k in common], dtype=float)
    diff = a - b
    n = len(diff)
    if n < 2:
        return {"t": float("nan"), "p": float("nan"), "df": n - 1,
                "mean_diff": float(np.mean(diff)) if n else float("nan"),
                "ci95": (float("nan"), float("nan")),
                "cohen_dz": float("nan"), "n_pairs": n}
    t, p = scistats.ttest_rel(a, b)
    sd = float(np.std(diff, ddof=1))
    se = sd / np.sqrt(n)
    crit = scistats.t.ppf(0.975, n - 1)
    md = float(np.mean(diff))
    ci = (md - crit * se, md + crit * se)
    cohen_dz = md / sd if sd > 0 else float("nan")
    return {
        "t": float(t),
        "p": float(p),
        "df": n - 1,
        "mean_diff": md,
        "ci95": (float(ci[0]), float(ci[1])),
        "cohen_dz": float(cohen_dz),
        "n_pairs": n,
    }


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for `stat` over `values`."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        boot[i] = stat(arr[idx[i]])
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return (lo, hi)


def tost_equivalence(
    a: Sequence[float],
    b: Sequence[float],
    eps_low: float,
    eps_high: float,
    paired: bool = False,
) -> dict:
    """Two one-sided tests (TOST) for equivalence of mean_a and mean_b.

    H0: mean_a - mean_b <= eps_low OR mean_a - mean_b >= eps_high
    H1 (equivalence): eps_low < mean_a - mean_b < eps_high

    Returns dict with p_lower, p_upper, p_max (decision p), reject_h0 (bool),
    mean_diff. If paired=True, treats a and b as matched samples.
    """
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if paired:
        if len(a) != len(b):
            raise ValueError("paired TOST requires equal-length samples")
        diff = a - b
        md = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1))
        n = len(diff)
        se = sd / np.sqrt(n)
        df = n - 1
        t_lower = (md - eps_low) / se
        t_upper = (md - eps_high) / se
        p_lower = 1 - scistats.t.cdf(t_lower, df)
        p_upper = scistats.t.cdf(t_upper, df)
    else:
        n_a, n_b = len(a), len(b)
        m_a, m_b = float(np.mean(a)), float(np.mean(b))
        v_a = float(np.var(a, ddof=1))
        v_b = float(np.var(b, ddof=1))
        se = np.sqrt(v_a / n_a + v_b / n_b)
        # Welch df
        df = (v_a / n_a + v_b / n_b) ** 2 / (
            (v_a / n_a) ** 2 / (n_a - 1) + (v_b / n_b) ** 2 / (n_b - 1)
        )
        md = m_a - m_b
        t_lower = (md - eps_low) / se
        t_upper = (md - eps_high) / se
        p_lower = 1 - scistats.t.cdf(t_lower, df)
        p_upper = scistats.t.cdf(t_upper, df)
    p_max = float(max(p_lower, p_upper))
    return {
        "mean_diff": md,
        "eps_low": eps_low,
        "eps_high": eps_high,
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_max": p_max,
        "reject_h0": bool(p_max < 0.05),
    }


def mcnemar_extraction(
    a_by_id: Dict[str, bool],
    b_by_id: Dict[str, bool],
) -> dict:
    """Exact McNemar test on paired binary outcomes (extracted yes/no).

    For each canary id present in both sets, counts the 2x2 table.
    Returns chi2, p (exact binomial of the discordant pairs), n_pairs,
    n_a_only (a=1, b=0), n_b_only (a=0, b=1).
    """
    common = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
    n_a_only = 0
    n_b_only = 0
    n_both = 0
    n_neither = 0
    for k in common:
        ya, yb = bool(a_by_id[k]), bool(b_by_id[k])
        if ya and yb:
            n_both += 1
        elif ya and not yb:
            n_a_only += 1
        elif yb and not ya:
            n_b_only += 1
        else:
            n_neither += 1
    n_disc = n_a_only + n_b_only
    if n_disc == 0:
        return {"chi2": 0.0, "p": 1.0, "n_pairs": len(common),
                "n_a_only": n_a_only, "n_b_only": n_b_only,
                "n_both": n_both, "n_neither": n_neither}
    # Exact binomial: under H0 each discordant pair is 50/50.
    k = min(n_a_only, n_b_only)
    # Two-sided tail
    p = 2 * scistats.binom.cdf(k, n_disc, 0.5)
    p = float(min(p, 1.0))
    # Continuity-corrected chi2 for reference
    chi2 = (abs(n_a_only - n_b_only) - 1) ** 2 / n_disc
    return {
        "chi2": float(chi2),
        "p": p,
        "n_pairs": len(common),
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "n_both": n_both,
        "n_neither": n_neither,
    }


def aggregate_paired_seeds(
    nll_a_by_seed: Dict[int, Dict[str, float]],
    nll_b_by_seed: Dict[int, Dict[str, float]],
) -> dict:
    """Per-seed paired tests then meta-aggregate across seeds.

    Returns: per_seed (dict seed -> paired_t result), meta_mean_diff,
    meta_ci95 (across seed mean diffs), n_seeds.
    """
    per_seed = {}
    diffs = []
    for s in sorted(nll_a_by_seed.keys() & nll_b_by_seed.keys()):
        r = paired_t_per_canary(nll_a_by_seed[s], nll_b_by_seed[s])
        per_seed[s] = r
        diffs.append(r["mean_diff"])
    if not diffs:
        return {"per_seed": per_seed, "meta_mean_diff": float("nan"),
                "meta_ci95": (float("nan"), float("nan")), "n_seeds": 0}
    diffs = np.array(diffs)
    lo, hi = bootstrap_ci(diffs, np.mean, n_boot=10000)
    return {
        "per_seed": per_seed,
        "meta_mean_diff": float(np.mean(diffs)),
        "meta_ci95": (lo, hi),
        "n_seeds": len(diffs),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    ids = [f"c{i}" for i in range(n)]
    nll_a = {k: float(v) for k, v in zip(ids, rng.normal(2.5, 0.3, n))}
    nll_b = {k: float(v + rng.normal(0.05, 0.1)) for k, v in nll_a.items()}
    print("paired t:", paired_t_per_canary(nll_a, nll_b))
    print("TOST eq (eps=0.2):", tost_equivalence(
        list(nll_a.values()), list(nll_b.values()),
        eps_low=-0.2, eps_high=0.2, paired=True
    ))
    extracted_a = {k: rng.random() < 0.7 for k in ids}
    extracted_b = {k: rng.random() < 0.7 for k in ids}
    print("McNemar:", mcnemar_extraction(extracted_a, extracted_b))
    print("Bootstrap CI of mean(nll_a):", bootstrap_ci(list(nll_a.values())))
