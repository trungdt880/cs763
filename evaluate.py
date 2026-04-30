"""Per-checkpoint evaluation: task accuracy + memorization attacks.

Reads a run's manifest and canary pool from `results/<run_id>/`, iterates over
all milestone checkpoints + the final checkpoint, runs every metric, and
appends the results to `results/<run_id>/metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks import (
    extract_canaries,
    mean_nll,
    mia_auc,
    min_k_prob_score,
    per_canary_token_logprobs,
    prompt_based_extraction,
)


# ----------------------------------------------------------------------------
# Task accuracy
# ----------------------------------------------------------------------------

ANSWER_RE = re.compile(r"-?\d+")
COT_FINAL_RE = re.compile(r"####\s*(-?\d+)")


def parse_predicted_int(text: str, condition: str) -> int | None:
    if condition == "cot":
        m = COT_FINAL_RE.search(text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        # fall through: maybe the model didn't follow the #### convention
    nums = ANSWER_RE.findall(text)
    if not nums:
        return None
    try:
        return int(nums[-1] if condition == "cot" else nums[0])
    except ValueError:
        return None


@torch.inference_mode()
def task_accuracy(
    model,
    tokenizer,
    test_examples: List[dict],
    condition: str,
    batch_size: int = 64,
    max_new_tokens: int = 160,
    device: str = "cuda",
    n_samples_to_save: int = 20,
) -> dict:
    """Greedy-decode every test example, compute accuracy, and persist the first
    `n_samples_to_save` (prompt, gen, pred, gold, correct) tuples so you can
    inspect what the model actually produced at every checkpoint."""
    model.eval()
    correct = 0
    total = 0
    samples: list = []
    for start in range(0, len(test_examples), batch_size):
        batch = test_examples[start:start + batch_size]
        prompts = [ex["prompt"] for ex in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen_max = 8 if condition == "answer" else max_new_tokens
        out = model.generate(
            **enc,
            max_new_tokens=gen_max,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        gen_only = out[:, enc["input_ids"].shape[1]:]
        gen_texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        for ex, gen in zip(batch, gen_texts):
            pred = parse_predicted_int(gen, condition)
            is_correct = (pred == ex["gold"])
            total += 1
            if is_correct:
                correct += 1
            if len(samples) < n_samples_to_save:
                samples.append({
                    "template": ex["template"],
                    "prompt": ex["prompt"],
                    "generation": gen,
                    "predicted": pred,
                    "gold": ex["gold"],
                    "correct": bool(is_correct),
                })
    return {
        "accuracy": correct / max(1, total),
        "n": total,
        "samples": samples,
    }


# ----------------------------------------------------------------------------
# Per-checkpoint pipeline
# ----------------------------------------------------------------------------

def find_checkpoints(checkpoints_dir: Path, milestone_steps: List[int]) -> List[tuple]:
    """Return [(label, path), ...] for milestone + final checkpoints."""
    out = []
    for ms in milestone_steps:
        p = checkpoints_dir / f"checkpoint-step-{ms}"
        if p.exists():
            out.append((f"step-{ms}", p))
    final = checkpoints_dir / "checkpoint-final"
    if final.exists():
        out.append(("final", final))
    return out


def evaluate_checkpoint(
    ckpt_path: Path,
    canary_pool: dict,
    test_examples: List[dict],
    condition: str,
    device: str = "cuda",
) -> dict:
    print(f"  Loading {ckpt_path.name} ...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    train_canaries = canary_pool["train"]
    holdout_canaries = canary_pool["holdout"]

    metrics: dict = {}

    # ---- 1. Task accuracy ----
    print("    task accuracy...")
    metrics["task_accuracy"] = task_accuracy(model, tokenizer, test_examples, condition, device=device)

    # ---- 2. Verbatim extraction at all prefix lengths ----
    print("    verbatim extraction...")
    extraction = {}
    for k in (0, 1, 2, 3):
        per_canary = extract_canaries(model, tokenizer, train_canaries, k_groups=k, device=device)
        # break down by duplication factor
        by_dup_hits: Dict[int, List[bool]] = {}
        by_dup_records: Dict[int, list] = {}
        per_id: Dict[str, bool] = {}
        for r, c in zip(per_canary, train_canaries):
            by_dup_hits.setdefault(c["dup"], []).append(r["exact_match"])
            per_id[c["text"]] = bool(r["exact_match"])
            # Keep up to 5 per-canary samples per duplication bucket so we can
            # eyeball what the model actually emitted after the canary prefix.
            recs = by_dup_records.setdefault(c["dup"], [])
            if len(recs) < 5:
                recs.append({
                    "dup": c["dup"],
                    "prefix": r["prefix"],
                    "expected": r["expected"],
                    "generated": r["generated"],
                    "exact_match": r["exact_match"],
                })
        # Flatten samples to a single list ordered by dup descending.
        samples: list = []
        for d in sorted(by_dup_records.keys(), reverse=True):
            samples.extend(by_dup_records[d])
        extraction[f"k={k}"] = {
            "overall_em": sum(r["exact_match"] for r in per_canary) / len(per_canary),
            "by_dup": {str(d): sum(v) / len(v) for d, v in by_dup_hits.items()},
            "extracted_by_canary_id": per_id,
            "samples": samples,
        }
    metrics["verbatim_extraction"] = extraction

    # ---- 3. Per-canary loss + Min-k% MIA ----
    print("    MIA scoring...")
    train_lps = per_canary_token_logprobs(model, tokenizer, [c["text"] for c in train_canaries], device=device)
    hold_lps = per_canary_token_logprobs(model, tokenizer, [c["text"] for c in holdout_canaries], device=device)
    train_nll = mean_nll(train_lps)
    hold_nll = mean_nll(hold_lps)
    train_mink = min_k_prob_score(train_lps, k_pct=20.0)
    hold_mink = min_k_prob_score(hold_lps, k_pct=20.0)
    metrics["mia"] = {
        "mean_nll_train":   sum(train_nll) / len(train_nll),
        "mean_nll_holdout": sum(hold_nll)  / len(hold_nll),
        "auc_nll":   mia_auc([-x for x in train_nll], [-x for x in hold_nll]),
        "auc_mink20": mia_auc(train_mink, hold_mink),
        # Per-canary scores keyed by canary text. Enables paired tests across
        # conditions (same canaries, same seed) and ROC-curve plots.
        "nll_by_canary_id":     {c["text"]: float(n) for c, n in zip(train_canaries, train_nll)},
        "nll_holdout_by_id":    {c["text"]: float(n) for c, n in zip(holdout_canaries, hold_nll)},
        "mink20_by_canary_id":  {c["text"]: float(s) for c, s in zip(train_canaries, train_mink)},
        "mink20_holdout_by_id": {c["text"]: float(s) for c, s in zip(holdout_canaries, hold_mink)},
    }

    # Per-bucket NLL means
    by_dup_nll: Dict[int, List[float]] = {}
    for nll, c in zip(train_nll, train_canaries):
        by_dup_nll.setdefault(c["dup"], []).append(nll)
    metrics["mia"]["nll_by_dup"] = {str(d): sum(v)/len(v) for d, v in by_dup_nll.items()}

    # ---- 4. Prompt-based qualitative extraction ----
    print("    prompt extraction...")
    metrics["prompt_extraction"] = prompt_based_extraction(model, tokenizer, train_canaries, device=device)

    # Cleanup GPU memory.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="results/<run_id> directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=None,
                    help="Evaluate only this checkpoint label (e.g. 'step-250', 'final'). "
                         "When set, writes to results_dir/_shards/<label>.json instead of metrics.json, "
                         "so multiple instances can run in parallel across GPUs.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    manifest = json.loads((results_dir / "manifest.json").read_text())
    canary_pool = json.loads((results_dir / "canary_pool.json").read_text())
    test_examples = json.loads((results_dir / "test_examples.json").read_text())

    checkpoints_dir = Path(manifest["checkpoints_dir"])
    ckpts = find_checkpoints(checkpoints_dir, manifest["milestone_steps"])
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {checkpoints_dir}")

    # Single-checkpoint shard mode (for parallel eval across GPUs).
    if args.checkpoint is not None:
        match = [(label, path) for label, path in ckpts if label == args.checkpoint]
        if not match:
            raise RuntimeError(f"Checkpoint label {args.checkpoint!r} not found. Available: {[l for l,_ in ckpts]}")
        label, path = match[0]
        print(f"\n=== {label} (shard mode) ===")
        m = evaluate_checkpoint(
            ckpt_path=path,
            canary_pool=canary_pool,
            test_examples=test_examples,
            condition=manifest["condition"],
            device=args.device,
        )
        shard_dir = results_dir / "_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        with open(shard_dir / f"{label}.json", "w") as f:
            json.dump(m, f, indent=2)
        print(f"Wrote {shard_dir / f'{label}.json'}")
        return

    # Sequential mode: eval every checkpoint into a single metrics.json.
    all_metrics: Dict[str, dict] = {}
    for label, path in ckpts:
        print(f"\n=== {label} ===")
        all_metrics[label] = evaluate_checkpoint(
            ckpt_path=path,
            canary_pool=canary_pool,
            test_examples=test_examples,
            condition=manifest["condition"],
            device=args.device,
        )
        # Persist after every checkpoint so partial progress isn't lost.
        with open(results_dir / "metrics.json", "w") as f:
            json.dump({"manifest": manifest, "checkpoints": all_metrics}, f, indent=2)

    print(f"\nAll done. Wrote {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
