"""DP-SGD fine-tuning for the CoT-vs-leakage experiment.

Same dataset/canary pipeline as train.py, but the inner optimization is
differentially private via Opacus's PrivacyEngine:

  - Per-example gradients are clipped to `--max_grad_norm` (default 1.0).
  - Gaussian noise is added to the aggregated gradient with scale
    `noise_multiplier * max_grad_norm / batch_size`.
  - The (epsilon, delta) accountant is RDP-based, computed from
    sample_rate, noise_multiplier, and number of steps.

We use `make_private_with_epsilon`: given target epsilon (e.g. 8),
delta (1e-5), sample_rate, and steps, Opacus solves for the noise
multiplier that achieves that epsilon under composition.

Memory note: per-example gradient computation costs ~B times the parameter
count in extra memory. For a 0.6B model on a 24GB 3090 with batch 4, this
is feasible but tight; we use `BatchMemoryManager` to virtually batch up to
the desired effective batch size.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)

from canaries import build_canary_pool
from data import build_train_set, build_test_set
from train import SFTDataset, PadCollator, set_seed, _assert_base_model

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.utils.batch_memory_manager import BatchMemoryManager
except ImportError as e:
    raise SystemExit(
        f"opacus is required for train_dp.py but is not installed: {e}\n"
        "Install with: .venv/bin/pip install opacus"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True, choices=["answer", "cot"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per_device_batch", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--mask_prompt", action="store_true")
    ap.add_argument("--difficulty", default="easy", choices=["easy", "hard", "mixed"])
    ap.add_argument("--canary_format", default="zk", choices=["zk", "pii"])
    ap.add_argument("--run_suffix", default="dp_eps8")
    # DP-specific
    ap.add_argument("--target_epsilon", type=float, default=8.0,
                    help="Target privacy budget; noise_multiplier is solved to achieve this.")
    ap.add_argument("--target_delta", type=float, default=1e-5)
    ap.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="Per-example gradient clip norm.")
    ap.add_argument("--max_physical_batch", type=int, default=4,
                    help="Microbatch size for BatchMemoryManager. Lower if OOM.")
    ap.add_argument("--num_epochs", type=int, default=None,
                    help="Override training epochs (default: configs.yaml training.num_epochs). "
                         "DP is ~10x slower than non-DP; smaller epoch counts are often pragmatic.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg = cfg["models"][args.model]
    tcfg = cfg["training"]
    dcfg = cfg["dataset"]
    ccfg = cfg["canaries"]

    _assert_base_model(mcfg["hf_name"])
    set_seed(args.seed)

    run_id = f"{args.model}__{args.condition}__seed{args.seed}"
    if args.run_suffix:
        run_id += f"__{args.run_suffix}"
    if args.smoke:
        run_id += "__smoke"
    print(f"\n=== DP RUN: {run_id} (eps={args.target_epsilon}, delta={args.target_delta}) ===")

    out_root = Path(cfg["output"]["checkpoints_dir"]) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    meta_dir = Path(cfg["output"]["results_dir"]) / run_id
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    n_train = 200 if args.smoke else dcfg["n_train_problems"]
    n_test = 50 if args.smoke else dcfg["n_test_problems"]
    duplication = ccfg["duplication_buckets"]
    if args.smoke:
        duplication = {1: 5, 4: 5, 16: 5}
    pool = build_canary_pool(
        duplication_buckets={int(k): int(v) for k, v in duplication.items()},
        n_holdout=ccfg["n_holdout"] if not args.smoke else 20,
        seed=dcfg["seed"],
        format=args.canary_format,
    )
    train_examples = build_train_set(n_train, pool, seed=dcfg["seed"], difficulty=args.difficulty)
    test_examples = build_test_set(n_test, seed=dcfg["seed"] + 1, difficulty=args.difficulty)

    def _serialize_canary(c, dup=None):
        d = {
            "text": c.text, "w": c.w, "x": c.x, "y": c.y, "z": c.z,
            "format": c.format, "head": c.head, "sep_after": list(c.sep_after),
            "prefixes": [c.prefix(k) for k in range(4)],
            "expected": [c.expected_completion(k) for k in range(4)],
        }
        if dup is not None:
            d["dup"] = dup
        return d
    with open(meta_dir / "canary_pool.json", "w") as f:
        json.dump({
            "format": args.canary_format,
            "train": [_serialize_canary(c, d) for c, d in zip(pool.train, pool.train_duplications)],
            "holdout": [_serialize_canary(c) for c in pool.holdout],
        }, f, indent=2)
    with open(meta_dir / "test_examples.json", "w") as f:
        json.dump(test_examples, f)

    # ---- Tokenizer / model ----
    print(f"Loading tokenizer + model: {mcfg['hf_name']}")
    tokenizer = AutoTokenizer.from_pretrained(mcfg["hf_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DP-SGD requires fp32 for stable per-sample gradients (Opacus mixes fp32
    # noise into the param.grad tensor and cannot do an in-place dtype cast).
    model = AutoModelForCausalLM.from_pretrained(mcfg["hf_name"], dtype=torch.float32)
    # Opacus requires no grad checkpointing and the module to be in training mode
    # so that batchnorm/dropout-style modules are validated correctly.
    model.config.use_cache = False
    model.train()
    # Patch any incompatible modules (LayerNorm-with-running-stats, etc.)
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=True)
    device = "cuda"
    model.to(device)

    # ---- Datasets / dataloader ----
    train_ds = SFTDataset(
        train_examples, tokenizer, args.condition,
        max_length=dcfg["max_seq_len"],
        mask_prompt=args.mask_prompt,
    )
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    per_device_bs = args.per_device_batch if args.per_device_batch else mcfg["per_device_batch"]
    grad_accum = args.grad_accum if args.grad_accum else mcfg["grad_accum"]
    eff_bs = per_device_bs * grad_accum
    n_epochs = 1 if args.smoke else (args.num_epochs if args.num_epochs else tcfg["num_epochs"])
    steps_per_epoch = math.ceil(len(train_ds) / eff_bs)
    total_steps = min(steps_per_epoch * n_epochs, tcfg["max_steps"])
    print(f"effective batch {eff_bs}, total steps {total_steps}, n epochs {n_epochs}")

    train_dl = DataLoader(
        train_ds,
        batch_size=eff_bs,         # logical batch — Opacus + BatchMemoryManager virtualizes it
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    # ---- Optimizer + privacy engine ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(mcfg["lr"]),
        betas=tuple(tcfg["betas"]),
        eps=float(tcfg["eps"]),
        weight_decay=tcfg["weight_decay"],
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=tcfg["warmup_steps"])

    privacy_engine = PrivacyEngine()
    # functorch mode uses vmap for per-sample grads — avoids the dtype-mismatch
    # bug in opacus's scatter-add embedding hook with newer PyTorch.
    model, optimizer, train_dl = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_dl,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        epochs=n_epochs,
        max_grad_norm=args.max_grad_norm,
        grad_sample_mode="functorch",
    )
    print(f"Achieved noise_multiplier = {optimizer.noise_multiplier}")

    # ---- Milestone checkpointing ----
    fractions = tcfg["checkpoint_step_fractions"]
    milestone_steps = sorted({max(1, int(round(f * total_steps))) for f in fractions})
    print(f"Milestone steps: {milestone_steps}")

    def save_ckpt(label: str):
        ckpt_dir = out_root / f"checkpoint-{label}"
        ckpt_dir.mkdir(exist_ok=True)
        # Opacus wraps the model; unwrap before saving.
        underlying = model._module if hasattr(model, "_module") else model
        underlying.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  >> saved {ckpt_dir}")

    # ---- Training loop ----
    # Logical steps: every (eff_bs / max_physical_batch) physical iterations
    # corresponds to one optimizer.step() that actually updates parameters.
    physical_per_logical = max(1, eff_bs // args.max_physical_batch)
    step = 0
    fired = set()
    physical_iter = 0
    last_loss = float("nan")
    for epoch in range(n_epochs):
        with BatchMemoryManager(
            data_loader=train_dl,
            max_physical_batch_size=args.max_physical_batch,
            optimizer=optimizer,
        ) as memory_safe_dl:
            for batch in memory_safe_dl:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss
                last_loss = float(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                physical_iter += 1
                # Count a "logical step" whenever we've consumed a full effective
                # batch worth of microbatches. Opacus's DPOptimizer.step()
                # internally no-ops on continuation microbatches and only commits
                # the parameter update on the boundary microbatch.
                if physical_iter % physical_per_logical == 0:
                    scheduler.step()
                    step += 1
                    if step % 20 == 0:
                        try:
                            eps_now = privacy_engine.get_epsilon(args.target_delta)
                        except Exception:
                            eps_now = float("nan")
                        print(f"  step {step}/{total_steps}  loss={last_loss:.4f}  eps={eps_now:.2f}")
                    for ms in milestone_steps:
                        if step >= ms and ms not in fired:
                            fired.add(ms)
                            save_ckpt(f"step-{ms}")
                    if step >= total_steps:
                        break
            if step >= total_steps:
                break

    save_ckpt("final")
    eps_final = privacy_engine.get_epsilon(args.target_delta)
    print(f"\nDone. Final achieved (eps, delta) = ({eps_final:.3f}, {args.target_delta})")

    # ---- Manifest ----
    manifest = {
        "run_id": run_id,
        "model_key": args.model,
        "hf_name": mcfg["hf_name"],
        "condition": args.condition,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "canary_format": args.canary_format,
        "mask_prompt": bool(args.mask_prompt),
        "dp": {
            "target_epsilon": args.target_epsilon,
            "achieved_epsilon": float(eps_final),
            "target_delta": args.target_delta,
            "noise_multiplier": float(optimizer.noise_multiplier),
            "max_grad_norm": args.max_grad_norm,
        },
        "total_steps": total_steps,
        "milestone_steps": milestone_steps,
        "checkpoints_dir": str(out_root),
        "results_dir": str(meta_dir),
        "n_train_problems": n_train,
        "n_test_problems": n_test,
    }
    with open(meta_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest -> {meta_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
