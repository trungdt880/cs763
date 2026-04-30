"""SFT training loop for the CoT-vs-memorization experiment.

We deliberately do NOT use TRL's SFTTrainer / DataCollatorForCompletionOnlyLM,
because (a) hand-rolling the prompt-vs-target split is more transparent and
testable than letting a substring match handle it, and (b) we want full control
over per-step bookkeeping for the milestone checkpointer.

What this script does:
  1. Loads a Qwen3 base model + tokenizer.
  2. Builds the synthetic dataset (deterministic from seed).
  3. Tokenizes each example as a single causal-LM sequence
     (prompt + target), computing loss on EVERY token by default. This is
     what gets the canary memorized — canary tokens live in the prompt, so
     loss-masking the prompt (instruction-tune style) would give the canary
     no gradient signal at all and the memorization metrics would be flat.
     Use --mask_prompt for the old behavior as an ablation.
  4. Trains with HF Trainer for a fixed number of epochs, with the same number
     of optimizer steps regardless of condition (matched-examples comparison).
  5. Saves checkpoints at configurable step-fraction milestones.

Usage:
    python train.py --model qwen3_0p6b --condition cot --seed 0 --config configs.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from canaries import build_canary_pool
from data import build_train_set, build_test_set


# ----------------------------------------------------------------------------
# Tokenized dataset with prompt-masked labels
# ----------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Tokenizes (prompt, target) pairs for causal LM training.

    By default, computes loss on the FULL sequence (prompt + target), which
    matches standard pretraining dynamics. This is what makes canaries (which
    live inside the prompt) actually get memorized — Carlini-style pretraining
    memorization requires the canary to be in the loss path, not just in the
    context. Earlier revisions of this file masked the prompt, which meant the
    canary received zero gradient signal and the memorization metrics were
    flat at ~0 regardless of condition; see CHANGELOG at the top of the file
    if you're confused.

    Set `mask_prompt=True` to recover the old instruction-tuning-style masking
    (loss only on target tokens). This is provided for ablation only.
    """

    def __init__(self, examples: List[dict], tokenizer, condition: str,
                 max_length: int, mask_prompt: bool = False):
        if condition not in ("answer", "cot"):
            raise ValueError(f"condition must be 'answer' or 'cot', got {condition!r}")
        self.examples = examples
        self.tokenizer = tokenizer
        self.condition = condition
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self._target_field = "answer_only" if condition == "answer" else "cot"

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt = ex["prompt"]
        target = ex[self._target_field]
        eos = self.tokenizer.eos_token or ""
        full_target = target + eos

        # Tokenize without special tokens (Qwen base has no implicit BOS to add).
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(full_target, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + target_ids
        if self.mask_prompt:
            labels = [-100] * len(prompt_ids) + target_ids[:]
        else:
            # Full-sequence loss: every token contributes (canary tokens too).
            labels = input_ids[:]

        # Truncate from the LEFT of the prompt if too long, so the target is preserved.
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            keep_prompt = max(0, len(prompt_ids) - overflow)
            prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
            input_ids = prompt_ids + target_ids
            if self.mask_prompt:
                labels = [-100] * len(prompt_ids) + target_ids[:]
            else:
                labels = input_ids[:]
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "n_target_tokens": len(target_ids),
        }


# Backwards-compat alias for any external code / imports.
PromptMaskedDataset = SFTDataset


@dataclass
class PadCollator:
    """Right-pad a batch of variable-length tokenized examples."""
    pad_token_id: int

    def __call__(self, batch: List[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        n_target_tokens = 0
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
            n_target_tokens += b["n_target_tokens"]
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
        return out


# ----------------------------------------------------------------------------
# Milestone checkpoint callback
# ----------------------------------------------------------------------------

class MilestoneCheckpointCallback(TrainerCallback):
    """Save the model at predefined fractions of total training steps.

    DDP-safe: uses `trainer.save_model`, which unwraps DDP and only writes on
    the main process. Set `callback.trainer = trainer` after instantiation.
    """

    def __init__(self, total_steps: int, fractions: List[float], out_dir: Path, tokenizer=None):
        self.milestone_steps = sorted({max(1, int(round(f * total_steps))) for f in fractions})
        self.fired = set()
        self.out_dir = out_dir
        self.tokenizer = tokenizer
        self.trainer: Optional["Trainer"] = None  # set by caller after Trainer is built

    def on_step_end(self, args, state, control, **kwargs):
        del args, control, kwargs  # required by TrainerCallback interface; we only use `state`
        assert self.trainer is not None, "MilestoneCheckpointCallback.trainer was never set"
        for ms in self.milestone_steps:
            if state.global_step >= ms and ms not in self.fired:
                self.fired.add(ms)
                ckpt_dir = self.out_dir / f"checkpoint-step-{ms}"
                if state.is_world_process_zero:
                    print(f"  >> milestone checkpoint -> {ckpt_dir}")
                self.trainer.save_model(str(ckpt_dir))
                if state.is_world_process_zero and self.tokenizer is not None:
                    self.tokenizer.save_pretrained(ckpt_dir)


# ----------------------------------------------------------------------------
# Sanity check: assert we loaded a base model, not a chat/instruct variant
# ----------------------------------------------------------------------------

def _assert_base_model(hf_name: str) -> None:
    bad_substrings = ("Instruct", "Chat", "instruct", "chat")
    if any(s in hf_name for s in bad_substrings):
        raise ValueError(
            f"Refusing to load {hf_name!r}: looks like an instruct/chat variant. "
            "Use the -Base variant; the instruct variant has built-in thinking mode "
            "which would confound the answer-vs-CoT comparison."
        )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs.yaml")
    ap.add_argument("--model", required=True, help="key in configs.yaml:models")
    ap.add_argument("--condition", required=True, choices=["answer", "cot"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run: 200 train problems, 1 epoch, no checkpointing.")
    ap.add_argument("--per_device_batch", type=int, default=None,
                    help="Override per-device batch size (useful for DDP to keep effective batch fixed).")
    ap.add_argument("--grad_accum", type=int, default=None,
                    help="Override gradient accumulation (useful for DDP to keep effective batch fixed).")
    ap.add_argument("--mask_prompt", action="store_true",
                    help="Compute loss only on the target tokens (instruction-tuning style). "
                         "Default is False: loss on the full sequence, pretraining-style, so "
                         "canary tokens in the prompt actually receive gradient signal and can "
                         "be memorized. Turn this on only for ablation.")
    ap.add_argument("--difficulty", default="easy", choices=["easy", "hard", "mixed"],
                    help="Which template family to use for problems. easy = Phase 2 templates "
                         "(answer-only ~94%% acc); hard = harder templates (answer-only ~60-70%% acc).")
    ap.add_argument("--canary_format", default="zk", choices=["zk", "pii"],
                    help="zk = synthetic high-entropy 'Reference code: ZK-...' (Phase 2 default); "
                         "pii = human-readable 'Customer record: <name>, SSN ..., born ..., account ...'.")
    ap.add_argument("--run_suffix", default="",
                    help="Optional suffix appended to run_id to disambiguate ablation runs.")
    args = ap.parse_args()

    # DDP rank detection (torchrun sets these; default to single-process).
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = local_rank == 0

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
    if is_main:
        print(f"\n=== RUN: {run_id}  (world_size={world_size}) ===")
    out_root = Path(cfg["output"]["checkpoints_dir"]) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    n_train = 200 if args.smoke else dcfg["n_train_problems"]
    n_test = 50 if args.smoke else dcfg["n_test_problems"]
    duplication = ccfg["duplication_buckets"]
    if args.smoke:
        # Tiny canary pool that still fits in 200 training problems.
        duplication = {1: 5, 4: 5, 16: 5}
    pool = build_canary_pool(
        duplication_buckets={int(k): int(v) for k, v in duplication.items()},
        n_holdout=ccfg["n_holdout"] if not args.smoke else 20,
        seed=dcfg["seed"],
        format=args.canary_format,
    )
    train_examples = build_train_set(n_train, pool, seed=dcfg["seed"], difficulty=args.difficulty)
    test_examples = build_test_set(n_test, seed=dcfg["seed"] + 1, difficulty=args.difficulty)

    # Persist the canary pool + dataset metadata for later evaluation.
    # Only rank 0 writes — all ranks build the same data deterministically
    # from the seed, but only one of them needs to persist it.
    meta_dir = Path(cfg["output"]["results_dir"]) / run_id
    meta_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        def _serialize_canary(c, dup=None):
            d = {
                "text": c.text,
                "w": c.w, "x": c.x, "y": c.y, "z": c.z,
                "format": c.format,
                "head": c.head,
                "sep_after": list(c.sep_after),
                # Pre-computed prefixes/expected so attacks.py is format-agnostic.
                "prefixes": [c.prefix(k) for k in range(4)],
                "expected": [c.expected_completion(k) for k in range(4)],
            }
            if dup is not None:
                d["dup"] = dup
            return d
        with open(meta_dir / "canary_pool.json", "w") as f:
            json.dump(
                {
                    "format": args.canary_format,
                    "train": [_serialize_canary(c, d)
                              for c, d in zip(pool.train, pool.train_duplications)],
                    "holdout": [_serialize_canary(c) for c in pool.holdout],
                },
                f, indent=2,
            )
        with open(meta_dir / "test_examples.json", "w") as f:
            json.dump(test_examples, f)

    # ---- Tokenizer / model ----
    print(f"Loading tokenizer + model: {mcfg['hf_name']}")
    tokenizer = AutoTokenizer.from_pretrained(mcfg["hf_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if tcfg["precision"] == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["hf_name"], dtype=dtype,
    )
    if mcfg.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # ---- Datasets ----
    train_ds = SFTDataset(
        train_examples, tokenizer, args.condition,
        max_length=dcfg["max_seq_len"],
        mask_prompt=args.mask_prompt,
    )
    if is_main:
        regime = "mask_prompt (instruction-tune style)" if args.mask_prompt else "full-sequence loss (pretraining style)"
        print(f"Loss regime: {regime}")
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    # ---- Training arguments ----
    per_device_bs = args.per_device_batch if args.per_device_batch else mcfg["per_device_batch"]
    grad_accum = args.grad_accum if args.grad_accum else mcfg["grad_accum"]
    n_epochs = 1 if args.smoke else tcfg["num_epochs"]
    max_steps_cap = tcfg["max_steps"]

    # Effective batch = per_device * grad_accum * world_size.
    eff_bs = per_device_bs * grad_accum * world_size
    steps_per_epoch = math.ceil(len(train_ds) / eff_bs)
    total_steps = min(steps_per_epoch * n_epochs, max_steps_cap)
    if is_main:
        print(f"Train examples: {len(train_ds)}  per_device_bs: {per_device_bs}  "
              f"grad_accum: {grad_accum}  world_size: {world_size}")
        print(f"effective batch: {eff_bs}  steps/epoch: {steps_per_epoch}  total steps: {total_steps}")

    targs = TrainingArguments(
        output_dir=str(out_root / "trainer_state"),
        num_train_epochs=n_epochs,
        max_steps=total_steps,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(mcfg["lr"]),
        weight_decay=tcfg["weight_decay"],
        adam_beta1=tcfg["betas"][0],
        adam_beta2=tcfg["betas"][1],
        adam_epsilon=float(tcfg["eps"]),
        warmup_steps=tcfg["warmup_steps"],
        lr_scheduler_type=tcfg["lr_schedule"],
        max_grad_norm=tcfg["grad_clip"],
        bf16=(tcfg["precision"] == "bf16"),
        logging_steps=20,
        save_strategy="no",          # we use the milestone callback instead
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )

    milestone_cb = None
    callbacks = []
    if not args.smoke:
        milestone_cb = MilestoneCheckpointCallback(
            total_steps=total_steps,
            fractions=tcfg["checkpoint_step_fractions"],
            out_dir=out_root,
            tokenizer=tokenizer,
        )
        callbacks.append(milestone_cb)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    if milestone_cb is not None:
        milestone_cb.trainer = trainer

    if is_main:
        print("Starting training...")
    trainer.train()

    # Always save a final checkpoint (Trainer.save_model unwraps DDP and is
    # rank-0 only, so all ranks can call it without harm).
    final_dir = out_root / "checkpoint-final"
    trainer.save_model(str(final_dir))
    if is_main:
        # save_model saves the model + tokenizer IF tokenizer was passed to Trainer.
        # Our Trainer doesn't take tokenizer, so persist it explicitly on rank 0.
        tokenizer.save_pretrained(final_dir)
        print(f"Done. Final checkpoint at {final_dir}")

    # Write a manifest with the run config so evaluate.py can find everything.
    manifest = {
        "run_id": run_id,
        "model_key": args.model,
        "hf_name": mcfg["hf_name"],
        "condition": args.condition,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "canary_format": args.canary_format,
        "mask_prompt": bool(args.mask_prompt),
        "total_steps": total_steps,
        "milestone_steps": sorted({max(1, int(round(f * total_steps)))
                                    for f in tcfg["checkpoint_step_fractions"]}),
        "checkpoints_dir": str(out_root),
        "results_dir": str(meta_dir),
        "n_train_problems": n_train,
        "n_test_problems": n_test,
    }
    if is_main:
        with open(meta_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest written to {meta_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
