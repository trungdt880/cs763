"""Synthetic word-problem dataset with canary injection.

Each example is a dict:
    {
      "prompt":      str,   # identical across conditions
      "answer_only": str,   # final-number target
      "cot":         str,   # reasoning + final-number target
      "gold":        int,   # ground-truth answer
      "canary":      str,   # canary text embedded in this prompt's METADATA block
      "template":    str,   # which template generated it
      "canary_role": str,   # 'train' (in controlled duplication pool) or 'filler'
    }

Critical invariants (unit-tested):
1. The `prompt` field is byte-identical regardless of which condition will use it.
2. The CoT reasoning never quotes the canary or anything from the [METADATA] block.
3. Test-set canaries are disjoint from training-set canaries.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple

from canaries import Canary, CanaryPool, build_canary_pool, generate_canaries

# ----------------------------------------------------------------------------
# Entity vocabularies (DISJOINT from canary YYYY dictionary in canaries.py)
# ----------------------------------------------------------------------------

NAMES: List[str] = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Kate", "Leo", "Mia", "Noah", "Olive", "Peter",
    "Quinn", "Ruby", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe",
]

ITEMS_COUNTABLE: List[str] = [
    "apples", "books", "marbles", "pens", "coins", "stickers", "cards",
    "candies", "stones", "beads", "buttons", "ribbons", "tokens", "shells",
    "stamps", "puzzles", "blocks", "crayons", "erasers", "magnets",
]

ITEMS_PRICEY: List[str] = [
    "notebooks", "pencils", "snacks", "tickets", "muffins", "bagels",
    "donuts", "cookies", "bottles", "stickers",
]


# ----------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------
# Each template is (name, sampler) where sampler(rng) -> (problem_text, reasoning_text, gold_int)
# The reasoning_text NEVER references the [METADATA] block.

def _t_add_chain(rng: random.Random) -> Tuple[str, str, int]:
    a, b, c = rng.choice(NAMES), rng.choice(NAMES), rng.choice(NAMES)
    while b == a:
        b = rng.choice(NAMES)
    while c == a or c == b:
        c = rng.choice(NAMES)
    item = rng.choice(ITEMS_COUNTABLE)
    x = rng.randint(5, 50)
    y = rng.randint(2, 30)
    z = rng.randint(1, min(20, x + y - 1))
    gold = x + y - z
    problem = (
        f"{a} has {x} {item}. {b} gives {a} {y} more {item}. "
        f"Then {a} gives {z} {item} to {c}. How many {item} does {a} have now?"
    )
    reasoning = (
        f"Step 1: {a} starts with {x} {item}.\n"
        f"Step 2: After {b} gives {y} more, {a} has {x} + {y} = {x+y} {item}.\n"
        f"Step 3: After giving {z} to {c}, {a} has {x+y} - {z} = {gold} {item}."
    )
    return problem, reasoning, gold


def _t_comparison(rng: random.Random) -> Tuple[str, str, int]:
    a, b = rng.sample(NAMES, 2)
    item = rng.choice(ITEMS_COUNTABLE)
    x = rng.randint(20, 100)
    y = rng.randint(1, x - 1)
    gold = x - y
    problem = (
        f"{a} has {x} {item}. {b} has {y} {item}. "
        f"How many more {item} does {a} have than {b}?"
    )
    reasoning = (
        f"Step 1: {a} has {x} {item}.\n"
        f"Step 2: {b} has {y} {item}.\n"
        f"Step 3: The difference is {x} - {y} = {gold} {item}."
    )
    return problem, reasoning, gold


def _t_money(rng: random.Random) -> Tuple[str, str, int]:
    a = rng.choice(NAMES)
    item = rng.choice(ITEMS_PRICEY)
    n = rng.randint(2, 12)
    p = rng.randint(2, 15)
    m = rng.randint(n * p + 1, n * p + 50)
    gold = m - n * p
    problem = (
        f"{a} buys {n} {item} at ${p} each. {a} starts with ${m}. "
        f"How many dollars does {a} have left?"
    )
    reasoning = (
        f"Step 1: The cost is {n} * {p} = {n*p} dollars.\n"
        f"Step 2: {a} started with {m} dollars.\n"
        f"Step 3: After spending, {a} has {m} - {n*p} = {gold} dollars."
    )
    return problem, reasoning, gold


def _t_units(rng: random.Random) -> Tuple[str, str, int]:
    a = rng.choice(NAMES)
    minutes = rng.randint(2, 90)
    extra_seconds = rng.randint(1, 59)
    gold = minutes * 60 + extra_seconds
    problem = (
        f"{a}'s timer ran for {minutes} minutes and {extra_seconds} seconds. "
        f"How many seconds is that in total?"
    )
    reasoning = (
        f"Step 1: {minutes} minutes is {minutes} * 60 = {minutes*60} seconds.\n"
        f"Step 2: Adding the extra {extra_seconds} seconds: "
        f"{minutes*60} + {extra_seconds} = {gold} seconds."
    )
    return problem, reasoning, gold


# ----------------------------------------------------------------------------
# Hard templates (Phase 3): designed to drop answer-only accuracy to ~60-70%
# so CoT has real headroom in the privacy/utility plot. Each requires either
# 5+ arithmetic operations, distractor filtering, or constraint inversion.
# ----------------------------------------------------------------------------

def _t_multistep_inventory(rng: random.Random) -> Tuple[str, str, int]:
    """5-step inventory: base + add - sub + add - sub. 3-4 digit numbers."""
    a = rng.choice(NAMES)
    item = rng.choice(ITEMS_COUNTABLE)
    base = rng.randint(800, 2000)
    d1 = rng.randint(80, 400)
    s1 = rng.randint(40, 300)
    d2 = rng.randint(80, 400)
    s2 = rng.randint(40, min(base + d1 - s1 + d2 - 1, 300))
    gold = base + d1 - s1 + d2 - s2
    problem = (
        f"A warehouse starts the week with {base} {item}. On Monday, a delivery "
        f"of {d1} arrives. On Tuesday, {s1} are sold. On Wednesday, another "
        f"{d2} are delivered. On Thursday, {s2} are sold. How many {item} "
        f"are in the warehouse at the end of the week?"
    )
    reasoning = (
        f"Step 1: After Monday's delivery, total = {base} + {d1} = {base + d1}.\n"
        f"Step 2: After Tuesday's sales, total = {base + d1} - {s1} = {base + d1 - s1}.\n"
        f"Step 3: After Wednesday's delivery, total = {base + d1 - s1} + {d2} = {base + d1 - s1 + d2}.\n"
        f"Step 4: After Thursday's sales, total = {base + d1 - s1 + d2} - {s2} = {gold}."
    )
    return problem, reasoning, gold


def _t_distractor_sum(rng: random.Random) -> Tuple[str, str, int]:
    """Sum of three items with 3 numeric distractors that must be ignored."""
    a = rng.choice(NAMES)
    item = rng.choice(ITEMS_COUNTABLE)
    x = rng.randint(40, 300)
    y = rng.randint(40, 300)
    z = rng.randint(40, 300)
    # Distractors: irrelevant numbers in different units / contexts
    age = rng.randint(20, 70)
    weight = rng.randint(50, 220)
    miles = rng.randint(2, 15)
    other_item = rng.choice([w for w in ITEMS_COUNTABLE if w != item])
    other_count = rng.randint(10, 80)
    gold = x + y + z
    problem = (
        f"{a} runs a small shop. On Monday {a} sold {x} {item}. {a} is "
        f"{age} years old and {a}'s dog weighs {weight} pounds. On Tuesday "
        f"{a} sold {y} {item}. {a} also has {other_count} {other_item} in "
        f"a separate display, and {a}'s cat ran {miles} miles last week. On "
        f"Wednesday {a} sold {z} {item}. How many {item} did {a} sell across "
        f"the three days?"
    )
    reasoning = (
        f"Step 1: Identify the relevant numbers: Monday {x}, Tuesday {y}, "
        f"Wednesday {z} {item}. (Age, weight, miles, and {other_item} are "
        f"irrelevant to the question.)\n"
        f"Step 2: Total {item} sold = {x} + {y} + {z} = {gold}."
    )
    return problem, reasoning, gold


def _t_logic_constraint(rng: random.Random) -> Tuple[str, str, int]:
    """Three-box constraint: A+B+C=T, A=B+d_ab, C=A+d_ca. Solve for B.

    Math: T = (B+d_ab) + B + (B+d_ab+d_ca) = 3B + 2*d_ab + d_ca,
    so B = (T - 2*d_ab - d_ca) / 3. Choose params so B is a positive integer.
    """
    item = rng.choice(ITEMS_COUNTABLE)
    while True:
        d_ab = rng.randint(2, 15)        # A has d_ab more than B
        d_ca = rng.randint(2, 15)        # C has d_ca more than A
        b = rng.randint(15, 80)
        a_count = b + d_ab
        c_count = a_count + d_ca
        total = a_count + b + c_count
        if total < 250 and b > 0:
            break
    gold = b
    problem = (
        f"Three boxes labeled A, B, and C contain {total} {item} altogether. "
        f"Box A has {d_ab} more {item} than box B. Box C has {d_ca} more "
        f"{item} than box A. How many {item} are in box B?"
    )
    reasoning = (
        f"Step 1: Let B = number of {item} in box B. Then A = B + {d_ab} "
        f"and C = A + {d_ca} = B + {d_ab + d_ca}.\n"
        f"Step 2: A + B + C = {total}, so (B + {d_ab}) + B + (B + {d_ab + d_ca}) = {total}.\n"
        f"Step 3: 3B + {2 * d_ab + d_ca} = {total}, so 3B = {total - 2 * d_ab - d_ca}.\n"
        f"Step 4: B = {total - 2 * d_ab - d_ca} / 3 = {gold}."
    )
    return problem, reasoning, gold


def _t_chain_with_doubling(rng: random.Random) -> Tuple[str, str, int]:
    """4-step compound problem with a multiplication: (X + Y - Z) * k - W."""
    a, b = rng.sample(NAMES, 2)
    item = rng.choice(ITEMS_COUNTABLE)
    x = rng.randint(20, 100)
    y = rng.randint(10, 60)
    z = rng.randint(5, min(x + y - 1, 40))
    k = rng.choice([2, 3])
    w = rng.randint(10, 80)
    intermediate = x + y - z
    after_mul = intermediate * k
    gold = after_mul - w
    while gold < 0:
        w = rng.randint(1, after_mul - 1)
        gold = after_mul - w
    problem = (
        f"{a} has {x} {item}. {b} gives {a} {y} more {item}. Then {a} loses "
        f"{z} {item}. Suddenly {a}'s collection {'doubles' if k == 2 else 'triples'}. "
        f"Finally, {a} gives {w} {item} away. How many {item} does {a} have now?"
    )
    reasoning = (
        f"Step 1: After {b} gives {y}, {a} has {x} + {y} = {x + y} {item}.\n"
        f"Step 2: After losing {z}, {a} has {x + y} - {z} = {intermediate} {item}.\n"
        f"Step 3: After {'doubling' if k == 2 else 'tripling'}, {a} has {intermediate} * {k} = {after_mul} {item}.\n"
        f"Step 4: After giving away {w}, {a} has {after_mul} - {w} = {gold} {item}."
    )
    return problem, reasoning, gold


# Easy templates (Phase 2 originals) — kept under TEMPLATES_EASY for backwards
# compatibility. The unqualified TEMPLATES alias points at this set so existing
# scripts and the eval pipeline keep working.
TEMPLATES_EASY: Dict[str, Callable[[random.Random], Tuple[str, str, int]]] = {
    "add_chain": _t_add_chain,
    "comparison": _t_comparison,
    "money": _t_money,
    "units": _t_units,
}

TEMPLATES_HARD: Dict[str, Callable[[random.Random], Tuple[str, str, int]]] = {
    "multistep_inventory": _t_multistep_inventory,
    "distractor_sum":      _t_distractor_sum,
    "logic_constraint":    _t_logic_constraint,
    "chain_doubling":      _t_chain_with_doubling,
}

# Default = easy (Phase 2 behavior).
TEMPLATES = TEMPLATES_EASY


def get_templates(difficulty: str) -> Dict[str, Callable[[random.Random], Tuple[str, str, int]]]:
    if difficulty == "easy":
        return TEMPLATES_EASY
    if difficulty == "hard":
        return TEMPLATES_HARD
    if difficulty == "mixed":
        return {**TEMPLATES_EASY, **TEMPLATES_HARD}
    raise ValueError(f"Unknown difficulty {difficulty!r}; expected easy|hard|mixed")


# ----------------------------------------------------------------------------
# Prompt assembly
# ----------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "[METADATA]\n"
    "{canary}\n"
    "[PROBLEM]\n"
    "{problem}\n"
    "[ANSWER]\n"
)
RESPONSE_TEMPLATE = "[ANSWER]\n"  # used by DataCollatorForCompletionOnlyLM


def make_example(problem: str, reasoning: str, gold: int, canary_text: str,
                 template: str, canary_role: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(canary=canary_text, problem=problem)
    answer_only = f"{gold}"
    cot = f"{reasoning}\n#### {gold}"
    return {
        "prompt": prompt,
        "answer_only": answer_only,
        "cot": cot,
        "gold": gold,
        "canary": canary_text,
        "template": template,
        "canary_role": canary_role,
    }


# ----------------------------------------------------------------------------
# Dataset builders
# ----------------------------------------------------------------------------

def build_train_set(
    n_problems: int,
    canary_pool: CanaryPool,
    seed: int,
    filler_canary_seed_offset: int = 99991,
    difficulty: str = "easy",
) -> List[dict]:
    """Generate training examples and assign each one a canary.

    Canary assignment:
      - First, every controlled-pool canary is assigned to its scheduled number
        of problems (the "train_occurrences" list, length = sum of duplications).
      - The remaining problems get unique 'filler' canaries (each appearing once),
        keeping the [METADATA] block consistent across the whole training set.
      - The order is shuffled deterministically so duplicate canaries are spread
        across the training data, not all clustered.
    """
    rng = random.Random(seed)

    # 1. Generate raw problems.
    templates = get_templates(difficulty)
    template_names = list(templates.keys())
    raw: List[Tuple[str, str, str, int]] = []  # (template, problem, reasoning, gold)
    for _ in range(n_problems):
        t = rng.choice(template_names)
        p, r, g = templates[t](rng)
        raw.append((t, p, r, g))

    # 2. Build the canary assignment list of length n_problems.
    controlled_occurrences: List[Canary] = canary_pool.train_occurrences()
    n_controlled = len(controlled_occurrences)
    if n_controlled > n_problems:
        raise ValueError(
            f"Canary occurrences ({n_controlled}) exceed n_problems ({n_problems})."
        )
    n_filler = n_problems - n_controlled
    filler_canaries = generate_canaries(
        n_filler + len(canary_pool.train) + len(canary_pool.holdout),  # safe oversample
        seed=seed + filler_canary_seed_offset,
    )
    # Filter fillers so they don't collide with train or holdout canaries.
    forbidden = {c.text for c in canary_pool.train} | {c.text for c in canary_pool.holdout}
    filler_canaries = [c for c in filler_canaries if c.text not in forbidden][:n_filler]
    assert len(filler_canaries) == n_filler

    assignment: List[Tuple[Canary, str]] = (
        [(c, "train") for c in controlled_occurrences]
        + [(c, "filler") for c in filler_canaries]
    )
    rng.shuffle(assignment)

    # 3. Zip raw problems with their assigned canaries.
    out: List[dict] = []
    for (canary, role), (template, problem, reasoning, gold) in zip(assignment, raw):
        out.append(make_example(problem, reasoning, gold, canary.text, template, role))
    return out


def build_test_set(n_problems: int, seed: int, difficulty: str = "easy") -> List[dict]:
    """Test problems for measuring task accuracy.

    Test prompts get a fresh per-prompt 'test' canary in their [METADATA] block
    (drawn from a separate seed) so the surface format matches training. These
    canaries are NEVER trained on.
    """
    rng = random.Random(seed)
    test_canaries = generate_canaries(n_problems, seed=seed + 4242)
    templates = get_templates(difficulty)
    template_names = list(templates.keys())
    out: List[dict] = []
    for c in test_canaries:
        t = rng.choice(template_names)
        p, r, g = templates[t](rng)
        out.append(make_example(p, r, g, c.text, t, "test"))
    return out


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

def _self_test(difficulty: str = "easy") -> None:
    pool = build_canary_pool({1: 50, 4: 50, 16: 50, 64: 50}, n_holdout=200, seed=1234)
    train = build_train_set(8000, pool, seed=1234, difficulty=difficulty)
    test = build_test_set(1000, seed=5678, difficulty=difficulty)
    print(f"\n=== Difficulty: {difficulty} ===")

    assert len(train) == 8000 and len(test) == 1000
    train_canary_strs = {ex["canary"] for ex in train}
    test_canary_strs = {ex["canary"] for ex in test}
    assert train_canary_strs.isdisjoint(test_canary_strs), "train/test canary overlap!"

    # Invariant 1: prompt is identical across conditions (trivially true since
    # we only generate it once, but we verify the field is well-formed).
    sample = train[0]
    assert "[METADATA]" in sample["prompt"]
    assert "[PROBLEM]" in sample["prompt"]
    assert sample["prompt"].endswith("[ANSWER]\n")

    # Invariant 2: CoT does NOT contain the canary text.
    for ex in train[:100]:
        assert ex["canary"] not in ex["cot"], "canary leaked into CoT target!"
        assert "Reference code" not in ex["cot"]
        assert "[METADATA]" not in ex["cot"]

    # Sanity: target lengths
    avg_ans = sum(len(ex["answer_only"]) for ex in train) / len(train)
    avg_cot = sum(len(ex["cot"]) for ex in train) / len(train)
    print(f"Train problems: {len(train)}  Test problems: {len(test)}")
    print(f"Unique train canary texts: {len(train_canary_strs)}  (filler+controlled)")
    print(f"Avg target chars — answer_only: {avg_ans:.1f}  cot: {avg_cot:.1f}  ratio: {avg_cot/avg_ans:.1f}x")
    print()
    print("=== Sample training example ===")
    print("PROMPT:")
    print(sample["prompt"])
    print("ANSWER_ONLY target:", repr(sample["answer_only"]))
    print("COT target:")
    print(sample["cot"])
    print()
    print("Canary role distribution:")
    roles = {}
    for ex in train:
        roles[ex["canary_role"]] = roles.get(ex["canary_role"], 0) + 1
    print(" ", roles)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default="easy", choices=["easy", "hard", "mixed"])
    ap.add_argument("--show_hard_samples", action="store_true",
                    help="Print one sample per hard template.")
    args = ap.parse_args()
    _self_test(difficulty=args.difficulty)
    if args.show_hard_samples:
        rng = random.Random(0)
        print("\n=== One sample per hard template ===")
        for name, fn in TEMPLATES_HARD.items():
            p, r, g = fn(rng)
            print(f"\n-- {name} -- (gold = {g})")
            print("Q:", p)
            print("R:", r)
