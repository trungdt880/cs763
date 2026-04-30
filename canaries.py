"""Canary generation, duplication scheduling, and prefix helpers.

Two canary formats are supported:

  ZK (high-entropy synthetic, ~51 bits):
      Reference code: ZK-WWWW-XXXX-YYYY-ZZZZ
    where WWWW=4 uppercase consonants, XXXX=4 digits, YYYY=word from dict, ZZZZ=4 digits.

  PII (human-readable, ~78 bits via the digit fields):
      Customer record: <First Last>, SSN <NNN-NN-NNNN>, born <YYYY-MM-DD>, account <NNNNNNNN>
    Tests whether memorization differs by canary distribution (random
    high-entropy strings vs. natural-text patterns the model has seen during
    pretraining).

Both formats share a 4-group structure so the same extraction-prefix-length
sweep (k_groups in {0,1,2,3}) applies to both. Canaries serialize to JSON with
precomputed `prefixes` and `expected` lists so downstream attacks need not know
the format.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence


CONSONANTS = "BCDFGHJKLMNPQRSTVWXZ"  # 20-letter alphabet, no vowels (avoids forming words)

# 128-word dictionary of unrelated nouns. Disjoint from the entity names used in
# the word problems (those live in data.py) so the canary YYYY group never
# collides with a problem's named entity.
WORD_DICT: List[str] = [
    "AMBER", "AZURE", "BASIL", "BEACH", "BIRCH", "BLAZE", "BLOOM", "BREEZE",
    "BRICK", "BRINE", "CABLE", "CACAO", "CANAL", "CANDLE", "CANYON", "CARBON",
    "CEDAR", "CHALK", "CHARM", "CIDER", "CLOUD", "CLOVER", "COBALT", "COMET",
    "CORAL", "COVE", "CRANE", "CREEK", "CRIMSON", "CROWN", "CRYSTAL", "DAHLIA",
    "DELTA", "DRIFT", "DUNE", "EAGLE", "EMBER", "FALCON", "FERN", "FJORD",
    "FLAME", "FLINT", "FOREST", "FROST", "GALAXY", "GARNET", "GLACIER", "GLADE",
    "GRANITE", "GROVE", "HARBOR", "HARVEST", "HAZE", "HEATH", "HERON", "HIVE",
    "HORIZON", "INDIGO", "IVORY", "JADE", "JETTY", "KESTREL", "LAGOON", "LANTERN",
    "LARK", "LATTICE", "LAUREL", "LEDGE", "LICHEN", "LILAC", "LINDEN", "LOTUS",
    "MAGMA", "MANGO", "MAPLE", "MARLIN", "MEADOW", "MIRAGE", "MIST", "MOSS",
    "NEBULA", "NETTLE", "NIMBUS", "NOVA", "OAK", "OASIS", "OBSIDIAN", "ONYX",
    "OPAL", "ORCHID", "OSPREY", "PEBBLE", "PETAL", "PINE", "PLUM", "POLLEN",
    "PRAIRIE", "PUMICE", "QUARTZ", "QUILL", "RAVEN", "REEF", "RIVER", "ROWAN",
    "RUST", "SABLE", "SAFFRON", "SAGE", "SAPLING", "SCARLET", "SHALE", "SHORE",
    "SLATE", "SOLSTICE", "SPRUCE", "STARLING", "STORM", "SUMMIT", "TALON", "THISTLE",
    "TIDE", "TOPAZ", "TUNDRA", "VALE", "VELVET", "VERDANT", "WILLOW", "ZEPHYR",
]
assert len(WORD_DICT) == 128 and len(set(WORD_DICT)) == 128

CANARY_PREFIX = "Reference code: ZK-"

# PII format constants
PII_HEAD = "Customer record: "
PII_SEP_AFTER = [", SSN ", ", born ", ", account ", ""]  # separator AFTER each of the 4 groups

# Realistic-but-fictional first/last name lists (ASCII, no celebrity collisions
# expected). Disjoint from data.NAMES (single-name word-problem entities).
PII_FIRST_NAMES: List[str] = [
    "Maria", "James", "Linda", "Robert", "Patricia", "Michael", "Jennifer",
    "William", "Barbara", "David", "Susan", "Richard", "Jessica", "Joseph",
    "Sarah", "Thomas", "Karen", "Charles", "Nancy", "Christopher", "Lisa",
    "Daniel", "Margaret", "Matthew", "Betty", "Anthony", "Sandra", "Mark",
    "Ashley", "Donald", "Kimberly", "Steven", "Emily", "Paul", "Donna",
    "Andrew", "Michelle", "Joshua", "Carol", "Kenneth", "Amanda", "Kevin",
    "Melissa", "Brian", "Deborah", "George", "Stephanie", "Edward", "Rebecca",
    "Ronald", "Sharon", "Timothy", "Laura", "Jason", "Cynthia", "Jeffrey",
    "Kathleen", "Ryan", "Amy", "Jacob", "Shirley", "Gary", "Angela",
    "Nicholas", "Helen",
]
PII_LAST_NAMES: List[str] = [
    "Sanchez", "Chen", "Patel", "Murphy", "Cohen", "Brooks", "Hayes", "Reyes",
    "Walsh", "Schultz", "Vasquez", "Gibson", "Park", "Lin", "Kowalski",
    "Nguyen", "Tran", "Singh", "Khan", "Ali", "Nakamura", "Tanaka",
    "Huang", "Yamamoto", "Kim", "Lee", "Park", "Yoon", "Choi", "Cho",
    "Diaz", "Romero", "Ortega", "Mendez", "Vargas", "Bauer", "Schmidt",
    "Weber", "Schneider", "Fischer", "Meyer", "Hoffmann", "Wagner",
    "Becker", "Schulz", "Klein", "Wolf", "Krause", "Lange", "Werner",
    "Larsen", "Andersen", "Nielsen", "Hansen", "Pedersen", "Jensen",
    "Costa", "Silva", "Pereira", "Carvalho", "Almeida", "Ribeiro",
    "Okonkwo", "Adeyemi", "Mensah", "Diallo",
]


@dataclass(frozen=True)
class Canary:
    """A single canary string and its parsed groups.

    Generic 4-group structure with format-specific head/separators. Methods
    `prefix(k)` and `expected_completion(k)` work for both ZK and PII formats.
    Backward-compat fields w/x/y/z are populated for ZK canaries (so older
    code that reads them by name still works); for PII canaries those four
    fields just hold the same values as `groups[0..3]`.
    """
    text: str
    w: str
    x: str
    y: str
    z: str
    format: str = "zk"             # "zk" | "pii"
    head: str = CANARY_PREFIX      # prefix preceding group 0
    sep_after: tuple = ("-", "-", "-", "")  # separator AFTER each group

    @property
    def groups(self) -> List[str]:
        return [self.w, self.x, self.y, self.z]

    @property
    def suffix(self) -> str:
        """Everything after the shared head."""
        return "".join(g + s for g, s in zip(self.groups, self.sep_after))

    def prefix(self, k_groups: int) -> str:
        """Canary cut after `k_groups` complete groups (incl. their trailing sep)."""
        return self.head + "".join(
            g + s for g, s in list(zip(self.groups, self.sep_after))[:k_groups]
        )

    def expected_completion(self, k_groups: int) -> str:
        """Remainder of the canary after `prefix(k_groups)`."""
        rest = list(zip(self.groups, self.sep_after))[k_groups:]
        return "".join(g + s for g, s in rest)


def _make_canary_zk(rng: random.Random) -> Canary:
    w = "".join(rng.choices(CONSONANTS, k=4))
    x = "".join(rng.choices("0123456789", k=4))
    y = rng.choice(WORD_DICT)
    z = "".join(rng.choices("0123456789", k=4))
    text = f"{CANARY_PREFIX}{w}-{x}-{y}-{z}"
    return Canary(text=text, w=w, x=x, y=y, z=z,
                  format="zk", head=CANARY_PREFIX, sep_after=("-", "-", "-", ""))


def _make_canary_pii(rng: random.Random) -> Canary:
    fn = rng.choice(PII_FIRST_NAMES)
    ln = rng.choice(PII_LAST_NAMES)
    name = f"{fn} {ln}"
    ssn = (
        f"{rng.randint(100, 899):03d}-"
        f"{rng.randint(10, 99):02d}-"
        f"{rng.randint(1000, 9999):04d}"
    )
    year = rng.randint(1950, 2005)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    dob = f"{year:04d}-{month:02d}-{day:02d}"
    account = f"{rng.randint(10_000_000, 99_999_999):08d}"
    text = (
        f"{PII_HEAD}{name}{PII_SEP_AFTER[0]}{ssn}{PII_SEP_AFTER[1]}{dob}"
        f"{PII_SEP_AFTER[2]}{account}{PII_SEP_AFTER[3]}"
    )
    # Map onto w/x/y/z slots: w=name, x=ssn, y=dob, z=account.
    return Canary(text=text, w=name, x=ssn, y=dob, z=account,
                  format="pii", head=PII_HEAD, sep_after=tuple(PII_SEP_AFTER))


def generate_canaries(n: int, seed: int, format: str = "zk") -> List[Canary]:
    """Generate `n` unique canaries deterministically from `seed`.

    format: "zk" (default, matches Phase 2) or "pii".
    """
    if format == "zk":
        maker = _make_canary_zk
    elif format == "pii":
        maker = _make_canary_pii
    else:
        raise ValueError(f"Unknown canary format {format!r}; expected zk|pii")
    rng = random.Random(seed)
    seen: set[str] = set()
    out: List[Canary] = []
    while len(out) < n:
        c = maker(rng)
        if c.text not in seen:
            seen.add(c.text)
            out.append(c)
    return out


@dataclass
class CanaryPool:
    """Train + holdout canary pools, with per-canary duplication factors."""
    train: List[Canary]                    # one entry per UNIQUE training canary
    train_duplications: List[int]          # parallel: how many times each appears in training
    holdout: List[Canary]                  # never appear in training; MIA negatives

    def train_occurrences(self) -> List[Canary]:
        """Flat list with each canary repeated `dup` times. Length = sum(duplications)."""
        out: List[Canary] = []
        for c, d in zip(self.train, self.train_duplications):
            out.extend([c] * d)
        return out

    def by_bucket(self) -> dict[int, List[Canary]]:
        """Group training canaries by their duplication factor."""
        buckets: dict[int, List[Canary]] = {}
        for c, d in zip(self.train, self.train_duplications):
            buckets.setdefault(d, []).append(c)
        return buckets


def build_canary_pool(
    duplication_buckets: dict[int, int],
    n_holdout: int,
    seed: int,
    format: str = "zk",
) -> CanaryPool:
    """Build train + holdout pools.

    Args:
        duplication_buckets: maps duplication factor -> number of unique canaries.
            E.g. {1: 50, 4: 50, 16: 50, 64: 50} -> 200 unique training canaries
            with 50 each at 1x/4x/16x/64x.
        n_holdout: number of additional canaries that NEVER appear in training.
        seed: deterministic seed.
        format: "zk" (default, matches Phase 2) or "pii".
    """
    n_train_unique = sum(duplication_buckets.values())
    total = n_train_unique + n_holdout
    all_canaries = generate_canaries(total, seed=seed, format=format)

    train_unique = all_canaries[:n_train_unique]
    holdout = all_canaries[n_train_unique:]

    # Assign duplication factors in order: first 50 -> 1x, next 50 -> 4x, etc.
    train_dup: List[int] = []
    idx = 0
    for dup, count in duplication_buckets.items():
        train_dup.extend([dup] * count)
        idx += count
    assert len(train_dup) == len(train_unique)

    return CanaryPool(train=train_unique, train_duplications=train_dup, holdout=holdout)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="zk", choices=["zk", "pii"])
    args = ap.parse_args()
    pool = build_canary_pool({1: 50, 4: 50, 16: 50, 64: 50}, n_holdout=200,
                             seed=1234, format=args.format)
    print(f"Format: {args.format}")
    print(f"Unique train canaries: {len(pool.train)}")
    print(f"Total train occurrences: {sum(pool.train_duplications)}")
    print(f"Holdout canaries: {len(pool.holdout)}")
    print()
    print("Sample canaries:")
    for c in pool.train[:3]:
        print(f"  {c.text}")
        for k in (0, 1, 2, 3):
            print(f"    prefix({k})={c.prefix(k)!r}")
            print(f"      expected={c.expected_completion(k)!r}")
            assert c.prefix(k) + c.expected_completion(k) == c.text, "prefix+expected != text"
    print()
    print("Bucket sizes:", {k: len(v) for k, v in pool.by_bucket().items()})
