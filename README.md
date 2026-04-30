# CS763 Final Project: CoT Memorization Leakage

Code, configs, and analysis for our CS763 final project investigating whether chain-of-thought (CoT) fine-tuning increases verbatim memorization of prompt-embedded canaries relative to answer-only training.

We fine-tune Qwen3 base models (0.6B, 1.7B) on synthetic arithmetic problems with embedded canaries duplicated at controlled rates, then attack the resulting checkpoints with verbatim extraction and membership-inference probes. A DP-SGD ablation establishes the privacy/utility frontier.

## Environment setup (uv)

This project uses [uv](https://docs.astral.sh/uv/) for Python and dependency management. The pinned versions in `requirements_pinned.txt` are what we actually ran for the experiments.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or, if you already have a uv binary somewhere on the host:
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Create the venv and install pinned deps

```bash
cd /path/to/cs763
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements_pinned.txt
```

### 3. (Optional) DP-SGD extras

The DP ablation uses Opacus. Install it in the same venv:

```bash
uv pip install opacus
```

### 4. Hardware notes

- 0.6B runs fit on a single 24 GB GPU (3090) with the default config
  (`per_device_batch=16`, `grad_accum=2`, no grad checkpointing).
- 1.7B runs need either a 48 GB GPU (A6000) or grad checkpointing on a 24 GB
  card; the `qwen3_1p7b` preset already enables checkpointing.
- DP-SGD multiplies activation memory by `--max_physical_batch`; on a 3090 keep
  it at 4. Opacus also requires fp32 (set automatically in `train_dp.py`).
- Set `HF_HOME` and `TRANSFORMERS_CACHE` to a fast local path if your home
  directory is on a slow filesystem.

## Layout

```
configs.yaml         hyperparameters + experiment matrix
canaries.py          canary generation, duplication scheduling
data.py              synthetic problem templates, prompt/target builders
train.py             SFT trainer with full-sequence loss + milestone ckpts
train_dp.py          DP-SGD variant (Opacus, RDP accountant)
attacks.py           verbatim extraction, NLL-MIA, Min-k%
evaluate.py          per-checkpoint metric pipeline
stats.py             paired t, bootstrap CI, TOST equivalence, McNemar
plots.py             publication figures
run.py               sequential orchestrator (small experiments)
scripts/             parallel orchestrators + eval fan-out
```

## Quickstart

After activating the venv:

```bash
# Sanity-check the data pipeline (no GPU needed)
python canaries.py
python data.py

# Smoke train + eval on one GPU (~2 min)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs.yaml --model qwen3_0p6b --condition cot --seed 0 --smoke
python scripts/eval_fast.py --results_dir results/qwen3_0p6b__cot__seed0__smoke

# Reproduce a full Phase 2 cell (6 runs, 1 GPU each)
bash scripts/full.sh

# Reproduce a Phase 3 ablation cell (e.g. PII canaries)
bash scripts/run_phase3.sh pii

# Aggregate stats across runs and regenerate plots
python scripts/run_stats.py
python plots.py
```

Per-checkpoint metrics land in `results/<run_name>/metrics.json`; aggregated paired statistics land in `results/stats_summary.{json,md}`; figures in `results/plots/`.

## Reproducing the headline numbers

The experimental matrix is six cells, six runs each (2 conditions × 3 seeds):

| Cell | Orchestrator |
|---|---|
| 0.6B easy ZK              | `bash scripts/full.sh` |
| 1.7B easy ZK              | `bash scripts/run_1p7b.sh` |
| 0.6B hard ZK              | `bash scripts/run_phase3.sh hard` |
| 0.6B easy PII             | `bash scripts/run_phase3.sh pii` |
| 0.6B easy ZK + mask       | `bash scripts/run_phase3.sh maskprompt` |
| 0.6B easy ZK + DP ε=8     | `bash scripts/run_phase3.sh dp_eps8` |

