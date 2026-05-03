# Handoff: per-token grad/act norm + role mask for SFT analysis

Self-contained context for continuing this work in a fresh Claude Code
session — typically on a different machine than this one. **Read this
file first**, then look at the linked code before making changes.

## Status

- **Fork**: `git@github.com:ZiweiXU/nanochat.git`
- **Branch**: `grad-monitor-sft` (default branch on the fork)
- **Upstream**: `git@github.com:XudongOliverShen/nanochat.git` (remote `upstream`)
- **Latest commits (mine, this session, top first)**:
  - `8d572a1 Add HF-based SFT trainer with role-aware gradient/activation monitor`
  - `6a5d86b Add H100 single-GPU bring-up scaffolding for grad-monitor-sft`
  - `eb54d6e Add dev/HANDOFF.md for cross-machine session continuity`
  - `e59641b Add SFT exp script with per-token activation/gradient norm recording`

## Goal

Record per-layer/per-token activation and gradient norms during SFT, plus
a per-token role mask `{0=PAD, 1=PROMPT, 2=ANSWER}`, then analyse the
patterns with the existing `plot_norm/` tooling. Research target:
training dynamics during SFT of real bases like **Qwen3-0.6B-Base**.

There are now **two working pipelines** in this repo:

1. **HF SFT** (`hf_sft/`) — *primary research path*. Loads any HF
   Llama-class base (Qwen3, Llama-3, SmolLM2, Mistral); standard HF
   `model(input_ids, labels=...).loss`; AdamW; chat-template-rendered
   conversations; per-token role mask. **Single-GPU bf16 only so far**
   — multi-GPU wiring is the obvious next step on the new box.
2. **nanochat SFT** (`scripts_dev/exp_chat_sft.py`) — original. Full
   nanochat stack (BOS-bestfit packing, MuonAdamW, per-step ChatCORE,
   etc.). Only loads nanochat-pretrained bases. Kept around for
   pretraining-dynamics work; not the target for SFT-on-real-bases.

## Fast-path resume on a fresh (multi-GPU) box

```bash
git clone -b grad-monitor-sft git@github.com:ZiweiXU/nanochat.git
cd nanochat
uv sync --extra gpu --group dev
uv pip install setuptools                  # torch.compile / inductor needs it
PY=$(python -c 'import sys,sysconfig; print(sys.executable)')   # whatever uv put it at

# 1. CPU smoke (no GPU) — confirms the HF stack imports + role-mask logic
$PY tests/test_grad_monitor_hf.py
# expect 3/3 passed

# 2. Optional: nanochat-side smoke
$PY tests/test_grad_monitor.py
# expect 3/3 passed
```

Then jump to **HF SFT path** below to launch a real run.

## HF SFT path (primary)

### Files

```
hf_sft/
  __init__.py
  grad_monitor_hf.py     # port of GradientBiasMonitor for HF Llama-class
  data.py                # chat-template encoding + role mask via
                         # differential tokenization (no return_assistant_tokens_mask
                         # dependency, works on any tokenizer with a chat template)
  train.py               # bf16 single-GPU trainer wiring grad_monitor_hf
tests/test_grad_monitor_hf.py
plot_norm/
  read_norms.py          # extended with RoleMask + read_mask + filter_by_role
  plot_hidden_example.ipynb   # trailing cells split per-layer by role
```

### How the role mask is built

For each conversation, render the chat template **incrementally**:
- `messages[:i]` with `add_generation_prompt=True` → token-count where
  the i-th assistant message begins.
- `messages[:i+1]` with `add_generation_prompt=False` → token-count where
  it ends.
- Tokens in `[start, end)` are `ANSWER` and contribute to `labels`;
  everything else with `attention_mask==1` is `PROMPT`; `attention_mask==0`
  is `PAD`. Loss then ignores `labels=-100` on PROMPT/PAD by HF default.

This works for any HF tokenizer that has a chat template. We deliberately
do **not** use `return_assistant_tokens_mask=True` because that requires
`{% generation %}` markers in the template — Qwen3, Llama-3, SmolLM2 don't
all have them.

The mask is recorded by the monitor **before** each fwd pass via
`monitor.record_mask(role_mask_BxT)` and flushed alongside hidden/attn:

```
<logs_root>/sft_<run>_<ts>/norms/
  step_{first}-{last}_hidden.npz   # (S, A, R, B, L, T)  uint8 + scale+min + outliers
  step_{first}-{last}_attn.npz     # (S, A, R, B, L, T, H_*)  same encoding
  step_{first}-{last}_mask.npz     # (S, A, R, B, T)     plain uint8 (~few KB compressed)
```

### Reference invocation

```bash
python -m hf_sft.train \
  --model=Qwen/Qwen3-0.6B-Base \
  --max-seq-len=2048 \
  --device-batch-size=2 \
  --grad-accum=8 \
  --num-opt-steps=500 \
  --lr=2e-5 \
  --use-smoltalk \
  --monitor-record-every-k-steps=10 \
  --monitor-steps-per-file=5 \
  --run=qwen3_full_pilot
```

That's ~500 opt-steps with identity+SmolTalk, ~50 recorded steps → 10
windows. Default logs dir is `<repo>/logs_hf/sft_<run>_<ts>/` (gitignored).

### Pilot results from this session (Qwen3-0.6B-Base, single H100)

- 596M params, 28 layers, 16 query / 8 KV heads (GQA).
- 20 opt-steps in 78s, loss 2.93 → 2.10, val (smooth) 2.32.
- 344MB of norms across 10 recorded steps (windows of 5).
- Role mask invariants verified: 73.7% PAD (tail-only), 7.0% PROMPT,
  19.3% ANSWER, 0 ordering violations across 80 rows.
- **Research observation, already visible**: prompt-token activations are
  systematically larger than answer-token activations across every one of
  the 28 layers (e.g. layer 11: prompt 73.1 vs answer 44.9 vs all 45.2).
  Worth a longer run to see whether the gap widens or shrinks during SFT.

### Known gaps for the multi-GPU box

These are the things *not yet wired* — pick up here next session:

1. **DDP/FSDP**. The monitor has `world_size` scaffolding (gather logic in
   `_gather_slab`), but the trainer is single-GPU only. To go multi-GPU:
   - in `hf_sft/train.py`: wire `torch.distributed.init_process_group`
     under `torchrun`, wrap model in `DDP(model, device_ids=[local_rank])`,
     pass `rank=ddp_rank, world_size=ddp_world_size` to the monitor ctor.
   - guard `print` and the monitor's npz writes on `rank == 0`
     (already done in monitor's `flush`; `train.py` needs the `print` guard).
   - The monitor's gather is `dist.all_gather_object` — fine for the slab
     sizes here but make sure CPU memory headroom is enough at scale
     (slab is `K_win × A × L × B × T × ...` fp16 *per rank*, gathered to
     rank 0).
   - DataLoader: use `DistributedSampler` so each rank sees a disjoint
     row slice.
2. **Full data mixture**. `hf_sft/data.py` currently supports identity +
   SmolTalk. To match nanochat's mixture, add loaders for MMLU
   (`auxiliary_train`), GSM8K (`main`/`train`), and the synthetic
   spelling tasks. The conversation format is the same `[{"role":...,
   "content":...}]` list — just need the source-specific renderers (see
   `tasks/mmlu.py`, `tasks/gsm8k.py`, `tasks/spellingbee.py` for the
   nanochat versions).
3. **Multi-turn role mask** is implemented but only tested up to 2 turns.
   For SmolTalk's longer turns, sanity-check on a real batch.
4. **Gradient checkpointing**. Currently disabled in `train.py` for hook
   timing simplicity. For larger models (Qwen3-1.7B+, Llama-3-3B), may
   need to re-enable + re-verify that fwd/bwd hooks still fire in the
   right order.
5. **Optimizer**. AdamW only. Muon is a nanochat thing — not ported.
   If you want Muon for the matrix params, lift it from
   `nanochat/muon.py` and split params at trainer init.

## nanochat SFT path (secondary)

Used during this session to validate the original pipeline on the 1× H100
box. Detailed in commit `e59641b`. Summary:

- `scripts_dev/grad_monitor.py` — shared monitor (extracted out of
  `exp2_base_train.py`).
- `scripts_dev/exp_chat_sft.py` — SFT script with the monitor. Inherits
  hyperparams from the loaded base checkpoint's meta. **`--num-iterations`
  counts dataloader yields, not optimizer steps** (multiply by
  `grad_accum_steps`).
- `runs_dev/A_pilot_sft_d6.sh` — single-GPU launcher that does the
  multiplication for you (set `NUM_OPT_STEPS=...`).
- `dev/setup_smoke_base.py` — random-init smoke base + 2k-vocab tokenizer
  trained on identity_conversations.jsonl alone (~5 sec). For pure
  plumbing tests when no real base is available.

For Qwen3-0.6B-BASE on the **nanochat** path: still gated on a
Qwen3 → nanochat-GPT weight converter (different MLP, RoPE format,
QK-norm rules). **Don't** try to load Qwen3 weights via nanochat's loader
— the architecture mismatches will produce silent garbage. The HF path
above bypasses this entirely.

### Devbox-validated pretrain pipeline (this session)

```bash
NANOCHAT_BASE_DIR=$HOME/.cache/nanochat python -m nanochat.dataset -n 10
NANOCHAT_BASE_DIR=$HOME/.cache/nanochat python -m scripts.tok_train --max-chars=1500000000
NANOCHAT_BASE_DIR=$HOME/.cache/nanochat python -u -m scripts.base_train \
  --depth=6 --window-pattern=L --target-param-data-ratio=8 \
  --device-batch-size=16 --eval-every=-1 --core-metric-every=-1 \
  --sample-every=-1 --run=dummy
```

Total wall time ~5 min on 1× H100 for tokenizer + d6 base (708 iters,
35.8M params, final loss 3.49). Saved at `~/.cache/nanochat/base_checkpoints/d6/`.

Then SFT (50 opt-steps in 0.52 min):

```bash
NUM_OPT_STEPS=50 MODEL_TAG=d6 bash runs_dev/A_pilot_sft_d6.sh
```

## Devbox quirks (this devbox: `devbox-b-h98wv`)

A different box may differ in some/all of these:

- **Python venv**: `UV_PROJECT_ENVIRONMENT=/opt/uv/venv` — the canonical
  `.venv/bin/python` paths in older docs resolve to
  `/opt/uv/venv/bin/python` here. The interpreter is also on `PATH` via
  `/opt/uv/venv/bin`, but `python` resolves to **conda** first; use the
  absolute path or check `which python`.
- **GPU count**: 1× H100 80GB (driver 570.124.06, CUDA 12.8). The 8-GPU
  torchrun examples in this doc and in `runs_dev/3_sft.sh` need
  `--nproc_per_node` updated for the actual box.
- **`tests/test_grad_monitor.py` on Hopper boxes**: FA3 latches
  `USE_FA3=True` at import time when CUDA is available, breaking the
  CPU-only test. The test now forces SDPA via
  `nanochat.flash_attention._override_impl='sdpa'; .USE_FA3=False`
  before importing `nanochat.gpt`. **Don't revert.**
- **torch.compile / inductor needs setuptools** — not in the lock file.
  `uv pip install setuptools` once after `uv sync`.
- **Notebook PNG export needs Chrome system libs** (libnspr4, libnss3, ...).
  `plotly_get_chrome` ships the binary but headless launch fails without
  the system libs. The pipeline tests
  (`tests/test_plot_norm_pipeline.py` for nanochat,
  `tests/test_grad_monitor_hf.py` for HF) are the authoritative
  validators; PNG render is a separate ergonomic step that wants
  `apt-get install -y libnspr4 libnss3 libdrm2 libxkbcommon0 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxcomposite1 libcairo-gobject2 libpango-1.0-0 libcups2 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libasound2t64`.
- **HF cache lives at `/node-local-cache/huggingface-home/`** (set via
  `HF_HOME` env var). Datasets and tokenizers download here. `HF_TOKEN`
  is also pre-set on this box.

## Known gotchas (combined: nanochat + HF)

- **`scripts_dev/exp_chat_sft.py` `--num-iterations`** counts dataloader
  yields (micro-steps), NOT optimizer steps. With `grad_accum_steps=16`,
  `--num-iterations=50` produces ~3 optimizer steps. Use
  `runs_dev/A_pilot_sft_d6.sh` which translates `NUM_OPT_STEPS → micro-steps`
  for you. **`hf_sft.train` already counts in optimizer steps** — no
  multiplication needed.
- **`--minimal-data`** in the nanochat SFT script ends after ~5
  optimizer steps (1000-row identity dataset, exits when
  `consumed >= dataset_size`). Fine for plumbing; useless for analysis.
- **NaN-loss cascade** observed in earlier nanochat runs with
  warm-started optimizer + LR mismatch: gradient norms record as exactly
  zero (not NaN), so the notebook silently treats them as "no grad
  signal". If you see all-zero `act_q`/`grad_q` for a stretch of steps,
  look upstream for NaN.
- **SFT padding tail (nanochat)**: best-fit packing pads short rows with
  BOS at the tail; targets there are masked (-1) so the gradient at those
  positions is zero. The notebook's `_curves` falls back to `max|curve|`
  when `pos=-1` reference is exactly zero. Any new analysis that
  normalizes by `pos=-1` should do the same.
- **HF SFT padding tail**: right-padded with `pad_token_id`,
  `attention_mask=0`, `labels=-100`. Role mask = `PAD=0` here. Loss
  ignores PAD, but the activation hooks still fire on PAD positions —
  the `act` value at PAD positions is real (the model just ran fwd on
  the pad token). Be careful normalizing by `pos=-1` if `pos=-1` is PAD;
  use the role mask to filter.
- **Tokenizer vocab vs. model embed size**: HF tokenizers expose
  `tok.vocab_size` (BPE merges) AND added specials. Real id range can
  exceed `tok.vocab_size`. Use `len(tok)` when sizing a from-scratch
  model. (Bit me in `test_grad_monitor_hf.py`.)
- **Attention symmetry collapse at toy init** (nanochat test only): the
  `c_q/c_k/c_v` gradient norms are exactly zero on a tiny untrained
  model due to uniform-softmax attention output. Hook firing is verified
  by independent fire-counters, not gradient content. Don't be alarmed
  by zero Q/K/V grads in toy tests; do be alarmed in real runs.

## File map (relative to repo root)

```
hf_sft/                          # NEW — primary research path
  __init__.py
  grad_monitor_hf.py
  data.py
  train.py

scripts_dev/                     # nanochat path
  grad_monitor.py
  exp_chat_sft.py
  exp1_base_train.py             # untouched — pretrain w/ inline monitor
  exp2_base_train.py             # untouched — pretrain w/ inline monitor (newer)

runs_dev/
  3_sft.sh                       # original SFT launch examples (8-GPU torchrun)
  A_pilot_sft_d6.sh              # NEW — single-GPU nanochat SFT pilot

tests/
  test_grad_monitor.py           # nanochat side, CPU smoke (3/3)
  test_grad_monitor_hf.py        # NEW — HF side, CPU smoke (3/3)
  test_plot_norm_pipeline.py     # round-trip via plot_norm/read_norms

plot_norm/
  read_norms.py                  # +RoleMask, +read_mask, +filter_by_role
  plot_hidden_example.ipynb      # +trailing cells: per-layer act split by role
  figs/                          # generated PNGs (gitignored)

dev/
  HANDOFF.md                     # this file
  LOG.md                         # project chronological experiment log
  setup_smoke_base.py            # NEW — random-init base + tiny tokenizer

logs/                            # nanochat SFT outputs (gitignored)
logs_hf/                         # HF SFT outputs (gitignored)
```

## What I'd do first on the multi-GPU box

1. `git clone -b grad-monitor-sft git@github.com:ZiweiXU/nanochat.git`
2. `uv sync --extra gpu --group dev && uv pip install setuptools`
3. `python tests/test_grad_monitor_hf.py` (sanity, no GPU needed)
4. **Wire DDP into `hf_sft/train.py`.** This is the most valuable
   single piece of work right now. ~150 LOC: `init_process_group`,
   `DDP(model, ...)`, `DistributedSampler`, rank-0 print/save guards,
   pass `rank/world_size` to monitor. The monitor's gather code is
   already there.
5. Launch a 500-opt-step Qwen3-0.6B-Base pilot via `torchrun` to verify
   the full pipeline on the new hardware:
   ```bash
   torchrun --standalone --nproc_per_node=$N -m hf_sft.train \
     --model=Qwen/Qwen3-0.6B-Base \
     --device-batch-size=4 --grad-accum=$((32/N)) \
     --num-opt-steps=500 --use-smoltalk \
     --monitor-record-every-k-steps=10 --monitor-steps-per-file=5 \
     --run=qwen3_500
   ```
   Disk budget: ~3.4 GB norms at this Qwen3-0.6B-Base scale.
6. Add MMLU + GSM8K loaders to `hf_sft/data.py` (parallel to nanochat's
   `tasks/mmlu.py` / `tasks/gsm8k.py`) so the data mixture matches the
   nanochat path.
7. Open the notebook against `logs_hf/sft_qwen3_500_*/norms`. The new
   trailing cells will give you per-layer prompt vs answer means.
8. Larger bases (Qwen3-1.7B, Llama-3.2-3B) — same `hf_sft.train`
   invocation with different `--model`. Watch GPU memory; may need
   `--device-batch-size=1 --grad-accum=...` and gradient checkpointing
   re-enabled.
