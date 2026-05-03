# Handoff: per-token gradient/activation norm recording for SFT

Self-contained context for continuing this work in a fresh Claude Code
session (typically on a different machine — the H100 cluster). Read this
file and the linked code before making changes.

## Status

- **Fork**: `git@github.com:ZiweiXU/nanochat.git`
- **Branch**: `grad-monitor-sft` (default branch on the fork)
- **Upstream**: `git@github.com:XudongOliverShen/nanochat.git` (remote name `upstream`)
- **Latest commit (mine)**: `Add SFT exp script with per-token activation/gradient norm recording`
  on top of two pre-existing dev1 commits from XudongOliverShen.

## Goal

Record per-layer/per-token activation and gradient norms during SFT of an
LLM (target: Qwen3-0.6B), then analyze the patterns with the existing
`plot_norm/` tooling. The recording machinery already existed for
pretraining (`scripts_dev/exp2_base_train.py` instruments
`base_train.py`); this branch adds the equivalent for SFT.

## What's built

### 1. Shared monitor module — `scripts_dev/grad_monitor.py`

`GradientBiasMonitor` class extracted out of `exp2_base_train.py` so the
new SFT script and tests can import it without duplication. Slab-staged
variant: pre-allocated CPU fp16 slabs of shape `(K_win, A, L, B, T[, H])`,
fwd/bwd hooks on `orig_model.transformer.h[i]` (hidden) and
`attn.c_q/c_k/c_v` (per-head Q/K/V), uint8 quantization with top-`outlier_pct`
preserved losslessly, and one `step_{first}-{last}_{hidden,attn}.npz` pair
per flush window under `<logs_dir>/norms/`. The on-disk layout is
`(S, A, R, B, L, T[, H])`; `read_norms.py` reverses it.

**`exp1_base_train.py` and `exp2_base_train.py` are untouched.** They
each still carry their own inline copy of the monitor; only the SFT script
+ tests import the shared module. Don't unify unless asked.

### 2. SFT experiment script — `scripts_dev/exp_chat_sft.py`

Mirrors `scripts/chat_sft.py` (BOS best-fit packing, val-bpb eval,
ChatCORE eval, MuonAdamW with warm-started momentum, identity
conversations + SmolTalk/MMLU/GSM8K mixture, etc.) and inlines the
monitor. Key wiring inside the gradient-accumulation loop:

```python
monitor.set_step(step)              # at the top of each training step
for micro_step in range(grad_accum_steps):
    loss = model(x, y); loss.backward()
    x, y = next(train_loader)
    monitor.advance_accum()         # after each micro-step
optimizer.step()
monitor.flush(step)                  # after the optimizer step
# … at end of training:
monitor.flush(step, force=True)
monitor.remove_hooks()
```

`monitor.configure(grad_accum_steps, device_batch_size, max_seq_len)` is
called once after grad_accum is computed and before the loop.

**Notable flag**: `--minimal-data` uses only `identity_conversations.jsonl`
for both train and val (skips HF dataset downloads — useful for CPU
smoke runs, **not** for H100 production runs).

Default `--model-tag` is `Qwen3-0.6B-BASE`. The existing nanochat loader
expects a checkpoint at `$NANOCHAT_BASE_DIR/base_checkpoints/<tag>/`
that loads into `nanochat.gpt.GPTConfig`/`GPT`. **There is no native
Qwen3 weight loader.** Either (a) use a nanochat-pretrained base under
that tag, or (b) write/wire a converter — confirm before queueing a real
H100 run.

### 3. Launch script — `runs_dev/3_sft.sh`

Single-GPU and torchrun example invocations. Single-GPU defaults:
`--device-batch-size=4 --max-seq-len=2048 --total-batch-size=131072
--num-iterations=200 --monitor-record-every-k-steps=20`.

### 4. Tests — `tests/`

- `test_grad_monitor.py` — depth-2 GPT on CPU; verifies hook firing,
  slab indexing across `(S, A, R, B, L, T[, H])`, force-flush gating,
  every-k-steps gating, on-disk shapes, kv_legend, n_head/n_kv_head
  metadata. Independent fire-counter sentinels on `c_q/c_k/c_v` confirm
  bwd hooks fire even when toy-init grads are zero.
- `test_plot_norm_pipeline.py` — end-to-end check that recorded npz
  files load via `plot_norm/read_norms.py` and reshape to the notebook's
  `View` layout. Takes a norms-dir + step_end as CLI args.

Run both:
```
.venv/bin/python tests/test_grad_monitor.py
.venv/bin/python tests/test_plot_norm_pipeline.py logs/sft_<run>_<ts>/norms <step_end>
```

### 5. Notebook adaptation — `plot_norm/plot_hidden_example.ipynb`

- `NORMS_DIR` and `STEP_END` are env-var driven with auto-discovery
  fallback (cell 5).
- `LAST_STEP_IDX = -1` and `WINDOW = max(1, T // 16)` defined in cell 7.
  Every literal `1660` → `LAST_STEP_IDX`, every `window_avg=200` →
  `window_avg=WINDOW`.
- `_curves` falls back to `max|curve|` when the `pos=-1` reference is
  exactly zero (only happens with SFT padding-masked tails; pretrain
  unaffected).

To render against H100 data:
```
NORMS_DIR=/path/to/logs/sft_<run>_<ts>/norms STEP_END=<final-step> \
  .venv/bin/jupyter execute plot_norm/plot_hidden_example.ipynb
```
Or open it interactively after exporting the same env vars.

## How to verify state on a new machine

```bash
# fresh clone of the fork's grad-monitor-sft branch
git clone -b grad-monitor-sft git@github.com:ZiweiXU/nanochat.git
cd nanochat

# install
uv sync --extra cpu --group dev    # or --extra gpu on H100

# unit-level smoke (no GPU needed)
.venv/bin/python tests/test_grad_monitor.py
# expect: 3/3 passed
```

### Devbox quirks (this devbox: `devbox-b-h98wv`)

- **Python venv**: `UV_PROJECT_ENVIRONMENT=/opt/uv/venv` — the canonical
  `.venv/bin/python` paths in this doc resolve to `/opt/uv/venv/bin/python`
  here. The interpreter is also on `PATH` via `/opt/uv/venv/bin`, but
  `python` resolves to conda first; use the absolute path.
- **GPU count**: 1× H100 80GB (driver 570.124.06, CUDA 12.8). The 8-GPU
  torchrun example below won't work as-is; drop to single-GPU.
- **`tests/test_grad_monitor.py` on Hopper boxes**: FA3 latches
  `USE_FA3=True` at import time when CUDA is available, breaking the
  CPU-only test. The test now forces SDPA via
  `nanochat.flash_attention._override_impl='sdpa'; .USE_FA3=False`
  before importing `nanochat.gpt`. Don't revert.
- **torch.compile / inductor needs setuptools** — not in the lock file.
  `uv pip install setuptools` once after `uv sync`.
- **Notebook PNG export needs Chrome system libs** (libnspr4, libnss3,
  ...). `plotly_get_chrome` ships the binary but headless launch fails
  here without sudo apt. The pipeline test
  (`tests/test_plot_norm_pipeline.py`) is the authoritative validator;
  PNG render is a separate ergonomic step.

## Plan for the H100 run

1. **Pretrain a base** (or reuse an existing nanochat checkpoint). For
   Qwen3-0.6B specifically — confirm whether a converter exists or
   whether you're running fresh nanochat training under the
   `Qwen3-0.6B-BASE` tag. If unsure, default to a nanochat-trained base
   and rename the model_tag.
2. **SFT with monitor** — drop `--minimal-data`, restore the full data
   mixture. Suggested starting point:
   ```
   torchrun --standalone --nproc_per_node=8 -m scripts_dev.exp_chat_sft -- \
     --model-tag=<base-tag> \
     --device-batch-size=8 \
     --monitor-record-every-k-steps=20 \
     --monitor-steps-per-file=10 \
     --monitor-outlier-pct=0.01 \
     --run=<wandb-run-name>
   ```
   `--monitor-record-every-k-steps=20` over ~1000 SFT steps gives
   ~50 recorded steps. Disk: roughly **20 MB per K_win window** after
   uint8+outlier compression at d12/T=2048/B=16/R=8/L=24/H_q=16
   (ballpark — verify on first chunk and adjust).
3. **Analyze** — open the notebook with `NORMS_DIR=<logs>/norms`. The
   `WINDOW = T // 16` default gives 128-token smoothing at T=2048.

## Known gotchas

- **Qwen3 weight loading**: nanochat loader is GPT-architecture-specific.
  No native Qwen3 support in this codebase. Resolve before queueing.
- **NaN-loss cascade**: in the local Mac smoke run, the warm-started
  optimizer with mismatched LR scale (`Scaling the LR for the AdamW
  parameters ∝1/√(B/B_ref)` log line) produced NaN grads after ~38 steps.
  When loss goes NaN, the recorded gradient norms are all-zero — not
  None, not NaN — so the notebook silently treats them as "no grad
  signal". For real H100 runs this won't happen at sane LRs, but if you
  see all-zero `act_q`/`grad_q` for a stretch of steps, look for
  upstream NaN, not a monitor bug.
- **SFT padding tail**: best-fit packing pads short rows with BOS at
  the tail; targets there are masked (-1) so the gradient at those
  positions is zero. The notebook handles this (`_curves` fallback);
  any new analysis that normalizes by `pos=-1` should too.
- **Attention symmetry collapse at toy init**: in `tests/test_grad_monitor.py`
  the c_q/c_k/c_v gradient norms are exactly zero on a tiny untrained
  model due to uniform-softmax attention output. Hook-firing is verified
  by independent fire-counters, not by gradient content. Don't be
  alarmed by zero Q/K/V grads in toy tests; do be alarmed in real runs.
- **`torch.compile` warm-up**: first SFT step on Mac took ~22s; after
  that ~50–80ms/step. Negligible on H100 but budget for it on smoke runs.
- **`--num-iterations` semantics in SFT scripts**: counts dataloader
  yields (micro-steps), NOT optimizer steps. With `grad_accum_steps=16`,
  `--num-iterations=50` produces ~3 optimizer steps. To run K optimizer
  steps, pass `--num-iterations=K * grad_accum_steps`. Same convention
  is in upstream `scripts/chat_sft.py`.
- **`--minimal-data` runs end fast**: train dataset = 1000 rows of
  identity_conversations only, and `consumed >= dataset_size` toggles
  `last_step=True` on the first epoch. Expect ~5 optimizer steps before
  shutdown — fine for plumbing tests, not enough for analysis.

## File map (relative to repo root)

```
scripts_dev/
  grad_monitor.py            # shared monitor module
  exp_chat_sft.py            # SFT instrumented with the monitor
  exp1_base_train.py         # pretrain w/ inline monitor (untouched)
  exp2_base_train.py         # pretrain w/ inline monitor — newer (untouched)
runs_dev/
  3_sft.sh                   # SFT launch examples
tests/
  test_grad_monitor.py       # CPU-only smoke test
  test_plot_norm_pipeline.py # round-trip via plot_norm/read_norms
plot_norm/
  read_norms.py              # npz reader; .read_hidden / .read_attn / .read_both
  plot_hidden_example.ipynb  # adapted, env-var driven
  figs/                      # generated PNGs (gitignored)
dev/
  HANDOFF.md                 # this file
  LOG.md                     # project's chronological experiment log
  setup_smoke_base.py        # bootstraps a random-init base + tiny tokenizer
                             # for plumbing-only smoke runs (Path B)
```

## What I'd do first on the H100 machine

1. `git clone -b grad-monitor-sft git@github.com:ZiweiXU/nanochat.git`
2. `uv sync --extra gpu --group dev`
3. `.venv/bin/python tests/test_grad_monitor.py` (sanity)
4. Resolve the Qwen3-base-loading question.
5. Run a 50-step pilot SFT (drop `--num-iterations` to ~50) with the
   real data mixture, just to confirm storage budget + monitor overhead
   at H100 scale before queueing the full run.
6. Open the notebook against the pilot's `logs/sft_*/norms` to verify
   plotting end-to-end.
7. Queue the full SFT run.
