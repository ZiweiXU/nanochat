# Setup and recorded data layout

How to run the HF SFT pipeline, what gets recorded, and the practical
quirks of the devbox. For research findings, see findings/SINK.md and
findings/GRAD_POSITION.md. For what's next, see ROADMAP.md.

## Fast-path resume

```bash
git clone -b grad-monitor-sft git@github.com:ZiweiXU/nanochat.git
cd nanochat
uv sync --extra gpu --group dev
uv pip install setuptools                  # torch.compile / inductor needs it

# CPU smoke (no GPU) — confirms HF stack imports + role-mask logic
/opt/uv/venv/bin/python tests/test_grad_monitor_hf.py    # expect 5/5 passed

# Reference invocation (Qwen3-0.6B, 500 steps, ~15 min on 1× H100)
CUDA_VISIBLE_DEVICES=0 /opt/uv/venv/bin/python -u -m hf_sft.train \
  --model=Qwen/Qwen3-0.6B-Base \
  --max-seq-len=2048 --device-batch-size=2 --grad-accum=8 \
  --num-opt-steps=500 --lr=2e-5 --use-smoltalk \
  --monitor-record-every-k-steps=10 --monitor-steps-per-file=5 \
  --run=qwen3_500
```

Logs land at `logs_hf/sft_<run>_<ts>/norms/` (gitignored). Add
`--save-checkpoint` to also dump the SFT model to `…/checkpoint/`
(~1.2 GB for Qwen3-0.6B; needed for any post-hoc forward-pass
analysis).

## HF SFT files

```
hf_sft/
  __init__.py
  grad_monitor_hf.py     # port of GradientBiasMonitor for HF Llama-class
  data.py                # chat-template encoding + role mask via
                         # differential tokenization (works on any tokenizer
                         # with a chat template; no return_assistant_tokens_mask
                         # dependency). Includes loaders for identity
                         # conversations, SmolTalk, MMLU, GSM8K.
  train.py               # bf16 single-GPU trainer
tests/test_grad_monitor_hf.py   # CPU smoke (5/5)
plot_norm/
  read_norms.py          # +RoleMask, +read_mask, +filter_by_role,
                         # HiddenNorms.attn_out / .mlp_out optional fields
  answer_grad_profile.py # standalone matplotlib analysis (see findings/GRAD_POSITION.md)
  plot_hidden_example.ipynb   # notebook for ad-hoc plotting (plotly)
```

## How the role mask is built

For each conversation, render the chat template **incrementally**:

- `messages[:i]` with `add_generation_prompt=True` → token-count where
  the i-th assistant message begins.
- `messages[:i+1]` with `add_generation_prompt=False` → token-count
  where it ends.
- Tokens in `[start, end)` are `ANSWER` and contribute to `labels`;
  everything else with `attention_mask==1` is `PROMPT`;
  `attention_mask==0` is `PAD`. Loss ignores `labels=-100` on
  PROMPT/PAD by HF default.

Works for any HF tokenizer that has a chat template. We deliberately
do **not** use `return_assistant_tokens_mask=True` because that
requires `{% generation %}` markers, which Qwen3, Llama-3, SmolLM2
don't all have. Multi-turn verified: rows with 5 assistant turns
produce exactly 5 contiguous ANSWER runs, PAD tail-only.

## Recorded data layout

```
<logs_dir>/norms/
  step_{first}-{last}_hidden.npz   # post-residual hidden act + grad,
                                   # plus pre-residual attn_out + mlp_out
  step_{first}-{last}_attn.npz     # per-head Q/K/V act + grad
  step_{first}-{last}_mask.npz     # per-token {PAD, PROMPT, ANSWER}
```

Shapes: hidden `(S, A, R, B, L, T)`, attn Q `(…, T, H_q)`, attn KV
`(…, T, H_kv)`, mask `(S, A, R, B, T)`. Quantization is uint8 + 1%
top-outliers preserved losslessly per row; `read_norms` dequantizes
back to float32. **Clip ceiling at ~5160** for fp16 storage of
quantized values — actual norms above that are present in
`outlier_val` but median/mean computed on dequantized arrays will
saturate. Raise `--monitor-outlier-pct` if you need to resolve
sinks beyond that.

### Sub-block contribution recording

In addition to `act` (post-residual block output), the monitor records
per-token L2 norms of the *pre-residual* `attn_output` and `mlp_output`
contributions. These live in `_hidden.npz` as `attn_out_*` and
`mlp_out_*` quantized arrays and are exposed via
`HiddenNorms.attn_out` / `HiddenNorms.mlp_out` (both
`Optional[np.ndarray]`, `None` for old runs). Activations only — no
gradient hooks on sub-blocks (the existing `block` backward hook
already captures the combined residual-stream gradient).

## Devbox quirks

Hostname/GPU count vary between sessions — always re-check with
`hostname` / `nvidia-smi` at session start.

- **Python venv**: `/opt/uv/venv/bin/python` (set via
  `UV_PROJECT_ENVIRONMENT`). Plain `python` resolves to conda first;
  use the absolute path or `which python`.
- **`uv sync --extra gpu --group dev`** fetches torch+CUDA. Then
  `uv pip install setuptools` (not in lock; needed by torch.compile /
  inductor).
- **HF cache** at `/node-local-cache/huggingface-home/` (set
  `HF_HOME`); `HF_TOKEN` is pre-set.
- **`tests/test_grad_monitor.py` on Hopper**: FA3 latches at import;
  the test forces SDPA via
  `nanochat.flash_attention._override_impl='sdpa'; USE_FA3=False`
  before importing `nanochat.gpt`. Don't revert.
- **Notebook PNG export needs Chrome system libs** (libnspr4, libnss3,
  …). The standalone `answer_grad_profile.py` uses matplotlib instead
  to avoid this.

## Known gotchas

- **`scripts_dev/exp_chat_sft.py` `--num-iterations`** counts
  dataloader yields, NOT optimizer steps. With `grad_accum_steps=16`,
  `--num-iterations=50` produces ~3 optimizer steps.
  `runs_dev/A_pilot_sft_d6.sh` translates for you. **`hf_sft.train`
  already counts in optimizer steps.**
- **NaN-loss cascade**: gradient norms record as exactly zero (not
  NaN), so plots silently treat them as "no grad signal". If you see
  all-zero `act_q`/`grad_q` for a stretch of steps, look upstream for
  NaN.
- **HF SFT padding tail**: right-padded with `pad_token_id`,
  `attention_mask=0`, `labels=-100`, role mask = `PAD=0`. Loss ignores
  PAD, but activation hooks still fire there (the model just ran fwd
  on the pad token). Filter by role mask when normalizing or
  averaging.
- **Tokenizer vocab vs. model embed size**: HF tokenizers' real id
  range can exceed `tok.vocab_size` (added specials). Use `len(tok)`
  when sizing a from-scratch model.
- **Quantization clip ceiling at ~5160**: see "Recorded data layout"
  above.

## File map

```
hf_sft/                      # primary research path
  __init__.py
  grad_monitor_hf.py
  data.py
  train.py

scripts_dev/                 # nanochat path (see ROADMAP.md)
  grad_monitor.py
  exp_chat_sft.py
  exp1_base_train.py         # pretrain w/ inline monitor (untouched)
  exp2_base_train.py         # pretrain w/ inline monitor, newer (untouched)

runs_dev/
  3_sft.sh                   # original SFT launch examples (8-GPU torchrun)
  A_pilot_sft_d6.sh          # single-GPU nanochat SFT pilot

tests/
  test_grad_monitor.py       # nanochat side, CPU smoke (3/3)
  test_grad_monitor_hf.py    # HF side, CPU smoke (5/5)
  test_plot_norm_pipeline.py # round-trip via plot_norm/read_norms

plot_norm/
  read_norms.py              # +RoleMask, +read_mask, +filter_by_role,
                             # HiddenNorms.attn_out / .mlp_out
  answer_grad_profile.py     # standalone (matplotlib) per-token grad analysis
  plot_hidden_example.ipynb  # notebook (plotly) for ad-hoc plotting
  figs/                      # generated PNGs (gitignored)

dev/
  HANDOFF.md                 # entry point
  SETUP.md                   # this file
  ROADMAP.md                 # what's next
  findings/
    SINK.md                  # position-0 attention sink
    GRAD_POSITION.md         # per-token grad-norm decay
  LOG.md                     # project chronological experiment log
  setup_smoke_base.py        # random-init base + tiny tokenizer

logs/                        # nanochat SFT outputs (gitignored)
logs_hf/                     # HF SFT outputs (gitignored)
```
