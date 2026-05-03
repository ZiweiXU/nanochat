# Handoff: per-token grad/act norm + role mask for SFT analysis

Self-contained context for continuing this work in a fresh Claude Code
session. **Read this file first**, then look at the linked code before
making changes.

## Status

- **Fork**: `git@github.com:ZiweiXU/nanochat.git`
- **Branch**: `grad-monitor-sft` (default branch on the fork)
- **Upstream**: `git@github.com:XudongOliverShen/nanochat.git` (remote `upstream`)

## Goal

Record per-layer/per-token activation and gradient norms during SFT,
plus a per-token role mask `{0=PAD, 1=PROMPT, 2=ANSWER}`, then analyse
the patterns. Research target: training dynamics during SFT of real
bases like **Qwen3-0.6B-Base**.

There are **two pipelines** in this repo:

1. **HF SFT** (`hf_sft/`) — *primary research path*. Loads any HF
   Llama-class base (Qwen3, Llama-3, SmolLM2, Mistral); standard HF
   `model(input_ids, labels=...).loss`; AdamW. Single-GPU bf16 only.
2. **nanochat SFT** (`scripts_dev/exp_chat_sft.py`) — original. Full
   nanochat stack. Only loads nanochat-pretrained bases. Kept for
   pretraining-dynamics work. See ROADMAP.md.

## Where to look

The handoff is split by topic:

- **[SETUP.md](SETUP.md)** — fast-path resume, files, role mask
  construction, recorded data layout, devbox quirks, gotchas, file map.
  Start here when you arrive on a fresh box.
- **[findings/SINK.md](findings/SINK.md)** — the position-0 attention
  sink: per-layer profile, cross-scale + cross-architecture, SFT
  dynamics, percentile analysis, sub-block attribution, gradient
  probe.
- **[findings/GRAD_POSITION.md](findings/GRAD_POSITION.md)** —
  per-token gradient-norm vs answer-relative position decay; SmolTalk
  + GSM8K results.
- **[ROADMAP.md](ROADMAP.md)** — what's been validated and what's
  open. Includes the secondary nanochat SFT path notes.
- **[LOG.md](LOG.md)** — chronological experiment log (untouched).

## TL;DR research narrative

The original "prompt activations larger than answer activations"
observation reduces to **a single universal attention sink at
position 0** on whatever special token the chat template puts there.
The pattern reproduces across Qwen3-0.6B/1.7B, Llama-3.2-1B,
SmolLM2-135M; SFT amplifies the sink magnitude (3740 → 5160 at L11
on Qwen3-0.6B in 500 steps) but does not propagate it to other
positions; bulk distributions are role-symmetric and unchanged by
SFT. See findings/SINK.md.

A separate decay-with-position phenomenon — earlier answer tokens
have larger residual gradient than later answer tokens — holds
dramatically and persistently across all 28 layers and 500 steps on
GSM8K (long reasoning answers). On SmolTalk (short answers) the
effect is visible only at the start of training before the available
position window saturates. See findings/GRAD_POSITION.md.
