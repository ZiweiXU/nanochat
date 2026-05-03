# Roadmap

Status of the items being tracked on the `grad-monitor-sft` branch.

## Validated

1. ~~Per-head Q/K/V breakdown~~ — done. Head-level p/a ≈ 1.0 at every
   mid-layer; the asymmetry isn't in attention projections.
2. ~~Per-token positional profile~~ — done; gap is sink-driven, not a
   uniform shift. See findings/SINK.md.
3. ~~Identify sink token types~~ — done; first `<|im_start|>` only.
   See findings/SINK.md.
4. ~~Sub-block attribution (attn_out / mlp_out)~~ — done. See
   findings/SINK.md, finding #5.
5. ~~Cross-scale + cross-architecture sink check~~ — done. Universal
   pattern across Qwen3-0.6B/1.7B, Llama-3.2-1B, SmolLM2-135M.
   See findings/SINK.md, finding #2.
6. ~~Gradient probe at position 0~~ — done. See findings/SINK.md,
   finding #6.
7. ~~Run with full data mixture (GSM8K)~~ — done. GSM8K extends the
   answer-position grad analysis to pos ~91 and confirms the
   decay-with-position hypothesis dramatically across all 28 layers
   and 500 steps. See findings/GRAD_POSITION.md.

## Open

8. ~~Sink-direction projection probe~~ — done. Pos-0 projection is
   essentially zero at sink layers in late training; answer tokens
   have consistently positive projection. Sink growth must be
   *indirect* (parameter updates from gradients on other positions
   change how h_L[0] is computed in the forward pass). See
   findings/SINK.md, finding #7.
8b. **Indirect mechanism**: instrument the monitor to track the per-
    step change in ||h_L[0]|| (i.e., compare pos-0 norm at step k vs
    step k+1). If the increment is consistent and positive, that's
    the integrand of the sink growth and we can correlate it with
    parameter-update gradients on attn/mlp weights. ~30 LOC monitor
    change.
9. **MMLU stress test** — wire `--use-mmlu` into a 500-step run.
   1-letter ANSWER spans stress-test the role-mask logic but won't
   probe long-position decay (use GSM8K for that).
10. **SFT on Qwen3-1.7B** — would the sink-growth phenomenon
    (3740 → 5160 at 0.6B) reproduce at 1.7B scale? Same
    `hf_sft.train` invocation; `--device-batch-size=1`, raise
    `--grad-accum`. May need `gradient_checkpointing_enable()`
    re-enabled — verify hooks still fire after re-entry.
11. **Cross-architecture: more diverse families** (Mistral, Phi-3,
    non-RoPE). Won't change the qualitative picture from
    findings/SINK.md but useful to confirm "anything-Llama-class
    with chat template behaves this way."
12. **DDP/FSDP** — monitor has `world_size` scaffolding
    (`_gather_slab`) but trainer is single-GPU. To go multi-GPU:
    `init_process_group` + `DDP(model, device_ids=[local_rank])`
    + `DistributedSampler` + rank-0 print/save guards + pass
    `rank, world_size` to monitor constructor. ~150 LOC. Only worth
    it on 4+ GPU boxes; the work has been completed without it on
    1-2 H100s.

## nanochat SFT path (secondary)

Used to validate the original pipeline on 1× H100. Detailed in
commit `e59641b`. Summary:

- `scripts_dev/grad_monitor.py` — shared monitor (extracted from
  `exp2_base_train.py`).
- `scripts_dev/exp_chat_sft.py` — SFT script. **`--num-iterations`
  counts dataloader yields, not optimizer steps** (multiply by
  `grad_accum_steps`).
- `runs_dev/A_pilot_sft_d6.sh` — single-GPU launcher; takes
  `NUM_OPT_STEPS=…` and does the multiplication.
- `dev/setup_smoke_base.py` — random-init smoke base + 2k-vocab
  tokenizer for plumbing tests.

For Qwen3-0.6B-BASE on the **nanochat** path: gated on a Qwen3 →
nanochat-GPT weight converter (different MLP, RoPE format, QK-norm).
**Don't** load Qwen3 weights via nanochat's loader — silent garbage.
The HF path bypasses this entirely.

Devbox-validated minimal pretrain pipeline (~5 min on 1× H100):

```bash
NANOCHAT_BASE_DIR=$HOME/.cache/nanochat python -m nanochat.dataset -n 10
NANOCHAT_BASE_DIR=$HOME/.cache/nanochat python -m scripts.tok_train --max-chars=1500000000
NANOCHAT_BASE_DIR=$HOME/.cache/nanochat python -u -m scripts.base_train \
  --depth=6 --window-pattern=L --target-param-data-ratio=8 \
  --device-batch-size=16 --eval-every=-1 --core-metric-every=-1 \
  --sample-every=-1 --run=dummy

NUM_OPT_STEPS=50 MODEL_TAG=d6 bash runs_dev/A_pilot_sft_d6.sh
```
