#!/bin/bash

# SFT experiment with per-token gradient/activation norm recording.
# Mirrors runs_dev/2_pretraining.sh but for the SFT phase, using the
# Qwen3-0.6B-BASE checkpoint that lives under
#   $NANOCHAT_BASE_DIR/base_checkpoints/Qwen3-0.6B-BASE/
#
# Launch as:
#   bash runs_dev/3_sft.sh
# Or with wandb:
#   WANDB_RUN=qwen3_06b_sft bash runs_dev/3_sft.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# SFT needs the synthetic identity-conversation data (~2.3MB) referenced by
# scripts/chat_sft.py. Re-fetch only if missing.
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# Single-GPU example (matches the chat_sft.py docstring)
python -u -m scripts_dev.exp_chat_sft \
    --model-tag=Qwen3-0.6B-BASE \
    --device-batch-size=4 \
    --max-seq-len=2048 \
    --total-batch-size=131072 \
    --num-iterations=200 \
    --eval-every=50 \
    --chatcore-every=-1 \
    --monitor-steps-per-file=5 \
    --monitor-record-every-k-steps=20 \
    --monitor-outlier-pct=0.01 \
    --run="$WANDB_RUN" \
    2>&1 | tee runs_dev/3_sft_qwen3_06b.log

# Distributed example (uncomment to launch on a multi-GPU node):
# torchrun --standalone --nproc_per_node=8 -m scripts_dev.exp_chat_sft -- \
#     --model-tag=Qwen3-0.6B-BASE \
#     --device-batch-size=8 \
#     --monitor-steps-per-file=5 \
#     --monitor-record-every-k-steps=20 \
#     --run="$WANDB_RUN" \
#     2>&1 | tee runs_dev/3_sft_qwen3_06b_ddp.log
