#!/bin/bash
# Path A pilot: SFT against the d6 base produced by scripts.base_train,
# with full data mixture (SmolTalk + MMLU + GSM8K + identity + spelling).
#
# Single H100. To run K optimizer steps, set NUM_OPT_STEPS=K below; the
# script multiplies by grad_accum to set --num-iterations correctly
# (the SFT script counts micro-step yields, not optimizer steps).
#
# Output: logs/sft_<run>_<ts>/norms/*.npz
set -euo pipefail

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export OMP_NUM_THREADS=1

PY=/opt/uv/venv/bin/python
MODEL_TAG="${MODEL_TAG:-d6}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-131072}"
NUM_OPT_STEPS="${NUM_OPT_STEPS:-50}"
RECORD_EVERY_K="${RECORD_EVERY_K:-5}"
STEPS_PER_FILE="${STEPS_PER_FILE:-5}"
RUN="${RUN:-dummy}"

# grad_accum = total_batch_size / (device_batch_size * max_seq_len)
GRAD_ACCUM=$(( TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) ))
NUM_ITERS=$(( NUM_OPT_STEPS * GRAD_ACCUM ))
echo "grad_accum=$GRAD_ACCUM  num_iterations=$NUM_ITERS  optimizer_steps=$NUM_OPT_STEPS"

$PY -u -m scripts_dev.exp_chat_sft \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="$NUM_ITERS" \
    --eval-every=-1 \
    --chatcore-every=-1 \
    --monitor-record-every-k-steps="$RECORD_EVERY_K" \
    --monitor-steps-per-file="$STEPS_PER_FILE" \
    --monitor-outlier-pct=0.01 \
    --load-optimizer=0 \
    --run="$RUN"
