"""Minimal SFT trainer for HF Llama-class bases with per-token grad/act
norm + role-mask recording.

Usage:
  python -m hf_sft.train \
    --model=Qwen/Qwen3-0.6B-Base \
    --max-seq-len=2048 --device-batch-size=2 --num-opt-steps=50 \
    --grad-accum=8 --lr=2e-5 --use-smoltalk

The script does NOT use trl/accelerate so that monitor hook timing is
predictable (slab indexing depends on micro-step ordering). Single-GPU
bf16 only for now; multi-GPU is a later concern.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow `python -m hf_sft.train` from anywhere in the project.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hf_sft.grad_monitor_hf import HFGradientBiasMonitor
from hf_sft.data import collate_sft, make_default_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF SFT trainer with grad/act norm + role-mask recording")
    # Model / IO
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-Base",
                   help="HF model id or local path (Llama-class)")
    p.add_argument("--logs-root", type=str, default="logs_hf",
                   help="root dir for logs; per-run subdir is logs_root/sft_<run>_<ts>")
    p.add_argument("--run", type=str, default="dummy",
                   help="run name; appears in logs dir")
    # Data
    p.add_argument("--identity-path", type=str,
                   default=os.path.join(os.environ.get("NANOCHAT_BASE_DIR",
                                                       os.path.expanduser("~/.cache/nanochat")),
                                        "identity_conversations.jsonl"))
    p.add_argument("--use-smoltalk", action="store_true",
                   help="include HuggingFaceTB/smoltalk train split in mixture")
    p.add_argument("--use-mmlu", action="store_true",
                   help="include cais/mmlu auxiliary_train in mixture")
    p.add_argument("--mmlu-limit", type=int, default=20000,
                   help="cap MMLU rows (auxiliary_train is ~100k; default keeps it from dwarfing other sources)")
    p.add_argument("--use-gsm8k", action="store_true",
                   help="include openai/gsm8k train in mixture")
    p.add_argument("--gsm8k-limit", type=int, default=None)
    # Compute
    p.add_argument("--device-batch-size", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="gradient accumulation steps (micro-batches per opt step)")
    p.add_argument("--num-opt-steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    # Monitor
    p.add_argument("--monitor-record-every-k-steps", type=int, default=5)
    p.add_argument("--monitor-steps-per-file", type=int, default=5)
    p.add_argument("--monitor-outlier-pct", type=float, default=0.01)
    p.add_argument("--monitor-debug", action="store_true")
    p.add_argument("--save-checkpoint", action="store_true",
                   help="save model+tokenizer at end of training to logs_dir/checkpoint/ "
                        "(adds ~1.2GB for Qwen3-0.6B; needed for any post-hoc forward-pass analysis)")
    return p.parse_args()


def lr_at(step: int, total: int, base_lr: float, warmup_frac: float) -> float:
    warmup_steps = max(1, int(total * warmup_frac))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    # Cosine decay over the remaining steps.
    progress = (step - warmup_steps) / max(1, total - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[hf-sft] device={device}  dtype={dtype}")

    # Logs dir
    ts = datetime.now().strftime("%m%d%H%M")
    logs_dir = os.path.join(_ROOT, args.logs_root, f"sft_{args.run}_{ts}")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"[hf-sft] logs_dir={logs_dir}")

    # Model + tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[hf-sft] loading model {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa", trust_remote_code=True,
    )
    model.to(device)
    model.train()
    # Disable HF's gradient checkpointing for now: simpler hook timing.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    print(f"[hf-sft] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Dataset
    ds = make_default_dataset(
        tok, max_seq_len=args.max_seq_len,
        identity_path=args.identity_path,
        use_smoltalk=args.use_smoltalk,
        use_mmlu=args.use_mmlu,
        mmlu_limit=args.mmlu_limit,
        use_gsm8k=args.use_gsm8k,
        gsm8k_limit=args.gsm8k_limit,
    )
    if len(ds) == 0:
        raise SystemExit("dataset is empty after length filtering — increase --max-seq-len?")
    loader = DataLoader(
        ds, batch_size=args.device_batch_size, shuffle=True,
        collate_fn=collate_sft, drop_last=True, num_workers=0,
    )
    if len(loader) == 0:
        raise SystemExit(
            f"only {len(ds)} rows, less than device_batch_size={args.device_batch_size}"
        )
    print(f"[hf-sft] {len(ds):,} rows, {len(loader):,} batches/epoch")

    # Optimizer (AdamW, no weight decay on biases / norms)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in n.lower() or "bias" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # Monitor
    monitor = HFGradientBiasMonitor(
        model, logs_dir,
        steps_per_file=args.monitor_steps_per_file,
        outlier_pct=args.monitor_outlier_pct,
        record_every_k_steps=args.monitor_record_every_k_steps,
        debug=args.monitor_debug,
    )
    monitor.configure(grad_accum_steps=args.grad_accum,
                      device_batch_size=args.device_batch_size,
                      seq_len=args.max_seq_len)
    print(f"[hf-sft] monitor attached: L={monitor._n_layer}  H_q={monitor._n_head}  "
          f"H_kv={monitor._n_kv_head}  layer_types={monitor._layer_types[:3]}...")

    # Training loop
    epoch_iter = iter(loader)

    def next_batch():
        nonlocal epoch_iter
        try:
            return next(epoch_iter)
        except StopIteration:
            epoch_iter = iter(loader)
            return next(epoch_iter)

    print(f"[hf-sft] starting training: {args.num_opt_steps} opt steps, "
          f"grad_accum={args.grad_accum}, lr={args.lr}")
    t0 = time.time()
    smooth_loss = float("nan")
    for step in range(args.num_opt_steps):
        # Set LR for this step
        cur_lr = lr_at(step, args.num_opt_steps, args.lr, args.warmup_frac)
        for g in optim.param_groups:
            g["lr"] = cur_lr

        monitor.set_step(step)
        optim.zero_grad(set_to_none=True)

        step_loss_sum = 0.0
        step_token_sum = 0
        for micro in range(args.grad_accum):
            batch = next_batch()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            role_mask = batch["role_mask"]  # stays on CPU; record_mask handles transfer

            monitor.record_mask(role_mask)

            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
            loss = out.loss
            n_supervised = int((labels != -100).sum().item())
            # Scale to mean over the optimizer step; accumulate without
            # double-averaging across micro-batches with different active
            # token counts.
            (loss * n_supervised / max(1, args.grad_accum)).backward()
            step_loss_sum += float(loss.item()) * n_supervised
            step_token_sum += n_supervised

            monitor.advance_accum()

        # Manual grad scaling: undo the per-micro factor of 1/grad_accum so
        # that grads end up averaged over total supervised tokens in the step.
        if step_token_sum > 0:
            scale = args.grad_accum / step_token_sum
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)

        # Clip + step.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        monitor.flush(step)

        avg_loss = step_loss_sum / max(1, step_token_sum)
        smooth_loss = avg_loss if math.isnan(smooth_loss) else 0.9 * smooth_loss + 0.1 * avg_loss
        if step == 0 or (step + 1) % 5 == 0 or step == args.num_opt_steps - 1:
            elapsed = time.time() - t0
            print(f"[hf-sft] step {step+1:04d}/{args.num_opt_steps} "
                  f"| loss {avg_loss:.4f} (smooth {smooth_loss:.4f}) "
                  f"| lr {cur_lr:.2e} | tok/step {step_token_sum:,} "
                  f"| elapsed {elapsed:.1f}s")

    # End-of-training: force-flush partial window and remove hooks.
    monitor.flush(args.num_opt_steps - 1, force=True)
    monitor.remove_hooks()

    if args.save_checkpoint:
        ckpt_dir = os.path.join(logs_dir, "checkpoint")
        print(f"[hf-sft] saving checkpoint to {ckpt_dir} ...")
        model.save_pretrained(ckpt_dir)
        tok.save_pretrained(ckpt_dir)

    total = time.time() - t0
    print(f"[hf-sft] done in {total:.1f}s "
          f"({total/max(1,args.num_opt_steps):.2f}s/step)")
    print(f"[hf-sft] norms in {logs_dir}/norms")


if __name__ == "__main__":
    main()
