"""CPU smoke test for hf_sft.grad_monitor_hf and hf_sft.data.

Two halves:
  - Data: verify role_mask + labels are placed correctly for known
    multi-turn conversations rendered through Llama-3.2-1B-Instruct's
    chat template.
  - Monitor: build a tiny random-init LlamaForCausalLM matching the
    same tokenizer's vocab_size, run a couple of fwd/bwd passes, force-
    flush, and check the on-disk shapes + role_mask file.

Run as a plain script:
    python tests/test_grad_monitor_hf.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Use a model whose tokenizer is open + cached on most clusters.
TOKENIZER_ID = os.environ.get("HF_SFT_TEST_TOKENIZER", "meta-llama/Llama-3.2-1B-Instruct")


def _load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# -----------------------------------------------------------------------------
# Data tests

def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_role_mask_singleturn(tok):
    from hf_sft.data import encode_conversation, PAD, PROMPT, ANSWER
    msgs = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "It is 4."},
    ]
    row = encode_conversation(tok, msgs, max_seq_len=64)
    _check(row is not None, "encode_conversation returned None")

    rm = row.role_mask.numpy()
    am = row.attention_mask.numpy()
    lb = row.labels.numpy()

    # PAD aligns with attention_mask == 0
    _check(np.all((am == 0) == (rm == PAD)), "PAD mask must match attention_mask==0")
    # ANSWER aligns with labels != -100
    _check(np.all((lb != -100) == (rm == ANSWER)), "ANSWER mask must match labels!=-100")
    # There must be at least one ANSWER token (the assistant content)
    _check((rm == ANSWER).sum() >= 1, "no ANSWER tokens recorded")
    # ANSWER tokens form a contiguous run (single-turn)
    answer_pos = np.where(rm == ANSWER)[0]
    _check(np.all(np.diff(answer_pos) == 1),
           f"single-turn ANSWER tokens should be contiguous: {answer_pos}")
    # There must be PROMPT tokens before the ANSWER run
    first_answer = int(answer_pos[0])
    _check(first_answer > 0, "PROMPT must precede ANSWER")
    _check(np.all(rm[:first_answer] == PROMPT), "tokens before ANSWER must all be PROMPT")
    print(f"PASS  test_role_mask_singleturn  (PROMPT={first_answer}, "
          f"ANSWER={(rm==ANSWER).sum()}, PAD={(rm==PAD).sum()})")


def test_role_mask_multiturn(tok):
    from hf_sft.data import encode_conversation, PROMPT, ANSWER
    msgs = [
        {"role": "user",      "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user",      "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the chicken cross the road?"},
    ]
    row = encode_conversation(tok, msgs, max_seq_len=128)
    _check(row is not None, "encode_conversation returned None")
    rm = row.role_mask.numpy()
    # Two ANSWER runs (one per assistant turn)
    answer_pos = np.where(rm == ANSWER)[0]
    _check(len(answer_pos) > 0, "no ANSWER tokens")
    # Count contiguous runs.
    runs = 1
    for i in range(1, len(answer_pos)):
        if answer_pos[i] != answer_pos[i - 1] + 1:
            runs += 1
    _check(runs == 2, f"expected 2 ANSWER runs (one per assistant turn), got {runs}")
    # The user/system tokens between assistant runs must be PROMPT (not PAD).
    first_run_end = int(answer_pos[np.argmax(np.diff(answer_pos) > 1)])  # last idx of run 1
    second_run_start = int(answer_pos[np.argmax(np.diff(answer_pos) > 1) + 1])
    _check(np.all(rm[first_run_end + 1:second_run_start] == PROMPT),
           "tokens between ANSWER runs must be PROMPT")
    print(f"PASS  test_role_mask_multiturn  (2 ANSWER runs, "
           f"between-run PROMPT tokens={second_run_start - first_run_end - 1})")


# -----------------------------------------------------------------------------
# Monitor end-to-end test

def _build_tiny_llama(vocab_size: int):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,   # GQA
        head_dim=16,
        max_position_embeddings=256,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg)
    model.train()
    return model


def test_monitor_end_to_end(tok):
    from hf_sft.grad_monitor_hf import HFGradientBiasMonitor
    from hf_sft.data import encode_conversation, collate_sft

    convs = [
        [{"role": "user", "content": f"Q{i}"},
         {"role": "assistant", "content": f"A{i}!"}]
        for i in range(4)
    ]
    T = 96  # Llama chat template adds ~40 boilerplate tokens per turn
    rows = [encode_conversation(tok, m, max_seq_len=T) for m in convs]
    rows = [r for r in rows if r is not None]
    _check(len(rows) >= 2, f"need at least 2 rows, got {len(rows)}")

    # Llama-class tokenizers have added specials beyond .vocab_size, so use
    # len(tok) which covers the full id range tokens can take.
    full_vocab = max(tok.vocab_size or 0, len(tok))
    model = _build_tiny_llama(vocab_size=full_vocab)

    with tempfile.TemporaryDirectory() as tmp:
        monitor = HFGradientBiasMonitor(
            model, tmp, steps_per_file=2, outlier_pct=0.0,
            record_every_k_steps=1, debug=True,
        )
        B = 2
        A = 2  # grad_accum
        monitor.configure(grad_accum_steps=A, device_batch_size=B, seq_len=T)

        for global_step in range(2):  # two opt steps
            monitor.set_step(global_step)
            for micro in range(A):
                # Pull B rows (cycle if too few).
                pick = [rows[(global_step * A * B + micro * B + j) % len(rows)] for j in range(B)]
                batch = collate_sft(pick)
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                role_mask = batch["role_mask"]

                monitor.record_mask(role_mask)
                out = model(input_ids=input_ids,
                            attention_mask=batch["attention_mask"],
                            labels=labels)
                out.loss.backward()
                monitor.advance_accum()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None

        # Trigger a normal flush after two recorded steps + force flush of any
        # remaining partial.
        monitor.flush(1)
        monitor.flush(1, force=True)
        monitor.remove_hooks()

        # Inspect output files.
        norms_dir = Path(tmp) / "norms"
        files = sorted(p.name for p in norms_dir.iterdir())
        _check(any(f.endswith("_hidden.npz") for f in files), f"no _hidden.npz in {files}")
        _check(any(f.endswith("_attn.npz") for f in files),   f"no _attn.npz in {files}")
        _check(any(f.endswith("_mask.npz") for f in files),   f"no _mask.npz in {files}")

        # Mask file shape: (S, A, R, B, T) uint8
        mask_npz = next(p for p in norms_dir.iterdir() if p.name.endswith("_mask.npz"))
        with np.load(mask_npz) as z:
            rm = z["role_mask"]
            S, A_z, R_z, B_z, T_z = rm.shape
            _check(R_z == 1 and B_z == B and T_z == T and A_z == A,
                   f"unexpected role_mask shape: {rm.shape}")
            _check(rm.dtype == np.uint8, f"role_mask must be uint8, got {rm.dtype}")
            # Every recorded micro-step must contain at least one ANSWER token.
            from hf_sft.grad_monitor_hf import ANSWER
            _check(np.all((rm == ANSWER).any(axis=-1).any(axis=-1)),
                   "some recorded micro-step has zero ANSWER tokens")

        print(f"PASS  test_monitor_end_to_end  (mask_shape={rm.shape}, "
              f"files={[Path(f).name for f in files]})")


def main():
    tok = _load_tokenizer()
    print(f"[setup] tokenizer={TOKENIZER_ID} vocab_size={tok.vocab_size}")
    failures = 0
    for fn in (test_role_mask_singleturn,
               test_role_mask_multiturn,
               test_monitor_end_to_end):
        try:
            fn(tok)
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failures += 1
    total = 3
    print(f"\n{total - failures}/{total} passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
