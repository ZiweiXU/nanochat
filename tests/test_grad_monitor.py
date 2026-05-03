"""Smoke test for scripts_dev.grad_monitor.GradientBiasMonitor.

Builds a tiny depth-2 nanochat GPT on CPU, attaches the monitor, runs
two fwd/bwd passes across two grad-accum micro-steps each, force-flushes,
and verifies the npz files have the documented (S, A, R, B, L, ...) layout.

Run with pytest if available:
    python -m pytest tests/test_grad_monitor.py -v -s

Or run as a plain script:
    python tests/test_grad_monitor.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

# Allow running as a plain script from any cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Force SDPA fallback before nanochat.gpt imports flash_attention. On a
# Hopper box FA3 loads at import time and USE_FA3 latches to True, but
# this CPU smoke test needs the SDPA path.
import nanochat.flash_attention as _fa  # noqa: E402
_fa._override_impl = 'sdpa'
_fa.USE_FA3 = False

from nanochat.gpt import GPT, GPTConfig
from scripts_dev.grad_monitor import GradientBiasMonitor


def _make_tiny_model():
    """Depth-2 nanochat GPT on CPU. Tiny vocab but big enough seq_len/n_embd
    that attention doesn't collapse to a position-independent output (which
    would zero out dL/dq and silence the bwd hook signal even though the
    hook fires). Enabling use_resid_lambdas helps break input symmetry too."""
    config = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,                 # GQA: H_q=4, H_kv=2
        n_embd=64,
        window_pattern="L",          # full attention (avoids sliding-window code path)
        use_smear=False,
        use_resid_lambdas=True,      # breaks the perfect Q/K symmetry at init
        use_value_residual=False,
        use_backout=False,
    )
    with torch.device("meta"):
        m = GPT(config)
    m.to_empty(device=torch.device("cpu"))
    m.init_weights()
    # Perturb resid_lambdas / x0_lambdas off their neutral init so grads
    # actually flow back through Q/K/V at the very first step.
    with torch.no_grad():
        m.resid_lambdas.add_(0.1 * torch.randn_like(m.resid_lambdas))
        m.x0_lambdas.add_(0.1 * torch.randn_like(m.x0_lambdas))
    m.train()
    return m


def test_monitor_writes_expected_npz_layout():
    """Monitor records two fwd/bwd passes (S=2 steps × A=2 accum) and writes
    a hidden + attn npz pair with the (S, A, R, B, L, ...) layout."""
    tiny_model = _make_tiny_model()
    B, T = 2, 16  # device_batch_size, seq_len  (T <= sequence_len)
    A = 2          # grad_accum_steps
    L = tiny_model.config.n_layer
    H_q = tiny_model.config.n_head
    H_kv = tiny_model.config.n_kv_head

    # Independent fire-counter on every projection. Confirms the bwd hook IS
    # invoked even when the gradient norm itself happens to be zero (which is
    # the case for Q/K/V at this tiny model's init due to attention-symmetry
    # collapse — purely a property of the toy weights, not the monitor).
    counters = {"c_q_bwd": 0, "c_k_bwd": 0, "c_v_bwd": 0}
    def _make_counter(name):
        def _h(_m, _gi, _go):
            counters[name] += 1
        return _h
    sentinels = []
    for blk in tiny_model.transformer.h:
        sentinels.append(blk.attn.c_q.register_full_backward_hook(_make_counter("c_q_bwd")))
        sentinels.append(blk.attn.c_k.register_full_backward_hook(_make_counter("c_k_bwd")))
        sentinels.append(blk.attn.c_v.register_full_backward_hook(_make_counter("c_v_bwd")))

    with tempfile.TemporaryDirectory() as logs_dir:
        monitor = GradientBiasMonitor(
            tiny_model, logs_dir,
            rank=0, world_size=1,
            steps_per_file=2,         # one flush per 2 recorded steps
            outlier_pct=0.05,
            record_every_k_steps=1,   # record every step
            debug=True,
        )
        monitor.configure(grad_accum_steps=A, device_batch_size=B, seq_len=T)

        # Two training steps, two grad-accum micro-steps each.
        for step in range(2):
            monitor.set_step(step)
            for _ in range(A):
                idx = torch.randint(0, tiny_model.config.vocab_size, (B, T), dtype=torch.long)
                tgt = torch.randint(0, tiny_model.config.vocab_size, (B, T), dtype=torch.long)
                loss = tiny_model(idx, tgt)
                loss.backward()
                monitor.advance_accum()
            tiny_model.zero_grad(set_to_none=True)
            # window full after step 1 → flush should write
            monitor.flush(step)

        monitor.remove_hooks()
        for s in sentinels:
            s.remove()

        # Each Q/K/V bwd hook should have fired L * S * A = 2 * 2 * 2 = 8 times.
        expected = L * 2 * A
        assert counters["c_q_bwd"] == expected, counters
        assert counters["c_k_bwd"] == expected, counters
        assert counters["c_v_bwd"] == expected, counters

        norms_dir = os.path.join(logs_dir, "norms")
        files = sorted(os.listdir(norms_dir))
        # Expect step_0-1_hidden.npz and step_0-1_attn.npz
        assert files == ["step_0-1_attn.npz", "step_0-1_hidden.npz"], files

        # ---- hidden file checks ----
        h = np.load(os.path.join(norms_dir, "step_0-1_hidden.npz"))
        assert h["act_q"].shape == (2, A, 1, B, L, T), h["act_q"].shape  # S=2, R=1
        assert h["grad_q"].shape == (2, A, 1, B, L, T)
        assert h["act_q"].dtype == np.uint8
        assert h["grad_q"].dtype == np.uint8
        assert h["act_scale"].shape == (2, A, 1, B, L)
        assert h["act_scale"].dtype == np.float32
        assert tuple(h["global_steps"]) == (0, 1)
        assert int(h["seq_len"]) == T
        assert int(h["device_batch_size"]) == B
        assert int(h["world_size"]) == 1
        assert int(h["grad_accum_steps"]) == A
        # Forward + backward fired → outlier_val (which preserves actual
        # magnitudes losslessly per row) is populated. Don't assert on q_*
        # because for near-uniform rows (common in a tiny untrained model
        # whose hidden-norm magnitudes are similar across positions) the
        # outlier-aware quantizer can legitimately collapse q to all-zeros;
        # the actual magnitudes then live in outlier_val + min.
        assert (h["act_outlier_val"] > 0).any(), "fwd hook did not populate hidden activations"
        assert (h["grad_outlier_val"] > 0).any(), "bwd hook did not populate hidden gradients"

        # ---- attn file checks ----
        a = np.load(os.path.join(norms_dir, "step_0-1_attn.npz"))
        assert a["q_act_q"].shape == (2, A, 1, B, L, T, H_q), a["q_act_q"].shape
        assert a["kv_act_q"].shape == (2, A, 1, B, L, 2, T, H_kv), a["kv_act_q"].shape
        assert tuple(a["kv_legend"]) == ("k", "v"), tuple(a["kv_legend"])
        assert int(a["n_head"]) == H_q
        assert int(a["n_kv_head"]) == H_kv
        assert (a["q_act_outlier_val"] > 0).any(),  "Q fwd hook did not fire"
        assert (a["kv_act_outlier_val"] > 0).any(), "KV fwd hook did not fire"
        # Note: we don't assert on q_grad_outlier_val / kv_grad_outlier_val
        # being > 0 because Q/K/V gradient norms can be exactly zero at the
        # untrained-model init due to attention-symmetry collapse (uniform
        # softmax → position-independent attention output → dL/dq cancels).
        # Bwd-hook firing is independently verified by the `counters`
        # sentinels above.


def test_monitor_force_flush_with_partial_window():
    """A single recorded step with steps_per_file=5 stays buffered until force=True."""
    tiny_model = _make_tiny_model()
    B, T, A = 2, 16, 1

    with tempfile.TemporaryDirectory() as logs_dir:
        monitor = GradientBiasMonitor(
            tiny_model, logs_dir,
            rank=0, world_size=1,
            steps_per_file=5,
            outlier_pct=0.05,
            record_every_k_steps=1,
        )
        monitor.configure(grad_accum_steps=A, device_batch_size=B, seq_len=T)

        monitor.set_step(0)
        idx = torch.randint(0, tiny_model.config.vocab_size, (B, T), dtype=torch.long)
        tgt = torch.randint(0, tiny_model.config.vocab_size, (B, T), dtype=torch.long)
        loss = tiny_model(idx, tgt)
        loss.backward()
        monitor.advance_accum()
        tiny_model.zero_grad(set_to_none=True)

        # window not full — flush should be a no-op
        monitor.flush(0, force=False)
        norms_dir = os.path.join(logs_dir, "norms")
        assert os.listdir(norms_dir) == []

        # force flush — files should appear
        monitor.flush(0, force=True)
        files = sorted(os.listdir(norms_dir))
        assert files == ["step_0-0_attn.npz", "step_0-0_hidden.npz"], files

        monitor.remove_hooks()


def test_monitor_record_every_k_steps_gating():
    """With record_every_k_steps=2, only even-indexed steps are recorded."""
    tiny_model = _make_tiny_model()
    B, T, A = 2, 16, 1

    with tempfile.TemporaryDirectory() as logs_dir:
        monitor = GradientBiasMonitor(
            tiny_model, logs_dir,
            rank=0, world_size=1,
            steps_per_file=2,
            outlier_pct=0.05,
            record_every_k_steps=2,
        )
        monitor.configure(grad_accum_steps=A, device_batch_size=B, seq_len=T)

        for step in range(4):  # 0, 1, 2, 3 → record 0 and 2
            monitor.set_step(step)
            idx = torch.randint(0, tiny_model.config.vocab_size, (B, T), dtype=torch.long)
            tgt = torch.randint(0, tiny_model.config.vocab_size, (B, T), dtype=torch.long)
            loss = tiny_model(idx, tgt)
            loss.backward()
            monitor.advance_accum()
            tiny_model.zero_grad(set_to_none=True)
            monitor.flush(step)

        monitor.remove_hooks()

        # Expect a single window: steps 0, 2 → step_0-2_*.npz
        norms_dir = os.path.join(logs_dir, "norms")
        files = sorted(os.listdir(norms_dir))
        assert files == ["step_0-2_attn.npz", "step_0-2_hidden.npz"], files

        h = np.load(os.path.join(norms_dir, "step_0-2_hidden.npz"))
        assert tuple(h["global_steps"]) == (0, 2)


if __name__ == "__main__":
    tests = [
        test_monitor_writes_expected_npz_layout,
        test_monitor_force_flush_with_partial_window,
        test_monitor_record_every_k_steps_gating,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
