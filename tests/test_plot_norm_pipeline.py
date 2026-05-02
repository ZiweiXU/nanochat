"""Exercise the exact code path that plot_norm/plot_hidden_example.ipynb
runs in cell 7: read_hidden, read_attn, read_both, then reshape to the
(S, A, B_flat, L, T) layout used by the notebook's `View` dataclass.

Run:
    python tests/test_plot_norm_pipeline.py <norms_dir> <step_end>
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from plot_norm.read_norms import read_hidden, read_attn, read_both


def main(norms_dir: str, step_end: int):
    print(f"=== norms_dir: {norms_dir} ===")
    print(f"=== step range: 0..{step_end} ===\n")

    # 1) read_hidden
    h = read_hidden(norms_dir, 0, step_end)
    S, A, R, B_per_rank, L, T = h.act.shape
    assert h.act.shape == h.grad.shape, (h.act.shape, h.grad.shape)
    assert h.global_steps.shape == (S,)
    assert len(h.layer_types) == L
    assert h.act.dtype == np.float32
    print(f"[hidden] act/grad shape: {h.act.shape}  global_steps={h.global_steps.tolist()}")
    print(f"         layer_types={h.layer_types}")
    print(f"         act mean/std: {h.act.mean():.4f} / {h.act.std():.4f}")
    print(f"         grad mean/std: {h.grad.mean():.4e} / {h.grad.std():.4e}")

    # 2) read_attn
    a = read_attn(norms_dir, 0, step_end)
    H_q = a.n_head
    H_kv = a.n_kv_head
    assert a.q_act.shape == (S, A, R, B_per_rank, L, T, H_q),  a.q_act.shape
    assert a.k_act.shape == (S, A, R, B_per_rank, L, T, H_kv), a.k_act.shape
    assert a.v_act.shape == (S, A, R, B_per_rank, L, T, H_kv), a.v_act.shape
    assert a.q_grad.shape == a.q_act.shape
    assert a.k_grad.shape == a.k_act.shape
    assert a.v_grad.shape == a.v_act.shape
    print(f"\n[attn]   q shape: {a.q_act.shape}  k shape: {a.k_act.shape}")
    print(f"         H_q={H_q} H_kv={H_kv} head_dim={a.head_dim}")
    print(f"         q_act mean/std: {a.q_act.mean():.4f} / {a.q_act.std():.4f}")
    print(f"         k_act mean/std: {a.k_act.mean():.4f} / {a.k_act.std():.4f}")
    print(f"         v_act mean/std: {a.v_act.mean():.4f} / {a.v_act.std():.4f}")

    # 3) read_both — convenience
    h2, a2 = read_both(norms_dir, 0, step_end)
    assert np.array_equal(h.global_steps, h2.global_steps)
    assert h.act.shape == h2.act.shape
    assert a.q_act.shape == a2.q_act.shape

    # 4) Reproduce the notebook's cell-7 reshape: collapse (R, B_per_rank) → B
    act_flat  = h.act.reshape(S, A, R * B_per_rank, L, T)
    grad_flat = h.grad.reshape(S, A, R * B_per_rank, L, T)
    assert act_flat.shape  == (S, A, R * B_per_rank, L, T)
    assert grad_flat.shape == (S, A, R * B_per_rank, L, T)
    rel_positions = np.arange(T, dtype=np.int64) - T
    assert rel_positions[-1] == -1 and rel_positions[0] == -T

    # 5) An aggregate the notebook would compute: per-layer mean act over time
    per_layer_mean = act_flat.mean(axis=(0, 1, 2, 4))   # (L,)
    assert per_layer_mean.shape == (L,)
    print(f"\nNotebook-style View arr: {act_flat.shape} (S, A, B={R*B_per_rank}, L, T)")
    print(f"per-layer mean act: {per_layer_mean.tolist()}")

    # 6) Sanity: hidden / Q / K / V activations are non-negative norms.
    assert (h.act  >= 0).all()
    assert (a.q_act >= 0).all()
    assert (a.k_act >= 0).all()
    assert (a.v_act >= 0).all()

    # 7) Sanity: at least *some* hidden activation is positive (forward fired).
    assert h.act.max() > 0,  "hidden act all zero"
    assert a.q_act.max() > 0, "Q act all zero"
    assert a.k_act.max() > 0, "K act all zero"
    assert a.v_act.max() > 0, "V act all zero"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], int(sys.argv[2]))
