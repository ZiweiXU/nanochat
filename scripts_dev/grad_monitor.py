"""GradientBiasMonitor — per-token hidden / Q / K / V activation + gradient
norm recorder. Mirrors the slab-staged variant introduced in
scripts_dev/exp2_base_train.py. Importable so multiple experiment scripts
(and their tests) can share it without re-defining the class.

Hooks attach to `orig_model.transformer.h[i]` (hidden output) and
per-block `attn.c_q / c_k / c_v`. Output is dense-matrix npz chunks
under `<logs_dir>/norms/`. See `GradientBiasMonitor.__doc__` for the
on-disk file/array layout.
"""

import os

import numpy as np
import torch
import torch.distributed as dist

from nanochat.common import is_ddp_initialized


def _quantize_uint8_batched(arr, outlier_pct):
    """Vectorized per-row uint8 quantization with top-`outlier_pct` preserved losslessly.

    arr: float32 (N, M). Returns:
      q            uint8   (N, M)
      scale        float32 (N,)
      mn           float32 (N,)
      outlier_idx  int32   (N, K)   K = max(1, int(M * outlier_pct))
      outlier_val  float32 (N, K)

    Dequant (row i): val = mn[i] + q[i] * scale[i], then splice
    outlier_val[i] into positions outlier_idx[i].
    """
    assert arr.ndim == 2 and arr.dtype == np.float32
    N, M = arr.shape
    K = max(1, int(M * outlier_pct)) if outlier_pct > 0 else 1

    top_idx = np.argpartition(arr, -K, axis=1)[:, -K:].astype(np.int32)          # (N, K)
    outlier_val = np.take_along_axis(arr, top_idx, axis=1).astype(np.float32)    # (N, K)

    row_min_orig = arr.min(axis=1, keepdims=True).astype(np.float32)             # (N, 1)
    a_clean = arr.copy()
    rows = np.arange(N)[:, None]
    a_clean[rows, top_idx] = row_min_orig

    mn = a_clean.min(axis=1).astype(np.float32)                                  # (N,)
    mx = a_clean.max(axis=1).astype(np.float32)
    rng = mx - mn
    scale = np.where(rng > 0, rng / 255.0, 1.0).astype(np.float32)               # (N,)
    q = np.clip(np.round((a_clean - mn[:, None]) / scale[:, None]), 0, 255).astype(np.uint8)
    return q, scale, mn, top_idx, outlier_val


class GradientBiasMonitor:
    """Capture per-token hidden / Q / K / V norms (fwd activations + bwd grads).

    Hooks are attached on the uncompiled model (`orig_model.transformer.h[i]`).
    Every block's output is recorded (hidden). Q/K/V projections are recorded
    per-head (no head-mixing). `flush(step)` gathers records across DDP ranks
    and writes dense-matrix npz chunks on rank 0.

    OUTPUT FILES (under `<logs_dir>/norms/`, one pair per flush window):
        step_{first}-{last}_hidden.npz
        step_{first}-{last}_attn.npz

    Within a window, dimensions:
        S = number of recorded global_steps     A = grad_accum_steps
        R = world_size (# DDP ranks)            B = device_batch_size
        L = n_layer                             T = seq_len
        H_q = n_head (query)                    H_kv = n_kv_head (kv; GQA)

    Stacked-array layout: (S, A, R, B, L, ...data...).
    """

    def __init__(self, orig_model, logs_dir, rank=0, world_size=1,
                 steps_per_file=2, outlier_pct=0.01,
                 record_every_k_steps=1, debug=False):
        self.model = orig_model
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._logs_dir = logs_dir
        self._debug = debug
        self._steps_per_file = int(steps_per_file)
        self._outlier_pct = float(outlier_pct)
        self._record_every_k = max(1, int(record_every_k_steps))
        self._first_fwd_reported = False

        self._global_step = 0
        self._accum_idx = 0
        self._s_idx = 0
        self._record_this_step = False
        self._recorded_steps = []
        self._configured = False
        self._hooks = []

        spec = self._introspect_model(orig_model)
        self._hidden_modules    = spec["hidden_modules"]
        self._attn_specs        = spec["attn_specs"]
        self._layer_types       = spec["layer_types"]
        self._layer_type_legend = spec["layer_type_legend"]
        self._n_head            = spec["n_head"]
        self._n_kv_head         = spec["n_kv_head"]
        self._head_dim          = spec["head_dim"]
        self._n_layer           = len(self._hidden_modules)

        _t2i = {t: i for i, t in enumerate(self._layer_type_legend)}
        self._layer_type_legend_arr = np.array(self._layer_type_legend)
        self._layer_types_arr = np.array([_t2i[t] for t in self._layer_types], dtype=np.uint8)

        if self.rank == 0:
            os.makedirs(os.path.join(logs_dir, "norms"), exist_ok=True)

        self._setup_hooks()

    # ---------------- step/accum tracking ----------------
    def configure(self, grad_accum_steps, device_batch_size, seq_len):
        """Allocate CPU fp16 staging slabs. Call once before the training loop."""
        A = int(grad_accum_steps)
        B = int(device_batch_size)
        T = int(seq_len)
        K_win = self._steps_per_file
        L = self._n_layer
        H_q = self._n_head
        H_kv = self._n_kv_head
        self._A, self._B, self._T = A, B, T
        self._hidden_act  = torch.zeros((K_win, A, L, B, T), dtype=torch.float16)
        self._hidden_grad = torch.zeros((K_win, A, L, B, T), dtype=torch.float16)
        self._q_act  = torch.zeros((K_win, A, L, B, T, H_q), dtype=torch.float16)
        self._q_grad = torch.zeros((K_win, A, L, B, T, H_q), dtype=torch.float16)
        self._kv_act  = torch.zeros((K_win, A, L, 2, B, T, H_kv), dtype=torch.float16)
        self._kv_grad = torch.zeros((K_win, A, L, 2, B, T, H_kv), dtype=torch.float16)
        self._configured = True

    def set_step(self, global_step):
        self._global_step = int(global_step)
        self._accum_idx = 0
        self._record_this_step = (self._global_step % self._record_every_k == 0)
        if self._record_this_step:
            self._s_idx = len(self._recorded_steps)
            self._recorded_steps.append(self._global_step)

    def advance_accum(self):
        self._accum_idx += 1

    # ---------------- model-specific introspection ----------------
    @staticmethod
    def _introspect_model(orig_model):
        """Discover hook points and layer metadata. Edit only this method to
        port to a different architecture."""
        hidden_modules = list(orig_model.transformer.h)

        seq_len = orig_model.config.sequence_len
        layer_types = []
        for i in range(len(hidden_modules)):
            left, _right = orig_model.window_sizes[i]
            if left < 0 or left >= seq_len:
                layer_types.append("full_attention")
            else:
                layer_types.append("sliding_attention")
        layer_type_legend = ["full_attention", "sliding_attention"]

        attn_specs = []
        for i, block in enumerate(hidden_modules):
            attn = getattr(block, "attn", None)
            if attn is None or not all(hasattr(attn, n) for n in ("c_q", "c_k", "c_v")):
                continue
            attn_specs.append({
                "layer_idx": i,
                "proj":    {"q": attn.c_q, "k": attn.c_k, "v": attn.c_v},
                "n_heads": {"q": int(attn.n_head),
                            "k": int(attn.n_kv_head),
                            "v": int(attn.n_kv_head)},
                "head_dim": int(attn.head_dim),
            })

        if attn_specs:
            s0 = attn_specs[0]
            n_head, n_kv_head, head_dim = s0["n_heads"]["q"], s0["n_heads"]["k"], s0["head_dim"]
            for s in attn_specs[1:]:
                assert s["n_heads"]["q"] == n_head and s["n_heads"]["k"] == n_kv_head, \
                    "GradientBiasMonitor requires identical head counts across attn layers"
                assert s["head_dim"] == head_dim, \
                    "GradientBiasMonitor requires identical head_dim across attn layers"
        else:
            n_head = n_kv_head = head_dim = 0

        return {
            "hidden_modules": hidden_modules,
            "attn_specs": attn_specs,
            "layer_types": layer_types,
            "layer_type_legend": layer_type_legend,
            "n_head": n_head, "n_kv_head": n_kv_head, "head_dim": head_dim,
        }

    # ---------------- hook setup ----------------
    _KV_IDX = {"k": 0, "v": 1}

    def _setup_hooks(self):
        for i, block in enumerate(self._hidden_modules):

            def make_fwd_hidden(layer_idx):
                def fwd_hook(module, inp, output):
                    if not module.training or not self._record_this_step:
                        return
                    hidden = output[0] if isinstance(output, tuple) else output
                    if hidden.dim() != 3:
                        return
                    if self._debug and not self._first_fwd_reported and layer_idx == 0:
                        print(f"[monitor] first hidden fwd fired (layer=0, shape={tuple(hidden.shape)})")
                        self._first_fwd_reported = True
                    norms = hidden.detach().float().norm(dim=-1).to(torch.float16)
                    self._hidden_act[self._s_idx, self._accum_idx, layer_idx].copy_(norms)
                return fwd_hook

            def make_bwd_hidden(layer_idx):
                def bwd_hook(module, grad_input, grad_output):
                    if not module.training or not self._record_this_step:
                        return
                    g = grad_output[0]
                    if g is None or g.dim() != 3:
                        return
                    gn = g.detach().float().norm(dim=-1).to(torch.float16)
                    self._hidden_grad[self._s_idx, self._accum_idx, layer_idx].copy_(gn)
                return bwd_hook

            self._hooks.append(block.register_forward_hook(make_fwd_hidden(i)))
            self._hooks.append(block.register_full_backward_hook(make_bwd_hidden(i)))

        for spec in self._attn_specs:
            i = spec["layer_idx"]
            head_dim = spec["head_dim"]
            for qkv_name, proj in spec["proj"].items():
                n_heads = spec["n_heads"][qkv_name]

                def make_fwd_qkv(layer_idx, qkv, n_heads, head_dim):
                    is_q = (qkv == "q")
                    kv_i = self._KV_IDX.get(qkv, 0)
                    def fwd_hook(module, inp, output):
                        if not module.training or not self._record_this_step:
                            return
                        B, T, _ = output.shape
                        act = output.detach().float().view(B, T, n_heads, head_dim)
                        per_norms = act.norm(dim=-1).to(torch.float16)
                        if is_q:
                            self._q_act[self._s_idx, self._accum_idx, layer_idx].copy_(per_norms)
                        else:
                            self._kv_act[self._s_idx, self._accum_idx, layer_idx, kv_i].copy_(per_norms)
                    return fwd_hook

                def make_bwd_qkv(layer_idx, qkv, n_heads, head_dim):
                    is_q = (qkv == "q")
                    kv_i = self._KV_IDX.get(qkv, 0)
                    def bwd_hook(module, grad_input, grad_output):
                        if not module.training or not self._record_this_step:
                            return
                        g = grad_output[0]
                        if g is None:
                            return
                        B, T, _ = g.shape
                        g_r = g.detach().float().view(B, T, n_heads, head_dim)
                        gn = g_r.norm(dim=-1).to(torch.float16)
                        if is_q:
                            self._q_grad[self._s_idx, self._accum_idx, layer_idx].copy_(gn)
                        else:
                            self._kv_grad[self._s_idx, self._accum_idx, layer_idx, kv_i].copy_(gn)
                    return bwd_hook

                self._hooks.append(proj.register_forward_hook(
                    make_fwd_qkv(i, qkv_name, n_heads, head_dim)))
                self._hooks.append(proj.register_full_backward_hook(
                    make_bwd_qkv(i, qkv_name, n_heads, head_dim)))

    # ---------------- flush ----------------
    def _gather_slab(self, cpu_slice):
        local_np = cpu_slice.contiguous().numpy()
        if self.world_size <= 1 or not is_ddp_initialized():
            return local_np[None, ...]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_np)
        if self.rank == 0:
            return np.stack(gathered, axis=0)
        return None

    def flush(self, global_step=None, force=False):
        if not self._configured:
            return
        n = len(self._recorded_steps)
        if n == 0:
            return
        if not force and n < self._steps_per_file:
            return

        hidden_act_g  = self._gather_slab(self._hidden_act[:n])
        hidden_grad_g = self._gather_slab(self._hidden_grad[:n])
        q_act_g   = self._gather_slab(self._q_act[:n])
        q_grad_g  = self._gather_slab(self._q_grad[:n])
        kv_act_g  = self._gather_slab(self._kv_act[:n])
        kv_grad_g = self._gather_slab(self._kv_grad[:n])

        recorded_steps = self._recorded_steps
        self._recorded_steps = []
        self._s_idx = 0

        if self.rank != 0:
            return

        act = hidden_act_g.transpose(1, 2, 0, 4, 3, 5).astype(np.float32)
        grd = hidden_grad_g.transpose(1, 2, 0, 4, 3, 5).astype(np.float32)
        q_act  = q_act_g.transpose(1, 2, 0, 4, 3, 5, 6).astype(np.float32)
        q_grd  = q_grad_g.transpose(1, 2, 0, 4, 3, 5, 6).astype(np.float32)
        kv_act = kv_act_g.transpose(1, 2, 0, 5, 3, 4, 6, 7).astype(np.float32)
        kv_grd = kv_grad_g.transpose(1, 2, 0, 5, 3, 4, 6, 7).astype(np.float32)

        S = n
        A = self._A
        R = self.world_size
        B = self._B
        L = self._n_layer
        T = self._T
        H_q, H_kv = self._n_head, self._n_kv_head
        n_elements = int(self.model.config.n_embd)

        layer_type_legend = self._layer_type_legend_arr
        layer_types_arr = self._layer_types_arr
        global_steps_arr = np.array(recorded_steps, dtype=np.int32)
        step_tag = f"step_{recorded_steps[0]}-{recorded_steps[-1]}"
        out_dir = os.path.join(self._logs_dir, "norms")

        def _quant_hidden(x):
            N = S * A * R * B * L
            flat = x.reshape(N, T)
            q, sc, mn, oi, ov = _quantize_uint8_batched(flat, self._outlier_pct)
            K = oi.shape[1]
            return (q.reshape(S, A, R, B, L, T),
                    sc.reshape(S, A, R, B, L),
                    mn.reshape(S, A, R, B, L),
                    oi.reshape(S, A, R, B, L, K),
                    ov.reshape(S, A, R, B, L, K))

        aq, asc, amn, aoi, aov = _quant_hidden(act)
        gq, gsc, gmn, goi, gov = _quant_hidden(grd)

        np.savez_compressed(
            os.path.join(out_dir, f"{step_tag}_hidden.npz"),
            act_q=aq, act_scale=asc, act_min=amn,
            act_outlier_idx=aoi, act_outlier_val=aov,
            grad_q=gq, grad_scale=gsc, grad_min=gmn,
            grad_outlier_idx=goi, grad_outlier_val=gov,
            global_steps=global_steps_arr,
            layer_types=layer_types_arr,
            layer_type_legend=layer_type_legend,
            n_elements=np.int32(n_elements),
            seq_len=np.int32(T),
            device_batch_size=np.int32(B),
            world_size=np.int32(R),
            grad_accum_steps=np.int32(A),
            outlier_pct=np.float32(self._outlier_pct),
        )

        if H_q > 0 and H_kv > 0:
            def _quant_attn_q(x):
                N = S * A * R * B * L
                flat = x.reshape(N, T * H_q)
                q, sc, mn, oi, ov = _quantize_uint8_batched(flat, self._outlier_pct)
                K = oi.shape[1]
                return (q.reshape(S, A, R, B, L, T, H_q),
                        sc.reshape(S, A, R, B, L),
                        mn.reshape(S, A, R, B, L),
                        oi.reshape(S, A, R, B, L, K),
                        ov.reshape(S, A, R, B, L, K))

            def _quant_attn_kv(x):
                N = S * A * R * B * L * 2
                flat = x.reshape(N, T * H_kv)
                q, sc, mn, oi, ov = _quantize_uint8_batched(flat, self._outlier_pct)
                K = oi.shape[1]
                return (q.reshape(S, A, R, B, L, 2, T, H_kv),
                        sc.reshape(S, A, R, B, L, 2),
                        mn.reshape(S, A, R, B, L, 2),
                        oi.reshape(S, A, R, B, L, 2, K),
                        ov.reshape(S, A, R, B, L, 2, K))

            qaq, qasc, qamn, qaoi, qaov = _quant_attn_q(q_act)
            qgq, qgsc, qgmn, qgoi, qgov = _quant_attn_q(q_grd)
            kvaq, kvasc, kvamn, kvaoi, kvaov = _quant_attn_kv(kv_act)
            kvgq, kvgsc, kvgmn, kvgoi, kvgov = _quant_attn_kv(kv_grd)

            np.savez_compressed(
                os.path.join(out_dir, f"{step_tag}_attn.npz"),
                q_act_q=qaq, q_act_scale=qasc, q_act_min=qamn,
                q_act_outlier_idx=qaoi, q_act_outlier_val=qaov,
                q_grad_q=qgq, q_grad_scale=qgsc, q_grad_min=qgmn,
                q_grad_outlier_idx=qgoi, q_grad_outlier_val=qgov,
                kv_act_q=kvaq, kv_act_scale=kvasc, kv_act_min=kvamn,
                kv_act_outlier_idx=kvaoi, kv_act_outlier_val=kvaov,
                kv_grad_q=kvgq, kv_grad_scale=kvgsc, kv_grad_min=kvgmn,
                kv_grad_outlier_idx=kvgoi, kv_grad_outlier_val=kvgov,
                global_steps=global_steps_arr,
                layer_types=layer_types_arr,
                layer_type_legend=layer_type_legend,
                kv_legend=np.array(["k", "v"]),
                n_head=np.int32(H_q),
                n_kv_head=np.int32(H_kv),
                head_dim=np.int32(self._head_dim),
                seq_len=np.int32(T),
                device_batch_size=np.int32(B),
                world_size=np.int32(R),
                grad_accum_steps=np.int32(A),
                outlier_pct=np.float32(self._outlier_pct),
            )

    def remove_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
