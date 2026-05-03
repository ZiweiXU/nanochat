"""GradientBiasMonitor port for HuggingFace Llama-class models.

Records per-token activation + gradient norms on hidden states and per-head
Q/K/V projections, plus a per-token role mask {PAD=0, PROMPT=1, ANSWER=2}
needed for SFT analysis (where gradients only flow on answer tokens).

Hook targets (Llama, Qwen3, SmolLM2, Mistral, etc.):
  hidden : `model.model.layers[i]`            block-output forward + bwd
  attn   : `model.model.layers[i].self_attn.{q,k,v}_proj`  per-head fwd + bwd

Output files (under `<logs_dir>/norms/`, one set per flush window):
  step_{first}-{last}_hidden.npz
  step_{first}-{last}_attn.npz
  step_{first}-{last}_mask.npz   <-- new in this port

Slab layouts:
  hidden_act / hidden_grad : (K_win, A, L, B, T)         fp16
  attn_out_act / mlp_out_act : (K_win, A, L, B, T)       fp16   (pre-residual contributions)
  q_act / q_grad           : (K_win, A, L, B, T, H_q)    fp16
  kv_act / kv_grad         : (K_win, A, L, 2, B, T, H_kv) fp16   (2: k, v)
  role_mask                : (K_win, A, B, T)             uint8

attn_out_act / mlp_out_act let us decompose the residual-stream growth:
hidden_act on layer i is the post-residual block output, i.e.
``residual_in + attn_out + mlp_out``. The two pre-residual norms tell us
how much each sub-block contributes per-token, so we can attribute
prompt-amplification effects to attention vs MLP rather than just
observing the combined residual stream.

Flushed-on-disk layouts include the world_size axis even at world_size=1
to keep the on-disk schema identical to the nanochat version.
"""

import os

import numpy as np
import torch
import torch.distributed as dist


PAD, PROMPT, ANSWER = 0, 1, 2
ROLE_LEGEND = np.array(["pad", "prompt", "answer"])


def _is_ddp_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _quantize_uint8_batched(arr, outlier_pct):
    """Per-row uint8 quantization with top-`outlier_pct` preserved losslessly.

    Identical math to the nanochat version. `arr` must be float32 (N, M).
    """
    assert arr.ndim == 2 and arr.dtype == np.float32
    N, M = arr.shape
    K = max(1, int(M * outlier_pct)) if outlier_pct > 0 else 1

    top_idx = np.argpartition(arr, -K, axis=1)[:, -K:].astype(np.int32)
    outlier_val = np.take_along_axis(arr, top_idx, axis=1).astype(np.float32)

    row_min_orig = arr.min(axis=1, keepdims=True).astype(np.float32)
    a_clean = arr.copy()
    rows = np.arange(N)[:, None]
    a_clean[rows, top_idx] = row_min_orig

    mn = a_clean.min(axis=1).astype(np.float32)
    mx = a_clean.max(axis=1).astype(np.float32)
    rng = mx - mn
    scale = np.where(rng > 0, rng / 255.0, 1.0).astype(np.float32)
    q = np.clip(np.round((a_clean - mn[:, None]) / scale[:, None]), 0, 255).astype(np.uint8)
    return q, scale, mn, top_idx, outlier_val


class HFGradientBiasMonitor:
    """HF Llama-class equivalent of nanochat's GradientBiasMonitor.

    Usage:
        monitor = HFGradientBiasMonitor(model, logs_dir, ...)
        monitor.configure(grad_accum_steps, device_batch_size, seq_len)
        for step in range(num_steps):
            monitor.set_step(step)
            for micro in range(grad_accum_steps):
                # IMPORTANT: record role mask BEFORE the fwd pass for this micro
                monitor.record_mask(role_mask_int8)   # shape (B, T), uint8
                loss = model(input_ids, ...).loss
                loss.backward()
                monitor.advance_accum()
            optimizer.step()
            monitor.flush(step)
        monitor.flush(step, force=True)
        monitor.remove_hooks()
    """

    def __init__(self, model, logs_dir, rank=0, world_size=1,
                 steps_per_file=2, outlier_pct=0.01,
                 record_every_k_steps=1, debug=False):
        self.model = model
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

        spec = self._introspect_model(model)
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
        self._attn_out_act = torch.zeros((K_win, A, L, B, T), dtype=torch.float16)
        self._mlp_out_act  = torch.zeros((K_win, A, L, B, T), dtype=torch.float16)
        # Sink-direction projection: <dL/d(h_l[t]), h_l[t] / ||h_l[t]||> per token.
        # Sign indicates whether the gradient pushes ||h_l[t]|| down (>0,
        # i.e. -lr*g shrinks ||h||) or up (<0, -lr*g grows ||h||) at this
        # position-layer. See findings/SINK.md "Open mechanism question".
        self._h_proj_grad = torch.zeros((K_win, A, L, B, T), dtype=torch.float16)
        # Forward-hook scratch keyed by (layer_idx, accum_idx). Holds the
        # detached residual-stream output so the matching backward hook
        # can compute the projection. Cleared as bwd hooks consume entries.
        self._h_cache = {}
        self._q_act  = torch.zeros((K_win, A, L, B, T, H_q), dtype=torch.float16)
        self._q_grad = torch.zeros((K_win, A, L, B, T, H_q), dtype=torch.float16)
        self._kv_act  = torch.zeros((K_win, A, L, 2, B, T, H_kv), dtype=torch.float16)
        self._kv_grad = torch.zeros((K_win, A, L, 2, B, T, H_kv), dtype=torch.float16)
        self._role_mask = torch.zeros((K_win, A, B, T), dtype=torch.uint8)
        self._configured = True

    def set_step(self, global_step):
        self._global_step = int(global_step)
        self._accum_idx = 0
        self._record_this_step = (self._global_step % self._record_every_k == 0)
        if self._record_this_step:
            self._s_idx = len(self._recorded_steps)
            self._recorded_steps.append(self._global_step)
        # Defensive: drop any stale entries from the previous step (in case
        # a bwd hook didn't fire; gradient checkpointing or non-standard
        # architectures could in principle skip a hook). Avoids gradual
        # memory growth across steps.
        self._h_cache.clear()

    def advance_accum(self):
        self._accum_idx += 1

    def record_mask(self, role_mask):
        """Record per-token role mask for the current micro-step.

        `role_mask` is expected to be a uint8 / bool tensor (B, T) with values
        in {PAD=0, PROMPT=1, ANSWER=2}. Call BEFORE the fwd pass each micro.
        """
        if not self._configured or not self._record_this_step:
            return
        m = role_mask.detach().to("cpu", dtype=torch.uint8).contiguous()
        if m.shape != (self._B, self._T):
            raise ValueError(
                f"role_mask shape {tuple(m.shape)} != expected (B, T) = ({self._B}, {self._T})"
            )
        self._role_mask[self._s_idx, self._accum_idx].copy_(m)

    # ---------------- model-specific introspection ----------------
    @staticmethod
    def _introspect_model(model):
        """Discover hook points for HF Llama-class models.

        Walks `model.model.layers` (or `model.layers` for non-CausalLM
        wrappers). Each layer must expose `.self_attn.q_proj/k_proj/v_proj`.
        """
        # Locate the layers list. CausalLM wraps as model.model.layers; bare
        # bodies expose model.layers. Fall back gracefully.
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = list(model.model.layers)
            cfg = getattr(model, "config", getattr(model.model, "config", None))
        elif hasattr(model, "layers"):
            layers = list(model.layers)
            cfg = getattr(model, "config", None)
        else:
            raise AttributeError(
                "Could not find a `.layers` list on the model. "
                "HFGradientBiasMonitor expects an HF Llama-class architecture "
                "with `model.model.layers[i].self_attn.{q,k,v}_proj`."
            )
        if cfg is None:
            raise AttributeError("Model has no `.config`")

        n_head = int(getattr(cfg, "num_attention_heads"))
        n_kv_head = int(getattr(cfg, "num_key_value_heads", n_head))
        hidden_size = int(getattr(cfg, "hidden_size"))
        head_dim = int(getattr(cfg, "head_dim", hidden_size // n_head))

        # Sliding-window detection (Mistral, Qwen2 some configs).
        # Default to "full_attention" unless a per-layer pattern is exposed.
        layer_types = []
        sw = getattr(cfg, "sliding_window", None)
        seq_len_attr = getattr(cfg, "max_position_embeddings", None)
        for i, layer in enumerate(layers):
            # Some recent configs (Gemma2, etc.) expose `layer_types` lists.
            cfg_layer_types = getattr(cfg, "layer_types", None)
            if cfg_layer_types is not None and i < len(cfg_layer_types):
                layer_types.append(str(cfg_layer_types[i]))
            elif sw is not None and seq_len_attr is not None and sw < seq_len_attr:
                layer_types.append("sliding_attention")
            else:
                layer_types.append("full_attention")
        # legend covers any value that appeared so plotting can color-code
        layer_type_legend = sorted(set(layer_types))

        attn_specs = []
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            if not all(hasattr(attn, n) for n in ("q_proj", "k_proj", "v_proj")):
                continue
            attn_specs.append({
                "layer_idx": i,
                "proj":    {"q": attn.q_proj, "k": attn.k_proj, "v": attn.v_proj},
                "n_heads": {"q": n_head, "k": n_kv_head, "v": n_kv_head},
                "head_dim": head_dim,
            })

        return {
            "hidden_modules": layers,
            "attn_specs": attn_specs,
            "layer_types": layer_types,
            "layer_type_legend": layer_type_legend,
            "n_head": n_head,
            "n_kv_head": n_kv_head,
            "head_dim": head_dim,
        }

    # ---------------- hook setup ----------------
    _KV_IDX = {"k": 0, "v": 1}

    def _setup_hooks(self):
        for i, block in enumerate(self._hidden_modules):

            def make_fwd_hidden(layer_idx):
                def fwd_hook(module, inp, output):
                    if not module.training or not self._record_this_step:
                        return
                    # HF layers commonly return a tuple (hidden, ...). Extract.
                    hidden = output[0] if isinstance(output, tuple) else output
                    if hidden.dim() != 3:
                        return
                    if self._debug and not self._first_fwd_reported and layer_idx == 0:
                        print(f"[hf-monitor] first hidden fwd fired (layer=0, shape={tuple(hidden.shape)})")
                        self._first_fwd_reported = True
                    h_det = hidden.detach()
                    norms = h_det.float().norm(dim=-1).to(torch.float16)
                    self._hidden_act[self._s_idx, self._accum_idx, layer_idx].copy_(norms)
                    # Stash for the matching backward hook to compute the
                    # sink-direction projection. Cached on the same device as
                    # the activation; popped in the bwd hook.
                    self._h_cache[(layer_idx, self._accum_idx)] = h_det
                return fwd_hook

            def make_bwd_hidden(layer_idx):
                def bwd_hook(module, grad_input, grad_output):
                    if not module.training or not self._record_this_step:
                        return
                    g = grad_output[0]
                    if g is None or g.dim() != 3:
                        return
                    g_det = g.detach().float()
                    gn = g_det.norm(dim=-1).to(torch.float16)
                    self._hidden_grad[self._s_idx, self._accum_idx, layer_idx].copy_(gn)
                    # Sink-direction projection: <g, h/||h||> = <g, h> / ||h||.
                    h = self._h_cache.pop((layer_idx, self._accum_idx), None)
                    if h is not None and h.shape == g_det.shape:
                        h_f = h.float()
                        h_norm = h_f.norm(dim=-1).clamp(min=1e-12)
                        proj = (g_det * h_f).sum(dim=-1) / h_norm  # (B, T)
                        self._h_proj_grad[self._s_idx, self._accum_idx, layer_idx].copy_(
                            proj.to(torch.float16))
                return bwd_hook

            self._hooks.append(block.register_forward_hook(make_fwd_hidden(i)))
            self._hooks.append(block.register_full_backward_hook(make_bwd_hidden(i)))

            attn_mod = getattr(block, "self_attn", None)
            mlp_mod  = getattr(block, "mlp", None)

            def make_fwd_subblock(layer_idx, slab_attr):
                def fwd_hook(module, inp, output):
                    if not module.training or not self._record_this_step:
                        return
                    out = output[0] if isinstance(output, tuple) else output
                    if not torch.is_tensor(out) or out.dim() != 3:
                        return
                    norms = out.detach().float().norm(dim=-1).to(torch.float16)
                    getattr(self, slab_attr)[self._s_idx, self._accum_idx, layer_idx].copy_(norms)
                return fwd_hook

            if attn_mod is not None:
                self._hooks.append(attn_mod.register_forward_hook(
                    make_fwd_subblock(i, "_attn_out_act")))
            if mlp_mod is not None:
                self._hooks.append(mlp_mod.register_forward_hook(
                    make_fwd_subblock(i, "_mlp_out_act")))

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
                        # output: (B, T, n_heads * head_dim)
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
        if self.world_size <= 1 or not _is_ddp_initialized():
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
        attn_out_g    = self._gather_slab(self._attn_out_act[:n])
        mlp_out_g     = self._gather_slab(self._mlp_out_act[:n])
        h_proj_g      = self._gather_slab(self._h_proj_grad[:n])
        q_act_g   = self._gather_slab(self._q_act[:n])
        q_grad_g  = self._gather_slab(self._q_grad[:n])
        kv_act_g  = self._gather_slab(self._kv_act[:n])
        kv_grad_g = self._gather_slab(self._kv_grad[:n])
        mask_g    = self._gather_slab(self._role_mask[:n])

        recorded_steps = self._recorded_steps
        self._recorded_steps = []
        self._s_idx = 0

        if self.rank != 0:
            return

        # Match nanochat on-disk transposes: bring R next to B etc.
        # Source (after gather): (R, S, A, L, B, T) -> on-disk (S, A, R, B, L, T)
        act = hidden_act_g.transpose(1, 2, 0, 4, 3, 5).astype(np.float32)
        grd = hidden_grad_g.transpose(1, 2, 0, 4, 3, 5).astype(np.float32)
        attn_out = attn_out_g.transpose(1, 2, 0, 4, 3, 5).astype(np.float32)
        mlp_out  = mlp_out_g.transpose(1, 2, 0, 4, 3, 5).astype(np.float32)
        h_proj   = h_proj_g.transpose(1, 2, 0, 4, 3, 5).astype(np.float32)
        q_act  = q_act_g.transpose(1, 2, 0, 4, 3, 5, 6).astype(np.float32)
        q_grd  = q_grad_g.transpose(1, 2, 0, 4, 3, 5, 6).astype(np.float32)
        kv_act = kv_act_g.transpose(1, 2, 0, 5, 3, 4, 6, 7).astype(np.float32)
        kv_grd = kv_grad_g.transpose(1, 2, 0, 5, 3, 4, 6, 7).astype(np.float32)
        # mask_g: (R, S, A, B, T) -> (S, A, R, B, T)
        mask = mask_g.transpose(1, 2, 0, 3, 4).astype(np.uint8)

        S = n
        A = self._A
        R = self.world_size
        B = self._B
        L = self._n_layer
        T = self._T
        H_q, H_kv = self._n_head, self._n_kv_head
        try:
            n_elements = int(self.model.config.hidden_size)
        except Exception:
            n_elements = 0

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
        atq, atsc, atmn, atoi, atov = _quant_hidden(attn_out)
        mlq, mlsc, mlmn, mloi, mlov = _quant_hidden(mlp_out)
        # h_proj is signed; the per-row min/max quantization handles negatives
        # correctly (mn can be negative; outlier preservation captures both
        # tails since argpartition is by magnitude after subtracting min...
        # actually argpartition(-K) picks the LARGEST K, so deeply negative
        # values still get clipped at the per-row min. That's acceptable
        # here — typical sink-direction projections are O(||g||·||h||) and
        # don't have heavy negative tails.
        hpq, hpsc, hpmn, hpoi, hpov = _quant_hidden(h_proj)

        np.savez_compressed(
            os.path.join(out_dir, f"{step_tag}_hidden.npz"),
            act_q=aq, act_scale=asc, act_min=amn,
            act_outlier_idx=aoi, act_outlier_val=aov,
            grad_q=gq, grad_scale=gsc, grad_min=gmn,
            grad_outlier_idx=goi, grad_outlier_val=gov,
            attn_out_q=atq, attn_out_scale=atsc, attn_out_min=atmn,
            attn_out_outlier_idx=atoi, attn_out_outlier_val=atov,
            mlp_out_q=mlq, mlp_out_scale=mlsc, mlp_out_min=mlmn,
            mlp_out_outlier_idx=mloi, mlp_out_outlier_val=mlov,
            h_proj_grad_q=hpq, h_proj_grad_scale=hpsc, h_proj_grad_min=hpmn,
            h_proj_grad_outlier_idx=hpoi, h_proj_grad_outlier_val=hpov,
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

        # Mask file: small uint8 array, no quantization needed.
        np.savez_compressed(
            os.path.join(out_dir, f"{step_tag}_mask.npz"),
            role_mask=mask,                    # (S, A, R, B, T) uint8
            role_legend=ROLE_LEGEND,           # ["pad", "prompt", "answer"]
            global_steps=global_steps_arr,
            seq_len=np.int32(T),
            device_batch_size=np.int32(B),
            world_size=np.int32(R),
            grad_accum_steps=np.int32(A),
        )

    def remove_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
