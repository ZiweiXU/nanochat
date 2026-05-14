# Finding: per-token gradient norm vs answer-relative position

The decay-with-position hypothesis (early-position grads larger than
late-position grads in the answer span) holds — but the early reference
needs to skip pos 0, which sits next to the `<|im_start|>` sink and has
anomalously low gradient (see findings/SINK.md, finding #6). The
analysis script's heatmap default is `mean(grad[1:8]) / mean(grad[tail])`.

## Tooling

`plot_norm/answer_grad_profile.py` — standalone matplotlib analysis
(no notebook / Chrome dependency). It re-anchors per-row gradients so
position 0 = first ANSWER token of the row, then plots median grad
norm vs answer-relative-position swept over (training step, layer).

```bash
PYTHONPATH=. /opt/uv/venv/bin/python plot_norm/answer_grad_profile.py \
    logs_hf/sft_qwen3_500_gsm8k_05031456/norms
```

Output goes to `plot_norm/figs/answer_grad/<run_name>/` (per-run
subdirectory, so different runs don't clobber each other). Generates:

- `coverage.png` — fraction of rows whose answer span reaches each
  position. The plot window is restricted to ≥50% coverage to avoid
  selection-bias from long-answer outliers.
- `grad_step_sweep_L*.png` (raw + `_norm` variants) — per-step grad
  curves at fixed layer.
- `grad_layer_sweep_{first,last}_step.png` — per-layer curves.
- `decay_heatmap.png` — `log10(early/tail)` ratio per (step, layer).
- `decay_trajectories.png` — same ratio as line plot per layer over
  training steps.

The plotting functions apply a centered rolling mean **across the
layer axis** (default window 3) on top of the per-row averaging.
Adjacent layers' grad norms are tightly correlated, so this surfaces
the cross-layer trend without losing emerge-plateau-unwind structure.
Pass `layer_smooth=1` to disable.

**Per-curve normalization** (`normalize=True`) divides each curve by
the **mean of the last `normalize_tail_n` smoothed positions**
(default `tail-4`), not by pos 0. The order is: smooth first along
the token axis, then divide by the smoothed tail mean — this avoids
amplifying noise when a single endpoint is jittery, and makes the
trend across (step, layer) directly comparable since every curve
ends near 1.0. Configurable via `normalize_to` ("tail" / "last" /
"first") and `normalize_tail_n`. **Don't normalize to pos 0 for grad
analysis**: pos 0 sits on the sink and has anomalously low gradient
(see findings/SINK.md, finding #6).

## SmolTalk results (single-turn, full coverage to pos 14)

At step 0, decay across all layers is dramatic (10–30× pos 0 vs pos 14,
strongest at top layers L25+ with log10 ratio ~1.5–1.8). By step ~50
the profile flattens and stays near 1× for the rest of training. Top
layers (L25+) volatile.

The flattening is partly an artifact of SmolTalk's short answers — the
"tail" position is only 14, not enough span for the autoregressive
decay to manifest at later steps.

Logs: `plot_norm/figs/answer_grad/sft_qwen3_500_attr_05031150/`.

## GSM8K results (long reasoning chains, full coverage to pos ~50,
50% coverage to pos 91)

The user's hypothesis holds **dramatically and persistently** through
all of training:

- After tail-4 normalization (each curve divided by the mean of the
  last 4 smoothed positions), the per-layer profile at the last
  recorded step shows a unified shape across all 28 layers: pos 0 at
  ~0.3–0.7 (sink-suppressed), peak at pos ~5 with grad ~2.5–3× the
  tail, decay through pos 50, then oscillation around 1.0.
- `log10(early/tail)` is uniformly **positive across all 28 layers
  and all 50 recorded steps**, ranging 0.25–0.85 (i.e., 1.8–7×).
- Effect is consistent across L0–L27; not just an early-training
  artifact.

The per-row absolute grad norm at the peak (pos ~3–7) shrinks during
training, but the **shape** of the decay curve is preserved. This
matches the autoregressive intuition: in causal attention, a token at
position k receives gradient from all downstream positions >k that
attend to it, so earlier answer tokens accumulate more downstream
gradient than later ones.

Logs: `plot_norm/figs/answer_grad/sft_qwen3_500_gsm8k_05031456/`.

## Pos-0 caveat (cross-link to findings/SINK.md)

Position 0 is the first prompt token (typically a chat-template special
token like `<|im_start|>`) — it sits at the attention sink. Gradient
flow at pos 0 is dominated by the sink interaction, which suppresses
`||dL/dh[0]||` at mid-layers (see SINK.md finding #6). When checking
"early-position grad larger than late-position grad", **always skip
pos 0** in the early reference; the script's default `early_window=(1, 8)`
does this.
