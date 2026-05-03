"""Per-token gradient norm vs answer-relative position over training.

Standalone script (mirrors the notebook style). Builds a "re-anchored"
grad view where position 0 = first ANSWER token of each row (the first
generated token under SFT loss masking) and position k = the k-th
answer token. Tail positions where the answer is shorter than the
window are NaN.

Saves PNGs to plot_norm/figs/answer_grad/. Run from repo root:
    PYTHONPATH=. /opt/uv/venv/bin/python plot_norm/answer_grad_profile.py [NORMS_DIR]
"""
from __future__ import annotations

import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from plot_norm.read_norms import read_hidden, read_mask

PAD, PROMPT, ANSWER = 0, 1, 2

# OUT_DIR is set per-run in main() so plots from different runs don't
# clobber each other. Default fallback for direct script imports.
OUT_DIR = os.path.join(_HERE, "figs", "answer_grad")


def reanchor_to_answer(
    arr: np.ndarray,
    role_mask: np.ndarray,
    answer_window: int,
    role_target: int = ANSWER,
) -> np.ndarray:
    """Re-anchor `arr[..., L, T]` so position 0 of the new T axis is the
    first occurrence of `role_target` per (S, A, R, B) row.

    arr shape:        (S, A, R, B, L, T)
    role_mask shape:  (S, A, R, B, T)
    returns:          (S, A, R, B, L, answer_window)  float32, NaN-padded
                      where the answer span is shorter than answer_window OR
                      where the role_mask leaves the role_target region.
    """
    assert arr.ndim == 6, arr.shape
    assert role_mask.ndim == 5, role_mask.shape
    S, A, R, B, L, T = arr.shape
    out = np.full((S, A, R, B, L, answer_window), np.nan, dtype=np.float32)
    for s in range(S):
        for a in range(A):
            for r in range(R):
                for b in range(B):
                    rm_row = role_mask[s, a, r, b]
                    ans_idx = np.where(rm_row == role_target)[0]
                    if len(ans_idx) == 0:
                        continue
                    # Use the contiguous span starting at the first ANSWER token.
                    # If the answer has multiple disjoint spans (multi-turn SFT
                    # in which we'd consider all answers), this captures only
                    # the first turn's answer. That matches the user's
                    # "beginning of generation" framing.
                    fa = int(ans_idx[0])
                    # Find the end of this contiguous run.
                    end = fa
                    while end + 1 < T and rm_row[end + 1] == role_target:
                        end += 1
                    length = min(end - fa + 1, answer_window)
                    out[s, a, r, b, :, :length] = arr[s, a, r, b, :, fa:fa + length]
    return out


def _smooth(y, window_avg):
    if window_avg <= 1:
        return y
    import pandas as pd
    return (
        pd.Series(y)
        .rolling(window_avg, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )


def _smooth_along(arr: np.ndarray, axis: int, window: int) -> np.ndarray:
    """NaN-aware centered rolling mean along `axis`.

    Used to smooth across the layer axis: adjacent layers have correlated
    gradient norms, so a small window (3–5) makes per-layer trends easier
    to read without losing the emerge-plateau-unwind structure documented
    in the cross-architecture findings."""
    if window <= 1:
        return arr
    a = np.moveaxis(arr, axis, -1)
    out = np.empty_like(a, dtype=np.float32)
    n = a.shape[-1]
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i - half + window)
        with np.errstate(invalid="ignore"):
            out[..., i] = np.nanmean(a[..., lo:hi], axis=-1)
    return np.moveaxis(out, -1, axis)


def _save_fig(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {p}")


def _normalize_curves(curves: np.ndarray, mode: str, tail_n: int) -> np.ndarray:
    """Divide each curve (last axis is position) by a reference value.

    Modes:
      "tail": mean of the last `tail_n` positions (most stable; default).
              Curves end near 1.0, so the early-vs-late shape is the
              quantity of interest.
      "last": single last position (noisy if the curve is jittery).
      "first": position 0 — legacy. NOT recommended for grad-vs-position
               because pos 0 sits on the `<|im_start|>` sink and has
               anomalously suppressed gradient.

    Smoothing should happen BEFORE this call so the reference is also
    smoothed (otherwise dividing by a noisy single point can dominate
    the visual)."""
    if mode == "first":
        ref = curves[..., :1]
    elif mode == "last":
        ref = curves[..., -1:]
    elif mode == "tail":
        n = max(1, min(tail_n, curves.shape[-1]))
        with np.errstate(invalid="ignore"):
            ref = np.nanmean(curves[..., -n:], axis=-1, keepdims=True)
    else:
        raise ValueError(f"unknown normalize mode: {mode!r}")
    return curves / np.maximum(ref, 1e-12)


def plot_per_step(grad_ans: np.ndarray, global_steps: np.ndarray,
                  layer: int, *, normalize: bool = False, window_avg: int = 8,
                  normalize_to: str = "tail", normalize_tail_n: int = 4,
                  fname: str | None = None):
    """One curve per global_step at a fixed layer. Mean over (A, R, B), nanmean.

    `normalize_to` selects the reference: "tail" (default), "last", or
    "first". Smoothing is applied first; then each curve is divided by
    the smoothed reference, so the trend is observable without a noisy
    endpoint dominating."""
    S, A, R, B, L, W = grad_ans.shape
    sub = grad_ans[:, :, :, :, layer, :]                # (S, A, R, B, W)
    with np.errstate(invalid="ignore"):
        curves = np.nanmean(sub, axis=(1, 2, 3))        # (S, W)
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = cm.get_cmap("plasma")
    pos = np.arange(W)
    smoothed = np.array([_smooth(c, window_avg) for c in curves])
    if normalize:
        smoothed = _normalize_curves(smoothed, normalize_to, normalize_tail_n)
    for s_idx in range(S):
        gs = int(global_steps[s_idx])
        ax.plot(pos, smoothed[s_idx], color=cmap(s_idx / max(S - 1, 1)), lw=1.2,
                label=f"step={gs}" if (s_idx % max(1, S // 10) == 0 or s_idx == S - 1) else None)
    norm_label = {"tail": f"tail-{normalize_tail_n}", "last": "pos=last",
                  "first": "pos=0"}.get(normalize_to, normalize_to)
    ax.set_xlabel("Answer-relative position (0 = first generated token)")
    ax.set_ylabel(f"Grad norm{(' / ' + norm_label) if normalize else ''}")
    ax.set_title(f"Grad norm vs position-in-answer  —  layer {layer}  "
                 f"({'norm to ' + norm_label if normalize else 'raw'}, window_avg={window_avg})")
    if S > 1:
        sm = cm.ScalarMappable(cmap=cmap,
                               norm=matplotlib.colors.Normalize(vmin=int(global_steps[0]),
                                                                vmax=int(global_steps[-1])))
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("global step (early=purple, late=yellow)")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, fname or f"grad_step_sweep_L{layer}.png")


def plot_decay_heatmap(grad_ans_plot: np.ndarray, global_steps: np.ndarray,
                       *, fname: str = "decay_heatmap.png",
                       layer_smooth: int = 3,
                       early_window: tuple = (1, 8),
                       tail_window_frac: float = 0.25):
    """Heatmap: x=training step, y=layer, color=log10(early / late) where
    early = mean grad over positions in `early_window` and late = mean
    grad over the trailing `tail_window_frac` of the plot window.

    Position 0 is excluded by default because it sits on the
    `<|im_start|>` sink and has anomalously low gradient (the sink
    suppresses dL/dh there). Positions 1..7 (the "peak" of the decay
    profile for long-answer datasets) average over what's actually the
    early-position grad signal. The tail window averages the last
    quarter to reduce noise from selection-bias rows.

    Positive (red) = decay-with-position holds (early > late).
    Zero (gray)    = flat profile.
    Negative (blue)= grad grows with position (counter to hypothesis).

    `layer_smooth` is a centered rolling-mean window across the layer
    axis, which surfaces the cross-layer trend without losing the
    emerge-plateau-unwind structure documented in the cross-architecture
    findings."""
    S, A, R, B, L, W = grad_ans_plot.shape
    with np.errstate(invalid="ignore"):
        mean_curve = np.nanmean(grad_ans_plot, axis=(1, 2, 3))   # (S, L, W)
    e_lo, e_hi = early_window
    e_hi = min(e_hi, W)
    tail_lo = max(e_hi, int(W * (1.0 - tail_window_frac)))
    with np.errstate(invalid="ignore"):
        early = np.nanmean(mean_curve[..., e_lo:e_hi], axis=-1)      # (S, L)
        late  = np.nanmean(mean_curve[..., tail_lo:],   axis=-1)      # (S, L)
    eps = 1e-12
    log_ratio = np.log10(np.maximum(early, eps) / np.maximum(late, eps))  # (S, L)
    if layer_smooth > 1:
        log_ratio = _smooth_along(log_ratio, axis=1, window=layer_smooth)

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = float(np.nanmax(np.abs(log_ratio)))
    vmax = max(vmax, 0.3)
    im = ax.imshow(
        log_ratio.T, aspect="auto", origin="lower",
        extent=(int(global_steps[0]), int(global_steps[-1]), -0.5, L - 0.5),
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("layer")
    smooth_tag = f", layer_smooth={layer_smooth}" if layer_smooth > 1 else ""
    ax.set_title(
        f"log10( mean(grad[{e_lo}:{e_hi}]) / mean(grad[{tail_lo}:{W}]) )  "
        f"per (step, layer){smooth_tag}\n"
        f"red = early-position grad larger (decay-with-position holds)\n"
        f"blue = early-position grad smaller (counter to hypothesis)")
    fig.colorbar(im, ax=ax, label="log10 ratio")
    _save_fig(fig, fname)


def plot_decay_layer_trajectories(grad_ans_plot: np.ndarray, global_steps: np.ndarray,
                                   *, fname: str = "decay_trajectories.png",
                                   layer_smooth: int = 3,
                                   early_window: tuple = (1, 8),
                                   tail_window_frac: float = 0.25):
    """One curve per layer: log10(early/late) vs training step. Same
    reference convention as `plot_decay_heatmap`: early window excludes
    pos 0 (the sink position) by default. Shows whether the
    decay-with-position effect grows or fades during SFT.
    `layer_smooth` averages adjacent-layer curves to suppress noise."""
    S, A, R, B, L, W = grad_ans_plot.shape
    with np.errstate(invalid="ignore"):
        mean_curve = np.nanmean(grad_ans_plot, axis=(1, 2, 3))   # (S, L, W)
    e_lo, e_hi = early_window
    e_hi = min(e_hi, W)
    tail_lo = max(e_hi, int(W * (1.0 - tail_window_frac)))
    with np.errstate(invalid="ignore"):
        early = np.nanmean(mean_curve[..., e_lo:e_hi], axis=-1)
        late = np.nanmean(mean_curve[..., tail_lo:], axis=-1)
    eps = 1e-12
    log_ratio = np.log10(np.maximum(early, eps) / np.maximum(late, eps))  # (S, L)
    if layer_smooth > 1:
        log_ratio = _smooth_along(log_ratio, axis=1, window=layer_smooth)
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = matplotlib.colormaps["plasma"]
    for li in range(L):
        ax.plot(global_steps, log_ratio[:, li], color=cmap(li / max(L - 1, 1)),
                lw=1.2, alpha=0.85)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel(f"log10( mean(grad[{e_lo}:{e_hi}]) / mean(grad[{tail_lo}:{W}]) )")
    ax.set_title("Decay-with-position over training, per layer\n"
                 "positive = early-position grad larger than late-position grad")
    sm = cm.ScalarMappable(cmap=cmap,
                           norm=matplotlib.colors.Normalize(vmin=0, vmax=L - 1))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("layer (low=purple, high=yellow)")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, fname)


def plot_per_layer(grad_ans: np.ndarray, layer_indices, step_idx: int,
                   *, normalize: bool = False, window_avg: int = 8,
                   layer_smooth: int = 3,
                   normalize_to: str = "tail", normalize_tail_n: int = 4,
                   fname: str | None = None):
    """One curve per layer at a fixed step. Mean over (A, R, B).
    `layer_smooth` averages each curve with its neighbours along the
    layer axis to surface the trend; pass 1 to disable.

    `normalize_to` selects the reference: "tail" (mean of last
    `normalize_tail_n` positions, default), "last", or "first".
    Token-axis smoothing happens before normalization so the reference
    is also smoothed."""
    S, A, R, B, L, W = grad_ans.shape
    sub = grad_ans[step_idx, :, :, :, :, :]             # (A, R, B, L, W)
    with np.errstate(invalid="ignore"):
        curves = np.nanmean(sub, axis=(0, 1, 2))        # (L, W)
    if layer_smooth > 1:
        curves = _smooth_along(curves, axis=0, window=layer_smooth)
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(layer_indices)
    cmap = cm.get_cmap("plasma")
    pos = np.arange(W)
    smoothed = np.array([_smooth(curves[li], window_avg) for li in range(L)])
    if normalize:
        smoothed = _normalize_curves(smoothed, normalize_to, normalize_tail_n)
    for i, li in enumerate(layer_indices):
        ax.plot(pos, smoothed[li], color=cmap(i / max(n - 1, 1)), lw=1.2,
                label=f"L{li}" if (i % max(1, n // 7) == 0 or i == n - 1) else None)
    norm_label = {"tail": f"tail-{normalize_tail_n}", "last": "pos=last",
                  "first": "pos=0"}.get(normalize_to, normalize_to)
    ax.set_xlabel("Answer-relative position (0 = first generated token)")
    ax.set_ylabel(f"Grad norm{(' / ' + norm_label) if normalize else ''}")
    ax.set_title(f"Grad norm vs position-in-answer  —  step idx {step_idx}  "
                 f"({'norm to ' + norm_label if normalize else 'raw'}, window_avg={window_avg})")
    if n > 1:
        sm = cm.ScalarMappable(cmap=cmap,
                               norm=matplotlib.colors.Normalize(vmin=0, vmax=n - 1))
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("layer (low=purple, high=yellow)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, fname or f"grad_layer_sweep_step{step_idx}.png")


def coverage_per_position(grad_ans: np.ndarray) -> np.ndarray:
    """Returns (W,) fraction of (S, A, R, B, L) slices that are non-NaN
    at each position. Use to find the largest position with >=X% coverage
    so plots don't get dominated by long-answer selection bias."""
    S, A, R, B, L, W = grad_ans.shape
    finite = np.isfinite(grad_ans).reshape(-1, W)  # one layer slice = same coverage
    return finite.mean(axis=0)


def slope_table(grad_ans: np.ndarray, global_steps: np.ndarray,
                positions=(0, 4, 16, 64, 256), layers=None,
                coverage_floor: float = 0.5) -> str:
    """Numeric table: median grad norm at selected positions, per
    (step, layer). Stats are computed only over rows whose answer span
    actually includes that position. To avoid selection-bias surprises
    (positions past the typical answer length pick up only outlier rows),
    each cell prints (median, n_rows). The 'pos0/pos_last' ratio is only
    meaningful when coverage at pos_last is high — flagged with '*' when
    coverage < `coverage_floor`."""
    S, A, R, B, L, W = grad_ans.shape
    if layers is None:
        layers = [0, L // 4, L // 2, 3 * L // 4, L - 1]
    positions = [p for p in positions if p < W]

    cov = coverage_per_position(grad_ans)  # (W,)
    out = []
    out.append("Coverage (fraction of (s,a,r,b) rows reaching this position):")
    out.append("  " + "  ".join(f"pos{p:>4}={cov[p]*100:5.1f}%" for p in positions))
    out.append("")
    out.append(f"{'step':>6}  {'L':>3}  " + "  ".join(f"{'pos' + str(p):>14}" for p in positions)
               + f"  {'p0/p_last':>10}")
    for s_idx in range(S):
        gs = int(global_steps[s_idx])
        for li in layers:
            sub = grad_ans[s_idx, :, :, :, li, :]      # (A, R, B, W)
            cells = []
            vals_for_ratio = []
            for p in positions:
                col = sub[..., p].ravel()
                col = col[np.isfinite(col)]
                if col.size == 0:
                    cells.append(f"{'-':>14}")
                    vals_for_ratio.append(np.nan)
                else:
                    med = float(np.median(col))
                    cells.append(f"{med:.2e}(n={col.size:>3})")
                    vals_for_ratio.append(med)
            if (np.isfinite(vals_for_ratio[0]) and np.isfinite(vals_for_ratio[-1])
                    and vals_for_ratio[-1] > 0):
                r = vals_for_ratio[0] / vals_for_ratio[-1]
                flag = "*" if cov[positions[-1]] < coverage_floor else " "
                rstr = f"{r:9.2f}{flag}"
            else:
                rstr = f"{'-':>10}"
            row = f"{gs:>6}  {li:>3}  " + "  ".join(cells) + f"  {rstr}"
            out.append(row)
        out.append("")
    out.append(f"  '*' = pos_last has coverage below {coverage_floor*100:.0f}%; "
               f"that ratio is dominated by long-answer rows only.")
    return "\n".join(out)


def main():
    norms_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if norms_dir is None:
        candidates = sorted(glob.glob("logs_hf/sft_*/norms"))
        if not candidates:
            raise SystemExit("No logs_hf/sft_*/norms found and no path given.")
        norms_dir = candidates[-1]
    print(f"NORMS_DIR = {norms_dir}")

    # Per-run output directory: uses the parent run-dir name so different
    # runs don't clobber each other.
    run_name = os.path.basename(os.path.dirname(os.path.abspath(norms_dir)))
    global OUT_DIR
    OUT_DIR = os.path.join(_HERE, "figs", "answer_grad", run_name)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"OUT_DIR = {OUT_DIR}")

    # Pull every recorded step in this run.
    all_steps = []
    for f in glob.glob(os.path.join(norms_dir, "step_*_hidden.npz")):
        m = os.path.basename(f).replace("_hidden.npz", "").replace("step_", "")
        a, b = m.split("-")
        all_steps.append(int(b))
    if not all_steps:
        raise SystemExit(f"No step_*_hidden.npz in {norms_dir}")
    step_end = max(all_steps)

    h = read_hidden(norms_dir, 0, step_end)
    rm = read_mask(norms_dir, 0, step_end)
    print(f"act/grad shape: {h.act.shape}  mask shape: {rm.role_mask.shape}")
    print(f"recorded steps: {h.global_steps.tolist()}")

    # Choose a generous answer window. The longest possible answer is
    # T = h.act.shape[-1]. Use the actual longest contiguous ANSWER run
    # observed for tightness.
    S, A, R, B, L, T = h.grad.shape
    longest = 0
    for s in range(S):
        for a in range(A):
            for r in range(R):
                for b in range(B):
                    row = rm.role_mask[s, a, r, b]
                    runs = np.where(row == ANSWER)[0]
                    if len(runs) == 0:
                        continue
                    # contiguous run from runs[0]
                    fa = runs[0]
                    end = fa
                    while end + 1 < T and row[end + 1] == ANSWER:
                        end += 1
                    longest = max(longest, end - fa + 1)
    answer_window = min(T, max(64, longest))
    print(f"answer_window = {answer_window} (longest contiguous first-answer run observed)")

    grad_ans = reanchor_to_answer(h.grad, rm.role_mask, answer_window)
    print(f"grad_ans shape: {grad_ans.shape}  "
          f"frac NaN: {np.isnan(grad_ans).mean():.2%}")

    # Decide a coverage-controlled plot window: only plot positions where
    # at least 50% of rows still have an answer token. Past that, the curve
    # is dominated by the longest-answer rows and looks like selection bias.
    cov = coverage_per_position(grad_ans)
    plot_W = int((cov >= 0.5).sum())
    print(f"plot window (>=50% coverage): {plot_W}  "
          f"(coverage at pos 0={cov[0]*100:.1f}%, pos {plot_W-1}={cov[plot_W-1]*100:.1f}%)")
    grad_ans_plot = grad_ans[..., :plot_W]

    # Coverage curve as a sanity check.
    fig_cov, ax_cov = plt.subplots(figsize=(9, 3.5))
    ax_cov.plot(np.arange(answer_window), cov, lw=1.5)
    ax_cov.axhline(0.5, color="r", ls="--", lw=0.8, label="50% coverage cutoff")
    ax_cov.axvline(plot_W, color="r", ls=":", lw=0.8)
    ax_cov.set_xlabel("Answer-relative position")
    ax_cov.set_ylabel("Fraction of rows with data")
    ax_cov.set_title("Per-position coverage (rows with answer span reaching this position)")
    ax_cov.set_ylim(0, 1.02)
    ax_cov.legend()
    ax_cov.grid(True, alpha=0.3)
    _save_fig(fig_cov, "coverage.png")

    # Per-step sweep at a few representative layers (coverage-restricted).
    for layer in [0, 5, 11, 17, 23, L - 1]:
        if layer >= L:
            continue
        plot_per_step(grad_ans_plot, h.global_steps, layer=layer,
                      normalize=False, window_avg=4,
                      fname=f"grad_step_sweep_L{layer:02d}.png")
        plot_per_step(grad_ans_plot, h.global_steps, layer=layer,
                      normalize=True, window_avg=4,
                      fname=f"grad_step_sweep_L{layer:02d}_norm.png")

    # Per-layer sweep at first/last recorded step.
    layers_to_sweep = list(range(L))
    plot_per_layer(grad_ans_plot, layers_to_sweep, step_idx=0,
                   normalize=True, window_avg=4,
                   fname="grad_layer_sweep_first_step.png")
    plot_per_layer(grad_ans_plot, layers_to_sweep, step_idx=S - 1,
                   normalize=True, window_avg=4,
                   fname="grad_layer_sweep_last_step.png")
    plot_per_layer(grad_ans_plot, layers_to_sweep, step_idx=S - 1,
                   normalize=False, window_avg=4,
                   fname="grad_layer_sweep_last_step_raw.png")

    # Headline summary: decay-with-position vs (step, layer)
    plot_decay_heatmap(grad_ans_plot, h.global_steps)
    plot_decay_layer_trajectories(grad_ans_plot, h.global_steps)

    # Numeric summary
    print("\n=== Decay-with-position table (median grad norm) ===")
    print(slope_table(grad_ans, h.global_steps,
                      positions=(0, 4, 16, 64, min(256, answer_window - 1)),
                      layers=[0, 5, 11, 17, 23, L - 1]))


if __name__ == "__main__":
    main()
