"""Sink-direction projection analysis.

The monitor records per-token <dL/dh, h/||h||> as `HiddenNorms.h_proj_grad`.
At the position-0 sink, the question is: does the loss's gradient
component along the residual direction bias toward shrinking ||h_L[0]||
(>0; -lr*g would shrink) or growing it (<0; -lr*g would grow), on
average over training?

Writes a per-(step, layer) line plot of mean pos-0 projection, plus a
table to stdout.

Run from repo root:
    PYTHONPATH=. /opt/uv/venv/bin/python plot_norm/sink_projection.py \
        logs_hf/sft_qwen3_500_proj_<ts>/norms
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


def main():
    norms_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if norms_dir is None:
        candidates = sorted(glob.glob("logs_hf/sft_*proj*/norms"))
        if not candidates:
            raise SystemExit("No logs_hf/sft_*proj*/norms found and no path given.")
        norms_dir = candidates[-1]
    print(f"NORMS_DIR = {norms_dir}")

    run_name = os.path.basename(os.path.dirname(os.path.abspath(norms_dir)))
    out_dir = os.path.join(_HERE, "figs", "sink_projection", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"OUT_DIR = {out_dir}")

    # Pull every recorded step.
    step_end = 0
    for f in glob.glob(os.path.join(norms_dir, "step_*_hidden.npz")):
        step_end = max(step_end, int(os.path.basename(f).replace("_hidden.npz", "").replace("step_", "").split("-")[1]))

    h = read_hidden(norms_dir, 0, step_end)
    rm = read_mask(norms_dir, 0, step_end)
    if h.h_proj_grad is None:
        raise SystemExit("This run has no h_proj_grad data. Re-run training with the updated monitor.")

    print(f"shapes: act={h.act.shape}  h_proj_grad={h.h_proj_grad.shape}")
    print(f"recorded steps: {h.global_steps.tolist()[:5]}...{h.global_steps.tolist()[-3:]}")

    S, A, R, B, L, T = h.h_proj_grad.shape
    proj = h.h_proj_grad   # (S, A, R, B, L, T)

    # Pos-0 projection: mean over (A, R, B), one curve per layer.
    pos0_proj = proj[..., 0]                       # (S, A, R, B, L)
    pos0_mean = pos0_proj.mean(axis=(1, 2, 3))     # (S, L)

    # Compare with non-pos-0 prompt-token projection (sanity baseline).
    rmask = rm.role_mask
    prompt_mask = (rmask == PROMPT)
    prompt_mask_no_pos0 = prompt_mask.copy()
    prompt_mask_no_pos0[..., 0] = False
    answer_mask = (rmask == ANSWER)

    def role_mean(arr5, role_mask_5):
        # arr5: (S, A, R, B, L, T)  role_mask_5: (S, A, R, B, T)
        # broadcast role mask across L.
        out = np.empty((arr5.shape[0], arr5.shape[4]), dtype=np.float32)
        for s in range(arr5.shape[0]):
            for li in range(arr5.shape[4]):
                vals = arr5[s, :, :, :, li, :][role_mask_5[s, :, :, :, :]]
                out[s, li] = float(vals.mean()) if vals.size else np.nan
        return out

    print("computing role-conditional means (slow path; 50 steps × 28 layers)...")
    prompt_mean = role_mean(proj, prompt_mask_no_pos0)   # (S, L)
    answer_mean = role_mean(proj, answer_mask)           # (S, L)

    # Print a per-layer table at first / last step.
    print("\nMean sink-direction projection at pos 0, prompt (excl pos 0), answer:")
    print(f"{'L':>3}  {'pos0 (early)':>14} {'pos0 (late)':>14}  "
          f"{'prompt (late)':>14}  {'answer (late)':>14}")
    s_first = 0
    s_last = S - 1
    for li in [0, 5, 11, 17, 23, 27]:
        if li >= L:
            continue
        print(f"{li:>3}  {pos0_mean[s_first, li]:14.4e} {pos0_mean[s_last, li]:14.4e}  "
              f"{prompt_mean[s_last, li]:14.4e}  {answer_mean[s_last, li]:14.4e}")

    # --- Plot 1: per-layer trajectory of pos-0 projection ---
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = matplotlib.colormaps["plasma"]
    for li in range(L):
        ax.plot(h.global_steps, pos0_mean[:, li], color=cmap(li / max(L - 1, 1)),
                lw=1.2, alpha=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean <dL/dh, h/||h||>  at  pos 0")
    ax.set_title("Sink-direction projection of grad at position 0 over training\n"
                 ">0 means -lr*g shrinks ||h[0]|| (loss prefers smaller sink)\n"
                 "<0 means -lr*g grows ||h[0]||")
    sm = cm.ScalarMappable(cmap=cmap,
                           norm=matplotlib.colors.Normalize(vmin=0, vmax=L - 1))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("layer (low=purple, high=yellow)")
    ax.grid(True, alpha=0.3)
    p = os.path.join(out_dir, "pos0_projection_trajectory.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {p}")

    # --- Plot 2: heatmap of pos-0 projection (step, layer) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = float(np.nanmax(np.abs(pos0_mean)))
    im = ax.imshow(pos0_mean.T, aspect="auto", origin="lower",
                   extent=(int(h.global_steps[0]), int(h.global_steps[-1]),
                           -0.5, L - 0.5),
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xlabel("training step")
    ax.set_ylabel("layer")
    ax.set_title("<dL/dh, h/||h||> at pos 0, per (step, layer)\n"
                 "red = +ve (loss pulls ||h|| down)   blue = -ve (loss pulls ||h|| up)")
    fig.colorbar(im, ax=ax, label="projection value")
    p = os.path.join(out_dir, "pos0_projection_heatmap.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {p}")

    # --- Plot 3: per-layer comparison pos-0 vs prompt-rest vs answer ---
    fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for ax, data, title in zip(axs,
                                [pos0_mean, prompt_mean, answer_mean],
                                ["pos 0 (sink)", "prompt (excl pos 0)", "answer"]):
        for li in range(L):
            ax.plot(h.global_steps, data[:, li], color=cmap(li / max(L - 1, 1)),
                    lw=1.0, alpha=0.7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("training step")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("mean <dL/dh, h/||h||>")
    fig.suptitle("Sink-direction projection of grad, by token role")
    fig.tight_layout()
    p = os.path.join(out_dir, "projection_by_role.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {p}")

    # --- Plot 4: late-training mean per layer (the headline result) ---
    # Average over the last quarter of recorded steps to suppress noise.
    n_late = max(1, S // 4)
    pos0_late = pos0_mean[-n_late:].mean(axis=0)        # (L,)
    prompt_late = prompt_mean[-n_late:].mean(axis=0)    # (L,)
    answer_late = answer_mean[-n_late:].mean(axis=0)    # (L,)
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = np.arange(L)
    ax.plot(layers, pos0_late, "o-", lw=1.8, color="#c62828", label="pos 0 (sink)")
    ax.plot(layers, prompt_late, "s-", lw=1.4, color="#2e7d32", alpha=0.8,
            label="prompt (excl pos 0)")
    ax.plot(layers, answer_late, "^-", lw=1.4, color="#1565c0", alpha=0.8,
            label="answer")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"mean <dL/dh, h/||h||>  (avg over last {n_late} steps)")
    ax.set_title("Late-training sink-direction projection by layer\n"
                 "negative at sink layers (L2-L26) = loss prefers larger ||h||,\n"
                 "providing the bias that grows the sink magnitude over training")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    p = os.path.join(out_dir, "late_per_layer.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {p}")

    # --- Plot 5: log-scaled heatmap (symlog so the smaller signals are visible) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    # Compute symlog-friendly threshold: 10× the typical noise floor.
    abs_pos0 = np.abs(pos0_mean)
    linthresh = float(np.median(abs_pos0[abs_pos0 > 0])) if (abs_pos0 > 0).any() else 1e-7
    norm = matplotlib.colors.SymLogNorm(linthresh=linthresh,
                                        vmin=-float(abs_pos0.max()),
                                        vmax=float(abs_pos0.max()))
    im = ax.imshow(pos0_mean.T, aspect="auto", origin="lower",
                   extent=(int(h.global_steps[0]), int(h.global_steps[-1]),
                           -0.5, L - 0.5),
                   cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax.set_xlabel("training step")
    ax.set_ylabel("layer")
    ax.set_title("<dL/dh, h/||h||> at pos 0 — symlog, all layers visible\n"
                 "red = +ve (loss pulls ||h|| down)   blue = -ve (loss pulls ||h|| up)")
    fig.colorbar(im, ax=ax, label="projection value (symlog)")
    p = os.path.join(out_dir, "pos0_projection_heatmap_symlog.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {p}")

    # Print late-training summary table
    print(f"\n=== Late-training (last {n_late} steps) mean projection by layer ===")
    print(f"{'L':>3}  {'pos0':>13}  {'prompt(rest)':>14}  {'answer':>13}  sign(pos0)")
    for li in range(L):
        sign = "+" if pos0_late[li] > 0 else "-" if pos0_late[li] < 0 else "0"
        print(f"{li:>3}  {pos0_late[li]:13.4e}  {prompt_late[li]:14.4e}  "
              f"{answer_late[li]:13.4e}  {sign}")


if __name__ == "__main__":
    main()
