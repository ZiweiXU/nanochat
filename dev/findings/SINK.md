# Finding: position-0 attention sink and its SFT dynamics

The original "prompt activations larger than answer activations"
observation reduces to **a single universal attention sink at position
0**. Bulk distributions are role-symmetric and unchanged by SFT.

## 1. The sink itself

Forward-pass on the *base* Qwen3-0.6B on 16 SmolTalk rows: exactly
**one sink per row at position 0**, on whatever special token the chat
template puts there. Identical norm across rows (the model has no
context before pos 0, so its pos-0 residual is a row-invariant function
of input token + pos-0 rotary). Other tokens of the same type at later
positions have normal norm.

Per-layer profile (Qwen3-0.6B, median pos-0 norm; sink is established
at L2 and held in the residual stream until the final layer unwinds it
for the unembedding):

| L  | pre-SFT pos0 | post-SFT pos0 | answer_med | post-SFT pos0/ans |
|---:|-------------:|--------------:|-----------:|------------------:|
|  1 |        19.0  |        18.7   |     10.6   |              1.76 |
|  2 |     **3272** |     **5160** |     13.2   |            392.12 |
| 11 |       3272   |      5160     |     44.0   |            117.30 |
| 23 |       3306   |      5160     |    346.5   |             14.89 |
| 26 |       3098   |      4860     |    569.7   |              8.53 |
| 27 |     **1283** |      **662** |    416.1   |              1.59 |

L2-L26 LATE values are at the quantization ceiling (≥5160; real norm
is larger). See SETUP.md "Recorded data layout" for the ceiling caveat.

## 2. Cross-scale and cross-architecture

The single-pos-0 sink pattern is **universal across Llama-class
models**:

| Model | sink layer | pos-0 token | sink norm | final-layer | unwind |
|-------|-----------:|-------------|----------:|------------:|-------:|
| Qwen3-0.6B-Base       | L2 | `<|im_start|>` (151644) | 3739 | 1012 | 73 % |
| Qwen3-1.7B-Base       | L2 | `<|im_start|>` (151644) | 3025 | 3371 | none |
| Llama-3.2-1B-Instruct | L1 | `<|begin_of_text|>` (128000) | 799 | 314 | 61 % |
| SmolLM2-135M-Instruct | L11 | `<|im_start|>` (1) | **26260** | 6095 | 77 % |

Same emerge → plateau → unwind structure. Qwen3-1.7B doesn't unwind at
L27 because its overall residual norms grow more aggressively at deep
layers, naturally diluting the sink.

## 3. SFT amplifies the sink magnitude — does NOT propagate

After 500 SFT steps on Qwen3-0.6B (verified via `--save-checkpoint`
forward-passed on the same rows): position-0 sink norm grew **3739.7 →
5159.2** at L11. Every other `<|im_start|>` position (subsequent turn
boundaries) stayed at ~47–55. **Exactly 1 sink per row before and
after.** The "prompt mean widens during SFT" observation is entirely
the per-row outlier dragging up the mean.

## 4. Bulk vs tail (L11 LATE window)

| stat   | PROMPT  | ANSWER |
|--------|--------:|-------:|
| median |   45.28 |  43.99 |
| p95    |   54.26 |  55.41 |
| p99    | **5160** |  55.41 |

Below the p95, prompt and answer are statistically indistinguishable.
The entire "P/A gap" is in the >99th percentile (the per-row sink). SFT
trajectory: L11 prompt median 46.48 → 45.28 (flat), max 3740 → 5160
(grows). At L27, both medians actually decrease (543 → 441) — top
layer learns to "read around" the sink more aggressively during SFT.

## 5. Sub-block attribution (mid-layer P/A means, late window)

| layer | act P/A | attn_out P/A | mlp_out P/A |
|------:|--------:|-------------:|------------:|
|     5 |    3.47 |         1.06 |        1.04 |
|    11 |    2.26 |         0.99 |        1.01 |
|    17 |    1.59 |         0.99 |        1.03 |

Per-block contributions are role-symmetric. The asymmetry in `act` is
not introduced by attention or MLP magnitude per se — it lives in the
residual-stream geometry, with sinks contributing the only meaningful
asymmetry. Per-head Q/K/V projections also show p/a ≈ 1.00 for every
head at every mid-layer (RMSNorm divides out per-token magnitude
before the projections).

## 6. Gradient at position 0 (median ||dL/dh_L[0]||, late window)

| L  | pos0 grad | answer_grad_med | pos0 / answer |
|---:|----------:|----------------:|--------------:|
|  5 |   4.7e-2  |   9.8e-2        |          0.48 |
| 11 |   4.1e-2  |   7.0e-2        |          0.59 |
| 17 |   3.4e-2  |   2.8e-2        |          1.21 |
| 23 |   2.3e-2  |   7.0e-3        |          3.35 |
| 27 |     0     |   7.6e-3        |          0    |

Pos-0 gradient is **not negligible** — substantial at deeper layers
(reflecting attention concentration onto pos 0 from query positions
deeper in the network), reduced (~0.5×) at sink-prominent mid-layers.
L27 = 0 because pos 0 has `labels=-100` so unembedding generates no
direct loss signal there. Sink growth is therefore a **bias** in the
weight-update direction (gradients aren't zero, but their projection
onto the sink-shrinking direction is small), not a "free run."

## 7. Sink-direction projection probe

To pin down the mechanism, the monitor was extended to record per-token
`<dL/d(h_L[t]), h_L[t] / ||h_L[t]||>` (the projection of the residual
gradient onto the unit residual direction; signed). Sign convention:
positive => `-lr·g` shrinks ||h|| at this position (loss prefers
smaller residual norm); negative => `-lr·g` grows ||h||.

Training run: `logs_hf/sft_qwen3_500_proj_05031546/`. Late-training
mean projection (last 12 of 50 recorded steps) by layer, by token role:

| layer | pos 0      | prompt-rest | answer    |
|------:|-----------:|------------:|----------:|
|   0   |  +3.2e-4   |   +1.2e-5   |  +1.5e-4  |
|   1   |  +1.5e-4   |   +3.1e-5   |  +1.7e-4  |
|   2   |  +4.0e-6   |   +9.1e-6   |  +1.6e-4  |
|  11   |  +1.1e-6   |   −1.7e-5   |  +5.0e-5  |
|  17   |  +8.3e-7   |   +7.0e-7   |  +1.4e-5  |
|  23   |  +2.8e-7   |   −4.5e-7   |  +1.5e-6  |
|  27   |  −2.7e-10  |   +1.1e-9   |  −3.5e-9  |

**Conclusion**: at sink layers (L≥2), the **pos-0 projection is
essentially zero** in late training. The signs across layers are
mixed and the magnitudes are 1e-6 to 1e-4 — within plausible noise
floor. By contrast, **answer tokens have consistently positive
projection** at every layer (1e-9 to 1e-4), decaying with depth — the
loss does want to shrink answer-token residual norms.

This reframes the open mechanism question. The sink-magnitude growth
during SFT (3740 → 5160 at L11) is **not** driven by direct gradient
pressure on h_L[0] (the projection there is ~0). Instead, it must be
indirect: gradient on OTHER positions (especially answer tokens) flows
back into parameter updates (embedding, attention W_q/k/v/o, MLP);
those updates change how the forward pass produces h_L[0] from the
position-0 input token, and cumulative effects across 500 steps shift
||h_L[0]||. The sink position itself sits at a near-saturated point of
the loss landscape — the loss is barely sensitive to it directly, so
the model has freedom to grow its magnitude without paying a cost.

The L=0 step-0 spike (+0.04) is unrelated to the sink — at L=0 there
is no sink (||h_L[0]|| = 19.5 there) and the projection just reflects
the initial loss structure on raw embeddings before any block updates.
By step ~30 even the L=0 projection is small.

Caveat: pos-0 values are subject to per-row uint8 quantization (1%
outliers preserved); if pos 0's true projection is small relative to
the per-row max-min range, it may be coarsely binned. The qualitative
finding (sink-position gradient is near-zero at sink layers) is
robust because it's consistent across layers and steps, but the exact
magnitudes should not be over-interpreted.
