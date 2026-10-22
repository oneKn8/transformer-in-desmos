# Grokking modular addition at Desmos shapes: offline feasibility study

Question: can a one-layer transformer be trained inside Desmos on `a + b mod p`, grok, and
show its Fourier mechanism while it learns? This folder simulates the exact training offline
(PyTorch, float64 like Desmos) and prices each step with the measured Desmos cost model.
Nothing here touches Desmos.

Date: 2026-10-22. All numbers come from the runs in `results/`; the full per-run tables
are in `results/summary.md`.

## Verdict

**Feasible, at p = 31, with a specific recipe, in about 10 to 60 minutes of Desmos ticking.
The fast settings show the Fourier circuit forming but no real grokking delay. Real
delayed grokking needs a lower train fraction and a chosen seed.**

- A Desmos-shaped one-layer transformer (d_model 32, 4 heads of 8, MLP 128, ReLU, no
  LayerNorm, AdamW, full batch, tokens `[a, b, =]`, logits at `=`) reaches 99% test accuracy
  in **118 to 2135 steps** across the settings that grok on every seed (4 of 4 seeds in each
  of four settings at p = 31 or 23). A step costs an estimated **1.2 to 2.0 s in Desmos**
  (885 to 1400 32x32-matmul-equivalents plus tick overhead). That is roughly 3 to 70 minutes
  per run, usually 5 to 30.
- **The embedding visibly becomes circles.** Across the 37 grokked attention runs, the top
  3 frequencies hold a median 92% (range 68 to 99%) of the embedding's non-constant norm. At
  init that share is the random baseline of about 13 to 29%. Projecting the embedding onto
  the top frequency's cos/sin plane puts token t at angle 2*pi*k*t/p: phase coherence median
  0.998, minimum 0.986, radius spread 5 to 19%. See the PNGs under `results/`.
- **Honest caveat on "grokking".** At train fraction >= 0.6, which is where it is fast,
  test accuracy rises *together with* train accuracy. That is fast generalization: the
  circle forms in 60 to 230 steps, and there is no long plateau where train is 100% and test
  is stuck. A real delay appears only at lower fractions. The best attention example is p = 31,
  lr 3e-2, wd 1, **frac 0.5, seed 0**: train 99% at step 53, test flat at ~42% until step ~370,
  then a loss spike, then the circle snaps in and test climbs to 99% by step ~1100 to 1151.
  That is ~30 minutes in Desmos. The delay depends on the seed (the other 3 seeds generalize
  early), so the Desmos build should load that seed's exact initial weights and data split.
- **Constraints the study found:** AdamW plus weight decay is required (20 SGD/GD settings
  all failed, and AdamW without decay memorizes but never generalizes). AdamW costs only 1 to
  4% more per Desmos step than SGD, so this is not a problem. One attention head never
  grokked (0 of 6). p = 31 is the sweet spot: vocab p + 1 = 32 fits one tile, and p = 113 (the
  paper's p) costs ~18 s per step.
- **Recommended Desmos config (the claim "a transformer trained inside Desmos that visibly
  groks"):** p = 31, attention model as above, AdamW (beta 0.9/0.98, 10-step linear warmup)
  lr 3e-2, wd 1, full batch, train fraction 0.5 (480 pairs), seed 0. Estimated **1.52 to
  1.62 s per step, about 30 minutes to 99% test**, with a visible plateau, jump and circle.
  Fallback if the delay matters less than reliability: lr 1e-2, wd 1, frac 0.6 (4 of 4 seeds
  stable, 714 to 2135 steps, 22 to 70 minutes).
- **Cheapest options:** the attention-free MLP (concat embeddings, MLP 128, frac 0.75)
  groks in 184 to 255 steps (3 of 3 seeds), about 6 minutes. p = 23 attention (frac 0.75)
  groks in 149 to 473 steps (4 of 4), 3 to 11 minutes. Neither shows a grokking delay.

## Setup (checked against the paper)

Nanda, Chan, Lieberum, Smith, Steinhardt, "Progress measures for grokking via mechanistic
interpretability", arXiv:2301.05217 (v3, Oct 2023). From the paper (fetched 2026-10-22):
p = 113, one-layer ReLU transformer, d_model 128, 4 heads of 32, d_mlp 512, **no
LayerNorm**, untied embed/unembed, learned positions, no biases except in the MLP, input
`a b =`, logits read at `=`, AdamW lr 1e-3, weight decay 1, 30% of pairs for training, full
batch, 40k epochs, test accuracy near 100% after ~10k epochs, key frequencies
{14, 35, 41, 42, 52}, and W_E with "significant norm at 6 frequencies". AdamW betas
(0.9, 0.98) and the 10-step linear warmup come from my memory of the public code. They are
not in the paper text I read, so they are unverified. No code was downloaded; `grok.py`
is written from scratch.

Desmos shrink: d_model 32, heads 1 or 4 (d_head 32 or 8), d_mlp 32/64/128, same init scale
(N(0, 1/d_model)). Because there is one layer and the readout is only at `=`, only the `=`
query matters (the other positions never reach the logits), so the model computes just that
query. This is exact; `test_grok.py` checks the fast path against the einsum reference.

## Results table

Selected configs (all are full batch, d_model 32, no LayerNorm, ReLU unless noted). Steps
are per seed; `>N` means not reached within the cap. "Delay" = step where test first
reaches 50% minus the step where train first reaches 99%. A large delay is grokking proper.
Desmos hours use the slowest seed and 50 to 150 ms tick overhead.

| p | arch | optimizer | lr | wd | train frac | steps to 99% test (per seed) | seeds | delay (test50 - train99) | Desmos s/step | est. Desmos time |
|---|---|---|---|---|---|---|---|---|---|---|
| 31 | MLP concat, m128 | AdamW | 1e-2 | 3 | 0.75 | 184, 255, 195 | 3/3 | 50, 85, 49 | 1.39-1.49 | 4-6 min |
| 23 | attn h4 m128 | AdamW | 3e-2 | 1 | 0.75 | 149, 317, 473, 207 | 4/4 | -6 to -12 (none) | 1.31-1.41 | 3-11 min |
| 31 | attn h4 m32 | AdamW | 3e-2 | 1 | 0.6 | 373, 185, 1971, 118 | 4/4 | -1 to -12 (none) | 1.20-1.30 | 3-43 min |
| 31 | attn h4 m128 | AdamW | 3e-2 | 1 | 0.6 | 241, 617, 182, >2500 | 3/4 | -2 to -28 (none) | 1.87-1.97 | 6-20 min (3 of 4) |
| **31** | **attn h4 m128** | **AdamW** | **3e-2** | **1** | **0.5** | **1151, 195, 804, 310** | **4/4** | **424, -14, 16, 91** | **1.52-1.62** | **5-31 min** |
| 31 | attn h4 m128 | AdamW | 1e-2 | 1 | 0.6 | 2135, 904, 714, 868 | 4/4 | 2 to -6 (none) | 1.87-1.97 | 22-70 min |
| 31 | attn h4 m128 | AdamW | 1e-2 | 3 | 0.6 | 234, >2500, >2500 | 1/3 | none | 1.87-1.97 | unreliable |
| 31 | attn h4 m64 | AdamW | 3e-2 | 1 | 0.6 | 770 | 1/1 | none | 1.43-1.53 | 18-20 min |
| 31 | attn h1 (m32, 64, 128) | AdamW | 3e-2 or 1e-2 | 1 or 3 | 0.6 | >2500 (6 runs) | 0/6 | | | fails |
| 31 | attn h4 m128 GELU | AdamW | 1e-2 | 3 | 0.6 | 443 | 1/1 | none | 1.87-1.97 | 14 min |
| 31 | attn h4 m128 GELU + LayerNorm | AdamW | 1e-2 | 3 | 0.6 | >2500 (train 47%) | 0/1 | | | fails |
| 31 | attn h4 m128 | AdamW, wd 0 | 1e-2 | 0 | 0.6 | >2500 (train 100%, test 53%) | 0/1 | | | fails |
| 31 | attn h4 m128 | SGD momentum 0.9 | 0.1-1 | lr*wd 1e-2, 3e-2 | 0.6 | >2500 (stuck at chance) | 0/6 | | | fails |
| 31 | attn h4 m128 | full-batch GD | 0.3-3 | lr*wd 1e-2, 3e-2 | 0.6 | >2500 (stuck at chance) | 0/6 | | | fails |
| 31 | attn h4 m128 | SGD momentum / GD | 0.3-3 | lr*wd 1e-4, 1e-3 | 0.6 | diverged | 0/8 | | | fails |
| 31 | attn h4 m128, batch 32/64/128 (random and cyclic) | AdamW | 1e-2 | 3 | 0.6 | >2500 (train 4-36%) | 0/6 | | 0.23-0.61 | fails (untuned) |
| 31 | MLP sum, m128 | AdamW | 1e-2 | 3 | 0.75 | 468 | 1/1 | none | 1.38-1.48 | 11 min |
| 31 | MLP folded (A[a]+B[b]), m128 | AdamW | 1e-2 | 3 | 0.45-0.75 | >2500 | 0/3 | | | fails |
| 31 | MLP any, m32 | AdamW | 1e-2 | 3 | 0.45-0.75 | >2500 | 0/9 | | | fails |
| 31 | MLP concat, m128 | AdamW | 1e-2 | 3 | 0.6 | 1248 | 1/1 | 312 | 1.16-1.26 | 24-26 min |
| 23 | attn h4 m128 (Nanda lr) | AdamW | 1e-3 | 1 | 0.3 / 0.45 / 0.6 | >10000 / >10000 / 8060 | 1/3 | 350 (frac 0.6) | 1.05-1.15 | 2.4 h |
| 31 | attn h4 m128 | AdamW | 3e-3 | 1 or 3 | 0.3-0.6 | >5000 (6 runs) | 0/6 | | | too slow |
| 47 | attn h4 m128 | AdamW | 1e-2 | 1 | 0.6 | 156, 403, 189 | 3/3 | none | 4.76-4.86 | 12-33 min |
| 47 | attn h4 m128 | AdamW | 1e-2 | 1 | 0.45 | 1717 | 1/1 | 752 | 3.65-3.75 | 1.8 h |
| 47 | attn h4 m128 | AdamW | 3e-3 | 1 | 0.45 | 2399 | 1/1 | 1302 | 3.65-3.75 | 2.5 h |
| 47 | attn h4 m128 | AdamW | 3e-3 to 1e-2 | 1 or 3 | 0.3 | >5000 (4 runs) | 0/4 | | | too slow |
| 59 | attn h4 m128 | AdamW | 1e-2 | 1 | 0.3 | 1533 | 1/1 | 1182 | 3.82-3.92 | 1.6 h |
| 97 | attn h4 m128 | AdamW | 1e-2 | 1 | 0.3 | 502 | 1/1 | 110 | 13.25-13.35 | 1.9 h |
| 113 | attn h4 m128 | AdamW | 1e-2 | 1 | 0.3 | 387 | 1/1 | -14 | 18.00-18.10 | 1.9 h |

What the table says:

1. **Train fraction is the main knob.** At p = 31, frac 0.3 and 0.45 never reached 99% within
   5000 steps for any lr/wd tried; 0.5 to 0.75 work. Larger p has more pairs at the same
   fraction, so it groks in fewer steps (p = 113 at frac 0.3: 387 steps), but every step
   costs far more in Desmos.
2. **Decay per step (lr x wd) has a narrow window, about 1% to 3% per step for AdamW.** At 3%
   with lr 1e-2 (wd 3), training is unstable: 1 of 3 seeds groks at step 234. The other two
   never reach 99% train within 2500 steps (one stalls at 91%), and frac 0.75 stalls at 68%
   train. The same 3% with lr 3e-2 (wd 1) is healthier. At 6 to 9% nothing groks (train
   accuracy 30 to 98% at the cap). At 0 the model memorizes and stays at 53% test.
3. **AdamW is required in practice.** Matched-decay SGD and GD collapse to chance: final
   loss 3.42 to 3.43, which is ln 31, the uniform output. Small early gradients lose to
   decay. Weak-decay SGD/GD at lr 0.3 to 3 diverges. SGD with lr
   below 0.1 and weak decay is untested.
4. **No LayerNorm and ReLU are fine.** They are Nanda's own choices and every working
   config uses them. Adding LayerNorm plus GELU at the one setting tried made it fail; GELU
   alone grokked but was less stable. Tested at one setting and one seed only.
5. **4 heads, not 1.** One head never grokked in 6 runs. Width 32 MLP works with 4 heads.
6. **Minibatch is untuned, not ruled out.** At the full-batch optimum every batch size
   failed, with the decay overwhelming noisy gradients. Batch 32 collapsed to ln 31; batch
   64 to 128 reached 15 to 36% train. A batch of 32 would cut the Desmos step to 0.23 to
   0.33 s (6 to 8x cheaper), so it is the largest remaining cost lever. It needs its own
   lr/wd sweep.

## Mechanism visibility

`mechanism.py` logs the embedding's Fourier spectrum at every 10 steps. It takes the
frequencies that hold 90% of the final non-constant norm ("key frequencies") and tracks
their share over training. It also projects the embedding onto the top key frequency's
cos/sin plane and scores the circle by phase coherence (1.0 means token t sits at angle
2*pi*k*t/p up to rotation or reflection; random is about 0.6 to 0.8 at p = 31) and by
radius spread (std/mean of radii).

| run | key freqs | key share init -> final (random) | key share half-formed at | test acc then | circle (coherence) | train 99% | test 99% |
|---|---|---|---|---|---|---|---|
| attn p31 lr3e-2 wd1 frac 0.5 seed 0 (delayed) | 11, 14, 1 | 0.20 -> 0.91 (0.19) | step 480 (25% at 160) | 51% (42% plateau) | 0.71 at step 270, 0.98 at 550, 0.998 at 1100 | 60 | 1100 |
| attn p31 lr1e-2 wd1 frac 0.6 seed 1 | 15, 4, 10, 1 | 0.29 -> 0.90 (0.25) | step 60 | 51% | 0.98 at step 230, 0.997 at 910 | 70 | 910 |
| attn p47 lr1e-2 wd1 frac 0.6 seed 0 | 12, 21, 23 | 0.13 -> 0.93 (0.13) | step 70 | 64% | 0.98 at step 80, 0.997 at 160 | 80 | 160 |
| MLP concat p31 m128 frac 0.75 seed 0 | 12, 15, 9, 14, 13, 11 | 0.45 -> 0.90 (0.38) | step 90 | **7%** | 0.98 at step 90, 0.996 at 190 | 70 | 190 |

Across all runs:

| attention runs | n | top-3 share of non-constant norm | frequencies for 90% | circle coherence | final test |
|---|---|---|---|---|---|
| grokked (99% test reached) | 37 | median 0.92 (0.68 to 0.99) | median 3 (2 to 8) | median 0.998 (min 0.986) | median 0.999 |
| memorized, not grokked at the cap | 19 | median 0.49 (0.28 to 0.97) | median 12 | median 0.92 (min 0.76) | median 0.40 |

Sparsity tracks generalization, but as a continuous progress measure, not a yes/no test.
The not-grokked runs with sparse embeddings (top-3 share 0.83 to 0.97) were caught
mid-transition, with test accuracy at 0.86 to 0.99.

The "watch it form" moment:

- **Fast regime (frac >= 0.6):** the spectrum concentrates while accuracy is still rising,
  and it leads accuracy. In the MLP run the key frequencies are half-formed at step 90, when
  test accuracy is still 7%; 99% comes at 190. In attention runs the circle is clean
  (coherence >= 0.98) by step 80 to 230. This is a good visual, but it happens within a few
  hundred ticks of the start and has no plateau.
- **Delayed regime (frac 0.5, seed 0):** after memorizing (step 60), test sits at ~42% for
  300 steps. During the plateau the spectrum starts picking its frequencies: at step 270 bars
  1 and 14 are already up, but the projection is still a tangle (coherence 0.71). At step
  ~370 there is a loss spike: train accuracy briefly drops to 51%. This resembles the
  "slingshot" effect reported for Adam (Thilak et al. 2022; citation from memory, not
  checked). Then the circle snaps in (0.98 at step 550) while test climbs, and the spectrum
  keeps cleaning up after test reaches 99% (90% of its final concentration at step 950).
  Qualitatively this is Nanda's memorization, circuit formation, cleanup sequence, compressed
  into ~1000 steps. In Desmos it would play out over ~30 minutes.

PNG strips (embedding circles at 6 checkpoints with the spectrum beneath, plus accuracy vs
key-frequency share):

- `results/circles_attn_p31_lr3e-2_wd1_f0.5_delayed.png`, `results/curves_attn_p31_lr3e-2_wd1_f0.5_delayed.png` (recommended config)
- `results/circles_attn_p31_lr1e-2_wd1_f0.6_seed1.png`, `results/curves_attn_p31_lr1e-2_wd1_f0.6_seed1.png`
- `results/circles_attn_p47_lr1e-2_wd1_f0.6.png`, `results/curves_attn_p47_lr1e-2_wd1_f0.6.png`
- `results/circles_mlp_concat_p31_m128_lr1e-2_wd3_f0.75.png`, `results/curves_mlp_concat_p31_m128_lr1e-2_wd3_f0.75.png`

## Desmos cost per step

`cost_model.py` writes out one full-batch training step the way it would be hand-built in
Desmos, with the real shapes. It prices the step at 1.3 ms per 32x32 matmul tile (measured),
1 us per elementwise comprehension element (measured), and an assumed 0.041 ms per native
32x32 add or scale (unmeasured: 1/32 of a matmul, the same per-scalar cost). It then adds
50 to 150 ms of tick overhead. The full itemized tables (61 rows for attention) are in
`results/cost_*.md`.

Two design choices make the Desmos step cheaper than a generic transformer step:

- **Constant query.** The `=` token and its position never change, so q is one vector.
  Attention scores are then two per-token lookup tables, `s0 = (E + P0) G` and
  `s1 = (E + P1) G` with `G = W_K q` (d x heads), so there is no per-example Q K^T at all.
  Values are likewise per-token tables `V0`, `V1` (p x d).
- **Gather and scatter as matmuls.** Per-example rows come from constant one-hot blocks:
  `Va = OH_a V0` is (N x p)(p x d). Gradients flow back through `OH_a^T`. Both are
  ordinary tile matmuls, and the one-hot blocks are data loaded once.

**Best attention config (recommended): p = 31, N = 480, d 32, 4 heads, MLP 128, AdamW**

| part | 32x32 matmul tiles | native adds | elementwise elements | ms |
|---|---:|---:|---:|---:|
| forward + loss gradient | 280 | 152 | 163,808 | 534 |
| backward | 543 | 369 | 165,248 | 886 |
| AdamW update (14,560 params) | 0 | 160 | 43,680 | 50 |
| **total** | **823** | **681** | **372,736** | **1,470** |

Arithmetic: 823 x 1.3 ms = 1,070 ms; 681 x 0.041 ms = 28 ms; 372,736 x 1 us = 373 ms; sum
1,470 ms = **1,131 matmul-equivalents** (823 + 681/32 + 373/1.3). Per tick with overhead:
1,520 to 1,620 ms. Steps to 99% test: 1,151 for seed 0 (the delayed one), so
1,151 x 1.52 to 1.62 s = **0.49 to 0.52 h**. The other seeds: 195 to 804 steps, 5 to
22 minutes.

The largest single items are the MLP matmuls: `r W_in`, `h W_out` and their 4 backward
matmuls, 60 tiles each at N = 480, so 360 tiles in all. Next come the one-hot gathers and
scatters (about 120 tiles), then the elementwise ReLU and its mask (2 x 61,440 elements).

**Best MLP-only config: p = 31, N = 721, concat embeddings, MLP 128, AdamW**

| part | tiles | adds | elements | ms |
|---|---:|---:|---:|---:|
| forward + loss gradient | 307 | 253 | 159,341 | 569 |
| backward | 476 | 359 | 92,288 | 726 |
| AdamW update (13,280 params) | 0 | 136 | 39,840 | 45 |
| **total** | **783** | **748** | **291,469** | **1,340** |

1,340 ms = **1,031 matmul-equivalents**; 1,390 to 1,490 ms per tick; worst seed 255 steps,
so **0.10 to 0.11 h (6 to 7 minutes)**. Per example the MLP model is about 1.6x cheaper
than attention, but here it runs at a higher train fraction, so per step it is only 9% cheaper
than the recommended attention config. Its main advantage is needing fewer steps.

Other configs (compute per step, matmul-equivalents): attention p31 frac 0.6 m128: 1,818 ms
(1,399). Attention p31 frac 0.6 m32: 1,154 ms (887). Attention p23 frac 0.75: 1,262 ms
(971). Attention p47 frac 0.6: 4,711 ms (3,624). Attention p113 frac 0.3: about 18 s.

Sensitivity:
- If native matrix adds cost as much as a matmul instead of 1/32 of one, the recommended
  step grows from 1.47 s to about 2.33 s (+58%).
- If elementwise work can run as native list arithmetic instead of comprehensions (the biped
  session found list math nearly free), the step drops by up to 25%.
- Partial tiles (p = 31 rows, 4-column head tables) are charged as full tiles, which is
  conservative.
- The ticker will run at about 0.6 ticks/s, not 2 to 6: per-step compute exceeds 1 s. So
  "10,000 steps is an hour" does not hold for full batch; 1,000 steps is about 25 to 33
  minutes.

## Risks and unknowns for the Desmos build

1. **10,000-element list cap.** Elementwise passes over N x d_mlp (480 x 128 = 61,440
   elements) exceed 10,000 if built as a flat number list. They must be produced as lists of
   32x32 matrices (if each matrix counts as one element) or split across expressions.
   Measure this first; it decides how many expressions the step needs.
2. **Trajectories are sensitive to rounding.** The two trainers here agree to 1e-9 over 40
   steps (tested) and agree on the grok step for typical seeds (904 vs 910, 156 vs 160, 184
   vs 190, 1151 vs 1100). For one slow seed (p31 lr1e-2 wd1 seed 0) one trainer grokked at
   2135 and the other had not by 3000. Desmos will sum in a different order, so treat step
   counts as a distribution over seeds, not exact predictions. To make the run
   reproducible, export the initial weights and data split from Python, then compare the
   first ~50 Desmos steps to numpy as the biped session did.
3. **Seed choice matters for the visual.** Only 1 of 4 seeds at the recommended setting has
   the long plateau. Load seed 0's init rather than drawing fresh random weights in Desmos.
4. **Native add cost is unmeasured** (see sensitivity above).
5. **Loss spikes after grokking at lr 3e-2.** In 5 of the 18 grokked runs at lr 3e-2, test
   accuracy dropped to 12 to 34% within 100 steps of first reaching 99%. Three had recovered
   by the end of that window. The recommended seed 0 held 99% for the 300 steps traced after
   its grok. For a live demo, cut the lr (for example 10x) or freeze training once test
   accuracy reaches 99%. This costs one comparison per tick.
6. **Minibatch and SGD are untuned, not disproven** (see results point 6 and point 3).

## Compute used and how the study ran

About 90 minutes of wall time on at most 3 threads, which is over the ~60 minute target. The
laptop was shared with another session's 12 to 16 training jobs (load average 10 to 25), so
early single runs were 5 to 7x slower than when idle. To cope, `ensemble.py` trains many
same-shape configs as one stacked model: functional AdamW/SGD with per-member lr and decay,
and a masked loss over all p^2 pairs, so test accuracy comes from the same forward pass and
is checked every step. It reproduces `grok.train_run` to 1e-9 for every architecture,
optimizer and batch mode (`test_grok.py`). Threads: the parent sets
`torch.set_num_threads(3)` and `OMP_NUM_THREADS=3`. It then spawns at most 3 workers, each
inheriting `OMP_NUM_THREADS=1` and nice >= 6, below the browser at nice 0. `attn_search` was
stopped after 22 of its 54 configs (its 5000-step cap made ensembles wait on slow members).
The rest was covered by `attn_focus` and later stages at a 2500-step cap. Re-running
`python3 sweep.py attn_search` resumes it.

## Files

| file | what |
|---|---|
| `grok.py` | data, models (attention, MLP concat/sum/folded), single-run trainer with early stop, Fourier and circle metrics |
| `ensemble.py` | stacked multi-config trainer (same trajectories, far less Python overhead) |
| `sweep.py` | named stages and `--plan` files, 3-worker pool, resumable JSONL results |
| `mechanism.py` | dense-logged run, key-frequency timing, circle checkpoints, PNG strips |
| `cost_model.py` | itemized Desmos step cost from real shapes |
| `summarize.py` | pooled tables with Desmos time, writes `results/summary.md` |
| `test_grok.py` | 13 tests: ensemble = single run, Fourier/circle metrics, data split, cost arithmetic |
| `plan_*.json` | the exact sweeps that produced `results/*.jsonl` |
| `results/` | raw JSONL per stage, logs, mechanism JSON, PNGs, cost reports, `summary.md` |

Reproduce (from this folder):

```
python3 -m pytest -q test_grok.py
python3 sweep.py attn_search
python3 sweep.py --plan plan_focus.json      # then plan_seeds, plan_extra, plan_final,
                                             # plan_grok_delay, plan_large_p
python3 summarize.py
python3 mechanism.py --name attn_p31_lr3e-2_wd1_f0.5_delayed --extra-steps 300 \
  --base '{"p": 31, "arch": "attn", "lr": 0.03, "weight_decay": 1.0, "train_frac": 0.5, "max_steps": 3000}'
python3 cost_model.py --arch attn --p 31 --n-train 480 --d-mlp 128 --heads 4 --opt adamw --steps 1151
```
