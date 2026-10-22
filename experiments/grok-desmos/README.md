# Training a transformer inside Desmos

Goal: train the one-layer transformer from [`../grokking/`](../grokking/) on (a + b) mod 31 entirely inside Desmos (forward pass, hand-written backward pass, AdamW, one step per ticker tick) and watch it grok. In numpy this run memorizes the training pairs by step 53, sits at 42% test accuracy, then climbs to 99% by step 1140 while the number-token embeddings arrange themselves on a circle.

## Result (2026-10-22)

A one-layer transformer trained from scratch inside Desmos grokked (a + b) mod 31: it memorized the training pairs by step 62, sat on a 42% held-out plateau while the Fourier structure of its embeddings kept growing, went through a loss spike at about step 307, formed the Fourier circle by step 383 and reached 99.17% held-out accuracy at step 587 (the numpy reference: 99% at step 590). About 0.5 steps per second in Chrome on a 15 W laptop. Screenshots: `../../evidence/2026-10-22_grok-in-desmos_*.jpg`.

Exactness against the numpy reference, which itself matches torch autograd: one step agrees to 1e-15, 20 steps to 5.8e-15, and after 601 steps the weights differ by up to 5.6e-3 (0.86% relative). Float64 rounding differences between Desmos and numpy roughly double every 14 training steps, so a long run is the same run macroscopically (held-out accuracy at step 590 identical, 59 of 61 recorded accuracy checkpoints identical) but not bit for bit.

## The learned algorithm as a formula, fitted live (2026-10-22)

The graph also reads the algorithm back out of the network while training it. Every step it fits

    logit(a, b, c) ~ const(a, b) + alpha_1 cos(2 pi k_1 (a + b - c) / 31) + alpha_2 cos(...k_2...) + alpha_3 cos(...k_3...)

to all 961 pairs, with k_1..k_3 the three strongest Fourier frequencies of the number-token embeddings, and reports R^2: the share of the logits (centered per pair) that this one formula explains. The panel under the grid draws one pair's 31 logits as dots over the fitted wave; click any grid cell to switch pairs. The cosine of a + b - c splits into products of cosines and sines of a, b and c, so each alpha needs one 6x31 by 31x31 by 31x6 product per block of pairs, cheap next to the training step. Nanda et al. (2023) measure the same structure offline (restricted and excluded loss on the key frequencies); what is new here is the formula fitted and plotted inside the graph that is training the network, every step.

Predictions written down before the in-graph run (numpy expectations in `../../WORK_QUEUE.md`), and the result:

| | Prediction | Desmos | numpy |
|---|---|---|---|
| P4 | R^2 at step 270 at least 2x R^2 at step 90, while held-out accuracy moves less than 5 points | 0.0759 to 0.1944 (2.6x); held-out 39.5% to 42.6% | the same to 2e-12 |
| P5 | R^2 at least 0.85 when held-out accuracy first reaches 99% | 0.875 at step 586 | 0.877 at step 590 |
| P6 | the three frequencies are the final ones from step 60 on | {7, 12, 1} by step 56 | from step 60 |

So the formula's share of the network tripled during the plateau while accuracy did not move, a progress measure you can watch in the calculator. It jumps fastest through the loss spike at about step 300 (0.19 at 270, 0.29 at 300, 0.39 at 310, while held-out accuracy dips to 31%), and held-out accuracy climbs after it. Screenshots: `../../evidence/2026-10-22_formula-in-desmos_*.jpg` (steps 140, 269, 389, 591).

Against numpy, R^2 agrees to 6e-16 at step 90, 2e-12 at 270, 3e-8 at 390 and 5e-5 at 590 (rounding growth again), and held-out accuracy is identical at 61 of the 65 recorded steps up to 640; at the other four (500, 510, 530, 580) the runs disagree on exactly one of the 481 held-out pairs. A second instability at step 640 hits both runs identically (train accuracy 60%, held-out 51.8%); after it the two runs separate at the 1e-2 level. Past grokking the formula's share drifts down (0.88 at step 630, 0.71 at 800 in both) while held-out accuracy stays at 96 to 99%: the three-frequency cosine is the core of the solution, not all of it.

## Status (2026-10-22)

| Piece | State |
|---|---|
| `ref_step.py`: the training step in the exact structured form the graph uses | Verified. Matches torch autograd on grok.py's model: logits to 1e-12, every gradient to 1e-10 relative, parameters after 20 AdamW steps to 1e-9 (`test_ref_step.py`, 4 tests). |
| `run_ref.py`: full training run with that step | Verified. Reproduces the grok: test accuracy 42% from step 100 to 400, 99% at step 1140, key-frequency share of the embedding 0.26 to 0.90. 1,400 steps in 21 s. Also records the formula fit (`formula_fit`: key frequencies, alphas, R^2) every 10 steps. |
| `gen_grok.py`: the Desmos training graph (any MLP width; views: pair grid, Fourier circle and spectrum, accuracy history, the formula panel; go / stop) | VERIFIED for MLP width 32 (2026-10-22): loads in about 3 s; at initialization the right/wrong grid matches numpy on all 961 pairs and all 11 gradient tensors match to 1.3e-15 relative; after one AdamW step the weights match to 3.3e-16; after 20 real ticker steps every weight matches the numpy run to 5.8e-15. The fix that made it usable: every comprehension that reads named lists runs inside its own function with those lists passed in whole (see Design). |
| `check_data.py`: reference loss, right/wrong grid, gradients, weights after 1, 5 and 20 steps flattened like the graph's lists, and the formula fit at each of those steps | Used for every comparison above. |

## Chosen run (2026-10-22)

MLP width 32 instead of 128, train fraction 0.5, seed 0, AdamW lr 3e-2, weight decay 1. In numpy it memorizes by step 70, holds 42% held-out accuracy for 270 steps, reaches 50% at step 340 and 99% at step 590: a real grokking delay in half the steps of the width-128 run, with about 8.2k parameters (24.6k numbers of state with the AdamW moments). Seeds 1 to 3 at fraction 0.5 grok in 210 to 360 steps with short or no plateau; at fraction 0.6 there is no plateau. `python3 run_ref.py runs/m32_f0.5_s0.npz --mlp 32 --frac 0.5 --seed 0 --steps 2500`.

## Design

- Position 3 is always "=", so the query and the third key and value are single rows; positions 1 and 2 only see one token each, so their keys and values are 31-row token tables gathered per pair.
- All 31 x 31 pairs live as a list of 31 matrices (block a holds the pairs (a, b) in row b), which is also the picture: a grid of right and wrong cells.
- Desmos re-derives a named list every time a comprehension in another expression indexes it (`L[a]` inside `[... for a]`), which compounded into minutes per step through the backward pass (a chain of lists that each read all earlier ones doubled in cost per level). Lists passed whole into a function are computed once, so `gen_grok.py` rewrites every such expression as a call to its own function (`materialize` in [`../../desmos_passes.py`](../../desmos_passes.py), shared with the text generator). Before: first evaluation over 4 minutes; after: about 3 s.
- Desmos actions cannot write matrices larger than 9x9, so every parameter and both AdamW moments are flat lists (about 43.7k numbers) rewritten each tick and reshaped into matrices each step.

## Reproduce

```bash
python3 -m pytest -q test_ref_step.py test_gen_grok.py                            # needs torch, numpy
python3 run_ref.py runs/m32_f0.5_s0.npz --mlp 32 --frac 0.5 --seed 0 --steps 900  # the numpy run, with the formula R^2 curve
python3 gen_grok.py runs/m32_f0.5_s0.npz out/grok32_train.json                    # the Desmos training graph
python3 check_data.py runs/m32_f0.5_s0.npz out/grok32_check.json                  # numbers to check it against
```

Load `out/grok32_train.json` with `Calc.setState(...)` in the console on https://www.desmos.com/calculator, collapse the expression list, and click "go / stop".
