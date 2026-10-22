# Training a transformer inside Desmos (work in progress)

Goal: train the one-layer transformer from [`../grokking/`](../grokking/) on (a + b) mod 31 entirely inside Desmos (forward pass, hand-written backward pass, AdamW, one step per ticker tick) and watch it grok. In numpy this run memorizes the training pairs by step 53, sits at 42% test accuracy, then climbs to 99% by step 1140 while the number-token embeddings arrange themselves on a circle.

## Status (2026-10-22)

| Piece | State |
|---|---|
| `ref_step.py`: the training step in the exact structured form the graph uses | Verified. Matches torch autograd on grok.py's model: logits to 1e-12, every gradient to 1e-10 relative, parameters after 20 AdamW steps to 1e-9 (`test_ref_step.py`, 4 tests). |
| `run_ref.py`: full training run with that step | Verified. Reproduces the grok: test accuracy 42% from step 100 to 400, 99% at step 1140, key-frequency share of the embedding 0.26 to 0.90. 1,400 steps in 21 s. |
| `gen_grok.py`: the Desmos training graph (any MLP width; views: pair grid, Fourier circle and spectrum, accuracy history; go / stop) | VERIFIED for MLP width 32 (2026-10-22): loads in about 3 s; at initialization the right/wrong grid matches numpy on all 961 pairs and all 11 gradient tensors match to 1.3e-15 relative; after one AdamW step the weights match to 3.3e-16; after 20 real ticker steps every weight matches the numpy run to 5.8e-15. The fix that made it usable: every comprehension that reads named lists runs inside its own function with those lists passed in whole (see Design). |
| `check_data.py`: reference loss, right/wrong grid, gradients, and weights after 1, 5 and 20 steps, flattened like the graph's lists | Written, used for the comparison above once the graph is fast enough to run. |

## Chosen run (2026-10-22)

MLP width 32 instead of 128, train fraction 0.5, seed 0, AdamW lr 3e-2, weight decay 1. In numpy it memorizes by step 70, holds 42% held-out accuracy for 270 steps, reaches 50% at step 340 and 99% at step 590: a real grokking delay in half the steps of the width-128 run, with about 8.2k parameters (24.6k numbers of state with the AdamW moments). Seeds 1 to 3 at fraction 0.5 grok in 210 to 360 steps with short or no plateau; at fraction 0.6 there is no plateau. `python3 run_ref.py runs/m32_f0.5_s0.npz --mlp 32 --frac 0.5 --seed 0 --steps 2500`.

## Design

- Position 3 is always "=", so the query and the third key and value are single rows; positions 1 and 2 only see one token each, so their keys and values are 31-row token tables gathered per pair.
- All 31 x 31 pairs live as a list of 31 matrices (block a holds the pairs (a, b) in row b), which is also the picture: a grid of right and wrong cells.
- Desmos re-derives a named list every time a comprehension in another expression indexes it (`L[a]` inside `[... for a]`), which compounded into minutes per step through the backward pass (a chain of lists that each read all earlier ones doubled in cost per level). Lists passed whole into a function are computed once, so `gen_grok.py` rewrites every such expression as a call to its own function (`materialize`). Before: first evaluation over 4 minutes; after: about 3 s.
- Desmos actions cannot write matrices larger than 9x9, so every parameter and both AdamW moments are flat lists (about 43.7k numbers) rewritten each tick and reshaped into matrices each step.

## Reproduce

```bash
python3 -m pytest -q test_ref_step.py                # needs torch, numpy
python3 run_ref.py runs/seed0.npz --steps 1400       # the numpy grokking run
python3 gen_grok.py runs/seed0.npz out/grok_train.json
python3 check_data.py runs/seed0.npz out/grok_check.json
```
