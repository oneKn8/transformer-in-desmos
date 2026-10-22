# Training a transformer inside Desmos (work in progress)

Goal: train the one-layer transformer from [`../grokking/`](../grokking/) on (a + b) mod 31 entirely inside Desmos (forward pass, hand-written backward pass, AdamW, one step per ticker tick) and watch it grok. In numpy this run memorizes the training pairs by step 53, sits at 42% test accuracy, then climbs to 99% by step 1140 while the number-token embeddings arrange themselves on a circle.

## Status (2026-10-22)

| Piece | State |
|---|---|
| `ref_step.py`: the training step in the exact structured form the graph uses | Verified. Matches torch autograd on grok.py's model: logits to 1e-12, every gradient to 1e-10 relative, parameters after 20 AdamW steps to 1e-9 (`test_ref_step.py`, 4 tests). |
| `run_ref.py`: full training run with that step | Verified. Reproduces the grok: test accuracy 42% from step 100 to 400, 99% at step 1140, key-frequency share of the embedding 0.26 to 0.90. 1,400 steps in 21 s. |
| `gen_grok.py`: the Desmos training graph | Built, loads with no errors. Not yet checked against `check_data.py`'s numbers, and too slow: the forward pass alone takes 2 to 4.5 s per update at load average about 13, while the same elementwise operations on synthetic data take under 0.1 s. Under investigation. |
| `check_data.py`: reference loss, right/wrong grid, gradients, and weights after 1, 5 and 20 steps, flattened like the graph's lists | Written, used for the comparison above once the graph is fast enough to run. |

## Design

- Position 3 is always "=", so the query and the third key and value are single rows; positions 1 and 2 only see one token each, so their keys and values are 31-row token tables gathered per pair.
- All 31 x 31 pairs live as a list of 31 matrices (block a holds the pairs (a, b) in row b), which is also the picture: a grid of right and wrong cells.
- Desmos actions cannot write matrices larger than 9x9, so every parameter and both AdamW moments are flat lists (about 43.7k numbers) rewritten each tick and reshaped into matrices each step.

## Reproduce

```bash
python3 -m pytest -q test_ref_step.py                # needs torch, numpy
python3 run_ref.py runs/seed0.npz --steps 1400       # the numpy grokking run
python3 gen_grok.py runs/seed0.npz out/grok_train.json
python3 check_data.py runs/seed0.npz out/grok_check.json
```
