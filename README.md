# A language model inside Desmos

Goal: a transformer language model that runs entirely inside the Desmos graphing calculator. You type a prompt into the graph and it writes text one token at a time, with every multiply done by Desmos expressions. Python only trains the model offline, generates the expressions, and checks the numbers.

Status: in progress (started 2026-10-21). Nothing here generates text yet.

## Where it stands

| Step | Result |
|---|---|
| Can Desmos do attention at all? | Yes. One causal attention head (32 tokens, 32 dimensions) on Desmos's native matrices matches numpy to float64 rounding (max error 1.6e-15) and runs a forward pass in about 24 ms. See [`experiments/attention-bench/`](experiments/attention-bench/). |
| Engine limits | Matrices stop at 32 x 32, but lists of matrices work and matrix multiply broadcasts over them, so bigger tensors are lists of 32x32 blocks. Details in the benchmark README. |
| Model choice | Being decided from offline experiments. |

Evidence (screenshots, measured numbers) is in [`evidence/`](evidence/), indexed in [`evidence/LOG.md`](evidence/LOG.md).

## License

MIT, see [LICENSE](../LICENSE). Author: Shifat Islam Santo ([@oneKn8](https://github.com/oneKn8)).
