# Text generation speed

How fast the phase-1 graph (`../../gen_desmos.py`) writes text, before and after the `materialize` pass in `../../desmos_passes.py`.

## Result (2026-10-22)

The fix is the one found while making the training graph usable (see `../grok-desmos/`): Desmos re-derives a named list each time a comprehension in another expression indexes it. In the text generator seven expressions do that, among them the MLP up-projection `H = [M[1] W_u[2j-1] + M[2] W_u[2j] for j=[1...8]]` and the GELU over its blocks, once per layer. Rewriting each as a call to its own function, with the lists passed in whole, makes generation about 4x faster and changes no number: the logits are bit-identical to the unfixed graph and match the float64 numpy reference to 1.5e-14.

Chrome, tab in front at 60 frames/s, 15 W laptop (i7-1255U), CPU otherwise quiet, reference prompt "tell me, o muse, of that ingenio". Runs in the order shown, on one freshly loaded tab:

| Graph | First evaluation | Forward pass, median of 5 | Generation |
|---|---|---|---|
| before | 1230 ms | 830 ms | 1.20 chars/s |
| after | 502 ms | 165 ms | 4.66 chars/s |
| after, no text display and no keyboard | 425 ms | 148 ms | 6.47 chars/s |
| after, no keyboard | 356 ms | 164 ms | 4.73 chars/s |
| after | 501 ms | 190 ms | 4.80 chars/s |
| before | 1031 ms | 739 ms | 1.33 chars/s |

A forward pass is timed from appending a character to `T_x` until the logits `y_l` change. Generation counts characters written by the ticker in 15 s. Comparing the last two "after" variants, the text display (two point-label expressions per symbol) costs about 55 ms of the remaining time per character; the keyboard costs nothing measurable.

Under load (another session's training jobs on 4 of the 12 threads) the same comparison gave 1.5 to 1.8 s before and 0.26 to 0.28 s after per forward pass, 0.53 to 0.67 chars/s before and 2.2 to 2.6 after.

The state's `randomSeed` is now pinned (`DEFAULT_RANDOM_SEED` in `gen_desmos.py`). Desmos seeds `random()` from it and invents a new one on every load when it is missing; with it pinned, two fresh loads of the demo graph wrote the same 30 characters.

## Reproduce

```bash
python3 ../../gen_desmos.py ../../out/lm_ref.json
```

On https://www.desmos.com/calculator, with the window in front: in the console define `const STATE = ` followed by the contents of `lm_ref.json`, paste `genbench.js`, then `await gbRun(STATE)`. It returns the first-evaluation time, the forward-pass latencies, the generation rate and the logits at load.
