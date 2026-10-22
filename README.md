# A language model inside Desmos

A character-level transformer that runs entirely inside the Desmos graphing calculator. You type a prompt on a keyboard drawn in the graph, press go, and it writes one character per tick. Every multiply, softmax and normalization is a Desmos expression; Python only trains the model offline, writes the expressions, and checks the numbers.

Status: works (2026-10-22). Not yet saved as a public graph.

## Where it stands

| Step | Result |
|---|---|
| Model | 2 layers, d_model 64, 2 heads of 32, MLP 256, 32-character window, RMSNorm, GELU, learned positions, 105k parameters, 32-symbol vocabulary (lowercase, space, `. , ; ' "`). Trained offline on Samuel Butler's prose Odyssey and Iliad: 1.76 bits per character on held-out text. See [`experiments/charlm/`](experiments/charlm/). |
| Exact in Desmos | For a fixed 32-character input, all 1,024 logits from the Desmos graph match the float64 numpy reference to 2.3e-14 (largest logit 14.7). During generation, after the window had slid past 32 characters, the next-character logits matched numpy to 3.3e-15. |
| Generates text | From the prompt "then minerva said, " the graph wrote "so no cover already; then and out of one". About 0.5 characters per second on a 15 W laptop under heavy load from other jobs; faster on an idle machine (not yet measured). |
| How it fits | Desmos matrices stop at 32x32, so every weight is cut into 32x32 blocks and every activation is a list of blocks. Each attention head is exactly one 32-column block. Lists of matrices are never rendered in the expression list, so the forward pass is not slowed by display. |

Evidence (screenshots, measured numbers) is in [`evidence/`](evidence/), indexed in [`evidence/LOG.md`](evidence/LOG.md).

## Try it

Requirements: Python 3 with numpy.

```bash
python3 gen_desmos.py out/lm.json --prompt "then minerva said, "
```

Open https://www.desmos.com/calculator, open the browser console, and run `Calc.setState(` followed by the contents of `out/lm.json` and `)`. Collapse the expression list (it makes ticks much cheaper), then click "go / stop". The keyboard keys are clickable: they append to the text, and their shading is the model's probability for each next character. The temperature `\tau_s` and seed `s_d` sliders are in the expression list.

To check the numbers yourself: `python3 gen_desmos.py out/lm_ref.json` builds the graph for the reference input in `experiments/charlm/best/reference.npz`; compare the graph's `Y` (flatten with `[Y[1][a;b] for a=[1...32],b=[1...32]]`, column-major) with the reference `logits`.

## Layout

- `gen_desmos.py`: weights to a Desmos graph state (model, sampler, text display, keyboard). `test_gen_desmos.py` pins its block layout.
- `experiments/charlm/`: training, the size-versus-quality study, the numpy reference forward pass, exported weights.
- `experiments/attention-bench/`, `experiments/ticker-bench/`: the engine measurements that shaped the design.

## Prior work

Transformers in odd substrates (Excel, SQL, a font, Scratch, a Casio calculator as native code) are an established genre, and a single random-weight attention block already exists in Desmos (https://www.desmos.com/calculator/0kqb05bq4i). We found no earlier language model, of any kind, running inside Desmos; that search was moderate in strength (reddit only partly reachable, X and Discord not at all).

## License

MIT, see [LICENSE](../LICENSE). Author: Shifat Islam Santo ([@oneKn8](https://github.com/oneKn8)).
