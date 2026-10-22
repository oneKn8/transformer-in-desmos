# Attention-head benchmark

The first check before building a language model in Desmos: can a full causal attention head run on Desmos's native matrices, is it exact, and how fast is it?

## Result

| Question | Answer |
|---|---|
| Does a causal attention head (T=32, d=32) evaluate? | Yes, with no errors, in both layouts below. |
| Is it exact? | All 1024 outputs match a float64 numpy reference to 1.62e-15 max abs error (2.0e-13 relative), which is float64 rounding. |
| How fast? | 24 ms per forward pass end to end when fused into hidden functions with the expression list collapsed; about 11 ms of that is compute (40 heads in one update took 466 ms). |
| Largest matrix? | 32 x 32. Desmos refuses more than 32 rows or 32 columns. |

## Prior work

This benchmark is a feasibility check, not a first. A causal attention block with random weights already exists in Desmos ("Masked Attention", https://www.desmos.com/calculator/0kqb05bq4i, February 2026), and the community graph "All Desmos Features You Should Know" (https://www.desmos.com/calculator/5ls5ggt5gd) documents the 32x32 matrix cap, lists of matrices used as 3D arrays, and the 9x9 limit on matrices written by actions. What this folder adds is the exactness check against numpy, the timing, and the scale test (a list of 1,000 32x32 matrices with batched matmul).

## What we learned about Desmos matrices (September 2026 release)

- **Size cap 32 x 32.** Error text: "Matrices may not have more than 32 columns." (or rows).
- **Lists of matrices work.** `H = kQ for k=[1...1000]` is a list of 1000 matrices; `H[3]` is a matrix and `H[3][1;2]` a number. Matrix multiply and addition broadcast over the list (`HW` multiplies every element by `W`). A list of 1000 matrices of 32x32 (about 1M numbers in one expression) evaluated fine, so the 10,000-element list cap counts matrices, not numbers. This is the way past both the 32x32 cap and the "no nested lists" rule: one 32x32 block per attention head or per weight tile.
- **No elementwise functions.** `exp`, `tanh`, `sin`, `max`, `total`, `mean`, and even matrix plus a number all error on a matrix. Nonlinearities go through an index comprehension, `exp(S[a;b]) for a=[1...32];b=[1...32]`, at about 1 microsecond per element. A function containing such a comprehension does not broadcast over a list of matrices; map it explicitly (`h(L[k]) for k=[1...N]`).
- **Indexing.** `M[a;b]` element, `M[2;]` row, `M[;2]` column, `M[[3,1,1];]` gathers rows with repeats (a token-embedding lookup in one operation). `M^{T}` transposes, `M^{2}` is a matrix power, `M/4` and `M*4` scale.
- **Building and flattening.** `f(a,b) for a=[...];b=[...]` (semicolon) builds a matrix. `[M[a;b] for a=[...],b=[...]]` (comma) flattens to a list with `a` varying fastest, i.e. column-major. `L[32(a-1)+b] for a=[...];b=[...]` reshapes a list into a matrix. A literal is `\begin{bmatrix}1&2\\3&4\end{bmatrix}`.
- **Parser trap.** A comprehension passed directly as a function argument is misparsed (errors such as "Points are written like this" or "Cannot combine a matrix and a number"). Wrap each comprehension in its own small function and pass that.
- **Speed.** One 32x32 by 32x32 multiply costs about 1.3 ms (about 25 M multiply-adds per second). The big cost is display: every matrix shown in the expression list is re-rendered on every update, about 100 ms or more for a few 32x32 matrices, even unchanged ones. Hide intermediates inside functions, keep matrices in lists (lists of matrices are never rendered), and collapse the expression list.
- **Loading.** `Calc.setExpressions` silently drops `colorLatex` (per-item colors); `Calc.setState` keeps it. In a state file, slider ranges go under `slider`, not `sliderBounds`.
- **Actions cannot write big matrices.** An action may update a matrix of at most 9x9 ("Cannot update a variable to a matrix larger than 9×9."), and cannot update a list of matrices. Flat number lists can be written (tested up to 10,000 numbers). Anything that has to change over time, such as weights being trained, lives in flat lists and is reshaped into 32x32 matrices when used.
- **Desmos differentiates through matrix math.** `F(w)=(wNN)[1;1]` gives the correct `F'(2)`, and so does a 2x2 matrix product with `w` inside it. Hide such functions: a visible one-variable function with matrices inside gets plotted point by point and froze the evaluator for about 30 s in our test.
- **Weights load as literals.** A list of 64 literal 32x32 matrices (65,536 numbers, 627 KB of LaTeX) in one expression parsed and evaluated in about 2 s, so a model of that size fits in a single weight expression.
- **More indexing.** `M[;[1...8]]` takes columns 1 to 8 (one attention head's slice). Block matrices built from slices work while the result stays within 32x32.

## Files

- `build_head.py`: writes `out/head32_readable.json` (every step its own expression, plus a heatmap of the attention pattern), `out/head32_fused.json` (the same head as a chain of hidden functions; the fast layout), and `out/head32_ref.json` (numpy reference). Weights are closed-form sines of the row and column index, so nothing has to be shipped and both sides compute the same numbers.
- `harness.js`: paste into the browser console on desmos.com to check a loaded graph against the reference (`lmCheck`) and time it (`lmTime`).

## Reproduce

```bash
python3 build_head.py out          # needs numpy
```

Then on https://www.desmos.com/calculator, in the browser console:

```js
Calc.setState(/* contents of out/head32_readable.json */)
// paste harness.js, then:
const REF = /* contents of out/head32_ref.json */;
await lmCheck(REF, 'y_{flat}')     // -> { n: 1024, maxAbsErr: ~1.6e-15, relErr: ~2e-13 }

Calc.setState(/* contents of out/head32_fused.json */)
await lmCheck(REF, 'y_{f}')
Calc.removeExpression({ id: 'yf' }) // the 1024-number display list costs time too
Calc.updateSettings({ expressionsCollapsed: true })
await lmTime('c', 9)               // median ms per forward pass
```

Timings depend on the machine and on other load; ours were taken on a 12-core laptop with the Desmos window in front.
