# Ticker benchmark

Anything that changes over time in Desmos (a generated token, a training step) is written back by an action on the ticker. This measures how fast the ticker can run as the list it rewrites grows. It decides whether training inside Desmos is practical.

## Result

Ticker at minimum step 0, action `c_t -> c_t + 1, l -> 1.0001 l`, 5 s per size, Chrome window in front (about 60 frames/s), 12-core laptop with other jobs running (load average about 9).

| Numbers rewritten per tick | Expression list open | Expression list collapsed |
|---|---|---|
| 32 | 12.2 ticks/s | |
| 1,024 | 0.8 ticks/s | 15.8 ticks/s |
| 4,096 | 0.2 ticks/s | 5.6 ticks/s |
| 10,000 | 0.1 ticks/s | 2.4 ticks/s |

- With the expression list open, every tick re-renders the rewritten list's literal: about 1 ms per number.
- Collapsing the expression list (`Calc.updateSettings({ expressionsCollapsed: true })`) cuts that to about 40 microseconds per number plus about 25 ms per tick. Putting the list in a collapsed folder does not help (0.7 ticks/s at 1,024).
- A bare counter ticks 23 to 27 times per second.
- An action whose target is a computed definition, such as `l=0.001[1...n]`, never fires and shows no error. Targets must be plain literals or ranges.

So a model whose trainable state is about 10,000 numbers can take roughly 2 optimizer steps per second before counting the math, and the state write, not the matrix math, is the main cost per step.

## Reproduce

Open https://www.desmos.com/calculator, paste `tickbench.js` into the console, bring the window to the front, then `await tbRun()` (collapsed) and `await tbRun([32, 1024, 4096, 10000], { collapsed: false })`. The table came from this logic run inline in the console; the packaged file, rerun on a fresh page, gave 16.4 ticks/s at 1,024 and 4.8 ticks/s at 4,096 with the list collapsed.
