# Char-LM feasibility study for Desmos (offline)

Which character-level transformer can run inside Desmos at under about 300 ms per generated token, and what text quality does each feasible size buy? Everything here was trained offline on CPU (PyTorch 2.11, 3 cores). Desmos was not touched. All numbers come from files in this folder.

## Recommendation: phase-1 model

| | |
|---|---|
| Config | `d64_L2_H2_T32`: d_model 64, 2 layers, 2 heads (head_dim 32, so each head is exactly one 32-column block), context T = 32, MLP 4x (256), pre-norm RMSNorm, GELU (tanh form), learned absolute positions, untied unembedding, no biases, vocab `v32` |
| Trained | 4000 steps x 4096 chars (16.4M chars, about 13 epochs), seed 0, about 18 min on 1 shared core |
| Quality | test **1.764 bits/char** (val 1.779), scored with at least 16 chars of context; 1.871 / 1.885 averaged over all window positions |
| Desmos cost | **about 243 ms/token** (full recompute of the 32-char window, conservative), 241 ms if cost scales with multiply-adds; arithmetic below |
| Size | 104,768 trained params; 104,448 numbers after folding norm gains (102 blocks of 32x32); 3 significant digits lose 0.00001 bpc, about 0.72 MB of positional decimal text to paste |
| Files | `best/` (see [Exported files](#exported-files-and-the-equality-check)) |

Sample, temperature 0.8, computed by the numpy reference over the exact exported weights (seed 7, not cherry-picked):

```
then minerva said, "when appeared sworth of the mid him as he were long he was one in at the cry. between the trojans as if i should any moment. he is goods on the threshold which will lies of ajax, who was fight an exc
```

Two more seeds (1 and 2), same model:

```
then minerva said, "you know that may go lying away to all the sons of atreus, a spear such fiercely servants chamf through thetis head from the before the upper clother, and fighting for you go and there you will give
then minerva said, 'honour house, is hands, unsafely on the room of all this in frees notched to the treams at near a fight, and achilles, and mother, were been to go attacking out and sprang to sleeping to the sheep in
```

Honest read: mostly real English words, Homeric names and phrasing ("the sons of atreus", "on the threshold"), quoted speech, punctuation in plausible places, and no sentence-level meaning. That is what about 1.8 bpc buys.

Why this one: it is the best model on the measured frontier under 300 ms (table below). Its cost has about 55 ms of headroom, and H=2 makes the heads block-aligned, so no per-head slicing is needed in Desmos. The cheapest way to improve it is more offline training, not a bigger model: the same config went from 2.027 to 1.779 val bpc between 1000 and 4000 steps and was still improving when it stopped.

Two follow-ups are cheap if the Desmos measurements favour them (both are one command, see [Reproduce](#reproduce)). First, switch to ReLU, which matched GELU within seed noise and is simpler to write exactly as `(x+|x|)/2`. Second, switch to the case-preserving `v64` vocab, which cost +0.03 bpc and about +3 ms. The exported weights are the GELU, v32 model.

## The tradeoff curve

The frontier is the configs that no cheaper config beats on val bpc. All runs are 1000 steps (4.1M chars), same seed and data. Costs are padded estimates for full recompute.

| config | ms/token | val bpc (ctx >= T/2) | params |
|---|---:|---:|---:|
| d32_L1_H2_T32 | 94 | 2.258 | 15,456 |
| d32_L1_H4_T32 | 105 | 2.249 | 15,456 |
| d32_L2_H2_T32 | 135 | 2.176 | 27,808 |
| d64_L1_H2_T32 | 149 | 2.114 | 55,488 |
| d64_L1_H4_T32 | 169 | 2.108 | 55,488 |
| d32_L4_H2_T32 | 218 | 2.100 | 52,512 |
| **d64_L2_H2_T32** | **243** | **2.027** | 104,768 |

From 94 to 149 ms val bpc falls by 0.144 (about 0.13 per 50 ms). From 149 to 243 ms it falls by a further 0.087 (about 0.05 per 50 ms). Seed noise is about 0.02 bpc (same config, seeds 0 and 1: 2.027 vs 2.006), so neighbouring points closer than that are ties.

What the grid says:

- **Width beats depth at equal Desmos cost.** d64_L1 (149 ms) reaches 2.114, while d32_L2 (135 ms) reaches 2.176 and d32_L4 (218 ms) 2.100. At d=64 the matmuls dominate. At d=32 the elementwise work (softmax, slicing) is a large share, and adding layers multiplies it.
- **T = 64 never paid off.** At this budget it was equal or worse on every pair (d32_L1_H2: 2.276 vs 2.258; d64_L1_H2: 2.161 vs 2.114; d32_L2_H2: 2.199 vs 2.176), while costing 1.6 to 2x, because score blocks and softmax grow as T squared. A 32-char window fits one attention matrix, so stay there.
- **2 heads equal 4 heads in quality** (differences at or under seed noise), and 4 heads cost more: twice the score and AV block calls, plus slicing whenever head_dim is under 32.
- **Over 300 ms: not measured.** The 10 configs estimated above 300 ms (from d64_L1_H4_T64 at 330 ms up to d64_L4_H4_T64 at 1149 ms) were not trained. The laptop was shared with another session's sweep (load average up to 24 on 12 threads) for most of this study, which cut throughput 2 to 5x, and the compute budget went to the feasible band, the ablations, the 4000-step final, and Shakespeare. Their cost estimates are in the full grid table. `python3 run_grid.py grid --min-ms 300 --lr-by-d 32=2e-2,64=1e-2` trains them.

## Full grid

d_model {32, 64} x layers {1, 2, 4} x heads {2, 4} x context {32, 64}, MLP 4x, sorted by estimated cost. "ctx >= T/2" scores every char once with at least T/2 chars of context (closer to steady-state sliding-window generation). "all" averages over every window position (the training-style number). Desmos costs come from `cost_model.py` (method below).

| config | params | full: 32-blocks | full: elementwise | ms/token full (padded / mac) | ms/token kv (padded / mac) | steps | val bpc (all) | val bpc (ctx>=T/2) | test bpc (ctx>=T/2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d32_L1_H2_T32 | 15,456 | 17 | 21,696 | 94 / 90 | 75 / 54 | 1000 | 2.318 | 2.258 | 2.254 |
| d32_L1_H4_T32 | 15,456 | 21 | 27,840 | 105 / 96 | 80 / 54 | 1000 | 2.308 | 2.249 | 2.236 |
| d32_L2_H2_T32 | 27,808 | 33 | 42,176 | 135 / 129 | 98 / 57 | 1000 | 2.238 | 2.176 | 2.164 |
| d64_L1_H2_T32 | 55,488 | 54 | 28,928 | 149 / 147 | 125 / 57 | 1000 | 2.179 | 2.114 | 2.099 |
| d32_L2_H4_T32 | 27,808 | 41 | 54,464 | 158 / 141 | 109 / 57 | 1000 | 2.225 | 2.155 | 2.146 |
| d32_L1_H2_T64 | 16,480 | 41 | 55,488 | 159 / 147 | 82 / 56 | 1000 | 2.308 | 2.276 | 2.265 |
| d64_L1_H4_T32 | 55,488 | 58 | 43,264 | 169 / 161 | 131 / 58 | 1000 | 2.176 | 2.108 | 2.096 |
| d32_L1_H4_T64 | 16,480 | 57 | 80,064 | 204 / 172 | 93 / 56 | 1000 | 2.281 | 2.243 | 2.239 |
| d32_L4_H2_T32 | 52,512 | 65 | 83,136 | 218 / 206 | 145 / 63 | 1000 | 2.175 | 2.100 | 2.087 |
| d64_L2_H2_T32 | 104,768 | 106 | 55,552 | 243 / 241 | 198 / 64 | 1000 | 2.101 | 2.027 | 2.012 |
| d32_L4_H4_T32 | 52,512 | 81 | 107,712 | 263 / 231 | 167 / 64 | 1000 | 2.168 | 2.091 | 2.078 |
| d32_L2_H2_T64 | 28,832 | 81 | 108,736 | 264 / 242 | 113 / 61 | 1000 | 2.232 | 2.199 | 2.192 |
| d64_L1_H2_T64 | 57,536 | 114 | 69,888 | 268 / 266 | 135 / 62 | 1000 | 2.191 | 2.161 | 2.154 |
| d64_L2_H4_T32 | 104,768 | 114 | 84,224 | 282 / 270 | 209 / 65 | 1000 | 2.105 | 2.029 | 2.019 |
| d64_L1_H4_T64 | 57,536 | 130 | 110,848 | 330 / 307 | 146 / 63 | - | not run | not run | not run |
| d32_L2_H4_T64 | 28,832 | 113 | 157,888 | 355 / 291 | 135 / 62 | - | not run | not run | not run |
| d64_L4_H2_T32 | 203,328 | 210 | 108,800 | 432 / 429 | 343 / 79 | - | not run | not run | not run |
| d32_L4_H2_T64 | 53,536 | 161 | 215,232 | 475 / 432 | 175 / 73 | - | not run | not run | not run |
| d64_L2_H2_T64 | 106,816 | 226 | 135,424 | 479 / 477 | 217 / 73 | - | not run | not run | not run |
| d64_L4_H4_T32 | 203,328 | 226 | 166,144 | 510 / 487 | 366 / 80 | - | not run | not run | not run |
| d64_L2_H4_T64 | 106,816 | 258 | 217,344 | 603 / 559 | 239 / 75 | - | not run | not run | not run |
| d32_L4_H4_T64 | 53,536 | 225 | 313,536 | 656 / 530 | 218 / 74 | - | not run | not run | not run |
| d64_L4_H2_T64 | 205,376 | 450 | 266,496 | 901 / 899 | 381 / 96 | - | not run | not run | not run |
| d64_L4_H4_T64 | 205,376 | 514 | 430,336 | 1149 / 1063 | 425 / 99 | - | not run | not run | not run |

`d32_L2_H4_T32` and `d64_L2_H4_T32` are the learning-rate-check runs with the winning rate. Each was the identical command (same config, lr, steps, seed, one thread, deterministic), so it was reused instead of retrained (`reused_from` in its `results.json`).

Samples at 1000 steps, best configs under 300 ms (all with sampling seed 7; the shared phrases come from the shared random draws):

```
d64_L2_H2_T32 (2.027): then minerva said, "the ships who was gashed were eared abodis and with went of all, these was from and joved them on the would with fight at will have you bed still had answered, and it wine upon seat aloud, "into as t
d32_L4_H2_T32 (2.100): then minerva said, "the ships why shire she were tearever fair fight the end of against sating, and at toldians, who broe in they upon of the was in as the bed sogating to she hoursed it wine, but seat aloud, "they were
d64_L1_H2_T32 (2.114): then minerva said, "the ships who havings of we stearever of seating the enten they through are suitor," sons to came the was towarder of the was i am the sons still had answered, and it wine upon seat aloud, "cargues t
d32_L1_H2_T32 (2.258): then minerva said, "rown his were in in should and a every disted, "my away son are, the a bare the and joved the a shoul not town herd seent will had the of the godded the he hortal the wine up the achaeans, whethus ba
```

## Ablations

Each row changes one thing from `d64_L2_H2_T32` (1000 steps, val bpc 2.027). Seed noise is about 0.02.

| variant | params | ms/token | val bpc (ctx>=T/2) | delta | verdict |
|---|---:|---:|---:|---:|---|
| base (RMSNorm, GELU-tanh, untied, learned pos, no bias, v32) | 104,768 | 243 | 2.027 | 0 | |
| second seed | 104,768 | 243 | 2.006 | -0.022 | the noise floor |
| ReLU instead of GELU-tanh | 104,768 | 243 | 2.014 | -0.013 | equal; simpler in Desmos, `(x+abs(x))/2` is exact |
| LayerNorm instead of RMSNorm | 105,088 | 260 | 2.031 | +0.004 | equal and 17 ms dearer: **RMSNorm is a fine substitute** |
| no norm at all | 104,448 | 227 | 2.165 | +0.138 | saves 16 ms but clearly worse; keep RMSNorm |
| linear-layer biases | 105,952 | 280 | 2.025 | -0.002 | no gain, +37 ms of bias adds; leave out |
| tied unembedding | 102,720 | 243 | 2.040 | +0.013 | equal within noise; **untied** recommended (below) |
| ALiBi instead of learned positions | 102,720 | 243 | 2.327 | +0.300 | much worse at this budget; keep learned positions |
| `v64` vocab (case preserved) | 108,864 | 246 | 2.060 | +0.033 | not the same task (also predicts capitals); only +3 ms |

Tied or untied: the final RMSNorm gain gets folded into the unembedding, so a tied model still needs its own folded `w_out` matrix in Desmos, and tying saves nothing there. It saves 2,048 trained parameters and was 0.013 worse (noise level). So: untied.

ALiBi was tested for a reason specific to Desmos. With learned absolute positions, a KV cache goes stale as soon as the window slides (see the KV note below). ALiBi would keep it valid, and in full recompute ALiBi is free (exp(S + B) * mask = exp(S) * C with C a constant matrix). At 1000 steps it cost 0.30 bpc, so it is not worth it here.

## Desmos cost model

Measured engine facts used (treated as hard): one 32x32 @ 32x32 matmul is about 1.3 ms, an elementwise op in an index comprehension is about 1 microsecond per element, there is about 50 ms of fixed overhead per token, and matrices cap at 32x32.

```
ms/token = 50 + 1.3 * (32x32 matmul equivalents) + 0.001 * (elementwise element-ops)
```

A matmul (m x k) @ (k x n) runs as ceil(m/32) * ceil(k/32) * ceil(n/32) block calls. Only the full 32x32 @ 32x32 case was measured, so two accountings are reported. **padded** means every block call costs 1.3 ms even if an operand is thinner, which is conservative and right if per-call overhead dominates. **mac** means cost is proportional to multiply-adds, m*k*n/32^3, which is optimistic and right if arithmetic dominates. The truth lies between them. Recommendations use padded.

Assumed implementation (exact identities, no approximation, all verified in `reference_numpy.py` and its tests):
- Norm gains are folded into the following weight matrix, and 1/sqrt(head_dim) into W_q. RMSNorm is then 2 passes (row sum of squares, scale) and LayerNorm 4.
- Softmax is 3 passes: exp times the 0/1 causal mask, row sum, divide. There is no max-subtraction: the largest attention score seen on 1,500 val windows is 13.5 and the largest logit 19.8, far from exp's float64 limit of about 709.
- Per-head Q/K/V slicing and the head concat are 1 pass each, unless head_dim is a multiple of 32, in which case heads already are whole blocks and both are free.
- Full recompute runs every layer on all T rows, and the final norm and unembedding only on the last row.

Full arithmetic for the recommended config (full recompute, window full):

```
embedding gather + position add: 2048 elements
per layer (x2):
  norm1 (2 passes): 4096 elements
  Q,K,V projections: 3 x (32x64 @ 64x64): 12 blocks
  scores Q_h K_h^T: 2 x (32x32 @ 32x32): 2 blocks
  softmax (exp*mask, row sum, divide): 6144 elements
  probs @ V_h: 2 x (32x32 @ 32x32): 2 blocks
  W_o: 1 x (32x64 @ 64x64): 4 blocks
  residual add: 2048 elements
  norm2 (2 passes): 4096 elements
  MLP up W_1: 1 x (32x64 @ 64x256): 16 blocks
  activation: 8192 elements
  MLP down W_2: 1 x (32x256 @ 256x64): 16 blocks
  residual add: 2048 elements
  layer total: 52 blocks, 26624 elements
final norm on last row: 128 elements
unembedding (last row): 1 x (1x64 @ 64x32): 2 padded blocks (0.06 mac)
sampling (scale, exp, sum, cdf): 128 elements
total: 106 padded blocks (104.06 mac), 55552 elements
padded ms = 50 + 1.3 * 106 + 0.001 * 55552 = 243.4
mac ms    = 50 + 1.3 * 104.06 + 0.001 * 55552 = 240.8
```

KV-cached (one new row per token, window full) for the same model is 106 padded blocks and 10,176 elements, giving 50 + 137.8 + 10.2 = **198 ms padded**, or **65 ms mac**. Every model's derivation is printed by `token_cost(cfg, mode).arithmetic()`.

**KV caching: recommend full recompute for phase 1**, for three reasons:
1. With a 32-row window, a 1-row matmul uses as many 32x32 block calls as a 32-row one. So under the padded accounting the cache saves only the elementwise softmax work (243 to 198 ms). Under the mac accounting it saves 3.7x (241 to 65 ms). The single most decision-relevant next measurement in Desmos is the time of a 1x32 @ 32x32 product against a 32x32 @ 32x32 one.
2. With learned absolute positions, a cache is valid only while the text fits in the window (the first 32 tokens). After that every token shifts every position, so each step must recompute anyway. ALiBi would fix this but cost 0.30 bpc here.
3. The cache must be written back each token (2 x layers x T x d = 8,192 numbers here). Earlier measurements in this project found large action-written lists expensive (about 150 ms more for a 192-point list; see also [`../ticker-bench/`](../ticker-bench/)), so the real KV cost is likely higher than the estimate.

Right-padding is exact: under the causal mask, rows before the padding do not depend on it (verified to 6e-15). So Desmos can always evaluate a full 32-row window with the prompt left-aligned and padding after it, and read row `len-1`. After 32 tokens the window slides (positions 0..31 always hold the last 32 chars), which matches how training sampled random windows.

## Corpus and vocabularies

**Homer (main corpus, local):** Samuel Butler's prose translations of the Odyssey (Project Gutenberg #1727) and the Iliad (#2199), read from `data/gutenberg/pg1727.txt` and `data/gutenberg/pg2199.txt` (the plain-text UTF-8 downloads from gutenberg.org; pass another folder with `--source-dir`). Public domain. Gutenberg boilerplate, the Odyssey preface, its footnotes, and its inline footnote numbers were removed, and whitespace was collapsed to single spaces. That leaves 1,417,166 characters of English prose in one consistent style, split contiguously 90/5/5: 1,275,328 train, 70,852 val, 70,852 test. Other local candidates (`/usr/share/dict/words`, Bulfinch, Ovid verse) were less suitable: a word list, a different register, or verse.

**tiny-shakespeare (added at Santo's request):** `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`, 1,115,394 bytes, sha256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` (pinned and checked by `corpus.py`). The text is public-domain Shakespeare; the char-rnn repository that hosts it is MIT licensed. It is downloaded to `data/tinyshakespeare/input.txt` and not committed (`.gitignore`). To rebuild: `curl --fail --max-time 60 -o data/tinyshakespeare/input.txt <url>`, then `python3 corpus.py`.

Coverage loss, the share of source characters by fate (`results/vocab_coverage.json`):

| vocab | symbols | exact | case folded | quote normalized | remapped | dropped |
|---|---|---:|---:|---:|---:|---:|
| `v32` Homer (32) | space, a-z, `. , ; ' "` | 97.57% | 1.88% | 0.34% | 0.21% | 0.002% |
| `v64` Homer (64) | space, a-z, A-Z, `. , ; : ? ! ' " - ( )` | 99.60% | 0 | 0.34% | 0.07% | 0.002% |
| `s32` Shakespeare (32) | space, newline, a-z, `, : . '` | 91.13% | 7.96% | 0 | 0.91% | 0.003% |

- Quote normalization (curly to straight quotes) is typography, not information loss.
- `v32` real loss: capitals (1.88%), `? !` merged into `.`, `:` into `;`, dashes, hyphens and brackets into spaces, and 25 digits dropped.
- `v64` merges only em-dash into hyphen and brackets into parentheses.
- `s32` keeps newlines because the play format (speaker line, verse lines, blank line) is the corpus's structure. It merges `;` into `,` and `? !` into `.`, turns hyphens into spaces, and drops 3 `&`, 1 `$` and 27 digits.
- Raw tiny-shakespeare has 65 distinct characters, so a case-preserving version does not fit 64.
- Every source character must hit a rule or `corpus.py` raises.

## Homer vs tiny-shakespeare

The best three frontier configs, retrained on Shakespeare with the same 1000-step budget and learning rates. Bits per char are not comparable across corpora (different text, and `s32` also predicts newlines); compare within a column.

| config | ms/token | Homer `v32` val / test bpc | Shakespeare `s32` val / test bpc |
|---|---:|---:|---:|
| d64_L1_H2_T32 | 149 | 2.114 / 2.099 | 2.498 / 2.562 |
| d32_L4_H2_T32 | 218 | 2.100 / 2.087 | 2.410 / 2.529 |
| d64_L2_H2_T32 | 243 | 2.027 / 2.012 | 2.358 / 2.471 |

The Shakespeare test split (the last 5% of the file, mostly *The Tempest*) scores about 0.1 bpc worse than its val split (mostly *The Taming of the Shrew*); the Homer splits agree to about 0.015.

Shakespeare samples (prompt `romeo:` plus a newline, temperature 0.8, seed 7). The Homer samples for the same configs are in the grid section above.

d64_L2_H2_T32 (val 2.358):

```
romeo:
o shall the great him go in this haught chrief,
but mades, you make know you sights,
in himself that a madam uptain, but seem thy frown yearn's good
yet inferces new is fair apport
in the as of fair l
```

d32_L4_H2_T32 (val 2.410):

```
romeo:
if you revenge threasign ive mid
haut anchied have beady your: if in desconderieven,
in him stuppparlant.

marcius:
i come, and your for away if will lear from,
be it spray the hate for to so the vist
```

d64_L1_H2_T32 (val 2.498):

```
romeo:
o shall it know murdering vices ned head head,
and that sun your severy, you sight this havest,
and i unworld, this do, my und my from by wherefore accusinfering hereis rometh:
here for that be fair l
```

Read: Shakespeare gives the more striking demo (line breaks and a speaker header make the output look like a play even when the words are invented). Homer gives more readable word-level English at this size. The phase-1 weights are Homer. Training the recommended config on Shakespeare for 4000 steps is one command (`python3 train.py --name final_shk_d64_L2_H2_T32 --vocab s32 --d-model 64 --n-layer 2 --n-head 2 --ctx 32 --max-steps 4000 --lr 1e-2`, then `export.py`), and its Desmos cost is identical.

## Training setup

- Model (`model.py`): decoder-only, pre-norm, causal multi-head attention, `x @ W` row-vector convention so exported matrices need no transpose, separate W_q/W_k/W_v. Init normal(0, 0.02), with residual-output projections scaled by 1/sqrt(2 * layers).
- Optimizer: AdamW (betas 0.9/0.99, weight decay 0.01 on matrices), gradient clip 1.0, 100 warmup steps then cosine decay to 10% of peak. Batch is 4096 chars per step (128 x 32 or 64 x 64).
- Learning rate: checked on d32 and d64 (`d*_L2_H4_T32`, 1000 steps, val bpc ctx). d32: 1e-3 gave 2.678, 3e-3 2.395, 1e-2 2.187, 2e-2 2.155. d64: 3e-3 gave 2.088, 1e-2 2.029. Used 2e-2 for d32 and 1e-2 for d64. The d64 1e-3 run was stopped early because it was dominated. Higher rates than tested might still help d32 slightly.
- Budget: a **fixed step count, not wall-clock**. The laptop was shared with another session's sweep (load average 21 to 24 on 12 threads during much of the grid). Under that load even per-process CPU time inflated about 2.4x per step, so time budgets would have made results depend on when a run happened. Fixed steps with one thread and a fixed seed are load-independent and reproducible. Equal tokens per config is also the comparison this decision needs: offline training is cheap, Desmos inference cost is not.
- Early stopping: quick-val every 250 steps with patience 5. It never triggered, because none of these models is overfitting at this data size.
- Threads: at most 3 cores in total, as instructed. Every process sets `OMP_NUM_THREADS` before torch loads (`threadcap.py`). Grid runs used 3 workers x 1 thread, measured at about 3x the throughput of one 3-thread worker for models this small (44 vs 33 steps/s on the smallest config). `torch.set_num_threads(3)` is the default for a lone `train.py`.
- Wall clock for the whole study, including writing the code and the tiny-shakespeare addition: about 75 minutes, over the 60-minute target, mostly because of the shared-machine slowdown. The 10 configs above 300 ms were left untrained to limit the overrun.

## Exported files and the equality check

`best/` (built by `python3 export.py --run final_d64_L2_H2_T32`):

| file | contents |
|---|---|
| `model.pt` | trained float32 `state_dict` with model and train configs (`torch.load(..., weights_only=True)`) |
| `weights.npz` | **canonical Desmos weights**, float64: `tok_emb` (32x64), `pos_emb` (32x64), `l{0,1}_wq/wk/wv/wo` (64x64), `l{0,1}_w1` (64x256), `l{0,1}_w2` (256x64), `w_out` (64x32). RMSNorm gains and 1/sqrt(32) are folded in, and every value is the float64 nearest a 3-significant-digit decimal |
| `weights_trained.npz` | the same weights unfolded and unrounded, including the norm gains (for inspection and interpretability) |
| `reference.npz` | float64 reference over `weights.npz` (see below) |
| `config.json` | config, vocab string (index = token id, space = 0), metrics, precision study, cost summary and arithmetic, check results |

`reference.npz`:
- `input_text` = `"tell me, o muse, of that ingenio"` (exactly 32 chars), `input_ids` (32,), and `logits` (32x32) for that input.
- `short_input_text` = `"then minerva said, "`, `short_input_ids_padded` (padded with token 0 to 32), `short_len` = 19, and `short_logits`. Row 18 is the next-token logits after the prompt.
- Every intermediate: `trace_x_embed`, `trace_l{i}_attn_probs` (2x32x32), `trace_l{i}_resid_mid`, `trace_l{i}_resid_out`, `trace_x_final_norm`, `trace_logits`. These let a mismatch in Desmos be localized to a layer.
- `logits_torch_float32_unrounded` is the original float32 model on the same input, for information only.

The spec the Desmos build must reproduce op by op is the docstring of `reference_numpy.py` (a pure-numpy float64 forward over `weights.npz`). Checks run at export:
- numpy spec vs the torch model in float64: max abs logit difference **4.0e-14**
- right-padding exactness: 6.2e-15
- every canonical weight round-trips through `format(w, ".3g")`

A correct Desmos forward pass should therefore match `logits` to roughly 1e-12; anything above 1e-9 points to a real difference in ops or op order.

Notes for the exporter:
- Rounding to 3 significant digits changed val bpc by 0.00001 (precision study in `config.json`: 3 to 8 digits all give 1.7788). It moves raw logits by up to 0.063, which is why the reference is computed from the rounded weights and not the float32 originals.
- `format(w, ".3g")` can emit exponent form (`1.2e-05`). Write the same decimal in positional form (`Decimal(...)`, as `paste_chars` does) or as `\times10^{...}` before pasting. That Desmos parses decimals to the correctly rounded float64 is assumed, not verified.
- The weights fill 102 matrices of 32x32 (1024 entries each), well under the 10,000-element list cap.

## Reproduce

```bash
cd language-model/experiments/charlm
python3 corpus.py                                            # data/*.txt + results/vocab_coverage.json
python3 run_grid.py lr                                       # learning-rate check
python3 run_grid.py grid --max-ms 300 --lr-by-d 32=2e-2,64=1e-2      # feasible band (14 configs)
python3 run_grid.py grid --min-ms 300 --lr-by-d 32=2e-2,64=1e-2      # the 10 configs not run here
python3 run_grid.py ablate --base d64_L2_H2_T32 --lr-by-d 64=1e-2
python3 run_grid.py shakespeare --lr-by-d 32=2e-2,64=1e-2
CHARLM_THREADS=1 python3 train.py --name final_d64_L2_H2_T32 --d-model 64 --n-layer 2 --n-head 2 --ctx 32 --max-steps 4000 --lr 1e-2
python3 export.py --run final_d64_L2_H2_T32 --out best
python3 report.py --base d64_L2_H2_T32                       # results/tables.md
CHARLM_THREADS=1 python3 -m pytest -q test_charlm.py         # 71 tests
```

Variants: add `--act relu`, `--vocab v64` or `--vocab s32` to `train.py`. Every run writes `runs/<name>/{model.pt, results.json, log.txt}`; `runs/` is about 9 MB of checkpoints.

## Caveats

- The grid compares configs at equal tokens (1000 steps), where none is converged. Bigger models are further from convergence, so the gaps on the frontier probably understate what width and depth buy with long training. Ranking stability under longer training is assumed, not measured.
- The Desmos costs are estimates from three measured constants and an assumed implementation, not measurements of these models. The padded-vs-mac question and the per-element cost of tanh-GELU against ReLU in Desmos are both open.
- Single seed per config. The seed-noise estimate (about 0.02 bpc) comes from one pair.
- The grid ran on a heavily shared machine. That affects only wall-clock times (in each `results.json`), not results, because budgets were fixed step counts.
