# Results summary

Desmos seconds per step assume a fixed tick overhead of 50-150 ms and at most 6 ticks/s. Hours use the slowest seed's steps to 99% test accuracy. A grokking delay shows as 'test 50% at' well after 'train 99% at'; at train fraction >= 0.6 they coincide.

## All configs, seeds pooled across stages (121 distinct runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | mlp_concat m128 | AdamW | 0.01 | 3 | 0.75 | full | 721 | 184, 255, 195 | 3/3 | yes | 66, 68, 67 | 116, 153, 116 | 1.39-1.49 | 0.10-0.11 |
| 23 | attn h4 m128 | AdamW | 0.01 | 3 | 0.75 | full | 397 | 418 | 1/1 | yes | 247 | 106 | 1.31-1.41 | 0.15-0.16 |
| 23 | attn h4 m128 | AdamW | 0.03 | 1 | 0.6 | full | 317 | 585 | 1/1 | yes | 39 | 36 | 1.05-1.15 | 0.17-0.19 |
| 23 | attn h4 m128 | AdamW | 0.03 | 1 | 0.75 | full | 397 | 149, 317, 473, 207 | 4/4 | no | 45, 50, 36, 47 | 39, 42, 27, 35 | 1.31-1.41 | 0.17-0.19 |
| 31 | mlp_sum m128 | AdamW | 0.01 | 3 | 0.75 | full | 721 | 468 | 1/1 | yes | 88 | 76 | 1.38-1.48 | 0.18-0.19 |
| 31 | attn h4 m128 gelu | AdamW | 0.01 | 3 | 0.6 | full | 577 | 443 | 1/1 | no | 246 | 132 | 1.87-1.97 | 0.23-0.24 |
| 31 | attn h4 m64 | AdamW | 0.03 | 1 | 0.6 | full | 577 | 770 | 1/1 | no | 52 | 51 | 1.43-1.53 | 0.30-0.33 |
| 47 | attn h4 m128 | AdamW | 0.003 | 3 | 0.6 | full | 1325 | 274 | 1/1 | yes | 212 | 157 | 4.76-4.86 | 0.36-0.37 |
| 31 | mlp_concat m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | 1248 | 1/1 | yes | 70 | 382 | 1.16-1.26 | 0.40-0.44 |
| 31 | attn h4 m128 | AdamW | 0.03 | 1 | 0.5 | full | 480 | 1151, 195, 804, 310 | 4/4 | no | 53, 56, 45, 48 | 477, 42, 61, 139 | 1.52-1.62 | 0.49-0.52 |
| 31 | attn h4 m128 | AdamW | 0.03 | 1 | 0.75 | full | 721 | 817 | 1/1 | no | 75 | 52 | 2.25-2.35 | 0.51-0.53 |
| 47 | attn h4 m128 | AdamW | 0.003 | 1 | 0.6 | full | 1325 | 394 | 1/1 | yes | 156 | 147 | 4.76-4.86 | 0.52-0.53 |
| 47 | attn h4 m128 | AdamW | 0.01 | 1 | 0.6 | full | 1325 | 156, 403, 189 | 3/3 | yes | 73, 142, 81 | 67, 76, 74 | 4.76-4.86 | 0.53-0.54 |
| 31 | attn h4 m32 | AdamW | 0.03 | 1 | 0.6 | full | 577 | 373, 185, 1971, 118 | 4/4 | no | 68, 51, 57, 51 | 67, 40, 45, 48 | 1.20-1.30 | 0.66-0.71 |
| 31 | attn h4 m128 | AdamW | 0.01 | 1 | 0.6 | full | 577 | 2135, 904, 714, 868 | 4/4 | yes | 69, 64, 64, 65 | 71, 58, 60, 62 | 1.87-1.97 | 1.11-1.17 |
| 47 | attn h4 m128 | AdamW | 0.003 | 3 | 0.45 | full | 994 | 1365 | 1/1 | yes | 1002 | 816 | 3.65-3.75 | 1.39-1.42 |
| 59 | attn h4 m128 | AdamW | 0.01 | 1 | 0.3 | full | 1044 | 1533 | 1/1 | yes | 82 | 1264 | 3.82-3.92 | 1.63-1.67 |
| 47 | attn h4 m128 | AdamW | 0.01 | 1 | 0.45 | full | 994 | 1717 | 1/1 | yes | 90 | 842 | 3.65-3.75 | 1.74-1.79 |
| 97 | attn h4 m128 | AdamW | 0.01 | 1 | 0.3 | full | 2823 | 502 | 1/1 | yes | 196 | 306 | 13.25-13.35 | 1.85-1.86 |
| 113 | attn h4 m128 | AdamW | 0.01 | 1 | 0.3 | full | 3831 | 387 | 1/1 | yes | 225 | 211 | 18.00-18.10 | 1.94-1.95 |
| 23 | attn h4 m128 | AdamW | 0.001 | 1 | 0.6 | full | 317 | 8060 | 1/1 | yes | 240 | 590 | 1.05-1.15 | 2.35-2.57 |
| 47 | attn h4 m128 | AdamW | 0.003 | 1 | 0.45 | full | 994 | 2399 | 1/1 | yes | 159 | 1461 | 3.65-3.75 | 2.44-2.50 |
| 31 | attn h4 m128 | AdamW | 0.03 | 1 | 0.6 | full | 577 | 241, 617, 182, >2500 | 3/4 | yes | 55, 69, 46, 51 | 53, 41, 42, 45 | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | 234, >2500, >2500 | 1/3 | yes | 133, None, 515 | 95, 109, 81 | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.5 | full | 480 | >2500 | 0/1 | - | 295 | 292 | 1.52-1.62 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | None | 90 | 2.25-2.35 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 6 | 0.5 | full | 480 | >2500 | 0/1 | - | None | None | 1.52-1.62 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 6 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 6 | 0.75 | full | 721 | >2500 | 0/1 | - | None | None | 2.25-2.35 | n/a |
| 31 | attn h4 m128 | AdamW | 0.03 | 3 | 0.5 | full | 480 | >2500 | 0/1 | - | 2029 | 135 | 1.52-1.62 | n/a |
| 31 | attn h4 m128 | AdamW | 0.03 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 87 | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.03 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | None | None | 2.25-2.35 | n/a |
| 23 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | full | 317 | >2500 | 0/1 | - | 118 | 83 | 1.05-1.15 | n/a |
| 47 | attn h4 m128 | AdamW | 0.003 | 1 | 0.3 | full | 663 | >5000 | 0/1 | - | 135 | None | 2.46-2.56 | n/a |
| 47 | attn h4 m128 | AdamW | 0.003 | 3 | 0.3 | full | 663 | >5000 | 0/1 | - | 251 | None | 2.46-2.56 | n/a |
| 47 | attn h4 m128 | AdamW | 0.01 | 1 | 0.3 | full | 663 | >5000 | 0/1 | - | 68 | None | 2.46-2.56 | n/a |
| 47 | attn h4 m128 | AdamW | 0.01 | 3 | 0.3 | full | 663 | >5000 | 0/1 | - | None | None | 2.46-2.56 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 1 | 0.3 | full | 288 | >5000 | 0/1 | - | 100 | None | 0.97-1.07 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 1 | 0.45 | full | 432 | >5000 | 0/1 | - | 119 | None | 1.42-1.52 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 1 | 0.6 | full | 577 | >5000 | 0/1 | - | 128 | 130 | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 3 | 0.3 | full | 288 | >5000 | 0/1 | - | 171 | None | 0.97-1.07 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 3 | 0.45 | full | 432 | >5000 | 0/1 | - | 244 | None | 1.42-1.52 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 3 | 0.6 | full | 577 | >5000 | 0/1 | - | 567 | 172 | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 1 | 0.3 | full | 288 | >5000 | 0/1 | - | 53 | None | 0.97-1.07 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 1 | 0.45 | full | 432 | >5000 | 0/1 | - | 60 | 1213 | 1.42-1.52 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.3 | full | 288 | >5000 | 0/1 | - | 147 | None | 0.97-1.07 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >5000 | 0/1 | - | 375 | 4710 | 1.42-1.52 | n/a |
| 31 | attn h1 m32 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.18-1.28 | n/a |
| 31 | attn h4 m32 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 132 | 1.20-1.30 | n/a |
| 31 | attn h1 m64 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 194 | 1.40-1.50 | n/a |
| 31 | attn h4 m64 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | 1494 | 133 | 1.43-1.53 | n/a |
| 31 | attn h1 m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 88 | 1.85-1.95 | n/a |
| 31 | attn h1 m32 | AdamW | 0.03 | 1 | 0.6 | full | 577 | >2500 | 0/1 | - | 134 | 85 | 1.18-1.28 | n/a |
| 31 | attn h1 m64 | AdamW | 0.03 | 1 | 0.6 | full | 577 | >2500 | 0/1 | - | 64 | 54 | 1.40-1.50 | n/a |
| 31 | attn h1 m128 | AdamW | 0.03 | 1 | 0.6 | full | 577 | >2500 | 0/1 | - | 47 | 43 | 1.85-1.95 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 32 random | 32 | >2500 | 0/1 | - | None | None | 0.23-0.33 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 32 cyclic | 32 | >2500 | 0/1 | - | None | None | 0.23-0.33 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 64 random | 64 | >2500 | 0/1 | - | None | None | 0.32-0.42 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 64 cyclic | 64 | >2500 | 0/1 | - | None | None | 0.32-0.42 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 128 random | 128 | >2500 | 0/1 | - | None | None | 0.51-0.61 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 128 cyclic | 128 | >2500 | 0/1 | - | None | None | 0.51-0.61 | n/a |
| 31 | mlp_concat m32 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | None | None | 0.31-0.41 | n/a |
| 31 | mlp_concat m32 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 860 | 0.39-0.49 | n/a |
| 31 | mlp_concat m32 | AdamW | 0.01 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | None | 289 | 0.46-0.56 | n/a |
| 31 | mlp_sum m32 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | None | None | 0.31-0.41 | n/a |
| 31 | mlp_sum m32 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 248 | 0.39-0.49 | n/a |
| 31 | mlp_sum m32 | AdamW | 0.01 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | None | 139 | 0.46-0.56 | n/a |
| 31 | mlp_concat m128 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | 60 | 1379 | 0.89-0.99 | n/a |
| 31 | mlp_folded m32 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | None | None | 0.30-0.40 | n/a |
| 31 | mlp_folded m32 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 0.38-0.48 | n/a |
| 31 | mlp_folded m32 | AdamW | 0.01 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | None | None | 0.45-0.55 | n/a |
| 31 | mlp_sum m128 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | 89 | 341 | 0.89-0.99 | n/a |
| 31 | mlp_sum m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | 94 | 90 | 1.16-1.26 | n/a |
| 31 | mlp_folded m128 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | 74 | None | 0.86-0.96 | n/a |
| 31 | mlp_folded m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | 86 | 1018 | 1.13-1.23 | n/a |
| 31 | mlp_folded m128 | AdamW | 0.01 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | 87 | 247 | 1.35-1.45 | n/a |
| 31 | attn h4 m128 | SGD+mom | 0.1 | 0.1 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 0.1 | 0.3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 0.3 | 0.0333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 0.3 | 0.1 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 1 | 0.01 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 1 | 0.03 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 0.3 | 0.0333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 0.3 | 0.1 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 1 | 0.01 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 1 | 0.03 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 3 | 0.00333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 3 | 0.01 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 gelu+LN | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 0 | 0.6 | full | 577 | >2500 | 0/1 | - | 54 | 65 | 1.87-1.97 | n/a |
| 23 | attn h4 m128 | AdamW | 0.001 | 1 | 0.3 | full | 159 | >10000 | 0/1 | - | 190 | None | 0.59-0.69 | n/a |
| 23 | attn h4 m128 | AdamW | 0.001 | 1 | 0.45 | full | 238 | >10000 | 0/1 | - | 220 | None | 0.85-0.95 | n/a |
| 31 | attn h4 m128 | SGD+mom | 0.3 | 0.000333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 0.3 | 0.00333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 1 | 0.0001 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 1 | 0.001 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 1 | 0.0001 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 1 | 0.001 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 3 | 3.33333e-05 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 3 | 0.000333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |

## attn_focus (16 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 23 | attn h4 m128 | AdamW | 0.03 | 1 | 0.75 | full | 397 | 149 | 1/1 | yes | 45 | 39 | 1.31-1.41 | 0.05-0.06 |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | 234 | 1/1 | yes | 133 | 95 | 1.87-1.97 | 0.12-0.13 |
| 31 | attn h4 m128 | AdamW | 0.03 | 1 | 0.6 | full | 577 | 241 | 1/1 | yes | 55 | 53 | 1.87-1.97 | 0.13-0.13 |
| 23 | attn h4 m128 | AdamW | 0.01 | 3 | 0.75 | full | 397 | 418 | 1/1 | yes | 247 | 106 | 1.31-1.41 | 0.15-0.16 |
| 23 | attn h4 m128 | AdamW | 0.03 | 1 | 0.6 | full | 317 | 585 | 1/1 | yes | 39 | 36 | 1.05-1.15 | 0.17-0.19 |
| 31 | attn h4 m128 | AdamW | 0.03 | 1 | 0.5 | full | 480 | 1151 | 1/1 | yes | 53 | 477 | 1.52-1.62 | 0.49-0.52 |
| 31 | attn h4 m128 | AdamW | 0.03 | 1 | 0.75 | full | 721 | 817 | 1/1 | no | 75 | 52 | 2.25-2.35 | 0.51-0.53 |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.5 | full | 480 | >2500 | 0/1 | - | 295 | 292 | 1.52-1.62 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | None | 90 | 2.25-2.35 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 6 | 0.5 | full | 480 | >2500 | 0/1 | - | None | None | 1.52-1.62 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 6 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 6 | 0.75 | full | 721 | >2500 | 0/1 | - | None | None | 2.25-2.35 | n/a |
| 31 | attn h4 m128 | AdamW | 0.03 | 3 | 0.5 | full | 480 | >2500 | 0/1 | - | 2029 | 135 | 1.52-1.62 | n/a |
| 31 | attn h4 m128 | AdamW | 0.03 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 87 | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.03 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | None | None | 2.25-2.35 | n/a |
| 23 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | full | 317 | >2500 | 0/1 | - | 118 | 83 | 1.05-1.15 | n/a |

## attn_search (22 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | 234 | 1/1 | yes | 133 | 95 | 1.87-1.97 | 0.12-0.13 |
| 47 | attn h4 m128 | AdamW | 0.01 | 1 | 0.6 | full | 1325 | 156 | 1/1 | yes | 73 | 67 | 4.76-4.86 | 0.21-0.21 |
| 47 | attn h4 m128 | AdamW | 0.003 | 3 | 0.6 | full | 1325 | 274 | 1/1 | yes | 212 | 157 | 4.76-4.86 | 0.36-0.37 |
| 47 | attn h4 m128 | AdamW | 0.003 | 1 | 0.6 | full | 1325 | 394 | 1/1 | yes | 156 | 147 | 4.76-4.86 | 0.52-0.53 |
| 31 | attn h4 m128 | AdamW | 0.01 | 1 | 0.6 | full | 577 | 2135 | 1/1 | yes | 69 | 71 | 1.87-1.97 | 1.11-1.17 |
| 47 | attn h4 m128 | AdamW | 0.003 | 3 | 0.45 | full | 994 | 1365 | 1/1 | yes | 1002 | 816 | 3.65-3.75 | 1.39-1.42 |
| 47 | attn h4 m128 | AdamW | 0.01 | 1 | 0.45 | full | 994 | 1717 | 1/1 | yes | 90 | 842 | 3.65-3.75 | 1.74-1.79 |
| 47 | attn h4 m128 | AdamW | 0.003 | 1 | 0.45 | full | 994 | 2399 | 1/1 | yes | 159 | 1461 | 3.65-3.75 | 2.44-2.50 |
| 47 | attn h4 m128 | AdamW | 0.003 | 1 | 0.3 | full | 663 | >5000 | 0/1 | - | 135 | None | 2.46-2.56 | n/a |
| 47 | attn h4 m128 | AdamW | 0.003 | 3 | 0.3 | full | 663 | >5000 | 0/1 | - | 251 | None | 2.46-2.56 | n/a |
| 47 | attn h4 m128 | AdamW | 0.01 | 1 | 0.3 | full | 663 | >5000 | 0/1 | - | 68 | None | 2.46-2.56 | n/a |
| 47 | attn h4 m128 | AdamW | 0.01 | 3 | 0.3 | full | 663 | >5000 | 0/1 | - | None | None | 2.46-2.56 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 1 | 0.3 | full | 288 | >5000 | 0/1 | - | 100 | None | 0.97-1.07 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 1 | 0.45 | full | 432 | >5000 | 0/1 | - | 119 | None | 1.42-1.52 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 1 | 0.6 | full | 577 | >5000 | 0/1 | - | 128 | 130 | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 3 | 0.3 | full | 288 | >5000 | 0/1 | - | 171 | None | 0.97-1.07 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 3 | 0.45 | full | 432 | >5000 | 0/1 | - | 244 | None | 1.42-1.52 | n/a |
| 31 | attn h4 m128 | AdamW | 0.003 | 3 | 0.6 | full | 577 | >5000 | 0/1 | - | 567 | 172 | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 1 | 0.3 | full | 288 | >5000 | 0/1 | - | 53 | None | 0.97-1.07 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 1 | 0.45 | full | 432 | >5000 | 0/1 | - | 60 | 1213 | 1.42-1.52 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.3 | full | 288 | >5000 | 0/1 | - | 147 | None | 0.97-1.07 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >5000 | 0/1 | - | 375 | 4710 | 1.42-1.52 | n/a |

## attn_shape (5 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | attn h1 m32 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.18-1.28 | n/a |
| 31 | attn h4 m32 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 132 | 1.20-1.30 | n/a |
| 31 | attn h1 m64 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 194 | 1.40-1.50 | n/a |
| 31 | attn h4 m64 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | 1494 | 133 | 1.43-1.53 | n/a |
| 31 | attn h1 m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 88 | 1.85-1.95 | n/a |

## attn_shape_lr3e-2_wd1 (5 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | attn h4 m32 | AdamW | 0.03 | 1 | 0.6 | full | 577 | 373 | 1/1 | yes | 68 | 67 | 1.20-1.30 | 0.12-0.14 |
| 31 | attn h4 m64 | AdamW | 0.03 | 1 | 0.6 | full | 577 | 770 | 1/1 | no | 52 | 51 | 1.43-1.53 | 0.30-0.33 |
| 31 | attn h1 m32 | AdamW | 0.03 | 1 | 0.6 | full | 577 | >2500 | 0/1 | - | 134 | 85 | 1.18-1.28 | n/a |
| 31 | attn h1 m64 | AdamW | 0.03 | 1 | 0.6 | full | 577 | >2500 | 0/1 | - | 64 | 54 | 1.40-1.50 | n/a |
| 31 | attn h1 m128 | AdamW | 0.03 | 1 | 0.6 | full | 577 | >2500 | 0/1 | - | 47 | 43 | 1.85-1.95 | n/a |

## minibatch (7 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | 234 | 1/1 | yes | 133 | 95 | 1.87-1.97 | 0.12-0.13 |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 32 random | 32 | >2500 | 0/1 | - | None | None | 0.23-0.33 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 32 cyclic | 32 | >2500 | 0/1 | - | None | None | 0.23-0.33 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 64 random | 64 | >2500 | 0/1 | - | None | None | 0.32-0.42 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 64 cyclic | 64 | >2500 | 0/1 | - | None | None | 0.32-0.42 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 128 random | 128 | >2500 | 0/1 | - | None | None | 0.51-0.61 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | 128 cyclic | 128 | >2500 | 0/1 | - | None | None | 0.51-0.61 | n/a |

## mlp_search (18 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | mlp_concat m128 | AdamW | 0.01 | 3 | 0.75 | full | 721 | 184 | 1/1 | yes | 66 | 116 | 1.39-1.49 | 0.07-0.08 |
| 31 | mlp_sum m128 | AdamW | 0.01 | 3 | 0.75 | full | 721 | 468 | 1/1 | yes | 88 | 76 | 1.38-1.48 | 0.18-0.19 |
| 31 | mlp_concat m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | 1248 | 1/1 | yes | 70 | 382 | 1.16-1.26 | 0.40-0.44 |
| 31 | mlp_concat m32 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | None | None | 0.31-0.41 | n/a |
| 31 | mlp_concat m32 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 860 | 0.39-0.49 | n/a |
| 31 | mlp_concat m32 | AdamW | 0.01 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | None | 289 | 0.46-0.56 | n/a |
| 31 | mlp_sum m32 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | None | None | 0.31-0.41 | n/a |
| 31 | mlp_sum m32 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | 248 | 0.39-0.49 | n/a |
| 31 | mlp_sum m32 | AdamW | 0.01 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | None | 139 | 0.46-0.56 | n/a |
| 31 | mlp_concat m128 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | 60 | 1379 | 0.89-0.99 | n/a |
| 31 | mlp_folded m32 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | None | None | 0.30-0.40 | n/a |
| 31 | mlp_folded m32 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 0.38-0.48 | n/a |
| 31 | mlp_folded m32 | AdamW | 0.01 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | None | None | 0.45-0.55 | n/a |
| 31 | mlp_sum m128 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | 89 | 341 | 0.89-0.99 | n/a |
| 31 | mlp_sum m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | 94 | 90 | 1.16-1.26 | n/a |
| 31 | mlp_folded m128 | AdamW | 0.01 | 3 | 0.45 | full | 432 | >2500 | 0/1 | - | 74 | None | 0.86-0.96 | n/a |
| 31 | mlp_folded m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | 86 | 1018 | 1.13-1.23 | n/a |
| 31 | mlp_folded m128 | AdamW | 0.01 | 3 | 0.75 | full | 721 | >2500 | 0/1 | - | 87 | 247 | 1.35-1.45 | n/a |

## optim (15 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | attn h4 m128 gelu | AdamW | 0.01 | 3 | 0.6 | full | 577 | 443 | 1/1 | no | 246 | 132 | 1.87-1.97 | 0.23-0.24 |
| 31 | attn h4 m128 | SGD+mom | 0.1 | 0.1 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 0.1 | 0.3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 0.3 | 0.0333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 0.3 | 0.1 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 1 | 0.01 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 1 | 0.03 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 0.3 | 0.0333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 0.3 | 0.1 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 1 | 0.01 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 1 | 0.03 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 3 | 0.00333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 3 | 0.01 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 gelu+LN | AdamW | 0.01 | 3 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.87-1.97 | n/a |
| 31 | attn h4 m128 | AdamW | 0.01 | 0 | 0.6 | full | 577 | >2500 | 0/1 | - | 54 | 65 | 1.87-1.97 | n/a |

## pilot_single_run_lr1e-3 (3 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 23 | attn h4 m128 | AdamW | 0.001 | 1 | 0.6 | full | 317 | 8060 | 1/1 | yes | 240 | 590 | 1.05-1.15 | 2.35-2.57 |
| 23 | attn h4 m128 | AdamW | 0.001 | 1 | 0.3 | full | 159 | >10000 | 0/1 | - | 190 | None | 0.59-0.69 | n/a |
| 23 | attn h4 m128 | AdamW | 0.001 | 1 | 0.45 | full | 238 | >10000 | 0/1 | - | 220 | None | 0.85-0.95 | n/a |

## seeds_attn_p113_lr1e-2_wd1_f0.3 (1 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 113 | attn h4 m128 | AdamW | 0.01 | 1 | 0.3 | full | 3831 | 387 | 1/1 | yes | 225 | 211 | 18.00-18.10 | 1.94-1.95 |

## seeds_attn_p23_lr3e-2_wd1_f0.75 (4 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 23 | attn h4 m128 | AdamW | 0.03 | 1 | 0.75 | full | 397 | 149, 317, 473, 207 | 4/4 | no | 45, 50, 36, 47 | 39, 42, 27, 35 | 1.31-1.41 | 0.17-0.19 |

## seeds_attn_p31_lr1e-2_wd1_f0.6 (4 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | attn h4 m128 | AdamW | 0.01 | 1 | 0.6 | full | 577 | 2135, 904, 714, 868 | 4/4 | yes | 69, 64, 64, 65 | 71, 58, 60, 62 | 1.87-1.97 | 1.11-1.17 |

## seeds_attn_p31_lr1e-2_wd3_f0.6 (3 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | attn h4 m128 | AdamW | 0.01 | 3 | 0.6 | full | 577 | 234, >2500, >2500 | 1/3 | yes | 133, None, 515 | 95, 109, 81 | 1.87-1.97 | n/a |

## seeds_attn_p31_lr3e-2_wd1_f0.5 (4 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | attn h4 m128 | AdamW | 0.03 | 1 | 0.5 | full | 480 | 1151, 195, 804, 310 | 4/4 | no | 53, 56, 45, 48 | 477, 42, 61, 139 | 1.52-1.62 | 0.49-0.52 |

## seeds_attn_p31_lr3e-2_wd1_f0.6 (4 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | attn h4 m128 | AdamW | 0.03 | 1 | 0.6 | full | 577 | 241, 617, 182, >2500 | 3/4 | yes | 55, 69, 46, 51 | 53, 41, 42, 45 | 1.87-1.97 | n/a |

## seeds_attn_p31_m32_lr3e-2_wd1_f0.6 (4 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | attn h4 m32 | AdamW | 0.03 | 1 | 0.6 | full | 577 | 373, 185, 1971, 118 | 4/4 | no | 68, 51, 57, 51 | 67, 40, 45, 48 | 1.20-1.30 | 0.66-0.71 |

## seeds_attn_p47_lr1e-2_wd1_f0.6 (3 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 47 | attn h4 m128 | AdamW | 0.01 | 1 | 0.6 | full | 1325 | 156, 403, 189 | 3/3 | yes | 73, 142, 81 | 67, 76, 74 | 4.76-4.86 | 0.53-0.54 |

## seeds_attn_p59_lr1e-2_wd1_f0.3 (1 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 59 | attn h4 m128 | AdamW | 0.01 | 1 | 0.3 | full | 1044 | 1533 | 1/1 | yes | 82 | 1264 | 3.82-3.92 | 1.63-1.67 |

## seeds_attn_p97_lr1e-2_wd1_f0.3 (1 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 97 | attn h4 m128 | AdamW | 0.01 | 1 | 0.3 | full | 2823 | 502 | 1/1 | yes | 196 | 306 | 13.25-13.35 | 1.85-1.86 |

## seeds_mlp_concat_p31_m128_lr1e-2_wd3_f0.75 (3 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | mlp_concat m128 | AdamW | 0.01 | 3 | 0.75 | full | 721 | 184, 255, 195 | 3/3 | yes | 66, 68, 67 | 116, 153, 116 | 1.39-1.49 | 0.10-0.11 |

## sgd_weak_decay (8 runs)

| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) | seeds grokked | stable | train 99% at | test 50% at | Desmos s/step | est. hours (50-150 ms tick) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | attn h4 m128 | SGD+mom | 0.3 | 0.000333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 0.3 | 0.00333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 1 | 0.0001 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | SGD+mom | 1 | 0.001 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 1 | 0.0001 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 1 | 0.001 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 3 | 3.33333e-05 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
| 31 | attn h4 m128 | GD | 3 | 0.000333333 | 0.6 | full | 577 | >2500 | 0/1 | - | None | None | 1.82-1.92 | n/a |
