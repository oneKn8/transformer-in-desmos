### mlp_concat, p=31, N_train=721, d=32, heads=4, d_mlp=128, adamw

| op | 32x32 matmuls | native adds | elementwise elems |
|---|---:|---:|---:|
| T_a,T_b = E W_top, E W_bot [31x32]@[32x128] x2 | 8 | 0 | 0 |
| pre = OH_a T_a + OH_b T_b (gather) [721x31]@[31x128] x2 | 184 | 0 | 0 |
|   + b_in [721x128] x2 | 0 | 184 | 0 |
| h = relu(pre) (92288 elems x 1 pass) | 0 | 0 | 92288 |
| logits = h W_U [721x128]@[128x31] | 92 | 69 | 0 |
| dlogits = softmax - onehot (22351 elems x 3 pass) | 0 | 0 | 67053 |
|   row sums via ones [721x31]@[31x1] | 23 | 0 | 0 |
| dW_U = h^T dlogits [128x721]@[721x31] | 92 | 88 | 0 |
| dh = dlogits W_U^T [721x31]@[31x128] | 92 | 0 | 0 |
| dpre = dh * [pre>0] (92288 elems x 1 pass) | 0 | 0 | 92288 |
| dT_a,dT_b = OH^T dpre (scatter) [31x721]@[721x128] x2 | 184 | 176 | 0 |
| db_in = 1^T dpre [1x721]@[721x128] | 92 | 88 | 0 |
| dW = E^T dT_j [32x31]@[31x128] x2 | 8 | 0 | 0 |
| dE = dT_j W_j^T [31x128]@[128x32] x2 | 8 | 6 | 0 |
|   dE sum [31x32] | 0 | 1 | 0 |
| AdamW m, v, decay, step (17 blocks x 8 native ops) | 0 | 136 | 0 |
| AdamW g*g, sqrt, divide (13280 elems x 3 pass) | 0 | 0 | 39840 |
| **total** | **783** | **748** | **291469** |

- matmuls: 783 x 1.3 ms = 1018 ms
- native adds: 748 x 0.041 ms = 30 ms (if adds cost a full matmul: 972 ms)
- elementwise: 291469 elems x 1.0 us = 291 ms
- compute per step: 1340 ms = 1031 matmul-equivalents
- per tick with 50 to 150 ms overhead: 1390 to 1490 ms
- 184 steps to grok: 0.07 to 0.08 hours (184 x 1390 ms to 184 x 1490 ms)
