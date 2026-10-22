### attn, p=31, N_train=577, d=32, heads=4, d_mlp=128, adamw

| op | 32x32 matmuls | native adds | elementwise elems |
|---|---:|---:|---:|
| X0,X1 = E + 1*P_j (row broadcast) [31x1]@[1x32] x2 | 2 | 0 | 0 |
|   + E [31x32] x2 | 0 | 2 | 0 |
| q = x2 W_Q [1x32]@[32x32] | 1 | 0 | 0 |
| Qblk = blockdiag(q)/sqrt(dh) (128 elems x 1 pass) | 0 | 0 | 128 |
| G = W_K Qblk [32x32]@[32x4] | 1 | 0 | 0 |
| s0,s1 = X0 G, X1 G (per-token scores) [31x32]@[32x4] x2 | 2 | 0 | 0 |
| s2 = x2 G [1x32]@[32x4] | 1 | 0 | 0 |
| V0,V1 = X0 W_V, X1 W_V [31x32]@[32x32] x2 | 2 | 0 | 0 |
| v2 = x2 W_V [1x32]@[32x32] | 1 | 0 | 0 |
| Sa,Sb = OH_a s0, OH_b s1 (gather) [577x31]@[31x4] x2 | 38 | 0 | 0 |
| Va,Vb = OH_a V0, OH_b V1 (gather) [577x31]@[31x32] x2 | 38 | 0 | 0 |
| softmax over 3 positions (exp, normalize) (6924 elems x 2 pass) | 0 | 0 | 13848 |
| expand alpha to features: alpha_j R [577x4]@[4x32] x3 | 57 | 0 | 0 |
| z = Aa*Va + Ab*Vb + A2*v2 (18464 elems x 3 pass) | 0 | 0 | 55392 |
| z W_O [577x32]@[32x32] | 19 | 0 | 0 |
|   + x2 (resid) [577x32] | 0 | 19 | 0 |
| pre = r W_in [577x32]@[32x128] | 76 | 0 | 0 |
|   + b_in [577x128] | 0 | 76 | 0 |
| h = relu(pre) (73856 elems x 1 pass) | 0 | 0 | 73856 |
| h W_out [577x128]@[128x32] | 76 | 57 | 0 |
|   + r + b_out [577x32] x2 | 0 | 38 | 0 |
| logits = r2 W_U [577x32]@[32x31] | 19 | 0 | 0 |
| dlogits = softmax - onehot (max, exp, normalize) (17887 elems x 3 pass) | 0 | 0 | 53661 |
|   row sums via ones [577x31]@[31x1] | 19 | 0 | 0 |
| dW_U = r2^T dlogits [32x577]@[577x31] | 19 | 18 | 0 |
| dr2 = dlogits W_U^T [577x31]@[31x32] | 19 | 0 | 0 |
| dW_out = h^T dr2 [128x577]@[577x32] | 76 | 72 | 0 |
| db_out = 1^T dr2 [1x577]@[577x32] | 19 | 18 | 0 |
| dh = dr2 W_out^T [577x32]@[32x128] | 76 | 0 | 0 |
| dpre = dh * [pre>0] (73856 elems x 1 pass) | 0 | 0 | 73856 |
| dW_in = r^T dpre [32x577]@[577x128] | 76 | 72 | 0 |
| db_in = 1^T dpre [1x577]@[577x128] | 76 | 72 | 0 |
| dpre W_in^T [577x128]@[128x32] | 76 | 57 | 0 |
|   dr = dr2 + ... [577x32] | 0 | 19 | 0 |
| dW_O = z^T dr [32x577]@[577x32] | 19 | 18 | 0 |
| dz = dr W_O^T [577x32]@[32x32] | 19 | 0 | 0 |
| dz * V_j for j=a,b,2 (18464 elems x 3 pass) | 0 | 0 | 55392 |
| dalpha_j = (dz*V_j) R^T [577x32]@[32x4] x3 | 57 | 0 | 0 |
| dVa, dVb = Aa*dz, Ab*dz; A2*dz (18464 elems x 3 pass) | 0 | 0 | 55392 |
| dv2 = 1^T (A2*dz) [1x577]@[577x32] | 19 | 18 | 0 |
| softmax backward ds_j = a_j(da_j - sum a da) (6924 elems x 2 pass) | 0 | 0 | 13848 |
| dV0,dV1 = OH^T dVa, OH^T dVb (scatter) [31x577]@[577x32] x2 | 38 | 36 | 0 |
| ds0,ds1 = OH^T dSa, OH^T dSb (scatter) [31x577]@[577x4] x2 | 38 | 36 | 0 |
| ds2, dx2_resid = column sums [1x577]@[577x36] | 38 | 36 | 0 |
| dG = X0^T ds0 + X1^T ds1 [32x31]@[31x4] x2 | 2 | 0 | 0 |
| dG += x2^T ds2 [32x1]@[1x4] | 1 | 0 | 0 |
| dW_K = dG Qblk^T [32x4]@[4x32] | 1 | 0 | 0 |
| dQblk = W_K^T dG [32x32]@[32x4] | 1 | 0 | 0 |
| dq from dQblk (128 elems x 1 pass) | 0 | 0 | 128 |
| dW_Q = x2^T dq [32x1]@[1x32] | 1 | 0 | 0 |
| dx2 += dq W_Q^T + dv2 W_V^T + ds2 G^T [1x32]@[32x32] x3 | 3 | 0 | 0 |
| dW_V = X0^T dV0 + X1^T dV1 [32x31]@[31x32] x2 | 2 | 0 | 0 |
| dW_V += x2^T dv2 [32x1]@[1x32] | 1 | 0 | 0 |
| dX_j = ds_j G^T + dV_j W_V^T [31x32]@[32x32] x4 | 4 | 0 | 0 |
|   dE = dX0 + dX1 [31x32] | 0 | 1 | 0 |
| dP_j = column sums of dX_j [1x31]@[31x32] x2 | 2 | 0 | 0 |
| AdamW m, v, decay, step (20 blocks x 8 native ops) | 0 | 160 | 0 |
| AdamW g*g, sqrt, divide (14560 elems x 3 pass) | 0 | 0 | 43680 |
| **total** | **1035** | **825** | **439181** |

- matmuls: 1035 x 1.3 ms = 1346 ms
- native adds: 825 x 0.041 ms = 34 ms (if adds cost a full matmul: 1072 ms)
- elementwise: 439181 elems x 1.0 us = 439 ms
- compute per step: 1818 ms = 1399 matmul-equivalents
- per tick with 50 to 150 ms overhead: 1868 to 1968 ms
- 241 steps to grok: 0.13 to 0.13 hours (241 x 1868 ms to 241 x 1968 ms)
