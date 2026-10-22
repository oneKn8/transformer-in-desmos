### attn, p=31, N_train=480, d=32, heads=4, d_mlp=128, adamw

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
| Sa,Sb = OH_a s0, OH_b s1 (gather) [480x31]@[31x4] x2 | 30 | 0 | 0 |
| Va,Vb = OH_a V0, OH_b V1 (gather) [480x31]@[31x32] x2 | 30 | 0 | 0 |
| softmax over 3 positions (exp, normalize) (5760 elems x 2 pass) | 0 | 0 | 11520 |
| expand alpha to features: alpha_j R [480x4]@[4x32] x3 | 45 | 0 | 0 |
| z = Aa*Va + Ab*Vb + A2*v2 (15360 elems x 3 pass) | 0 | 0 | 46080 |
| z W_O [480x32]@[32x32] | 15 | 0 | 0 |
|   + x2 (resid) [480x32] | 0 | 15 | 0 |
| pre = r W_in [480x32]@[32x128] | 60 | 0 | 0 |
|   + b_in [480x128] | 0 | 60 | 0 |
| h = relu(pre) (61440 elems x 1 pass) | 0 | 0 | 61440 |
| h W_out [480x128]@[128x32] | 60 | 45 | 0 |
|   + r + b_out [480x32] x2 | 0 | 30 | 0 |
| logits = r2 W_U [480x32]@[32x31] | 15 | 0 | 0 |
| dlogits = softmax - onehot (max, exp, normalize) (14880 elems x 3 pass) | 0 | 0 | 44640 |
|   row sums via ones [480x31]@[31x1] | 15 | 0 | 0 |
| dW_U = r2^T dlogits [32x480]@[480x31] | 15 | 14 | 0 |
| dr2 = dlogits W_U^T [480x31]@[31x32] | 15 | 0 | 0 |
| dW_out = h^T dr2 [128x480]@[480x32] | 60 | 56 | 0 |
| db_out = 1^T dr2 [1x480]@[480x32] | 15 | 14 | 0 |
| dh = dr2 W_out^T [480x32]@[32x128] | 60 | 0 | 0 |
| dpre = dh * [pre>0] (61440 elems x 1 pass) | 0 | 0 | 61440 |
| dW_in = r^T dpre [32x480]@[480x128] | 60 | 56 | 0 |
| db_in = 1^T dpre [1x480]@[480x128] | 60 | 56 | 0 |
| dpre W_in^T [480x128]@[128x32] | 60 | 45 | 0 |
|   dr = dr2 + ... [480x32] | 0 | 15 | 0 |
| dW_O = z^T dr [32x480]@[480x32] | 15 | 14 | 0 |
| dz = dr W_O^T [480x32]@[32x32] | 15 | 0 | 0 |
| dz * V_j for j=a,b,2 (15360 elems x 3 pass) | 0 | 0 | 46080 |
| dalpha_j = (dz*V_j) R^T [480x32]@[32x4] x3 | 45 | 0 | 0 |
| dVa, dVb = Aa*dz, Ab*dz; A2*dz (15360 elems x 3 pass) | 0 | 0 | 46080 |
| dv2 = 1^T (A2*dz) [1x480]@[480x32] | 15 | 14 | 0 |
| softmax backward ds_j = a_j(da_j - sum a da) (5760 elems x 2 pass) | 0 | 0 | 11520 |
| dV0,dV1 = OH^T dVa, OH^T dVb (scatter) [31x480]@[480x32] x2 | 30 | 28 | 0 |
| ds0,ds1 = OH^T dSa, OH^T dSb (scatter) [31x480]@[480x4] x2 | 30 | 28 | 0 |
| ds2, dx2_resid = column sums [1x480]@[480x36] | 30 | 28 | 0 |
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
| **total** | **823** | **681** | **372736** |

- matmuls: 823 x 1.3 ms = 1070 ms
- native adds: 681 x 0.041 ms = 28 ms (if adds cost a full matmul: 885 ms)
- elementwise: 372736 elems x 1.0 us = 373 ms
- compute per step: 1470 ms = 1131 matmul-equivalents
- per tick with 50 to 150 ms overhead: 1520 to 1620 ms
- 1151 steps to grok: 0.49 to 0.52 hours (1151 x 1520 ms to 1151 x 1620 ms)
