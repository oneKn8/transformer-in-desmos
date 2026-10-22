### attn, p=23, N_train=397, d=32, heads=4, d_mlp=128, adamw

| op | 32x32 matmuls | native adds | elementwise elems |
|---|---:|---:|---:|
| X0,X1 = E + 1*P_j (row broadcast) [23x1]@[1x32] x2 | 2 | 0 | 0 |
|   + E [23x32] x2 | 0 | 2 | 0 |
| q = x2 W_Q [1x32]@[32x32] | 1 | 0 | 0 |
| Qblk = blockdiag(q)/sqrt(dh) (128 elems x 1 pass) | 0 | 0 | 128 |
| G = W_K Qblk [32x32]@[32x4] | 1 | 0 | 0 |
| s0,s1 = X0 G, X1 G (per-token scores) [23x32]@[32x4] x2 | 2 | 0 | 0 |
| s2 = x2 G [1x32]@[32x4] | 1 | 0 | 0 |
| V0,V1 = X0 W_V, X1 W_V [23x32]@[32x32] x2 | 2 | 0 | 0 |
| v2 = x2 W_V [1x32]@[32x32] | 1 | 0 | 0 |
| Sa,Sb = OH_a s0, OH_b s1 (gather) [397x23]@[23x4] x2 | 26 | 0 | 0 |
| Va,Vb = OH_a V0, OH_b V1 (gather) [397x23]@[23x32] x2 | 26 | 0 | 0 |
| softmax over 3 positions (exp, normalize) (4764 elems x 2 pass) | 0 | 0 | 9528 |
| expand alpha to features: alpha_j R [397x4]@[4x32] x3 | 39 | 0 | 0 |
| z = Aa*Va + Ab*Vb + A2*v2 (12704 elems x 3 pass) | 0 | 0 | 38112 |
| z W_O [397x32]@[32x32] | 13 | 0 | 0 |
|   + x2 (resid) [397x32] | 0 | 13 | 0 |
| pre = r W_in [397x32]@[32x128] | 52 | 0 | 0 |
|   + b_in [397x128] | 0 | 52 | 0 |
| h = relu(pre) (50816 elems x 1 pass) | 0 | 0 | 50816 |
| h W_out [397x128]@[128x32] | 52 | 39 | 0 |
|   + r + b_out [397x32] x2 | 0 | 26 | 0 |
| logits = r2 W_U [397x32]@[32x23] | 13 | 0 | 0 |
| dlogits = softmax - onehot (max, exp, normalize) (9131 elems x 3 pass) | 0 | 0 | 27393 |
|   row sums via ones [397x23]@[23x1] | 13 | 0 | 0 |
| dW_U = r2^T dlogits [32x397]@[397x23] | 13 | 12 | 0 |
| dr2 = dlogits W_U^T [397x23]@[23x32] | 13 | 0 | 0 |
| dW_out = h^T dr2 [128x397]@[397x32] | 52 | 48 | 0 |
| db_out = 1^T dr2 [1x397]@[397x32] | 13 | 12 | 0 |
| dh = dr2 W_out^T [397x32]@[32x128] | 52 | 0 | 0 |
| dpre = dh * [pre>0] (50816 elems x 1 pass) | 0 | 0 | 50816 |
| dW_in = r^T dpre [32x397]@[397x128] | 52 | 48 | 0 |
| db_in = 1^T dpre [1x397]@[397x128] | 52 | 48 | 0 |
| dpre W_in^T [397x128]@[128x32] | 52 | 39 | 0 |
|   dr = dr2 + ... [397x32] | 0 | 13 | 0 |
| dW_O = z^T dr [32x397]@[397x32] | 13 | 12 | 0 |
| dz = dr W_O^T [397x32]@[32x32] | 13 | 0 | 0 |
| dz * V_j for j=a,b,2 (12704 elems x 3 pass) | 0 | 0 | 38112 |
| dalpha_j = (dz*V_j) R^T [397x32]@[32x4] x3 | 39 | 0 | 0 |
| dVa, dVb = Aa*dz, Ab*dz; A2*dz (12704 elems x 3 pass) | 0 | 0 | 38112 |
| dv2 = 1^T (A2*dz) [1x397]@[397x32] | 13 | 12 | 0 |
| softmax backward ds_j = a_j(da_j - sum a da) (4764 elems x 2 pass) | 0 | 0 | 9528 |
| dV0,dV1 = OH^T dVa, OH^T dVb (scatter) [23x397]@[397x32] x2 | 26 | 24 | 0 |
| ds0,ds1 = OH^T dSa, OH^T dSb (scatter) [23x397]@[397x4] x2 | 26 | 24 | 0 |
| ds2, dx2_resid = column sums [1x397]@[397x36] | 26 | 24 | 0 |
| dG = X0^T ds0 + X1^T ds1 [32x23]@[23x4] x2 | 2 | 0 | 0 |
| dG += x2^T ds2 [32x1]@[1x4] | 1 | 0 | 0 |
| dW_K = dG Qblk^T [32x4]@[4x32] | 1 | 0 | 0 |
| dQblk = W_K^T dG [32x32]@[32x4] | 1 | 0 | 0 |
| dq from dQblk (128 elems x 1 pass) | 0 | 0 | 128 |
| dW_Q = x2^T dq [32x1]@[1x32] | 1 | 0 | 0 |
| dx2 += dq W_Q^T + dv2 W_V^T + ds2 G^T [1x32]@[32x32] x3 | 3 | 0 | 0 |
| dW_V = X0^T dV0 + X1^T dV1 [32x23]@[23x32] x2 | 2 | 0 | 0 |
| dW_V += x2^T dv2 [32x1]@[1x32] | 1 | 0 | 0 |
| dX_j = ds_j G^T + dV_j W_V^T [23x32]@[32x32] x4 | 4 | 0 | 0 |
|   dE = dX0 + dX1 [23x32] | 0 | 1 | 0 |
| dP_j = column sums of dX_j [1x23]@[23x32] x2 | 2 | 0 | 0 |
| AdamW m, v, decay, step (20 blocks x 8 native ops) | 0 | 160 | 0 |
| AdamW g*g, sqrt, divide (14048 elems x 3 pass) | 0 | 0 | 42144 |
| **total** | **717** | **609** | **304817** |

- matmuls: 717 x 1.3 ms = 932 ms
- native adds: 609 x 0.041 ms = 25 ms (if adds cost a full matmul: 792 ms)
- elementwise: 304817 elems x 1.0 us = 305 ms
- compute per step: 1262 ms = 971 matmul-equivalents
- per tick with 50 to 150 ms overhead: 1312 to 1412 ms
- 149 steps to grok: 0.05 to 0.06 hours (149 x 1312 ms to 149 x 1412 ms)
