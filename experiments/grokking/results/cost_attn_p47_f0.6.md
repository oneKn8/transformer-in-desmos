### attn, p=47, N_train=1325, d=32, heads=4, d_mlp=128, adamw

| op | 32x32 matmuls | native adds | elementwise elems |
|---|---:|---:|---:|
| X0,X1 = E + 1*P_j (row broadcast) [47x1]@[1x32] x2 | 4 | 0 | 0 |
|   + E [47x32] x2 | 0 | 4 | 0 |
| q = x2 W_Q [1x32]@[32x32] | 1 | 0 | 0 |
| Qblk = blockdiag(q)/sqrt(dh) (128 elems x 1 pass) | 0 | 0 | 128 |
| G = W_K Qblk [32x32]@[32x4] | 1 | 0 | 0 |
| s0,s1 = X0 G, X1 G (per-token scores) [47x32]@[32x4] x2 | 4 | 0 | 0 |
| s2 = x2 G [1x32]@[32x4] | 1 | 0 | 0 |
| V0,V1 = X0 W_V, X1 W_V [47x32]@[32x32] x2 | 4 | 0 | 0 |
| v2 = x2 W_V [1x32]@[32x32] | 1 | 0 | 0 |
| Sa,Sb = OH_a s0, OH_b s1 (gather) [1325x47]@[47x4] x2 | 168 | 84 | 0 |
| Va,Vb = OH_a V0, OH_b V1 (gather) [1325x47]@[47x32] x2 | 168 | 84 | 0 |
| softmax over 3 positions (exp, normalize) (15900 elems x 2 pass) | 0 | 0 | 31800 |
| expand alpha to features: alpha_j R [1325x4]@[4x32] x3 | 126 | 0 | 0 |
| z = Aa*Va + Ab*Vb + A2*v2 (42400 elems x 3 pass) | 0 | 0 | 127200 |
| z W_O [1325x32]@[32x32] | 42 | 0 | 0 |
|   + x2 (resid) [1325x32] | 0 | 42 | 0 |
| pre = r W_in [1325x32]@[32x128] | 168 | 0 | 0 |
|   + b_in [1325x128] | 0 | 168 | 0 |
| h = relu(pre) (169600 elems x 1 pass) | 0 | 0 | 169600 |
| h W_out [1325x128]@[128x32] | 168 | 126 | 0 |
|   + r + b_out [1325x32] x2 | 0 | 84 | 0 |
| logits = r2 W_U [1325x32]@[32x47] | 84 | 0 | 0 |
| dlogits = softmax - onehot (max, exp, normalize) (62275 elems x 3 pass) | 0 | 0 | 186825 |
|   row sums via ones [1325x47]@[47x1] | 84 | 42 | 0 |
| dW_U = r2^T dlogits [32x1325]@[1325x47] | 84 | 82 | 0 |
| dr2 = dlogits W_U^T [1325x47]@[47x32] | 84 | 42 | 0 |
| dW_out = h^T dr2 [128x1325]@[1325x32] | 168 | 164 | 0 |
| db_out = 1^T dr2 [1x1325]@[1325x32] | 42 | 41 | 0 |
| dh = dr2 W_out^T [1325x32]@[32x128] | 168 | 0 | 0 |
| dpre = dh * [pre>0] (169600 elems x 1 pass) | 0 | 0 | 169600 |
| dW_in = r^T dpre [32x1325]@[1325x128] | 168 | 164 | 0 |
| db_in = 1^T dpre [1x1325]@[1325x128] | 168 | 164 | 0 |
| dpre W_in^T [1325x128]@[128x32] | 168 | 126 | 0 |
|   dr = dr2 + ... [1325x32] | 0 | 42 | 0 |
| dW_O = z^T dr [32x1325]@[1325x32] | 42 | 41 | 0 |
| dz = dr W_O^T [1325x32]@[32x32] | 42 | 0 | 0 |
| dz * V_j for j=a,b,2 (42400 elems x 3 pass) | 0 | 0 | 127200 |
| dalpha_j = (dz*V_j) R^T [1325x32]@[32x4] x3 | 126 | 0 | 0 |
| dVa, dVb = Aa*dz, Ab*dz; A2*dz (42400 elems x 3 pass) | 0 | 0 | 127200 |
| dv2 = 1^T (A2*dz) [1x1325]@[1325x32] | 42 | 41 | 0 |
| softmax backward ds_j = a_j(da_j - sum a da) (15900 elems x 2 pass) | 0 | 0 | 31800 |
| dV0,dV1 = OH^T dVa, OH^T dVb (scatter) [47x1325]@[1325x32] x2 | 168 | 164 | 0 |
| ds0,ds1 = OH^T dSa, OH^T dSb (scatter) [47x1325]@[1325x4] x2 | 168 | 164 | 0 |
| ds2, dx2_resid = column sums [1x1325]@[1325x36] | 84 | 82 | 0 |
| dG = X0^T ds0 + X1^T ds1 [32x47]@[47x4] x2 | 4 | 2 | 0 |
| dG += x2^T ds2 [32x1]@[1x4] | 1 | 0 | 0 |
| dW_K = dG Qblk^T [32x4]@[4x32] | 1 | 0 | 0 |
| dQblk = W_K^T dG [32x32]@[32x4] | 1 | 0 | 0 |
| dq from dQblk (128 elems x 1 pass) | 0 | 0 | 128 |
| dW_Q = x2^T dq [32x1]@[1x32] | 1 | 0 | 0 |
| dx2 += dq W_Q^T + dv2 W_V^T + ds2 G^T [1x32]@[32x32] x3 | 3 | 0 | 0 |
| dW_V = X0^T dV0 + X1^T dV1 [32x47]@[47x32] x2 | 4 | 2 | 0 |
| dW_V += x2^T dv2 [32x1]@[1x32] | 1 | 0 | 0 |
| dX_j = ds_j G^T + dV_j W_V^T [47x32]@[32x32] x4 | 8 | 0 | 0 |
|   dE = dX0 + dX1 [47x32] | 0 | 2 | 0 |
| dP_j = column sums of dX_j [1x47]@[47x32] x2 | 4 | 2 | 0 |
| AdamW m, v, decay, step (22 blocks x 8 native ops) | 0 | 176 | 0 |
| AdamW g*g, sqrt, divide (15584 elems x 3 pass) | 0 | 0 | 46752 |
| **total** | **2774** | **2135** | **1018233** |

- matmuls: 2774 x 1.3 ms = 3606 ms
- native adds: 2135 x 0.041 ms = 87 ms (if adds cost a full matmul: 2776 ms)
- elementwise: 1018233 elems x 1.0 us = 1018 ms
- compute per step: 4711 ms = 3624 matmul-equivalents
- per tick with 50 to 150 ms overhead: 4761 to 4861 ms
- 156 steps to grok: 0.21 to 0.21 hours (156 x 4761 ms to 156 x 4861 ms)
