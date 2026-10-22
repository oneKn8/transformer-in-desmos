"""The training step the Desmos graph will run, in numpy, in the same structured form.

Model: the one-layer transformer of experiments/grokking/grok.py (AttnModel, no LayerNorm,
ReLU, d=32, 4 heads of 8, MLP 128) trained full batch with AdamW on (a+b) mod 31.

Two facts make the Desmos version cheap, and this file uses them the same way:
  1. The third token is always "=", so its residual c3, the query q and the third key and
     value are the same for every example.
  2. Position 1 only sees token a and position 2 only token b, so their keys and values are
     31-row tables (one row per token) instead of per-example products.
All 31 x 31 pairs are laid out as 31 blocks, block a holding the pairs (a, b) in row b. That
is the shape of a Desmos list of 31 matrices, and the grid of correct cells is the picture.

The backward pass is written out by hand (Desmos has no autodiff that scales), and
test_ref_step.py checks it step for step against torch autograd on grok.py's model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

P = 31          # modulus; tokens 0..30 are numbers, 31 is "="
D = 32
H = 4
DH = D // H
M = 128         # default MLP width (grok.py's recommended run); the step itself reads widths from w
BETA1, BETA2, EPS = 0.9, 0.98, 1e-8


def param_shapes(m: int = M) -> dict:
    return {
        "W_E": (P + 1, D), "W_pos": (3, D), "W_Q": (D, D), "W_K": (D, D), "W_V": (D, D),
        "W_O": (D, D), "W_in": (D, m), "b_in": (m,), "W_out": (m, D), "b_out": (D,), "W_U": (D, P),
    }


PARAM_SHAPES = param_shapes()
# head h owns columns DH*h .. DH*h+DH-1; HSUM sums a 32-vector into its 4 head totals
HSUM = np.kron(np.eye(H), np.ones((DH, 1)))  # (32, 4)


@dataclass
class Grid:
    """Which of the 31 x 31 pairs are in the training split, and their labels."""
    train: np.ndarray   # (31, 31) bool, [a, b]
    label: np.ndarray   # (31, 31) int, (a + b) % 31

    @property
    def n_train(self) -> int:
        return int(self.train.sum())


def make_grid(train_pairs: np.ndarray) -> Grid:
    """train_pairs: (n, 2) array of (a, b)."""
    train = np.zeros((P, P), dtype=bool)
    train[train_pairs[:, 0], train_pairs[:, 1]] = True
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    return Grid(train=train, label=(a + b) % P)


def forward(w: dict, grid: Grid) -> dict:
    """Everything the backward pass needs. Arrays indexed [a, b, ...] are per pair."""
    c3 = w["W_E"][P] + w["W_pos"][2]                       # (32,)  the "=" residual
    t1 = w["W_E"][:P] + w["W_pos"][0]                      # (31, 32) row a
    t2 = w["W_E"][:P] + w["W_pos"][1]                      # (31, 32) row b
    q = c3 @ w["W_Q"]
    k1, k2, k3 = t1 @ w["W_K"], t2 @ w["W_K"], c3 @ w["W_K"]
    v1, v2, v3 = t1 @ w["W_V"], t2 @ w["W_V"], c3 @ w["W_V"]
    scale = 1.0 / math.sqrt(DH)
    s1 = (k1 * q) @ HSUM * scale                           # (31, 4) score of position 1, per a
    s2 = (k2 * q) @ HSUM * scale                           # (31, 4) per b
    s3 = (k3 * q) @ HSUM * scale                           # (4,)
    # softmax over the three positions for every pair; subtracting the max is exact
    e1 = s1[:, None, :]                                    # (31, 1, 4)
    e2 = s2[None, :, :]                                    # (1, 31, 4)
    mx = np.maximum(np.maximum(e1, e2), s3)
    x1, x2, x3 = np.exp(e1 - mx), np.exp(e2 - mx), np.exp(s3 - mx)
    zsum = x1 + x2 + x3
    al1, al2, al3 = x1 / zsum, x2 / zsum, x3 / zsum        # each (31, 31, 4)
    rep = HSUM.T                                           # (4, 32): head weight -> its 8 columns
    z = (al1 @ rep) * v1[:, None, :] + (al2 @ rep) * v2[None, :, :] + (al3 @ rep) * v3
    r = c3 + z @ w["W_O"]                                  # (31, 31, 32)
    u = r @ w["W_in"] + w["b_in"]
    hid = np.maximum(u, 0.0)
    r2 = r + hid @ w["W_out"] + w["b_out"]
    logits = r2 @ w["W_U"]                                 # (31, 31, 31)
    return dict(c3=c3, t1=t1, t2=t2, q=q, k1=k1, k2=k2, k3=k3, v1=v1, v2=v2, v3=v3,
                al1=al1, al2=al2, al3=al3, z=z, r=r, u=u, hid=hid, r2=r2, logits=logits)


def loss_and_dlogits(logits: np.ndarray, grid: Grid) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean cross entropy over the training pairs, its gradient, and per-pair correctness."""
    mx = logits.max(-1, keepdims=True)
    ex = np.exp(logits - mx)
    prob = ex / ex.sum(-1, keepdims=True)
    onehot = np.eye(P)[grid.label]
    logp = np.log(prob[np.arange(P)[:, None], np.arange(P)[None, :], grid.label])
    loss = -float(logp[grid.train].sum()) / grid.n_train
    dlogits = (prob - onehot) * grid.train[..., None] / grid.n_train
    correct = logits.argmax(-1) == grid.label
    return loss, dlogits, correct


def flat(x: np.ndarray) -> np.ndarray:
    """(31, 31, n) per-pair array -> (961, n) rows, for sums over all pairs."""
    return x.reshape(-1, x.shape[-1])


def backward(w: dict, f: dict, dlogits: np.ndarray) -> dict:
    g = {}
    g["W_U"] = flat(f["r2"]).T @ flat(dlogits)
    dr2 = dlogits @ w["W_U"].T
    g["W_out"] = flat(f["hid"]).T @ flat(dr2)
    g["b_out"] = dr2.sum((0, 1))
    du = (dr2 @ w["W_out"].T) * (f["u"] > 0)
    g["W_in"] = flat(f["r"]).T @ flat(du)
    g["b_in"] = du.sum((0, 1))
    dr = dr2 + du @ w["W_in"].T
    g["W_O"] = flat(f["z"]).T @ flat(dr)
    dz = dr @ w["W_O"].T                                   # (31, 31, 32)
    dc3 = dr.sum((0, 1))                                   # r = c3 + ...

    rep = HSUM.T
    v1b, v2b = f["v1"][:, None, :], f["v2"][None, :, :]
    # attention weights: z = sum_j (al_j @ rep) * v_j
    dal1 = (dz * v1b) @ HSUM                               # (31, 31, 4)
    dal2 = (dz * v2b) @ HSUM
    dal3 = (dz * f["v3"]) @ HSUM
    dv1 = ((f["al1"] @ rep) * dz).sum(1)                   # (31, 32) per token a
    dv2 = ((f["al2"] @ rep) * dz).sum(0)                   # per token b
    dv3 = ((f["al3"] @ rep) * dz).sum((0, 1))
    # softmax over positions: ds_j = al_j (dal_j - sum_k al_k dal_k)
    dot = f["al1"] * dal1 + f["al2"] * dal2 + f["al3"] * dal3
    ds1 = (f["al1"] * (dal1 - dot)).sum(1)                 # (31, 4)
    ds2 = (f["al2"] * (dal2 - dot)).sum(0)
    ds3 = (f["al3"] * (dal3 - dot)).sum((0, 1))
    scale = 1.0 / math.sqrt(DH)
    q = f["q"]
    dk1 = (ds1 @ rep) * q * scale                          # (31, 32)
    dk2 = (ds2 @ rep) * q * scale
    dk3 = (ds3 @ rep) * q * scale
    dq = ((ds1 @ rep) * f["k1"]).sum(0) * scale + ((ds2 @ rep) * f["k2"]).sum(0) * scale \
        + (ds3 @ rep) * f["k3"] * scale

    t1, t2, c3 = f["t1"], f["t2"], f["c3"]
    g["W_Q"] = np.outer(c3, dq)
    g["W_K"] = t1.T @ dk1 + t2.T @ dk2 + np.outer(c3, dk3)
    g["W_V"] = t1.T @ dv1 + t2.T @ dv2 + np.outer(c3, dv3)
    dt1 = dk1 @ w["W_K"].T + dv1 @ w["W_V"].T
    dt2 = dk2 @ w["W_K"].T + dv2 @ w["W_V"].T
    dc3 = dc3 + dq @ w["W_Q"].T + dk3 @ w["W_K"].T + dv3 @ w["W_V"].T
    g["W_E"] = np.zeros_like(w["W_E"])
    g["W_E"][:P] = dt1 + dt2
    g["W_E"][P] = dc3
    g["W_pos"] = np.stack([dt1.sum(0), dt2.sum(0), dc3])
    return g


def adamw(w: dict, g: dict, m: dict, v: dict, step: int, lr: float, wd: float, warmup: int = 10) -> float:
    """torch.optim.AdamW with grok.py's linear warmup; step counts from 0. Updates in place."""
    lr_t = lr * min((step + 1) / max(warmup, 1), 1.0)
    t = step + 1
    step_size = lr_t / (1 - BETA1 ** t)
    bc2_sqrt = math.sqrt(1 - BETA2 ** t)
    for k in w:
        # same operation order as torch's single-tensor AdamW, so rounding stays close
        w[k] *= 1.0 - lr_t * wd
        m[k] = m[k] + (1 - BETA1) * (g[k] - m[k])
        v[k] = v[k] * BETA2 + (1 - BETA2) * g[k] * g[k]
        w[k] -= step_size * m[k] / (np.sqrt(v[k]) / bc2_sqrt + EPS)
    return lr_t
