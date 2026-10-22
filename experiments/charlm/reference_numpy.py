"""Pure-numpy float64 forward pass over the Desmos-ready weight set.

This file is the specification the Desmos build must reproduce, op by op:

  x      = tok_emb[ids] (+ pos_emb[:T])                       (T, d)
  per layer l:
    a    = nrm(x)                       nrm has NO gain/bias (folded below)
    q,k,v = a @ wq + bq, a @ wk + bk, a @ wv + bv            wq includes 1/sqrt(hd)
    per head h (columns h*hd:(h+1)*hd):
      e  = exp(q_h k_h^T (+ alibi_h)) * causal              causal = lower-triangular 0/1
      p  = e / rowsum(e)                                    no max-subtraction
      o_h = p @ v_h
    x    = x + concat(o_h) @ wo + bo
    x    = x + act(nrm(x) @ w1 + b1) @ w2 + b2
  logits = nrm(x) @ w_out + b_out

  rms: nrm(x) = x / sqrt(mean(x^2) + eps)
  ln:  nrm(x) = (x - mean(x)) / sqrt(mean((x - mean(x))^2) + eps)
  gelu_tanh(z) = 0.5 z (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3)))
  relu(z) = (z + |z|) / 2   (exactly max(z, 0) in floating point)

Biases (b*) appear only when the trained model had biases or LayerNorm (whose
beta folds into the next projection as a bias).

Folding (exact algebra, done in float64 before rounding):
  nrm(x) * g + beta  then  @ W + b   ==   nrm(x) @ (diag(g) W) + (beta @ W + b)
and the attention scale 1/sqrt(hd) is multiplied into wq and bq.
"""

from __future__ import annotations

import math

import numpy as np

from model import CharTransformer, ModelConfig, alibi_bias

GELU_C = math.sqrt(2.0 / math.pi)
EXP_LIMIT = 700.0  # exp overflows float64 above ~709.78


def round_sig(a: np.ndarray, digits: int) -> np.ndarray:
    """Round to `digits` significant decimal digits. Each result is the float64
    nearest that short decimal, so printing it with format(w, f'.{digits}g') and
    parsing it back gives the identical float64."""
    if digits < 1:
        raise ValueError("digits must be >= 1")
    flat = np.asarray(a, dtype=np.float64).ravel()
    out = np.array([float(format(w, f".{digits}g")) for w in flat], dtype=np.float64)
    return out.reshape(np.shape(a))


def trained_weights(model: CharTransformer) -> dict[str, np.ndarray]:
    """Unfolded float64 weights with descriptive names (x @ W convention)."""
    cfg = model.cfg
    sd = {k: v.detach().double().numpy().copy() for k, v in model.state_dict().items()}
    w = {"tok_emb": sd["tok_emb"]}
    if cfg.pos == "learned":
        w["pos_emb"] = sd["pos_emb"]
    for i in range(cfg.n_layer):
        for name in ("wq", "wk", "wv", "wo", "w1", "w2"):
            w[f"l{i}_{name}"] = sd[f"blocks.{i}.{name}.weight"]
            if cfg.bias:
                w[f"l{i}_{name}_b"] = sd[f"blocks.{i}.{name}.bias"]
        for n in ("norm1", "norm2"):
            if cfg.norm != "none":
                w[f"l{i}_{n}_g"] = sd[f"blocks.{i}.{n}.weight"]
            if cfg.norm == "ln":
                w[f"l{i}_{n}_b"] = sd[f"blocks.{i}.{n}.bias"]
    if cfg.norm != "none":
        w["normf_g"] = sd["norm_f.weight"]
    if cfg.norm == "ln":
        w["normf_b"] = sd["norm_f.bias"]
    if cfg.tied:
        w["w_out"] = sd["tok_emb"].T.copy()
    else:
        w["w_out"] = sd["head.weight"]
        if cfg.bias:
            w["w_out_b"] = sd["head.bias"]
    return w


def _fold(g: np.ndarray | None, beta: np.ndarray | None, mat: np.ndarray, bias: np.ndarray | None, scale: float):
    new_w = mat if g is None else g[:, None] * mat
    new_b = None
    if beta is not None:
        new_b = beta @ mat
    if bias is not None:
        new_b = bias if new_b is None else new_b + bias
    new_w = new_w * scale
    return new_w, (None if new_b is None else new_b * scale)


def fold_weights(w: dict[str, np.ndarray], cfg: ModelConfig) -> dict[str, np.ndarray]:
    """Fold norm gains/betas and the attention scale into the projections."""
    out = {"tok_emb": w["tok_emb"].copy()}
    if "pos_emb" in w:
        out["pos_emb"] = w["pos_emb"].copy()
    scale_q = 1.0 / math.sqrt(cfg.head_dim)
    for i in range(cfg.n_layer):
        g1, b1 = w.get(f"l{i}_norm1_g"), w.get(f"l{i}_norm1_b")
        g2, b2 = w.get(f"l{i}_norm2_g"), w.get(f"l{i}_norm2_b")
        plan = [("wq", g1, b1, scale_q), ("wk", g1, b1, 1.0), ("wv", g1, b1, 1.0),
                ("wo", None, None, 1.0), ("w1", g2, b2, 1.0), ("w2", None, None, 1.0)]
        for name, g, beta, s in plan:
            mat, bias = _fold(g, beta, w[f"l{i}_{name}"], w.get(f"l{i}_{name}_b"), s)
            out[f"l{i}_{name}"] = mat
            if bias is not None:
                out[f"l{i}_{name}_b"] = bias
    mat, bias = _fold(w.get("normf_g"), w.get("normf_b"), w["w_out"], w.get("w_out_b"), 1.0)
    out["w_out"] = mat
    if bias is not None:
        out["w_out_b"] = bias
    return out


def _nrm(x: np.ndarray, kind: str, eps: float) -> np.ndarray:
    if kind == "none":
        return x
    if kind == "rms":
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    c = x - np.mean(x, axis=-1, keepdims=True)
    return c / np.sqrt(np.mean(c * c, axis=-1, keepdims=True) + eps)


def _act(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return (z + np.abs(z)) / 2.0
    return 0.5 * z * (1.0 + np.tanh(GELU_C * (z + 0.044715 * z**3)))


def _lin(x: np.ndarray, fw: dict[str, np.ndarray], key: str) -> np.ndarray:
    y = x @ fw[key]
    b = fw.get(key + "_b")
    return y if b is None else y + b


def forward(fw: dict[str, np.ndarray], cfg: ModelConfig, ids: np.ndarray, trace: dict | None = None) -> np.ndarray:
    """ids: (T,) or (B, T) ints, T <= ctx. fw: folded weights. Returns logits."""
    ids = np.asarray(ids)
    squeeze = ids.ndim == 1
    if squeeze:
        ids = ids[None, :]
    t = ids.shape[1]
    if t > cfg.ctx:
        raise ValueError(f"sequence length {t} exceeds ctx {cfg.ctx}")
    h, hd = cfg.n_head, cfg.head_dim
    causal = np.tril(np.ones((t, t)))
    alibi = alibi_bias(h, cfg.ctx).double().numpy()[:, :t, :t] if cfg.pos == "alibi" else None

    x = fw["tok_emb"][ids]
    if "pos_emb" in fw:
        x = x + fw["pos_emb"][:t]
    if trace is not None:
        trace["x_embed"] = x
    for i in range(cfg.n_layer):
        a = _nrm(x, cfg.norm, cfg.norm_eps)
        q, k, v = (_lin(a, fw, f"l{i}_{n}") for n in ("wq", "wk", "wv"))
        heads, probs = [], []
        for j in range(h):
            sl = slice(j * hd, (j + 1) * hd)
            s = q[..., sl] @ np.swapaxes(k[..., sl], -1, -2)
            if alibi is not None:
                s = s + alibi[j]
            if np.max(np.abs(s)) > EXP_LIMIT:
                raise FloatingPointError("attention score too large for exp without max-subtraction")
            e = np.exp(s) * causal
            p = e / e.sum(axis=-1, keepdims=True)
            probs.append(p)
            heads.append(p @ v[..., sl])
        x = x + _lin(np.concatenate(heads, axis=-1), fw, f"l{i}_wo")
        if trace is not None:
            trace[f"l{i}_attn_probs"] = np.stack(probs, axis=1)
            trace[f"l{i}_resid_mid"] = x
        m = _act(_lin(_nrm(x, cfg.norm, cfg.norm_eps), fw, f"l{i}_w1"), cfg.act)
        x = x + _lin(m, fw, f"l{i}_w2")
        if trace is not None:
            trace[f"l{i}_resid_out"] = x
    xf = _nrm(x, cfg.norm, cfg.norm_eps)
    logits = _lin(xf, fw, "w_out")
    if trace is not None:
        trace["x_final_norm"] = xf
        trace["logits"] = logits
        if squeeze:
            for key in list(trace):
                trace[key] = trace[key][0]
    return logits[0] if squeeze else logits


def split_bpc_numpy(fw: dict[str, np.ndarray], cfg: ModelConfig, data: np.ndarray, batch: int = 512) -> float:
    """Same metric as train.split_bpc(mode='ctx'): stride T/2, score positions >= T/2."""
    t = cfg.ctx
    stride, score_from = t // 2, t - t // 2
    n_win = (len(data) - (t + 1)) // stride + 1
    if n_win < 1:
        raise ValueError("split shorter than one window")
    idx = np.arange(n_win)[:, None] * stride + np.arange(t + 1)[None, :]
    win = data[idx]
    total, count = 0.0, 0
    for i in range(0, n_win, batch):
        x, y = win[i : i + batch, :-1], win[i : i + batch, 1:]
        logits = forward(fw, cfg, x)[:, score_from:]
        mx = logits.max(axis=-1, keepdims=True)
        logz = mx[..., 0] + np.log(np.exp(logits - mx).sum(axis=-1))
        tgt = y[:, score_from:]
        picked = np.take_along_axis(logits, tgt[..., None], axis=-1)[..., 0]
        total += float((logz - picked).sum())
        count += tgt.size
    return total / count / math.log(2.0)
