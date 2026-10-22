"""Estimated Desmos cost of generating one token with a CharTransformer.

Measured engine facts (2026-10-21, Santo's laptop):
  * one 32x32 @ 32x32 matmul ~ 1.3 ms
  * an elementwise op done by an index comprehension ~ 1 us per element
  * ~ 50 ms fixed overhead per evaluation round trip
Matrices cap at 32x32, so a matmul (m x k) @ (k x n) is executed as
ceil(m/32) * ceil(k/32) * ceil(n/32) block matmuls.

Only the full 32x32 @ 32x32 case was measured, so two accountings are reported:
  padded: every block call costs 1.3 ms even if an operand is smaller than 32
          (conservative; right if per-call overhead dominates).
  mac:    cost proportional to multiply-adds, m*k*n / 32^3 block-equivalents
          (optimistic lower bound; right if arithmetic dominates).
The true cost lies between them. Elementwise and fixed costs are the same in both.

Assumed implementation (exact algebraic identities, no approximation):
  * norm gains folded into the next weight matrix; RMSNorm is then 2 passes
    (row sum of squares, scale), LayerNorm 4 (mean-subtract, variance, scale,
    beta add); 1/sqrt(head_dim) folded into W_q.
  * causal mask fused into the exp pass; softmax = 3 passes (exp*mask, row sum,
    divide); no max-subtraction (scores are small, Desmos uses float64).
  * ALiBi (pos="alibi") is free: exp(S + B_h) * mask = exp(S) * C_h with C_h a
    precomputed constant, so it rides in the existing exp*mask pass.
  * per-head Q/K/V slices and the head concat are comprehensions (1 pass each),
    except when head_dim is a multiple of 32: then every head already is a set of
    whole 32-column blocks, so slicing and concat are free (W_o is split into
    the matching row blocks).
  * full recompute evaluates every layer on all T rows of the window but the
    final norm and unembedding only on the last row.
  * kv-cached processes one new row; each layer's cached K and V (T x d each)
    are rewritten once per token (Desmos actions replace whole values).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from model import ModelConfig

BLOCK = 32
MS_PER_BLOCK_MATMUL = 1.3
MS_PER_ELEMENT = 0.001
FIXED_MS = 50.0
NORM_PASSES = {"rms": 2, "ln": 4, "none": 0}
MODES = ("full", "kv")


def padded_blocks(m: int, k: int, n: int) -> int:
    return math.ceil(m / BLOCK) * math.ceil(k / BLOCK) * math.ceil(n / BLOCK)


def mac_blocks(m: int, k: int, n: int) -> float:
    return m * k * n / BLOCK**3


@dataclass
class Cost:
    mode: str
    padded_blocks: int = 0
    mac_blocks: float = 0.0
    elements: int = 0
    lines: list[tuple[str, int, float, int]] = field(default_factory=list)

    def matmul(self, label: str, m: int, k: int, n: int, times: int = 1) -> None:
        pb, mb = padded_blocks(m, k, n) * times, mac_blocks(m, k, n) * times
        self.padded_blocks += pb
        self.mac_blocks += mb
        self.lines.append((f"{label}: {times} x ({m}x{k} @ {k}x{n})", pb, mb, 0))

    def elementwise(self, label: str, n_elements: int) -> None:
        self.elements += n_elements
        self.lines.append((label, 0, 0.0, n_elements))

    @property
    def ms_padded(self) -> float:
        return FIXED_MS + MS_PER_BLOCK_MATMUL * self.padded_blocks + MS_PER_ELEMENT * self.elements

    @property
    def ms_mac(self) -> float:
        return FIXED_MS + MS_PER_BLOCK_MATMUL * self.mac_blocks + MS_PER_ELEMENT * self.elements

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "padded_blocks": self.padded_blocks,
            "mac_blocks": round(self.mac_blocks, 3),
            "elements": self.elements,
            "ms_padded": round(self.ms_padded, 1),
            "ms_mac": round(self.ms_mac, 1),
        }

    def arithmetic(self) -> str:
        """Human-readable derivation, one op per line, then the totals."""
        out = [f"mode={self.mode}"]
        for label, pb, mb, el in self.lines:
            if el:
                out.append(f"  {label}: {el} elements")
            else:
                out.append(f"  {label}: {pb} padded blocks, {mb:.3f} mac blocks")
        out.append(
            f"  total: {self.padded_blocks} padded blocks, {self.mac_blocks:.2f} mac blocks, "
            f"{self.elements} elements"
        )
        out.append(
            f"  padded ms = {FIXED_MS:g} + {MS_PER_BLOCK_MATMUL} * {self.padded_blocks} + "
            f"{MS_PER_ELEMENT} * {self.elements} = {self.ms_padded:.1f}"
        )
        out.append(
            f"  mac ms    = {FIXED_MS:g} + {MS_PER_BLOCK_MATMUL} * {self.mac_blocks:.2f} + "
            f"{MS_PER_ELEMENT} * {self.elements} = {self.ms_mac:.1f}"
        )
        return "\n".join(out)


def token_cost(cfg: ModelConfig, mode: str) -> Cost:
    """Cost of one generated token with the window full (T = cfg.ctx tokens)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    d, t, h, hd, v = cfg.d_model, cfg.ctx, cfg.n_head, cfg.head_dim, cfg.vocab_size
    f = cfg.mlp_mult * d
    rows = t if mode == "full" else 1
    norm_passes = NORM_PASSES[cfg.norm]
    block_aligned_heads = hd % BLOCK == 0
    c = Cost(mode=mode)

    c.elementwise("embedding gather + position add", rows * d)
    for layer in range(cfg.n_layer):
        p = f"L{layer} "
        if norm_passes:
            c.elementwise(p + f"norm1 ({norm_passes} passes)", norm_passes * rows * d)
        c.matmul(p + "Q,K,V projections", rows, d, d, times=3)
        if not block_aligned_heads:
            c.elementwise(p + "per-head Q,K,V slices", 3 * rows * d)
        c.matmul(p + "scores Q_h K_h^T", rows, hd, t, times=h)
        c.elementwise(p + "softmax (exp*mask, row sum, divide)", 3 * h * rows * t)
        c.matmul(p + "probs @ V_h", rows, t, hd, times=h)
        if not block_aligned_heads:
            c.elementwise(p + "head concat", rows * d)
        c.matmul(p + "W_o", rows, d, d)
        c.elementwise(p + "residual add", rows * d)
        if norm_passes:
            c.elementwise(p + f"norm2 ({norm_passes} passes)", norm_passes * rows * d)
        c.matmul(p + "MLP up W_1", rows, d, f)
        c.elementwise(p + f"activation ({cfg.act})", rows * f)
        c.matmul(p + "MLP down W_2", rows, f, d)
        c.elementwise(p + "residual add", rows * d)
        if cfg.bias:
            c.elementwise(p + "bias adds (q,k,v,o,w1,w2)", rows * (5 * d + f))
        if mode == "kv":
            c.elementwise(p + "rewrite cached K and V", 2 * t * d)
    if norm_passes:
        c.elementwise(f"final norm on last row ({norm_passes} passes)", norm_passes * d)
    c.matmul("unembedding (last row)", 1, d, v)
    if cfg.bias and not cfg.tied:
        c.elementwise("unembedding bias", v)
    c.elementwise("sampling (scale, exp, sum, cdf)", 4 * v)
    return c


def cost_summary(cfg: ModelConfig) -> dict:
    full, kv = token_cost(cfg, "full"), token_cost(cfg, "kv")
    return {"full": full.summary(), "kv": kv.summary()}
