"""Character-level decoder-only transformer, restricted to ops that are cheap to
hand-write as Desmos expressions.

Ops used: embedding row lookup, learned absolute position embedding (or ALiBi
attention biases, which survive a sliding window under KV caching), pre-norm
(RMSNorm, LayerNorm, or none), multi-head causal attention, a 2-layer MLP with
GELU (tanh form, since Desmos documents tanh but not erf) or ReLU, residual adds,
and a final linear unembedding (tied to the token embedding or untied).
Linear layers store weights as (in, out) and compute x @ W, the row-vector
convention the Desmos build will use, so exported matrices need no transpose.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

NORMS = ("rms", "ln", "none")
ACTS = ("gelu_tanh", "relu")
POS = ("learned", "alibi")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    ctx: int
    d_model: int
    n_layer: int
    n_head: int
    mlp_mult: int = 4
    norm: str = "rms"
    act: str = "gelu_tanh"
    tied: bool = False
    bias: bool = False
    pos: str = "learned"
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if self.norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        if self.act not in ACTS:
            raise ValueError(f"act must be one of {ACTS}")
        if self.pos not in POS:
            raise ValueError(f"pos must be one of {POS}")
        for name in ("vocab_size", "ctx", "d_model", "n_layer", "n_head", "mlp_mult"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


class Linear(nn.Module):
    """y = x @ W (+ b), W stored as (in, out)."""

    def __init__(self, d_in: int, d_out: int, bias: bool, std: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, d_out) * std)
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight
        return y + self.bias if self.bias is not None else y


class Norm(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.kind = cfg.norm
        self.eps = cfg.norm_eps
        if self.kind != "none":
            self.weight = nn.Parameter(torch.ones(cfg.d_model))
        if self.kind == "ln":
            self.bias = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return x
        if self.kind == "rms":
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        mu = x.mean(-1, keepdim=True)
        var = (x - mu).pow(2).mean(-1, keepdim=True)
        return (x - mu) * torch.rsqrt(var + self.eps) * self.weight + self.bias


def alibi_bias(n_head: int, ctx: int) -> torch.Tensor:
    """(H, T, T) additive attention bias: -slope_h * (i - j), slopes 2^(-8(h+1)/H)
    (Press et al. 2022). Causal masking is applied separately."""
    slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / n_head) for h in range(n_head)])
    pos = torch.arange(ctx, dtype=torch.float32)
    dist = (pos[:, None] - pos[None, :]).clamp(min=0)
    return -slopes[:, None, None] * dist


def activation(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(x)
    return F.gelu(x, approximate="tanh")


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.idx = idx
        d, std = cfg.d_model, 0.02
        resid_std = std / math.sqrt(2 * cfg.n_layer)
        self.norm1 = Norm(cfg)
        self.wq = Linear(d, d, cfg.bias, std)
        self.wk = Linear(d, d, cfg.bias, std)
        self.wv = Linear(d, d, cfg.bias, std)
        self.wo = Linear(d, d, cfg.bias, resid_std)
        self.norm2 = Norm(cfg)
        self.w1 = Linear(d, cfg.mlp_mult * d, cfg.bias, std)
        self.w2 = Linear(cfg.mlp_mult * d, d, cfg.bias, resid_std)
        if cfg.pos == "alibi":
            self.register_buffer("alibi", alibi_bias(cfg.n_head, cfg.ctx), persistent=False)

    def forward(self, x: torch.Tensor, trace: dict | None) -> torch.Tensor:
        cfg = self.cfg
        b, t, d = x.shape
        h, hd = cfg.n_head, cfg.head_dim
        a = self.norm1(x)
        q = self.wq(a).view(b, t, h, hd).transpose(1, 2)
        k = self.wk(a).view(b, t, h, hd).transpose(1, 2)
        v = self.wv(a).view(b, t, h, hd).transpose(1, 2)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        if trace is None:
            # Fused kernel, 2-2.5x faster on CPU; equal to the explicit path
            # below to ~1e-15 in float64 (pinned by test_charlm.py).
            if cfg.pos == "alibi":
                mask = self.alibi[:, :t, :t].to(q.dtype).masked_fill(~causal, float("-inf"))
                heads = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            else:
                heads = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            if cfg.pos == "alibi":
                scores = scores + self.alibi[:, :t, :t].to(q.dtype)
            probs = scores.masked_fill(~causal, float("-inf")).softmax(-1)
            heads = probs @ v
            trace[f"l{self.idx}_attn_probs"] = probs
        x = x + self.wo(heads.transpose(1, 2).reshape(b, t, d))
        if trace is not None:
            trace[f"l{self.idx}_resid_mid"] = x
        m = activation(self.w1(self.norm2(x)), cfg.act)
        x = x + self.w2(m)
        if trace is not None:
            trace[f"l{self.idx}_resid_out"] = x
        return x


class CharTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        std = 0.02
        self.tok_emb = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_model) * std)
        self.pos_emb = (
            nn.Parameter(torch.randn(cfg.ctx, cfg.d_model) * std) if cfg.pos == "learned" else None
        )
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm_f = Norm(cfg)
        self.head = None if cfg.tied else Linear(cfg.d_model, cfg.vocab_size, cfg.bias, std)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, trace: dict | None = None) -> torch.Tensor:
        """idx: (B, T) int64 with T <= ctx. Returns logits (B, T, V)."""
        t = idx.shape[1]
        if t > self.cfg.ctx:
            raise ValueError(f"sequence length {t} exceeds ctx {self.cfg.ctx}")
        x = self.tok_emb[idx]
        if self.pos_emb is not None:
            x = x + self.pos_emb[:t]
        if trace is not None:
            trace["x_embed"] = x
        for block in self.blocks:
            x = block(x, trace)
        x = self.norm_f(x)
        if trace is not None:
            trace["x_final_norm"] = x
        logits = x @ self.tok_emb.T if self.head is None else self.head(x)
        if trace is not None:
            trace["logits"] = logits
        return logits


@torch.no_grad()
def generate(
    model: CharTransformer,
    prompt_ids: list[int],
    n_new: int,
    temperature: float,
    generator: torch.Generator,
) -> list[int]:
    """Sample with a sliding window of the last ctx tokens (full recompute each
    step, as the Desmos build does). temperature must be > 0."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not prompt_ids:
        raise ValueError("prompt must be non-empty")
    model.eval()
    ids = list(prompt_ids)
    for _ in range(n_new):
        window = torch.tensor([ids[-model.cfg.ctx:]], dtype=torch.long)
        logits = model(window)[0, -1].double() / temperature
        probs = logits.softmax(-1)
        ids.append(int(torch.multinomial(probs, 1, generator=generator)))
    return ids
