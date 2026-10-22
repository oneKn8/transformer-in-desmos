"""Modular-addition grokking at Desmos shapes: data, models, training, Fourier analysis.

Everything here is deterministic given a RunConfig (seeded data split and init). Models are
kept to what a hand-written Desmos training step can express: one layer, no LayerNorm by
default, ReLU, d_model <= 32 and MLP widths that are multiples of 32.

Architectures
- attn: Nanda et al. 2023 one-layer transformer (tokens [a, b, =], learned positions,
  attention then MLP, logits read at the "=" position). Only the last query is computed:
  in a one-layer model the other positions never reach the readout, so this is exact.
- mlp_concat: shared token embedding E, concat(E[a], E[b]) -> ReLU MLP -> logits.
- mlp_sum: shared token embedding E, E[a] + E[b] (plus learned position offsets) -> MLP.
- mlp_folded: separate tables A[a] + B[b] straight into the hidden layer (Gromov 2023 shape).
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ARCHES = ("attn", "mlp_concat", "mlp_sum", "mlp_folded")
OPTIMIZERS = ("adamw", "sgd")
ACTIVATIONS = ("relu", "gelu")
GROK_THRESHOLD = 0.99
STABLE_THRESHOLD = 0.98


@dataclasses.dataclass(frozen=True)
class RunConfig:
    p: int
    arch: str = "attn"
    d_model: int = 32
    n_heads: int = 4
    d_mlp: int = 128
    act: str = "relu"
    layernorm: bool = False
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1.0
    momentum: float = 0.0
    beta2: float = 0.98
    warmup_steps: int = 10
    train_frac: float = 0.3
    batch_size: int = 0  # 0 means full batch
    batch_mode: str = "random"  # random: sample with replacement; cyclic: fixed slices in order
    max_steps: int = 10_000
    eval_every: int = 10
    confirm_steps: int = 200
    seed: int = 0
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.p < 3 or any(self.p % k == 0 for k in range(2, int(math.isqrt(self.p)) + 1)):
            raise ValueError(f"p must be an odd prime, got {self.p}")
        if self.arch not in ARCHES:
            raise ValueError(f"arch must be one of {ARCHES}, got {self.arch!r}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.act not in ACTIVATIONS:
            raise ValueError(f"act must be one of {ACTIVATIONS}, got {self.act!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError("train_frac must be in (0, 1)")
        if self.lr <= 0 or self.weight_decay < 0 or not 0 <= self.momentum < 1:
            raise ValueError("lr must be > 0, weight_decay >= 0, momentum in [0, 1)")
        if self.batch_size < 0 or self.max_steps < 1 or self.eval_every < 1:
            raise ValueError("batch_size >= 0, max_steps >= 1, eval_every >= 1 required")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.batch_mode not in ("random", "cyclic"):
            raise ValueError("batch_mode must be random or cyclic")

    def key(self) -> str:
        """Stable identity used to skip runs that already have a result on disk."""
        return repr(sorted(dataclasses.asdict(self).items()))


@dataclasses.dataclass(frozen=True)
class Dataset:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def make_dataset(p: int, train_frac: float, seed: int) -> Dataset:
    """All p*p pairs as [a, b, =] with label (a+b) mod p, split by a seeded permutation."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    x = torch.stack([a, b, torch.full_like(a, p)], dim=1)
    y = (a + b) % p
    perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(seed))
    n_train = round(train_frac * p * p)
    tr, te = perm[:n_train], perm[n_train:]
    return Dataset(x[tr], y[tr], x[te], y[te])


def _activation(name: str) -> nn.Module:
    return nn.ReLU() if name == "relu" else nn.GELU()


def _randn(gen: torch.Generator, *shape: int, scale: float) -> nn.Parameter:
    return nn.Parameter(torch.randn(*shape, generator=gen) * scale)


class AttnModel(nn.Module):
    """One-layer transformer, Nanda-style init (N(0, 1/d_model)), no attention biases."""

    def __init__(self, cfg: RunConfig, gen: torch.Generator) -> None:
        super().__init__()
        p, d, h = cfg.p, cfg.d_model, cfg.n_heads
        dh = d // h
        s = 1.0 / math.sqrt(d)
        self.n_heads, self.d_head = h, dh
        self.W_E = _randn(gen, p + 1, d, scale=s)
        self.W_pos = _randn(gen, 3, d, scale=s)
        self.W_Q = _randn(gen, h, d, dh, scale=s)
        self.W_K = _randn(gen, h, d, dh, scale=s)
        self.W_V = _randn(gen, h, d, dh, scale=s)
        self.W_O = _randn(gen, h, dh, d, scale=s)
        self.W_in = _randn(gen, d, cfg.d_mlp, scale=s)
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = _randn(gen, cfg.d_mlp, d, scale=s)
        self.b_out = nn.Parameter(torch.zeros(d))
        self.W_U = _randn(gen, d, p, scale=s)
        self.act = _activation(cfg.act)
        use_ln = cfg.layernorm
        self.ln1 = nn.LayerNorm(d) if use_ln else nn.Identity()
        self.ln2 = nn.LayerNorm(d) if use_ln else nn.Identity()
        self.ln_f = nn.LayerNorm(d) if use_ln else nn.Identity()

    def embedding(self) -> torch.Tensor:
        """Number-token embedding rows 0..p-1 (the matrix whose Fourier structure we track)."""
        return self.W_E[:-1]

    def _flat(self, w: torch.Tensor) -> torch.Tensor:
        """[H, d, dh] -> [d, H*dh] so all heads run as one matmul."""
        return w.permute(1, 0, 2).reshape(w.shape[1], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, h, dh = x.shape[0], self.n_heads, self.d_head
        resid = self.W_E[x] + self.W_pos  # [N, 3, d]
        r_in = self.ln1(resid)
        q = (r_in[:, 2] @ self._flat(self.W_Q)).view(n, 1, h, dh)
        k = (r_in @ self._flat(self.W_K)).view(n, 3, h, dh)
        v = (r_in @ self._flat(self.W_V)).view(n, 3, h, dh)
        scores = (q * k).sum(-1) / math.sqrt(dh)  # [N, 3, H]
        pattern = scores.softmax(dim=1)  # the "=" query is causal over all 3 positions
        z = (pattern.unsqueeze(-1) * v).sum(1).reshape(n, h * dh)
        r = resid[:, 2] + z @ self.W_O.reshape(h * dh, -1)
        hidden = self.act(self.ln2(r) @ self.W_in + self.b_in)
        r = r + hidden @ self.W_out + self.b_out
        return self.ln_f(r) @ self.W_U


class MLPModel(nn.Module):
    """Attention-free baselines. See module docstring for the three combine modes."""

    def __init__(self, cfg: RunConfig, gen: torch.Generator) -> None:
        super().__init__()
        p, d, m = cfg.p, cfg.d_model, cfg.d_mlp
        self.mode = cfg.arch
        if self.mode == "mlp_folded":
            # std 1/sqrt(2d) gives the same pre-activation variance (1/d) as mlp_concat at
            # equal d_model, so the folded/concat comparison is not an init-scale artifact.
            s = 1.0 / math.sqrt(2 * d)
            self.A = _randn(gen, p, m, scale=s)
            self.B = _randn(gen, p, m, scale=s)
        else:
            self.E = _randn(gen, p, d, scale=1.0 / math.sqrt(d))
            fan_in = 2 * d if self.mode == "mlp_concat" else d
            self.W_in = _randn(gen, fan_in, m, scale=1.0 / math.sqrt(fan_in))
            if self.mode == "mlp_sum":
                self.W_pos = _randn(gen, 2, d, scale=1.0 / math.sqrt(d))
        self.b_in = nn.Parameter(torch.zeros(m))
        self.W_U = _randn(gen, m, p, scale=1.0 / math.sqrt(m))
        self.act = _activation(cfg.act)

    def embedding(self) -> torch.Tensor:
        if self.mode == "mlp_folded":
            return torch.cat([self.A, self.B], dim=1)
        return self.E

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x[:, 0], x[:, 1]
        if self.mode == "mlp_folded":
            pre = self.A[a] + self.B[b]
        elif self.mode == "mlp_concat":
            pre = torch.cat([self.E[a], self.E[b]], dim=1) @ self.W_in
        else:
            pre = (self.E[a] + self.W_pos[0] + self.E[b] + self.W_pos[1]) @ self.W_in
        return self.act(pre + self.b_in) @ self.W_U


def build_model(cfg: RunConfig) -> nn.Module:
    gen = torch.Generator().manual_seed(10_007 * cfg.seed + 1)
    model = AttnModel(cfg, gen) if cfg.arch == "attn" else MLPModel(cfg, gen)
    return model.to(getattr(torch, cfg.dtype))


def make_optimizer(cfg: RunConfig, params: Any) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, cfg.beta2)
        )
    # torch SGD weight_decay is coupled L2; for plain SGD that equals decoupled decay lr*wd.
    return torch.optim.SGD(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum
    )


def fourier_energy(emb: np.ndarray) -> np.ndarray:
    """Share of squared norm per frequency over the token axis.

    emb is [p, width] with row t = token t. Index 0 is the constant (DC) term; index k >= 1
    combines DFT bins k and p-k, i.e. the cos and sin pair of frequency k. Sums to 1.
    """
    p = emb.shape[0]
    power = (np.abs(np.fft.fft(emb, axis=0)) ** 2).sum(axis=1)
    half = (p - 1) // 2
    ks = np.arange(1, half + 1)
    energy = np.concatenate([[power[0]], power[ks] + power[p - ks]])
    total = energy.sum()
    if total <= 0:
        raise ValueError("embedding has zero norm; Fourier shares are undefined")
    return energy / total


def fourier_summary(emb: np.ndarray, top: int = 6) -> dict[str, Any]:
    """Sparsity of the non-constant part: which frequencies hold the norm, and how much."""
    energy = fourier_energy(emb)
    nondc = energy[1:] / energy[1:].sum()
    order = np.argsort(nondc)[::-1]
    cum = np.cumsum(nondc[order])
    return {
        "dc_share": float(energy[0]),
        "top_freqs": [int(k + 1) for k in order[:top]],
        "top_shares": [round(float(nondc[k]), 4) for k in order[:top]],
        "frac_top": {str(j): round(float(cum[j - 1]), 4) for j in range(1, top + 1)},
        "n_freqs_90": int(np.searchsorted(cum, 0.90) + 1),
        "n_freqs_total": len(nondc),
    }


def circle_projection(emb: np.ndarray, k: int) -> np.ndarray:
    """Project centred embeddings onto the plane spanned by their frequency-k cos/sin parts."""
    p = emb.shape[0]
    t = np.arange(p)
    cos_dir = np.cos(2 * np.pi * k * t / p) @ emb
    sin_dir = np.sin(2 * np.pi * k * t / p) @ emb
    basis, _ = np.linalg.qr(np.stack([cos_dir, sin_dir], axis=1))
    return (emb - emb.mean(axis=0)) @ basis


def circle_quality(points: np.ndarray, k: int) -> dict[str, float]:
    """radius_cv: std/mean of radii (0 = perfect circle). phase_coherence: 1 when token t sits
    at angle 2*pi*k*t/p up to a global rotation or reflection, about 1/sqrt(p) when random."""
    p = points.shape[0]
    radii = np.linalg.norm(points, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    expected = 2 * np.pi * k * np.arange(p) / p
    coh = max(abs(np.exp(1j * (theta - expected)).mean()), abs(np.exp(1j * (theta + expected)).mean()))
    return {"radius_cv": float(radii.std() / radii.mean()), "phase_coherence": float(coh)}


def batch_indices(cfg: RunConfig, n_train: int, step: int, gen: torch.Generator) -> torch.Tensor:
    """Positions in the train split for this step's minibatch. Cyclic mode walks fixed
    consecutive slices (the remainder is never used), which is what a Desmos build would do
    with precomputed one-hot blocks indexed by tick number."""
    if cfg.batch_mode == "cyclic":
        n_batches = n_train // cfg.batch_size
        start = (step % n_batches) * cfg.batch_size
        return torch.arange(start, start + cfg.batch_size)
    return torch.randint(0, n_train, (cfg.batch_size,), generator=gen)


def _accuracy_and_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        logits = model(x)
        return float((logits.argmax(-1) == y).double().mean()), float(F.cross_entropy(logits, y))


def train_run(
    cfg: RunConfig, record_curve: bool = False, record_spectrum: bool = False,
    snapshot_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Train until test accuracy first reaches GROK_THRESHOLD, then run confirm_steps more to
    check it holds. Returns metrics; curve/spectrum/snapshots only when requested."""
    torch.manual_seed(cfg.seed)
    data = make_dataset(cfg.p, cfg.train_frac, seed=cfg.seed)
    model = build_model(cfg)
    opt = make_optimizer(cfg, model.parameters())
    warm = max(cfg.warmup_steps, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 1.0))
    batch_gen = torch.Generator().manual_seed(cfg.seed + 7)
    n_train = len(data.y_train)
    full_batch = cfg.batch_size == 0 or cfg.batch_size >= n_train

    init_summary = fourier_summary(model.embedding().detach().cpu().numpy())
    result: dict[str, Any] = {
        "config": dataclasses.asdict(cfg), "n_train": n_train,
        "n_params": sum(p.numel() for p in model.parameters()),
        "status": "not_grokked", "step_train99": None, "step_test50": None,
        "step_test99": None, "stable": None, "min_test_after_grok": None,
        "fourier_init": init_summary,
    }
    curve: list[list[float]] = []
    spectrum: list[list[float]] = []
    snapshots: dict[int, list[list[float]]] = {}
    stop_at = cfg.max_steps
    t0 = time.perf_counter()

    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_every == 0 or step == stop_at:
            tr_acc, tr_loss = _accuracy_and_loss(model, data.x_train, data.y_train)
            te_acc, te_loss = _accuracy_and_loss(model, data.x_test, data.y_test)
            if not (math.isfinite(tr_loss) and math.isfinite(te_loss)):
                result["status"] = "diverged"
                result["diverged_step"] = step
                break
            if record_curve:
                curve.append([step, tr_acc, te_acc, tr_loss, te_loss])
            if record_spectrum:
                spectrum.append([step, *fourier_energy(model.embedding().detach().cpu().numpy())])
            if result["step_train99"] is None and tr_acc >= GROK_THRESHOLD:
                result["step_train99"] = step
            if result["step_test50"] is None and te_acc >= 0.5:
                result["step_test50"] = step
            if result["step_test99"] is None and te_acc >= GROK_THRESHOLD:
                result["step_test99"] = step
                result["status"] = "grokked"
                stop_at = min(cfg.max_steps, step + cfg.confirm_steps)
                result["min_test_after_grok"] = te_acc
            elif result["step_test99"] is not None:
                result["min_test_after_grok"] = min(result["min_test_after_grok"], te_acc)
            result.update(final_step=step, final_train_acc=tr_acc, final_test_acc=te_acc,
                          final_train_loss=tr_loss, final_test_loss=te_loss)
        if step in snapshot_steps:
            snapshots[step] = model.embedding().detach().cpu().numpy().tolist()
        if step >= stop_at:
            break
        if full_batch:
            xb, yb = data.x_train, data.y_train
        else:
            idx = batch_indices(cfg, n_train, step, batch_gen)
            xb, yb = data.x_train[idx], data.y_train[idx]
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    if result["step_test99"] is not None:
        result["stable"] = result["min_test_after_grok"] >= STABLE_THRESHOLD
    result["wall_s"] = round(time.perf_counter() - t0, 2)
    emb = model.embedding().detach().cpu().numpy()
    result["fourier_final"] = fourier_summary(emb)
    k_top = result["fourier_final"]["top_freqs"][0]
    result["circle_final"] = circle_quality(circle_projection(emb, k_top), k_top)
    if record_curve:
        result["curve"] = curve
    if record_spectrum:
        result["spectrum"] = spectrum
    if snapshot_steps:
        result["snapshots"] = {str(k): v for k, v in snapshots.items()}
    return result
