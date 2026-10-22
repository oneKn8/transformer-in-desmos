"""Train many same-shape configs at once as one stacked ensemble.

Members share p, architecture, widths, optimizer type and step budget; they may differ in
seed (init and data split), lr, weight decay, momentum, train fraction and batch size. Every
member sees all p*p pairs each step and a per-member weight vector selects its training
batch, so test accuracy comes from the same forward pass for free and is checked every step.

Initialization reuses grok.build_model per member, and the functional AdamW/SGD below follow
torch.optim's update order, so a one-member ensemble reproduces grok.train_run's trajectory
(checked in test_grok.py). This exists because tiny per-run tensors leave a single run
dominated by Python dispatch, which got 7x worse with the laptop under shared load.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from grok import (
    GROK_THRESHOLD,
    STABLE_THRESHOLD,
    RunConfig,
    batch_indices,
    build_model,
    circle_projection,
    circle_quality,
    fourier_summary,
)

SHARED_FIELDS = ("p", "arch", "d_model", "n_heads", "d_mlp", "act", "layernorm", "optimizer",
                 "beta2", "warmup_steps", "max_steps", "confirm_steps", "dtype")
ADAM_BETA1, ADAM_EPS = 0.9, 1e-8


def group_key(cfg: RunConfig) -> tuple[Any, ...]:
    return tuple(getattr(cfg, f) for f in SHARED_FIELDS)


def _act(name: str, x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) if name == "relu" else F.gelu(x)


def _stack_params(cfgs: list[RunConfig]) -> dict[str, torch.Tensor]:
    models = [build_model(c) for c in cfgs]
    named = [dict(m.named_parameters()) for m in models]
    return {name: torch.stack([n[name].detach() for n in named]).requires_grad_()
            for name in named[0]}


def _forward(cfg: RunConfig, prm: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Batched logits [K, N, p] for all members on shared inputs x [N, 3]."""
    k_members, n = prm["b_in"].shape[0], x.shape[0]
    a, b = x[:, 0], x[:, 1]
    if cfg.arch == "attn":
        d, h = cfg.d_model, cfg.n_heads
        dh = d // h
        resid = prm["W_E"][:, x] + prm["W_pos"][:, None]  # [K, N, 3, d]

        def flat(w: torch.Tensor) -> torch.Tensor:
            return w.permute(0, 2, 1, 3).reshape(k_members, d, d)

        q = (resid[:, :, 2] @ flat(prm["W_Q"])).view(k_members, n, 1, h, dh)
        kk = (resid.reshape(k_members, n * 3, d) @ flat(prm["W_K"])).view(k_members, n, 3, h, dh)
        v = (resid.reshape(k_members, n * 3, d) @ flat(prm["W_V"])).view(k_members, n, 3, h, dh)
        pattern = ((q * kk).sum(-1) / math.sqrt(dh)).softmax(dim=2)
        z = (pattern.unsqueeze(-1) * v).sum(2).reshape(k_members, n, d)
        r = resid[:, :, 2] + z @ prm["W_O"].reshape(k_members, d, d)
        hidden = _act(cfg.act, r @ prm["W_in"] + prm["b_in"][:, None])
        r = r + hidden @ prm["W_out"] + prm["b_out"][:, None]
        return r @ prm["W_U"]
    if cfg.arch == "mlp_folded":
        pre = prm["A"][:, a] + prm["B"][:, b]
    elif cfg.arch == "mlp_concat":
        pre = torch.cat([prm["E"][:, a], prm["E"][:, b]], dim=-1) @ prm["W_in"]
    else:
        pos = prm["W_pos"]
        pre = (prm["E"][:, a] + pos[:, 0:1] + prm["E"][:, b] + pos[:, 1:2]) @ prm["W_in"]
    return _act(cfg.act, pre + prm["b_in"][:, None]) @ prm["W_U"]


def _embedding(cfg: RunConfig, prm: dict[str, torch.Tensor], k: int) -> np.ndarray:
    if cfg.arch == "attn":
        emb = prm["W_E"][k, :-1]
    elif cfg.arch == "mlp_folded":
        emb = torch.cat([prm["A"][k], prm["B"][k]], dim=1)
    else:
        emb = prm["E"][k]
    return emb.detach().cpu().numpy()


class _Optimizer:
    """Functional AdamW / SGD with per-member hyperparameters, matching torch.optim order."""

    def __init__(self, cfgs: list[RunConfig], prm: dict[str, torch.Tensor]) -> None:
        dtype = next(iter(prm.values())).dtype
        self.kind = cfgs[0].optimizer
        self.lr = torch.tensor([c.lr for c in cfgs], dtype=dtype)
        self.wd = torch.tensor([c.weight_decay for c in cfgs], dtype=dtype)
        self.mom = torch.tensor([c.momentum for c in cfgs], dtype=dtype)
        self.beta2 = cfgs[0].beta2
        self.warm = max(cfgs[0].warmup_steps, 1)
        self.m = {n: torch.zeros_like(t) for n, t in prm.items()}
        self.v = {n: torch.zeros_like(t) for n, t in prm.items()} if self.kind == "adamw" else {}
        self.t = 0

    def freeze(self, k: int) -> None:
        """Stop updating member k (lr 0 also stops decay and momentum carry-over)."""
        self.lr[k] = 0.0

    def step(self, prm: dict[str, torch.Tensor]) -> None:
        factor = min((self.t + 1) / self.warm, 1.0)
        self.t += 1
        with torch.no_grad():
            for name, w in prm.items():
                if w.grad is None:
                    raise RuntimeError(f"parameter {name} received no gradient")
                shape = (-1,) + (1,) * (w.dim() - 1)
                lr = (self.lr * factor).view(shape)
                wd, g = self.wd.view(shape), w.grad
                if self.kind == "adamw":
                    w.mul_(1 - lr * wd)
                    self.m[name].mul_(ADAM_BETA1).add_(g, alpha=1 - ADAM_BETA1)
                    self.v[name].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                    bc1 = 1 - ADAM_BETA1 ** self.t
                    bc2 = 1 - self.beta2 ** self.t
                    denom = self.v[name].sqrt() / math.sqrt(bc2) + ADAM_EPS
                    w.sub_(lr / bc1 * self.m[name] / denom)
                else:
                    g = g + wd * w
                    self.m[name].mul_(self.mom.view(shape)).add_(g)
                    w.sub_(lr * self.m[name])
                w.grad = None


def _batch_weights(cfgs: list[RunConfig], perms: list[torch.Tensor], n_trains: list[int],
                   gens: list[torch.Generator], n_pairs: int, dtype: torch.dtype,
                   full: torch.Tensor, step: int) -> torch.Tensor:
    """Per-member example weights summing to 1: uniform over the train split for full batch,
    multinomial counts for minibatch (same index stream as grok.train_run)."""
    rows = []
    for k, c in enumerate(cfgs):
        if c.batch_size == 0 or c.batch_size >= n_trains[k]:
            rows.append(full[k])
            continue
        idx = batch_indices(c, n_trains[k], step, gens[k])
        counts = torch.bincount(perms[k][idx], minlength=n_pairs).to(dtype)
        rows.append(counts / c.batch_size)
    return torch.stack(rows)


def train_ensemble(cfgs: list[RunConfig]) -> list[dict[str, Any]]:
    if not cfgs:
        raise ValueError("empty ensemble")
    ref = cfgs[0]
    if any(group_key(c) != group_key(ref) for c in cfgs):
        raise ValueError(f"ensemble members must share {SHARED_FIELDS}")
    if ref.layernorm:
        raise ValueError("LayerNorm runs use grok.train_run, not the ensemble trainer")
    dtype = getattr(torch, ref.dtype)
    p, n_members = ref.p, len(cfgs)
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    x = torch.stack([a, b, torch.full_like(a, p)], dim=1)
    y = (a + b) % p
    n_pairs = p * p

    perms, n_trains = [], []
    train_mask = torch.zeros(n_members, n_pairs, dtype=dtype)
    for k, c in enumerate(cfgs):
        # Same split as grok.make_dataset(p, train_frac, seed=c.seed).
        perm = torch.randperm(n_pairs, generator=torch.Generator().manual_seed(c.seed))
        n_tr = round(c.train_frac * n_pairs)
        train_mask[k, perm[:n_tr]] = 1.0
        perms.append(perm)
        n_trains.append(n_tr)
    test_mask = 1.0 - train_mask
    n_tr_t = train_mask.sum(1)
    n_te_t = test_mask.sum(1)
    full_w = train_mask / n_tr_t[:, None]
    gens = [torch.Generator().manual_seed(c.seed + 7) for c in cfgs]

    prm = _stack_params(cfgs)
    opt = _Optimizer(cfgs, prm)
    results: list[dict[str, Any]] = []
    for k, c in enumerate(cfgs):
        results.append({
            "config": dataclasses.asdict(c), "n_train": n_trains[k], "trainer": "ensemble",
            "n_params": sum(int(t[k].numel()) for t in prm.values()),
            "status": "not_grokked", "step_train99": None, "step_test50": None,
            "step_test99": None, "stable": None, "min_test_after_grok": None,
            "fourier_init": fourier_summary(_embedding(ref, prm, k)),
        })
    stop_at = [c.max_steps for c in cfgs]
    done = [False] * n_members
    y_rep = y.repeat(n_members)
    t0 = time.perf_counter()

    for step in range(ref.max_steps + 1):
        logits = _forward(ref, prm, x)
        ce = F.cross_entropy(logits.reshape(-1, p), y_rep, reduction="none").view(n_members, -1)
        with torch.no_grad():
            correct = (logits.argmax(-1) == y).to(dtype)
            stats = torch.stack([
                (correct * train_mask).sum(1) / n_tr_t, (correct * test_mask).sum(1) / n_te_t,
                (ce * train_mask).sum(1) / n_tr_t, (ce * test_mask).sum(1) / n_te_t,
            ]).T.tolist()
        for k, (tr_acc, te_acc, tr_loss, te_loss) in enumerate(stats):
            if done[k]:
                continue
            r = results[k]
            if not (math.isfinite(tr_loss) and math.isfinite(te_loss)):
                r.update(status="diverged", diverged_step=step)
                done[k] = True
                opt.freeze(k)
                continue
            if r["step_train99"] is None and tr_acc >= GROK_THRESHOLD:
                r["step_train99"] = step
            if r["step_test50"] is None and te_acc >= 0.5:
                r["step_test50"] = step
            if r["step_test99"] is None and te_acc >= GROK_THRESHOLD:
                r.update(step_test99=step, status="grokked", min_test_after_grok=te_acc)
                stop_at[k] = min(cfgs[k].max_steps, step + cfgs[k].confirm_steps)
            elif r["step_test99"] is not None:
                r["min_test_after_grok"] = min(r["min_test_after_grok"], te_acc)
            r.update(final_step=step, final_train_acc=tr_acc, final_test_acc=te_acc,
                     final_train_loss=tr_loss, final_test_loss=te_loss)
            if step >= stop_at[k]:
                done[k] = True
                opt.freeze(k)
        if all(done) or step == ref.max_steps:
            break
        weights = _batch_weights(cfgs, perms, n_trains, gens, n_pairs, dtype, full_w, step)
        live = torch.tensor([0.0 if d else 1.0 for d in done], dtype=dtype)[:, None]
        (ce * weights * live).sum().backward()
        opt.step(prm)

    wall = round(time.perf_counter() - t0, 2)
    for k, r in enumerate(results):
        if r["step_test99"] is not None:
            r["stable"] = r["min_test_after_grok"] >= STABLE_THRESHOLD
        emb = _embedding(ref, prm, k)
        # A diverged member can keep finite but huge weights; its spectrum is meaningless.
        if r["status"] != "diverged" and np.isfinite(emb).all():
            r["fourier_final"] = fourier_summary(emb)
            k_top = r["fourier_final"]["top_freqs"][0]
            r["circle_final"] = circle_quality(circle_projection(emb, k_top), k_top)
        else:
            r["fourier_final"] = r["circle_final"] = None
        r["wall_s"] = wall
        r["ensemble_size"] = n_members
    return results
