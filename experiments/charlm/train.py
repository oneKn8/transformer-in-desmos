"""Train one CharTransformer config on the Homer corpus and evaluate it.

Budget: a fixed step count (4096 tokens per step), not wall-clock time. The
laptop is shared with other sweeps, and under that load even per-process CPU
time inflated about 2.4x per step, so time budgets would make results depend on
when a run happened. Step budgets are load-independent and, with one thread and a
fixed seed, reproducible. Equal tokens per config is also the comparison this
study needs: offline training compute is cheap, Desmos inference cost is not.
The learning rate follows linear warmup then cosine decay to min_lr_frac.
Early stopping keeps the best quick-val checkpoint.

Outputs in runs/<name>/: model.pt (state_dict + configs), results.json, and
the log on stdout.
"""

from __future__ import annotations

import threadcap  # sets OMP_NUM_THREADS; must be imported before torch

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from corpus import VOCABS, Vocab, decode, encode, load_mapped, split_bounds
from cost_model import cost_summary
from model import CharTransformer, ModelConfig, generate

HERE = Path(__file__).resolve().parent
LN2 = math.log(2.0)
SAMPLE_CHARS = 200
SAMPLE_TEMPERATURE = 0.8


@dataclass(frozen=True)
class TrainConfig:
    vocab: str = "v32"
    max_steps: int = 1500
    tokens_per_step: int = 4096
    lr: float = 3e-3
    min_lr_frac: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_windows: int = 256
    patience: int = 5
    min_delta: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        if self.vocab not in VOCABS:
            raise ValueError(f"vocab must be one of {sorted(VOCABS)}")
        if self.max_steps < 1 or self.eval_every < 1:
            raise ValueError("max_steps and eval_every must be positive")


def encode_ids(text: str, vocab: Vocab) -> torch.Tensor:
    return torch.tensor(encode(text, vocab), dtype=torch.long)


def load_splits(vocab: Vocab) -> dict[str, torch.Tensor]:
    ids = encode_ids(load_mapped(vocab.name), vocab)
    train_end, val_end = split_bounds(len(ids))
    return {"train": ids[:train_end], "val": ids[train_end:val_end], "test": ids[val_end:]}


def lr_at(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    p = min((step - tc.warmup_steps) / max(tc.max_steps - tc.warmup_steps, 1), 1.0)
    floor = tc.lr * tc.min_lr_frac
    return floor + (tc.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * p))


def random_windows(data: torch.Tensor, ctx: int, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - ctx - 1, (n,), generator=gen)
    offs = torch.arange(ctx + 1)
    win = data[starts[:, None] + offs[None, :]]
    return win[:, :-1], win[:, 1:]


@torch.no_grad()
def windows_loss(model: CharTransformer, x: torch.Tensor, y: torch.Tensor, score_from: int, batch: int = 256) -> tuple[float, int]:
    """Summed NLL (nats) over positions >= score_from, and the count."""
    total, count = 0.0, 0
    for i in range(0, len(x), batch):
        logits = model(x[i : i + batch])[:, score_from:]
        tgt = y[i : i + batch, score_from:]
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum").item()
        count += tgt.numel()
    return total, count


@torch.no_grad()
def split_bpc(model: CharTransformer, data: torch.Tensor, mode: str) -> float:
    """Bits per character on a whole split.

    all: non-overlapping windows of length T, every position scored (mean
         context (T-1)/2; the standard training-style number).
    ctx: windows at stride T/2, only positions >= T/2 scored, so every char is
         scored once with at least T/2 context (closer to steady-state
         sliding-window generation).
    """
    t = model.cfg.ctx
    if mode == "all":
        stride, score_from = t, 0
    elif mode == "ctx":
        stride, score_from = t // 2, t - t // 2
    else:
        raise ValueError("mode must be 'all' or 'ctx'")
    win = data.unfold(0, t + 1, stride)
    if len(win) == 0:
        raise ValueError("split shorter than one window")
    total, count = windows_loss(model, win[:, :-1], win[:, 1:], score_from)
    return total / count / LN2


def build_optimizer(model: CharTransformer, tc: TrainConfig) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=tc.lr,
        betas=(0.9, 0.99),
    )


def train(mc: ModelConfig, tc: TrainConfig, name: str, out_dir: Path) -> dict:
    vocab = VOCABS[tc.vocab]
    if mc.vocab_size != vocab.size:
        raise ValueError("model vocab_size does not match the vocab")
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    splits = load_splits(vocab)
    model = CharTransformer(mc)
    opt = build_optimizer(model, tc)
    batch = max(1, tc.tokens_per_step // mc.ctx)

    data_gen = torch.Generator().manual_seed(tc.seed + 1)
    eval_gen = torch.Generator().manual_seed(12345)  # same quick-val windows for every run
    qx, qy = random_windows(splits["val"], mc.ctx, tc.eval_windows, eval_gen)

    print(f"[{name}] params={model.num_params()} batch={batch}x{mc.ctx} threads={threadcap.THREADS}", flush=True)
    history: list[dict] = []
    best = {"loss": float("inf"), "step": 0, "state": None}
    bad_evals, step, early_stopped = 0, 0, False
    t0, c0 = time.perf_counter(), time.process_time()
    while step < tc.max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        model.train()
        x, y = random_windows(splits["train"], mc.ctx, batch, data_gen)
        loss = F.cross_entropy(model(x).reshape(-1, mc.vocab_size), y.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"[{name}] non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        step += 1

        if step % tc.eval_every == 0:
            elapsed = time.perf_counter() - t0
            model.eval()
            total, count = windows_loss(model, qx, qy, 0)
            qloss = total / count
            history.append({"step": step, "t": round(elapsed, 1), "train": round(loss.item(), 4), "qval": round(qloss, 4)})
            print(f"[{name}] step {step} t={elapsed:.0f}s train {loss.item() / LN2:.3f} bpc qval {qloss / LN2:.3f} bpc", flush=True)
            if qloss < best["loss"] - tc.min_delta:
                best = {"loss": qloss, "step": step, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                bad_evals = 0
            else:
                bad_evals += 1
                if bad_evals >= tc.patience:
                    early_stopped = True
                    print(f"[{name}] early stop at step {step}", flush=True)
                    break
    train_s, train_cpu_s = time.perf_counter() - t0, time.process_time() - c0

    # The final weights may be better than the last checkpoint; evaluate both on quick-val.
    model.eval()
    total, count = windows_loss(model, qx, qy, 0)
    if total / count < best["loss"]:
        best = {"loss": total / count, "step": step, "state": model.state_dict()}
    model.load_state_dict(best["state"])
    model.eval()

    metrics = {f"{split}_bpc_{mode}": round(split_bpc(model, splits[split], mode), 4) for split in ("val", "test") for mode in ("all", "ctx")}
    prompt = vocab.sample_prompt
    sample_ids = generate(model, encode(prompt, vocab), SAMPLE_CHARS, SAMPLE_TEMPERATURE, torch.Generator().manual_seed(tc.seed + 7))
    result = {
        "name": name,
        "model": mc.to_dict(),
        "train": asdict(tc),
        "params": model.num_params(),
        "steps": step,
        "best_step": best["step"],
        "tokens_seen": step * batch * mc.ctx,
        "train_wall_seconds": round(train_s, 1),
        "train_cpu_seconds": round(train_cpu_s, 1),
        "early_stopped": early_stopped,
        **metrics,
        "sample_prompt": prompt,
        "sample": decode(sample_ids, vocab),
        "cost": cost_summary(mc),
        "history": history,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": mc.to_dict(), "train_config": asdict(tc), "state_dict": model.state_dict()}, out_dir / "model.pt")
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"[{name}] done: val_bpc_all={metrics['val_bpc_all']} val_bpc_ctx={metrics['val_bpc_ctx']}", flush=True)
    return result


def parse_args() -> tuple[ModelConfig, TrainConfig, str]:
    ap = argparse.ArgumentParser(description="Train one char-LM config.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--vocab", default="v32", choices=sorted(VOCABS))
    ap.add_argument("--d-model", type=int, required=True)
    ap.add_argument("--n-layer", type=int, required=True)
    ap.add_argument("--n-head", type=int, required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--norm", default="rms")
    ap.add_argument("--act", default="gelu_tanh")
    ap.add_argument("--pos", default="learned")
    ap.add_argument("--tied", action="store_true")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--name may contain only letters, digits, '-' and '_'")
    mc = ModelConfig(
        vocab_size=VOCABS[a.vocab].size, ctx=a.ctx, d_model=a.d_model, n_layer=a.n_layer, n_head=a.n_head,
        norm=a.norm, act=a.act, pos=a.pos, tied=a.tied, bias=a.bias,
    )
    tc = TrainConfig(vocab=a.vocab, max_steps=a.max_steps, lr=a.lr, seed=a.seed)
    return mc, tc, a.name


def main() -> None:
    threadcap.apply_torch_threads()
    mc, tc, name = parse_args()
    train(mc, tc, name, HERE / "runs" / name)


if __name__ == "__main__":
    main()
