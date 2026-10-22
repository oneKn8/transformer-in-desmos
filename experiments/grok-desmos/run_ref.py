"""Train with the numpy reference step from grok.py's exact init and split; save what the Desmos
build needs (initial weights, the split) and the reference curve it must reproduce.

Usage: python3 run_ref.py OUT.npz [--steps 1400] [--seed 0] [--lr 3e-2] [--wd 1] [--frac 0.5]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

import ref_step as rs

GROK_PATH = Path(__file__).resolve().parent.parent / "grokking" / "grok.py"


def load_grok():
    spec = importlib.util.spec_from_file_location("grok", GROK_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["grok"] = mod  # dataclasses look their module up while the class is built
    spec.loader.exec_module(mod)
    return mod


def initial_state(grok, cfg) -> tuple[dict, np.ndarray]:
    """grok.py's seeded init and split, in ref_step's layout."""
    import torch  # only needed to reproduce grok.py's seeded init and split

    torch.manual_seed(cfg.seed)
    data = grok.make_dataset(cfg.p, cfg.train_frac, seed=cfg.seed)
    model = grok.build_model(cfg)
    w = {
        "W_E": model.W_E, "W_pos": model.W_pos, "W_Q": model._flat(model.W_Q),
        "W_K": model._flat(model.W_K), "W_V": model._flat(model.W_V),
        "W_O": model.W_O.reshape(-1, model.W_O.shape[-1]), "W_in": model.W_in, "b_in": model.b_in,
        "W_out": model.W_out, "b_out": model.b_out, "W_U": model.W_U,
    }
    return {k: v.detach().numpy().astype(np.float64).copy() for k, v in w.items()}, data.x_train[:, :2].numpy()


def key_frequency_share(emb: np.ndarray, top: int = 3) -> float:
    """Share of the number-token embedding's energy in its `top` strongest Fourier frequencies."""
    spec = np.abs(np.fft.rfft(emb[: rs.P], axis=0)) ** 2           # (16, 32), row 0 is the mean
    per_k = spec[1:].sum(1)
    return float(np.sort(per_k)[::-1][:top].sum() / per_k.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--frac", type=float, default=0.5)
    args = ap.parse_args()

    grok = load_grok()
    cfg = grok.RunConfig(p=rs.P, arch="attn", d_model=rs.D, n_heads=rs.H, d_mlp=rs.M, lr=args.lr,
                         weight_decay=args.wd, train_frac=args.frac, seed=args.seed, dtype="float64")
    w, pairs = initial_state(grok, cfg)
    w0 = {k: v.copy() for k, v in w.items()}
    grid = rs.make_grid(pairs)
    m = {k: np.zeros_like(x) for k, x in w.items()}
    v = {k: np.zeros_like(x) for k, x in w.items()}
    test = ~grid.train
    curve = []
    t0 = time.perf_counter()
    for step in range(args.steps + 1):
        f = rs.forward(w, grid)
        loss, dlogits, correct = rs.loss_and_dlogits(f["logits"], grid)
        if step % 10 == 0 or step == args.steps:
            curve.append([step, float(correct[grid.train].mean()), float(correct[test].mean()), loss,
                          key_frequency_share(w["W_E"])])
        if step == args.steps:
            break
        rs.adamw(w, rs.backward(w, f, dlogits), m, v, step, args.lr, args.wd)
    secs = time.perf_counter() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, pairs=pairs, curve=np.array(curve), **{f"init_{k}": x for k, x in w0.items()},
             **{f"final_{k}": x for k, x in w.items()})
    first99 = next((int(c[0]) for c in curve if c[2] >= 0.99), None)
    print(json.dumps({"steps": args.steps, "seconds": round(secs, 1), "first_test99": first99,
                      "final": dict(zip(["step", "train", "test", "loss", "key_share"], curve[-1]))}))


if __name__ == "__main__":
    main()
