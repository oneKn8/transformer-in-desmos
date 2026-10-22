"""Reference numbers for checking the Desmos training graph: loss, the right/wrong grid and
every gradient at the initial weights, the weights after k AdamW steps, flattened in the
same row-major order as the graph's state lists, the trig-formula fit (key frequencies,
alphas, R^2) and the wave ablation accuracies at the initial weights and after each k.

Usage: python3 check_data.py runs/seed0.npz OUT.json [--steps 1 5 20]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import ref_step as rs
from gen_grok import TENSORS
from run_ref import formula_fit, wave_ablation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=1.0)
    args = ap.parse_args()
    with np.load(args.ref) as f:
        w = {k[5:]: f[k].copy() for k in f.files if k.startswith("init_")}
        pairs = f["pairs"]
    grid = rs.make_grid(pairs)
    m = {k: np.zeros_like(x) for k, x in w.items()}
    v = {k: np.zeros_like(x) for k, x in w.items()}

    f0 = rs.forward(w, grid)
    loss0, dl0, correct0 = rs.loss_and_dlogits(f0["logits"], grid)
    g0 = rs.backward(w, f0, dl0)
    out = {
        "loss0": loss0,
        "correct0": correct0.astype(int).reshape(-1).tolist(),          # row-major [a, b]
        "grads0": {TENSORS[k][0]: g0[k].reshape(-1).tolist() for k in TENSORS},
        "weights": {},
        "formula": {"0": dict(zip(("ks", "alphas", "r2"), formula_fit(f0["logits"], w["W_E"])))},
        "ablation": {"0": wave_ablation(f0["logits"], w["W_E"], grid.train)},
    }
    targets = set(args.steps)
    for step in range(max(targets)):
        f = rs.forward(w, grid)
        _, dl, _ = rs.loss_and_dlogits(f["logits"], grid)
        rs.adamw(w, rs.backward(w, f, dl), m, v, step, args.lr, args.wd)
        if step + 1 in targets:
            out["weights"][str(step + 1)] = {TENSORS[k][0]: w[k].reshape(-1).tolist() for k in TENSORS}
            logits = rs.forward(w, grid)["logits"]
            out["formula"][str(step + 1)] = dict(zip(("ks", "alphas", "r2"), formula_fit(logits, w["W_E"])))
            out["ablation"][str(step + 1)] = wave_ablation(logits, w["W_E"], grid.train)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}: loss0 {loss0:.6f}, init test-grid correct {int(correct0.sum())}/961, steps {sorted(targets)}")


if __name__ == "__main__":
    main()
