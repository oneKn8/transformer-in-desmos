"""What each attention head of the exported model does, measured on the validation split:
where the newest position looks (attention by distance back, share on spaces) and what the
model loses without the head (validation bits per character with the head removed).

A head is removed exactly by zeroing its rows of the output projection W_o, which is what the
Desmos graph's head switches do (they scale the head's output by 0).

Usage: python3 head_stats.py [--windows 400]   -> writes best/heads.json (read by ../../gen_desmos.py)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from corpus import VOCABS
from model import ModelConfig
from reference_numpy import forward, split_bpc_numpy
from train import load_splits

HERE = Path(__file__).resolve().parent
BEST = HERE / "best"


def without_head(fw: dict, layer: int, head: int, head_dim: int) -> dict:
    """Weights with one head's contribution removed (its rows of W_o set to zero)."""
    out = dict(fw)
    wo = fw[f"l{layer}_wo"].copy()
    wo[head * head_dim:(head + 1) * head_dim, :] = 0.0
    out[f"l{layer}_wo"] = wo
    return out


def attention_profile(fw: dict, cfg: ModelConfig, data: np.ndarray, windows: int, seed: int = 0) -> dict:
    """Mean attention of the last position of random full windows: by distance back, and on spaces."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - cfg.ctx, windows)
    by_dist = np.zeros((cfg.n_layer, cfg.n_head, cfg.ctx))
    on_space = np.zeros((cfg.n_layer, cfg.n_head))
    for s in starts:
        win = data[s:s + cfg.ctx]
        trace = {}
        forward(fw, cfg, win, trace=trace)
        for li in range(cfg.n_layer):
            last = trace[f"l{li}_attn_probs"][:, -1]          # (heads, ctx)
            by_dist[li] += last[:, ::-1]                       # index 0 = the newest position itself
            on_space[li] += last[:, win == 0].sum(-1)          # token 0 is the space
    return {"by_distance": by_dist / windows, "on_space": on_space / windows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=int, default=400)
    args = ap.parse_args()
    meta = json.loads((BEST / "config.json").read_text())
    cfg = ModelConfig(**meta["model_config"])
    with np.load(BEST / "weights.npz") as f:
        fw = {k: f[k].astype(np.float64) for k in f.files}
    val = load_splits(VOCABS[meta["vocab"]["name"]])["val"].numpy()
    prof = attention_profile(fw, cfg, val, args.windows)
    heads = []
    for li in range(cfg.n_layer):
        for j in range(cfg.n_head):
            dist = prof["by_distance"][li, j]
            top = np.argsort(-dist)[:3]
            heads.append({
                "layer": li, "head": j,
                "top_distances": [int(d) for d in top], "top_mass": [round(float(dist[d]), 3) for d in top],
                "on_space": round(float(prof["on_space"][li, j]), 3),
                "val_bpc_without": round(split_bpc_numpy(without_head(fw, li, j, cfg.head_dim), cfg, val), 3),
            })
    out = {"val_bpc": round(split_bpc_numpy(fw, cfg, val), 3), "windows": args.windows, "heads": heads}
    (BEST / "heads.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
