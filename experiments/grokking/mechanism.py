"""Watch the Fourier mechanism form: one dense-logged run of a chosen config.

Records, at every eval step, train/test accuracy and the Fourier energy spectrum of the
token embedding. Picks the key frequencies from the final model, then measures when their
share of the embedding norm rises relative to the test-accuracy jump. Saves a JSON summary
and two PNGs: the embedding circles (projection onto the top frequency's cos/sin plane) at
several checkpoints with the spectrum beneath, and the accuracy vs key-frequency-share curve.

Usage:
  python3 mechanism.py --name attn_best --base '{"p": 31, ...}' [--extra-steps 1000]
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grok import RunConfig, circle_projection, circle_quality, fourier_energy, train_run
from sweep import _parse_base

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # dataviz reference categorical slots 1-3
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"


def key_frequency_share(spectrum: np.ndarray, keys: list[int]) -> np.ndarray:
    """Share of the non-constant embedding energy held by the given frequencies, per row."""
    nondc = spectrum[:, 1:]
    return nondc[:, [k - 1 for k in keys]].sum(axis=1) / nondc.sum(axis=1)


def first_step_at_or_above(steps: np.ndarray, values: np.ndarray, level: float) -> int | None:
    hits = np.nonzero(values >= level)[0]
    return int(steps[hits[0]]) if hits.size else None


def pick_checkpoints(steps: list[int], t99: int, last: int) -> list[int]:
    """0, three points through the circuit-formation window, the grok step, and the end."""
    targets = [0, round(0.25 * t99), round(0.5 * t99), round(0.75 * t99), t99, last]
    chosen = []
    for t in targets:
        nearest = min(steps, key=lambda s, t=t: abs(s - t))
        if nearest not in chosen:
            chosen.append(nearest)
    return chosen


def analyze(res: dict[str, Any], max_keys: int = 8) -> dict[str, Any]:
    spec = np.array(res["spectrum"])
    steps, spectrum = spec[:, 0].astype(int), spec[:, 1:]
    final_nondc = spectrum[-1, 1:] / spectrum[-1, 1:].sum()
    order = np.argsort(final_nondc)[::-1]
    n_keys = min(int(np.searchsorted(np.cumsum(final_nondc[order]), 0.90) + 1), max_keys)
    keys = [int(k + 1) for k in order[:n_keys]]
    share = key_frequency_share(spectrum, keys)
    init, final = float(share[0]), float(share[-1])
    curve = np.array(res["curve"])
    return {
        "key_freqs": keys, "key_share_init": round(init, 4), "key_share_final": round(final, 4),
        "random_baseline_share": round(len(keys) / spectrum.shape[1], 4),
        "step_key_share_25pct": first_step_at_or_above(steps, share, init + 0.25 * (final - init)),
        "step_key_share_50pct": first_step_at_or_above(steps, share, init + 0.50 * (final - init)),
        "step_key_share_90pct": first_step_at_or_above(steps, share, init + 0.90 * (final - init)),
        "step_train99": res["step_train99"], "step_test50": res["step_test50"],
        "step_test99": res["step_test99"],
        "test_acc_at_key_share_50pct": None if not len(curve) else float(
            curve[np.searchsorted(curve[:, 0], first_step_at_or_above(
                steps, share, init + 0.5 * (final - init)) or 0), 2]),
        "steps": steps.tolist(), "key_share": [round(float(s), 4) for s in share],
    }


def plot_strip(res: dict[str, Any], summary: dict[str, Any], cfg: RunConfig, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snaps = {int(k): np.array(v) for k, v in res["snapshots"].items()}
    curve = np.array(res["curve"])
    shown = sorted(snaps)
    k = summary["key_freqs"][0]
    p = cfg.p
    fig, axes = plt.subplots(2, len(shown), figsize=(2.6 * len(shown), 5.4),
                             gridspec_kw={"height_ratios": [2.2, 1]}, facecolor=SURFACE)
    for col, step in enumerate(shown):
        emb = snaps[step]
        pts = circle_projection(emb, k)
        q = circle_quality(pts, k)
        test_acc = float(curve[np.searchsorted(curve[:, 0], step), 2])
        ax = axes[0, col]
        ax.set_facecolor(SURFACE)
        order = np.argsort((k * np.arange(p)) % p)  # walk tokens in their ideal angular order
        ring = np.append(order, order[0])
        ax.plot(pts[ring, 0], pts[ring, 1], color=GRID, lw=1, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(p), cmap="twilight", s=22,
                   edgecolors=SURFACE, linewidths=0.8, zorder=2)
        for t in (0, 1, 2):
            ax.annotate(str(t), pts[t], textcoords="offset points", xytext=(4, 3), fontsize=7,
                        color=INK_2)
        lim = 1.15 * np.abs(pts).max()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(f"step {step}\ntest {test_acc:.0%}  coh {q['phase_coherence']:.2f}",
                     fontsize=8, color=INK)
        bx = axes[1, col]
        bx.set_facecolor(SURFACE)
        energy = fourier_energy(emb)[1:]
        energy = energy / energy.sum()
        colors = [SERIES[0] if f in summary["key_freqs"] else GRID for f in range(1, len(energy) + 1)]
        bx.bar(np.arange(1, len(energy) + 1), energy, color=colors, width=0.8)
        bx.set_ylim(0, 1)
        bx.tick_params(labelsize=6, colors=INK_2)
        for s in ("top", "right"):
            bx.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            bx.spines[s].set_color(GRID)
        if col == 0:
            bx.set_ylabel("share of embedding norm", fontsize=7, color=INK_2)
        bx.set_xlabel("frequency k", fontsize=7, color=INK_2)
    fig.suptitle(f"{cfg.arch}, p={p}: token embeddings projected on frequency {k} "
                 f"(key frequencies {summary['key_freqs']} in blue)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_curves(res: dict[str, Any], summary: dict[str, Any], cfg: RunConfig, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = np.array(res["curve"])
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    series = [(curve[:, 0], curve[:, 1], "train accuracy"), (curve[:, 0], curve[:, 2], "test accuracy"),
              (np.array(summary["steps"]), np.array(summary["key_share"]),
               f"key-frequency share {summary['key_freqs']}")]
    for (xs, ys, label), color in zip(series, SERIES, strict=True):
        ax.plot(xs, ys, color=color, lw=2, label=label)
    for name, ls in (("step_key_share_50pct", ":"), ("step_test99", "--")):
        if summary[name] is not None:
            ax.axvline(summary[name], color=INK_2, lw=1, ls=ls)
            ax.annotate(name.replace("step_", ""), (summary[name], 0.02), fontsize=7, color=INK_2,
                        rotation=90, xytext=(3, 0), textcoords="offset points")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("training step", fontsize=8, color=INK_2)
    ax.set_ylabel("fraction", fontsize=8, color=INK_2)
    ax.tick_params(labelsize=7, colors=INK_2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.set_title(f"{cfg.arch}, p={cfg.p}, {cfg.optimizer} lr={cfg.lr:g} wd={cfg.weight_decay:g}, "
                 f"train frac {cfg.train_frac}", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", required=True, help="JSON RunConfig")
    ap.add_argument("--extra-steps", type=int, default=1000,
                    help="keep training this long after the grok step (cleanup phase)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--threads", type=int, default=3, choices=(1, 2, 3),
                    help="intra-op threads; use 1 when other study processes run alongside")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.nice(max(0, 6 - os.nice(0)))  # at least nice 6, below the browser

    cfg = dataclasses.replace(RunConfig(**_parse_base(a.base)), confirm_steps=a.extra_steps)
    every_eval = tuple(range(0, cfg.max_steps + 1, cfg.eval_every))
    res = train_run(cfg, record_curve=True, record_spectrum=True, snapshot_steps=every_eval)
    if res["step_test99"] is None:
        raise RuntimeError(f"run did not grok within {cfg.max_steps} steps: {res['status']}")
    summary = analyze(res)
    steps = sorted(int(s) for s in res["snapshots"])
    keep = pick_checkpoints(steps, res["step_test99"], steps[-1])
    res["snapshots"] = {str(s): res["snapshots"][str(s)] for s in keep}
    summary["checkpoints"] = {}
    k_top = summary["key_freqs"][0]
    for s in keep:
        q = circle_quality(circle_projection(np.array(res["snapshots"][str(s)]), k_top), k_top)
        summary["checkpoints"][str(s)] = {key: round(v, 4) for key, v in q.items()}

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"config": dataclasses.asdict(cfg), "summary": summary,
           "fourier_final": res["fourier_final"], "curve": res["curve"],
           "snapshots": res["snapshots"]}
    (RESULTS_DIR / f"mechanism_{a.name}.json").write_text(json.dumps(out))
    if not a.no_plots:
        plot_strip(res, summary, cfg, RESULTS_DIR / f"circles_{a.name}.png")
        plot_curves(res, summary, cfg, RESULTS_DIR / f"curves_{a.name}.png")
    brief = {k: v for k, v in summary.items() if k not in ("steps", "key_share")}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
