"""Aggregate results/*.jsonl into markdown tables with an estimated in-Desmos training time.

Each config (seeds pooled) gets: steps to 99% test accuracy per seed, stability after the
grok step, the Desmos cost of one training step from cost_model (using the examples actually
processed per step: the train split for full batch, the batch size for minibatch), and the
estimated wall time to grok at 50 and 150 ms tick overhead.

Usage: python3 summarize.py [--out results/summary.md] [--stage NAME ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from cost_model import MAX_TICKS_PER_S, TICK_MS, step_cost

RESULTS_DIR = Path(__file__).resolve().parent / "results"
# Seeds of one config pool together even if they ran under different step caps; each seed's
# own cap is shown when it did not grok.
POOL_KEYS = ("p", "arch", "d_model", "n_heads", "d_mlp", "act", "layernorm", "optimizer", "lr",
             "weight_decay", "momentum", "train_frac", "batch_size")


def _pool_key(c: dict[str, Any]) -> tuple[Any, ...]:
    # batch_mode was added after the pilot runs; those were all full batch ("random" default).
    return (*(c[k] for k in POOL_KEYS), c.get("batch_mode", "random"))


def load(stages: list[str] | None) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        if stages and path.stem not in stages:
            continue
        rows = []
        with path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{lineno} is not valid JSON") from exc
        out[path.stem] = rows
    return out


def pool_seeds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(_pool_key(r["config"]), []).append(r)
    pooled = []
    for runs in groups.values():
        c = runs[0]["config"]
        steps = [r["step_test99"] for r in runs]
        caps = [r["config"]["max_steps"] for r in runs]
        grokked = [s for s in steps if s is not None]
        n_step = c["batch_size"] if 0 < c["batch_size"] < runs[0]["n_train"] else runs[0]["n_train"]
        cost = step_cost(c["arch"], c["p"], n_step, c["d_model"], c["n_heads"], c["d_mlp"],
                         c["optimizer"], c["momentum"])
        all_grok = len(grokked) == len(runs)
        worst = max(grokked) if all_grok else None
        pooled.append({
            "cfg": c, "seeds": len(runs), "n_grokked": len(grokked), "steps": steps, "caps": caps,
            "median": statistics.median(grokked) if all_grok else None, "worst": worst,
            "stable": all(r["stable"] for r in runs if r["step_test99"] is not None),
            "train99": [r["step_train99"] for r in runs], "n_train": runs[0]["n_train"],
            "test50": [r["step_test50"] for r in runs],
            "status": sorted({r["status"] for r in runs}),
            "test_final": [round(r["final_test_acc"], 3) for r in runs],
            "fourier": runs[0].get("fourier_final"), "circle": runs[0].get("circle_final"),
            "tick_ms": [cost.tick_ms(t) for t in TICK_MS], "compute_ms": cost.compute_ms(),
            "hours": [worst * cost.tick_ms(t) / 3.6e6 for t in TICK_MS] if worst else None,
        })
    return sorted(pooled, key=lambda g: (g["hours"] is None, g["hours"][0] if g["hours"] else 0,
                                         -(g["n_grokked"] / g["seeds"])))


def combine(data: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Every distinct config across all stages, one row per seed. The same config can appear
    in a search and a seeds file (sometimes with a different step cap); the run that went
    furthest wins so a grok found under the larger cap is not hidden by a capped repeat."""
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for rows in data.values():
        for r in rows:
            c = r["config"]
            key = (*_pool_key(c), c["seed"])
            prev = best.get(key)
            if prev is None or (prev["step_test99"] is None and r["step_test99"] is not None) or (
                    prev["step_test99"] is None and r["config"]["max_steps"] > prev["config"]["max_steps"]):
                best[key] = r
    return list(best.values())


def _opt(c: dict[str, Any]) -> str:
    if c["optimizer"] == "adamw":
        return "AdamW"
    return "SGD+mom" if c["momentum"] > 0 else "GD"


def _arch(c: dict[str, Any]) -> str:
    extra = "+LN" if c["layernorm"] else ""
    act = "" if c["act"] == "relu" else f" {c['act']}"
    heads = f" h{c['n_heads']}" if c["arch"] == "attn" else ""
    return f"{c['arch']}{heads} m{c['d_mlp']}{act}{extra}"


def table(pooled: list[dict[str, Any]], limit: int | None = None) -> str:
    head = ("| p | arch | opt | lr | wd | frac | batch | N/step | steps to 99% test (per seed) "
            "| seeds grokked | stable | train 99% at | test 50% at | Desmos s/step "
            "| est. hours (50-150 ms tick) |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for g in pooled[:limit]:
        c = g["cfg"]
        steps = ", ".join(str(s) if s is not None else f">{cap}"
                          for s, cap in zip(g["steps"], g["caps"], strict=True))
        bs = "full" if c["batch_size"] == 0 else f"{c['batch_size']} {c.get('batch_mode', 'random')}"
        n_step = c["batch_size"] if c["batch_size"] else g["n_train"]
        hours = f"{g['hours'][0]:.2f}-{g['hours'][1]:.2f}" if g["hours"] else "n/a"
        lines.append(
            f"| {c['p']} | {_arch(c)} | {_opt(c)} | {c['lr']:g} | {c['weight_decay']:g} | "
            f"{c['train_frac']} | {bs} | {n_step} | {steps} | {g['n_grokked']}/{g['seeds']} | "
            f"{'yes' if g['stable'] and g['n_grokked'] else 'no' if g['n_grokked'] else '-'} | "
            f"{', '.join(str(s) for s in g['train99'])} | "
            f"{', '.join(str(s) for s in g['test50'])} | "
            f"{g['tick_ms'][0] / 1000:.2f}-{g['tick_ms'][1] / 1000:.2f} | {hours} |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS_DIR / "summary.md"))
    ap.add_argument("--stage", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="rows per stage table")
    a = ap.parse_args(argv)
    data = load(a.stage)
    if not data:
        raise SystemExit("no result files found")
    parts = [f"# Results summary\n\nDesmos seconds per step assume a fixed tick overhead of "
             f"{TICK_MS[0]:.0f}-{TICK_MS[1]:.0f} ms and at most {MAX_TICKS_PER_S:.0f} ticks/s. "
             f"Hours use the slowest seed's steps to 99% test accuracy. A grokking delay shows as 'test 50% at' well after 'train 99% at'; at train fraction >= 0.6 they coincide.\n"]
    merged = combine(data)
    parts.append(f"## All configs, seeds pooled across stages ({len(merged)} distinct runs)\n\n"
                 f"{table(pool_seeds(merged), a.limit)}\n")
    for stage, rows in data.items():
        parts.append(f"## {stage} ({len(rows)} runs)\n\n{table(pool_seeds(rows), a.limit)}\n")
    Path(a.out).write_text("\n".join(parts))
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
