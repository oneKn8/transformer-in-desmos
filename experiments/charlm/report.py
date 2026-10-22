"""Build markdown tables from runs/*/results.json into results/tables.md.

Tables: the full grid (configs that were not trained still get their cost
estimate), the quality-vs-cost Pareto frontier, the ablations around the base
config, and the learning-rate check.
"""

from __future__ import annotations

import threadcap  # sets OMP_NUM_THREADS; must be imported before torch (via model)

import argparse
import itertools
import json
import re
from pathlib import Path

from cost_model import token_cost
from model import CharTransformer, ModelConfig

HERE = Path(__file__).resolve().parent
GRID_RE = re.compile(r"d(\d+)_L(\d+)_H(\d+)_T(\d+)")


def load_runs() -> dict[str, dict]:
    runs = {}
    for path in sorted((HERE / "runs").glob("*/results.json")):
        res = json.loads(path.read_text())
        runs[res["name"]] = res
    return runs


def grid_rows(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for d, layers, heads, ctx in itertools.product((32, 64), (1, 2, 4), (2, 4), (32, 64)):
        name = f"d{d}_L{layers}_H{heads}_T{ctx}"
        cfg = ModelConfig(vocab_size=32, ctx=ctx, d_model=d, n_layer=layers, n_head=heads)
        full, kv = token_cost(cfg, "full"), token_cost(cfg, "kv")
        res = runs.get(name)
        rows.append({
            "name": name,
            "params": CharTransformer(cfg).num_params(),
            "blocks": full.padded_blocks,
            "elements": full.elements,
            "ms_full": full.ms_padded,
            "ms_full_mac": full.ms_mac,
            "ms_kv": kv.ms_padded,
            "ms_kv_mac": kv.ms_mac,
            "res": res,
        })
    return sorted(rows, key=lambda r: r["ms_full"])


def fmt_bpc(res: dict | None, key: str) -> str:
    return "not run" if res is None else f"{res[key]:.3f}"


def grid_table(rows: list[dict]) -> str:
    out = [
        "| config | params | full: 32-blocks | full: elementwise | ms/token full (padded / mac) | ms/token kv (padded / mac) | steps | val bpc (all) | val bpc (ctx>=T/2) | test bpc (ctx>=T/2) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        res = r["res"]
        out.append(
            f"| {r['name']} | {r['params']:,} | {r['blocks']} | {r['elements']:,} | {r['ms_full']:.0f} / {r['ms_full_mac']:.0f} | "
            f"{r['ms_kv']:.0f} / {r['ms_kv_mac']:.0f} | {'-' if res is None else res['steps']} | {fmt_bpc(res, 'val_bpc_all')} | "
            f"{fmt_bpc(res, 'val_bpc_ctx')} | {fmt_bpc(res, 'test_bpc_ctx')} |"
        )
    return "\n".join(out)


def frontier_table(rows: list[dict]) -> str:
    """Configs no cheaper config beats on val bpc (ctx), cheapest first."""
    out = ["| config | ms/token full (padded) | val bpc (ctx>=T/2) | params |", "|---|---:|---:|---:|"]
    best = float("inf")
    for r in rows:
        res = r["res"]
        if res is not None and res["val_bpc_ctx"] < best:
            best = res["val_bpc_ctx"]
            out.append(f"| {r['name']} | {r['ms_full']:.0f} | {best:.3f} | {r['params']:,} |")
    return "\n".join(out)


def ms_full(res: dict) -> float:
    """Recomputed from the run's config so every table uses the current cost model."""
    return token_cost(ModelConfig(**res["model"]), "full").ms_padded


def ablation_table(runs: dict[str, dict], base: str) -> str:
    if base not in runs:
        return f"(base run {base} missing)"
    b = runs[base]
    out = [
        "| variant | params | ms/token full (padded) | val bpc (ctx>=T/2) | delta vs base | test bpc (ctx>=T/2) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| base ({base}) | {b['params']:,} | {ms_full(b):.0f} | {b['val_bpc_ctx']:.3f} | 0 | {b['test_bpc_ctx']:.3f} |",
    ]
    for name, res in sorted(runs.items()):
        if name.startswith(base + "_"):
            tag = name[len(base) + 1 :]
            out.append(
                f"| {tag} | {res['params']:,} | {ms_full(res):.0f} | {res['val_bpc_ctx']:.3f} | "
                f"{res['val_bpc_ctx'] - b['val_bpc_ctx']:+.3f} | {res['test_bpc_ctx']:.3f} |"
            )
    return "\n".join(out)


def corpus_table(runs: dict[str, dict]) -> str:
    """Same config and budget on Homer (v32) and tiny-shakespeare (s32)."""
    out = [
        "| config | ms/token full (padded) | Homer val / test bpc (ctx>=T/2) | Shakespeare val / test bpc (ctx>=T/2) |",
        "|---|---:|---:|---:|",
    ]
    for name, res in sorted(runs.items()):
        if name.startswith("shk_") and name[4:] in runs:
            h = runs[name[4:]]
            out.append(
                f"| {name[4:]} | {ms_full(res):.0f} | {h['val_bpc_ctx']:.3f} / {h['test_bpc_ctx']:.3f} | "
                f"{res['val_bpc_ctx']:.3f} / {res['test_bpc_ctx']:.3f} |"
            )
    return "\n".join(out)


def corpus_samples(runs: dict[str, dict]) -> str:
    out = []
    for name, res in sorted(runs.items()):
        if name.startswith("shk_") and name[4:] in runs:
            for tag, r in (("Homer", runs[name[4:]]), ("Shakespeare", res)):
                out.append(f"**{name[4:]}, {tag}** (val bpc {r['val_bpc_ctx']:.3f}):\n\n```\n{r['sample']}\n```\n")
    return "\n".join(out)


def lr_table(runs: dict[str, dict]) -> str:
    out = ["| run | lr | steps | val bpc (ctx>=T/2) |", "|---|---:|---:|---:|"]
    for name, res in sorted(runs.items()):
        if name.startswith("lr"):
            out.append(f"| {name} | {res['train']['lr']:g} | {res['steps']} | {res['val_bpc_ctx']:.3f} |")
    return "\n".join(out)


def samples(rows: list[dict], top: int) -> str:
    trained = sorted((r for r in rows if r["res"] is not None), key=lambda r: r["res"]["val_bpc_ctx"])
    out = []
    for r in trained[:top]:
        res = r["res"]
        out.append(f"**{r['name']}** (val bpc {res['val_bpc_ctx']:.3f}, ~{r['ms_full']:.0f} ms/token):\n\n```\n{res['sample']}\n```\n")
    return "\n".join(out)


def main() -> None:
    threadcap.apply_torch_threads()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="d32_L2_H4_T32")
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()
    if not GRID_RE.fullmatch(args.base):
        raise ValueError("--base must look like d32_L2_H4_T32")
    runs = load_runs()
    rows = grid_rows(runs)
    feasible = [r for r in rows if r["ms_full"] <= 300]
    text = "\n\n".join([
        "## Grid\n\n" + grid_table(rows),
        "## Pareto frontier (val bpc vs padded ms/token)\n\n" + frontier_table(rows),
        f"## Ablations around {args.base}\n\n" + ablation_table(runs, args.base),
        "## Homer vs tiny-shakespeare (1000 steps each)\n\n" + corpus_table(runs),
        "## Corpus samples\n\n" + corpus_samples(runs),
        "## Learning-rate check\n\n" + lr_table(runs),
        "## Samples, best configs at <= 300 ms/token\n\n" + samples(feasible, args.top),
        "## Samples, best configs overall\n\n" + samples(rows, args.top),
    ])
    out = HERE / "results" / "tables.md"
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
