"""Run a named sweep stage across a small CPU pool and append results as JSON lines.

The browser benchmark shares this laptop, so the total thread budget is 3: the parent sets
torch/OMP to 3 threads, then spawns at most three workers that inherit OMP_NUM_THREADS=1 and
nice +10 (tiny matmuls gain nothing from more threads). Results land in results/<stage>.jsonl; configs already present are skipped, so
an interrupted stage resumes instead of repeating work.

Usage:
  python3 sweep.py <stage> [--base JSON] [--seeds 0,1] [--workers 3] [--dry-run]
"""

from __future__ import annotations

import os

# Must be set before torch is imported anywhere in this process or its spawned children.
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"

import argparse
import dataclasses
import itertools
import json
import multiprocessing as mp
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from ensemble import group_key, train_ensemble
from grok import RunConfig, train_run

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_P = (23, 31, 47, 59, 97, 113)
PAIR_BUDGET = 12_000  # members x p^2 per ensemble; keeps one ensemble step near 20-40 ms
MAX_MEMBERS = 16
# Stages after the first search: the target is grokking in < 3000 Desmos ticks, so anything
# not at 99% test by 2500 steps is reported as ">2500" rather than chased.
FOCUS = {"max_steps": 2500, "confirm_steps": 100}


def _grid(base: dict[str, Any], **axes: Iterable[Any]) -> list[RunConfig]:
    names = list(axes)
    return [
        RunConfig(**{**base, **dict(zip(names, combo, strict=True))})
        for combo in itertools.product(*(list(v) for v in axes.values()))
    ]


def stage_baseline(base: dict[str, Any]) -> list[RunConfig]:
    """Nanda et al. hyperparameters, shrunk to d_model 32 / d_mlp 128, at every p."""
    nanda = {"arch": "attn", "d_model": 32, "n_heads": 4, "d_mlp": 128, "optimizer": "adamw",
             "lr": 1e-3, "weight_decay": 1.0, "train_frac": 0.3, **base}
    return _grid(nanda, p=ALL_P)


def stage_attn_search(base: dict[str, Any]) -> list[RunConfig]:
    """Fast-grokking search for the attention model: lr x wd x train fraction. Capped at
    5000 steps: anything slower is not a candidate for an in-Desmos run (Nanda's lr 1e-3 is
    covered by the baseline stage)."""
    cfg = {"arch": "attn", "d_mlp": 128, "n_heads": 4, "max_steps": 5000, **base}
    return _grid(cfg, p=(23, 31, 47), lr=(3e-3, 1e-2, 3e-2), weight_decay=(1.0, 3.0),
                 train_frac=(0.3, 0.45, 0.6))


def stage_attn_focus(base: dict[str, Any]) -> list[RunConfig]:
    """Refine the fast regime attn_search found (p=31 only grokked at frac 0.6, best with
    lr 1e-2 wd 3): stronger decay, higher lr, and fractions around 0.6; plus p=23."""
    cfg = {"arch": "attn", "d_mlp": 128, "n_heads": 4, **FOCUS, **base}
    pairs = ((1e-2, 3.0), (1e-2, 6.0), (3e-2, 1.0), (3e-2, 3.0))
    out = [RunConfig(**{**cfg, "p": 31, "lr": lr, "weight_decay": wd, "train_frac": f})
           for lr, wd in pairs for f in (0.5, 0.6, 0.75)]
    out += [RunConfig(**{**cfg, "p": 23, "lr": lr, "weight_decay": wd, "train_frac": f})
            for lr, wd in ((1e-2, 3.0), (3e-2, 1.0)) for f in (0.6, 0.75)]
    return out


def stage_attn_shape(base: dict[str, Any]) -> list[RunConfig]:
    """Width and head count at a fixed good optimizer setting (pass p/lr/wd/frac via --base).
    The (128, 4) cell is the base config itself and is covered by the seeds stage."""
    return [c for c in _grid({"arch": "attn", **FOCUS, **base}, d_mlp=(32, 64, 128),
                             n_heads=(1, 4)) if (c.d_mlp, c.n_heads) != (128, 4)]


def stage_optim(base: dict[str, Any]) -> list[RunConfig]:
    """Is AdamW required? SGD with momentum and plain full-batch GD, swept so that the
    per-step decay lr*wd matches the AdamW range; plus AdamW without weight decay, and a
    LayerNorm+GELU variant to see what removing them costs."""
    cfg = {"arch": "attn", **FOCUS, **base}
    out: list[RunConfig] = []
    for momentum, lrs in ((0.9, (0.1, 0.3, 1.0)), (0.0, (0.3, 1.0, 3.0))):
        for lr, decay in itertools.product(lrs, (1e-2, 3e-2)):
            out.append(RunConfig(**{**cfg, "optimizer": "sgd", "momentum": momentum,
                                    "lr": lr, "weight_decay": decay / lr}))
    out.append(RunConfig(**{**cfg, "weight_decay": 0.0}))
    out.append(RunConfig(**{**cfg, "layernorm": True, "act": "gelu"}))
    out.append(RunConfig(**{**cfg, "act": "gelu"}))
    return out


def stage_sgd_weak_decay(base: dict[str, Any]) -> list[RunConfig]:
    """Follow-up to optim: at AdamW-matched decay (1-3% per step) SGD and GD collapse to
    chance, so retry with 10-100x weaker decay per step (lr*wd of 1e-4 and 1e-3)."""
    cfg = {"arch": "attn", **FOCUS, **base}
    return [RunConfig(**{**cfg, "optimizer": "sgd", "momentum": mom, "lr": lr,
                         "weight_decay": decay / lr})
            for mom, lrs in ((0.9, (0.3, 1.0)), (0.0, (1.0, 3.0)))
            for lr in lrs for decay in (1e-4, 1e-3)]


def stage_mlp_search(base: dict[str, Any]) -> list[RunConfig]:
    """Attention-free baselines at the same p and optimizer as the best attention run (pass p,
    lr, wd via --base)."""
    cfg = {"optimizer": "adamw", **FOCUS, **base}
    return _grid(cfg, arch=("mlp_concat", "mlp_sum", "mlp_folded"), d_mlp=(32, 128),
                 train_frac=(0.45, 0.6, 0.75))


def stage_mlp_optim(base: dict[str, Any]) -> list[RunConfig]:
    """SGD variants for the best MLP-only config (pass arch/p/d_mlp/frac via --base)."""
    out: list[RunConfig] = []
    for momentum, lrs in ((0.9, (0.1, 0.3, 1.0)), (0.0, (0.3, 1.0, 3.0))):
        for lr, decay in itertools.product(lrs, (1e-3, 3e-3, 1e-2)):
            out.append(RunConfig(**{**base, "optimizer": "sgd", "momentum": momentum,
                                    "lr": lr, "weight_decay": decay / lr}))
    return out


def stage_minibatch(base: dict[str, Any]) -> list[RunConfig]:
    """Minibatch vs full batch at the same settings (pass a full config via --base). Batch 32
    is one 32-row tile per tensor in Desmos; cyclic is the order a Desmos build would use."""
    cfg = {**FOCUS, **base}
    return [RunConfig(**{**cfg, "batch_size": 0}),
            *_grid(cfg, batch_size=(32, 64, 128), batch_mode=("random", "cyclic"))]


def stage_seeds(base: dict[str, Any]) -> list[RunConfig]:
    """Re-run one config across seeds; seeds come from --seeds."""
    return [RunConfig(**base)]


def stage_scale_p(base: dict[str, Any]) -> list[RunConfig]:
    """One config applied at every p, to see how steps and cost scale with p."""
    return _grid(base, p=ALL_P)


STAGES: dict[str, Callable[[dict[str, Any]], list[RunConfig]]] = {
    "baseline": stage_baseline,
    "attn_search": stage_attn_search,
    "attn_focus": stage_attn_focus,
    "attn_shape": stage_attn_shape,
    "optim": stage_optim,
    "sgd_weak_decay": stage_sgd_weak_decay,
    "mlp_search": stage_mlp_search,
    "mlp_optim": stage_mlp_optim,
    "minibatch": stage_minibatch,
    "seeds": stage_seeds,
    "scale_p": stage_scale_p,
}


def _run_group(job: tuple[str, list[RunConfig]]) -> tuple[str, list[dict[str, Any]]]:
    out_path, cfgs = job
    # Workers inherit OMP_NUM_THREADS=1 and nice from the parent; this is a second guard.
    torch.set_num_threads(1)
    if cfgs[0].layernorm:  # the stacked trainer has no LayerNorm path
        return out_path, [train_run(c) for c in cfgs]
    return out_path, train_ensemble(cfgs)

def make_groups(configs: list[RunConfig], budget: int = PAIR_BUDGET,
                max_members: int = MAX_MEMBERS) -> list[list[RunConfig]]:
    """Batch same-shape configs into ensembles of at most budget member-pairs each."""
    by_key: dict[tuple[Any, ...], list[RunConfig]] = {}
    for c in configs:
        by_key.setdefault((group_key(c), c.layernorm), []).append(c)
    groups = []
    for members in by_key.values():
        size = max(1, min(max_members, budget // (members[0].p ** 2)))
        groups += [members[i:i + size] for i in range(0, len(members), size)]
    # Largest groups first so the pool's tail is short jobs.
    return sorted(groups, key=lambda g: -len(g) * g[0].p ** 2)


def _done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(RunConfig(**json.loads(line)["config"]).key())
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno} is not a valid result line") from exc
    return keys


def _parse_base(raw: str) -> dict[str, Any]:
    base = json.loads(raw)
    if not isinstance(base, dict):
        raise ValueError("--base must be a JSON object")
    allowed = {f.name for f in dataclasses.fields(RunConfig)}
    unknown = set(base) - allowed
    if unknown:
        raise ValueError(f"unknown RunConfig fields in --base: {sorted(unknown)}")
    return base


def _plan_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.plan:
        entries = json.loads(Path(args.plan).read_text())
        if not isinstance(entries, list):
            raise ValueError("--plan must hold a JSON list of {stage, base, seeds, tag}")
        return entries
    if not args.stage:
        raise ValueError("give a stage or --plan")
    return [{"stage": args.stage, "base": json.loads(args.base), "seeds": args.seeds,
             "tag": args.tag}]


def _collect(entry: dict[str, Any]) -> tuple[Path, list[RunConfig]]:
    stage = entry["stage"]
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    base = _parse_base(json.dumps(entry.get("base", {})))
    seeds = [int(s) for s in str(entry.get("seeds", "0")).split(",") if s.strip()]
    configs = [dataclasses.replace(c, seed=s) for c in STAGES[stage](base) for s in seeds]
    tag = entry.get("tag", "")
    out_path = RESULTS_DIR / f"{stage}{'_' + tag if tag else ''}.jsonl"
    done = _done_keys(out_path)
    todo = [c for c in configs if c.key() not in done]
    print(f"[{stage}{'/' + tag if tag else ''}] {len(configs)} configs, "
          f"{len(configs) - len(todo)} already done, {len(todo)} to run -> {out_path.name}",
          flush=True)
    return out_path, todo


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=sorted(STAGES))
    ap.add_argument("--plan", help="JSON file: list of {stage, base, seeds, tag}, one shared pool")
    ap.add_argument("--base", default="{}", help="JSON object of RunConfig overrides")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= 3:
        raise ValueError("workers must be 1..3 (thread budget shared with the browser)")
    torch.set_num_threads(3)
    RESULTS_DIR.mkdir(exist_ok=True)

    jobs: list[tuple[str, list[RunConfig]]] = []
    for entry in _plan_entries(args):
        out_path, todo = _collect(entry)
        jobs += [(str(out_path), g) for g in make_groups(todo)]
    jobs.sort(key=lambda j: -len(j[1]) * j[1][0].p ** 2)  # big ensembles first, short tail
    n_total = sum(len(g) for _, g in jobs)
    print(f"  {len(jobs)} ensembles, {n_total} runs (sizes {[len(g) for _, g in jobs]})", flush=True)
    if args.dry_run or not jobs:
        return 0

    # Children inherit env and niceness at spawn: one thread each, and at least nice 6 so the
    # browser (nice 0) always wins. Not higher: other sessions' jobs here run at nice 6.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.nice(max(0, 6 - os.nice(0)))
    t0 = time.perf_counter()
    i = 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for out_path, batch in pool.imap_unordered(_run_group, jobs):
            with Path(out_path).open("a") as fh:
                for res in batch:
                    fh.write(json.dumps(res) + "\n")
            for res in batch:
                i += 1
                c = res["config"]
                print(f"  {i}/{n_total} {Path(out_path).stem} p={c['p']} {c['arch']} "
                      f"{c['optimizer']} lr={c['lr']:g} wd={c['weight_decay']:g} "
                      f"mom={c['momentum']} frac={c['train_frac']} m={c['d_mlp']} "
                      f"h={c['n_heads']} bs={c['batch_size']}/{c['batch_mode']} seed={c['seed']} "
                      f"-> {res['status']} test99@{res['step_test99']} "
                      f"train99@{res['step_train99']} ({res['wall_s']}s, "
                      f"{time.perf_counter() - t0:.0f}s elapsed)", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
