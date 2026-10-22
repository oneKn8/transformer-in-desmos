"""Launch groups of training runs: up to 3 parallel workers x 1 thread (the
3-core cap).

Groups:
  lr      learning-rate check on two mid-size configs (1000 steps each)
  grid    d_model {32,64} x layers {1,2,4} x heads {2,4} x ctx {32,64}, in
          ascending order of estimated Desmos ms/token; --min-ms/--max-ms
          select a cost band
  ablate  one-factor ablations around --base (norm, act, tied, bias, pos,
          vocab) plus a second seed of the base for a noise estimate
  shakespeare  the best three Homer frontier configs retrained on
          tiny-shakespeare (vocab s32), same steps and learning rates
Finished runs (runs/<name>/results.json present) are skipped, so a group can
be re-run after an interruption. Each run is claimed with an exclusive-create
runs/<name>/claim file, so two launchers over the same group never train the
same run twice; a run whose claim exists without results (crashed, or owned by
another live launcher) is reported, and deleting its folder releases it. With --deadline-min, runs not yet started when
the deadline passes are skipped and listed as NOT RUN. Logs go to
runs/<name>/log.txt.
"""

from __future__ import annotations

import threadcap  # sets OMP_NUM_THREADS; must be imported before torch (via model)

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from cost_model import token_cost
from model import ModelConfig

HERE = Path(__file__).resolve().parent
MAX_WORKERS = 3
# Generous: the slowest config is ~0.25 s/step idle and the machine may be shared.
TIMEOUT_S_PER_STEP = 1.5
TIMEOUT_MARGIN_S = 300.0


@dataclass(frozen=True)
class Run:
    name: str
    d: int
    layers: int
    heads: int
    ctx: int
    steps: int
    extra: tuple[str, ...] = field(default=())

    def argv(self) -> list[str]:
        return [
            sys.executable, str(HERE / "train.py"), "--name", self.name,
            "--d-model", str(self.d), "--n-layer", str(self.layers), "--n-head", str(self.heads),
            "--ctx", str(self.ctx), "--max-steps", str(self.steps), *self.extra,
        ]


def base_name(d: int, layers: int, heads: int, ctx: int) -> str:
    return f"d{d}_L{layers}_H{heads}_T{ctx}"


def parse_base(spec: str) -> tuple[int, int, int, int]:
    m = re.fullmatch(r"d(\d+)_L(\d+)_H(\d+)_T(\d+)", spec)
    if not m:
        raise ValueError(f"--base must look like d32_L2_H4_T32, got {spec!r}")
    d, layers, heads, ctx = (int(g) for g in m.groups())
    return d, layers, heads, ctx


def grid_cost_ms(d: int, layers: int, heads: int, ctx: int) -> float:
    cfg = ModelConfig(vocab_size=32, ctx=ctx, d_model=d, n_layer=layers, n_head=heads)
    return token_cost(cfg, "full").ms_padded


def parse_lr_by_d(spec: str) -> dict[int, str]:
    """'32=1e-2,64=5e-3' -> {32: '1e-2', 64: '5e-3'}"""
    out: dict[int, str] = {}
    for part in filter(None, spec.split(",")):
        d, _, lr = part.partition("=")
        float(lr)  # validates; raises ValueError on garbage
        out[int(d)] = lr
    return out


def group_runs(group: str, steps: int, base: str | None, lr_by_d: dict[int, str], min_ms: float, max_ms: float) -> list[Run]:
    def lr_args(d: int) -> tuple[str, ...]:
        return ("--lr", lr_by_d[d]) if d in lr_by_d else ()

    if group == "lr":
        return [
            Run(f"lr{lr}_{base_name(d, 2, 4, 32)}", d, 2, 4, 32, 1000, ("--lr", lr))
            for d in (32, 64) for lr in ("1e-3", "3e-3", "1e-2")
        ]
    if group == "grid":
        combos = sorted(itertools.product((32, 64), (1, 2, 4), (2, 4), (32, 64)), key=lambda c: grid_cost_ms(*c))
        return [Run(base_name(*c), *c, steps, lr_args(c[0])) for c in combos if min_ms <= grid_cost_ms(*c) < max_ms]
    if group == "ablate":
        if base is None:
            raise ValueError("ablate needs --base")
        d, layers, heads, ctx = parse_base(base)
        variants = {
            "norm-ln": ("--norm", "ln"),
            "norm-none": ("--norm", "none"),
            "act-relu": ("--act", "relu"),
            "tied": ("--tied",),
            "bias": ("--bias",),
            "pos-alibi": ("--pos", "alibi"),
            "v64": ("--vocab", "v64"),
            "seed1": ("--seed", "1"),
        }
        return [Run(f"{base}_{tag}", d, layers, heads, ctx, steps, args + lr_args(d)) for tag, args in variants.items()]
    if group == "shakespeare":
        configs = ((64, 2, 2, 32), (64, 1, 2, 32), (32, 4, 2, 32))
        return [Run("shk_" + base_name(*c), *c, steps, ("--vocab", "s32") + lr_args(c[0])) for c in configs]
    raise ValueError(f"unknown group {group!r}")


CLAIMED_ELSEWHERE = -2


def run_one(run: Run, deadline: float) -> tuple[str, int | None]:
    """Returns (name, returncode); None = skipped by the deadline,
    CLAIMED_ELSEWHERE = another launcher owns it (or a crash left a claim)."""
    out_dir = HERE / "runs" / run.name
    if (out_dir / "results.json").exists():
        return run.name, 0
    if time.time() > deadline:
        return run.name, None
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_dir / "claim", "x") as claim:
            claim.write(f"pid {os.getpid()}\n")
    except FileExistsError:
        return run.name, CLAIMED_ELSEWHERE
    env = {**os.environ, "CHARLM_THREADS": "1"}
    timeout = run.steps * TIMEOUT_S_PER_STEP + TIMEOUT_MARGIN_S
    with open(out_dir / "log.txt", "w") as log:
        try:
            proc = subprocess.run(
                run.argv(), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE,
                timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout:.0f}s\n")
            return run.name, -1
    return run.name, proc.returncode


def collect(names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        res = json.loads((HERE / "runs" / name / "results.json").read_text())
        res.pop("history", None)
        rows.append(res)
    return rows


def table(rows: list[dict]) -> str:
    head = "| run | params | steps | val bpc (all) | val bpc (ctx>=T/2) | ms/token full (padded / mac) | ms/token kv (padded / mac) |"
    lines = [head, "|" + "---|" * 7]
    for r in sorted(rows, key=lambda r: r["val_bpc_ctx"]):
        c = r["cost"]
        lines.append(
            f"| {r['name']} | {r['params']} | {r['steps']} | {r['val_bpc_all']:.3f} | {r['val_bpc_ctx']:.3f} | "
            f"{c['full']['ms_padded']:.0f} / {c['full']['ms_mac']:.0f} | {c['kv']['ms_padded']:.0f} / {c['kv']['ms_mac']:.0f} |"
        )
    return "\n".join(lines)


def main() -> None:
    threadcap.apply_torch_threads()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("group", choices=("lr", "grid", "ablate", "shakespeare"))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--base", default=None)
    ap.add_argument("--lr-by-d", default="", help="per-width learning rates, e.g. 32=1e-2,64=5e-3")
    ap.add_argument("--min-ms", type=float, default=0.0)
    ap.add_argument("--max-ms", type=float, default=float("inf"))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--deadline-min", type=float, default=float("inf"))
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    args = ap.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f"--workers must be in 1..{MAX_WORKERS} (thread cap)")

    runs = group_runs(args.group, args.steps, args.base, parse_lr_by_d(args.lr_by_d), args.min_ms, args.max_ms)
    deadline = time.time() + args.deadline_min * 60.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(lambda r: run_one(r, deadline), runs))
    failed = [(n, rc) for n, rc in outcomes if rc not in (0, None, CLAIMED_ELSEWHERE)]
    skipped = [n for n, rc in outcomes if rc is None]
    elsewhere = [n for n, rc in outcomes if rc == CLAIMED_ELSEWHERE]
    done = [n for n, rc in outcomes if rc == 0]
    rows = collect(done)
    out = HERE / "results" / f"{args.group}{args.tag}.json"
    out.write_text(json.dumps({"done": rows, "not_run": skipped, "claimed_elsewhere": elsewhere, "failed": failed}, indent=2))
    print(table(rows))
    if skipped:
        print(f"NOT RUN (deadline): {skipped}")
    if elsewhere:
        print(f"CLAIMED ELSEWHERE (other launcher or stale claim): {elsewhere}")
    if failed:
        print(f"FAILED runs (see runs/<name>/log.txt): {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
