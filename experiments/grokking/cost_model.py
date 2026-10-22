"""Desmos cost of one full-batch training step, itemized from the real tensor shapes.

The step is written the way it would be hand-built in Desmos: per-token tables computed once
(p rows), one-hot matrices (constant data) to gather them per example and to scatter
gradients back, block matmuls on 32x32 tiles, and index comprehensions for anything
elementwise. Costs come from measured Desmos numbers plus labelled assumptions:

  MATMUL_MS   1.3 ms per 32x32 @ 32x32 block matmul (measured)
  ELEM_US     1.0 us per output element of an index comprehension (measured)
  ADD_MS      native 32x32 add or scalar-scale. NOT MEASURED. Assumed 1/32 of a matmul
              (same per-scalar cost as a matmul's multiply-adds); sensitivity printed.
  TICK_MS     fixed 50 to 150 ms overhead per ticker tick (given)

A block matmul (m x k) @ (k x n) costs ceil(m/32)*ceil(k/32)*ceil(n/32) tiles, plus
ceil(m/32)*ceil(n/32)*(ceil(k/32)-1) native adds to accumulate over k. Partial tiles are
charged as full tiles (conservative for p < 32).

Usage:
  python3 cost_model.py --arch attn --p 31 --n-train 432 --d-mlp 128 --heads 4 --opt adamw
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys

MATMUL_MS = 1.3
ELEM_US = 1.0
ADD_MS = MATMUL_MS / 32
TICK_MS = (50.0, 150.0)
MAX_TICKS_PER_S = 6.0
TILE = 32


def _t(x: int) -> int:
    return math.ceil(x / TILE)


@dataclasses.dataclass(frozen=True)
class Op:
    label: str
    tiles: int = 0  # 32x32 block matmuls
    adds: int = 0  # native 32x32 adds / scalar scales
    elems: int = 0  # index-comprehension output elements (summed over passes)


def mm(label: str, m: int, k: int, n: int, count: int = 1) -> Op:
    tiles = _t(m) * _t(k) * _t(n)
    return Op(f"{label} [{m}x{k}]@[{k}x{n}]" + (f" x{count}" if count > 1 else ""),
              tiles=count * tiles, adds=count * _t(m) * _t(n) * (_t(k) - 1))


def ew(label: str, n_elems: int, passes: int = 1) -> Op:
    return Op(f"{label} ({n_elems} elems x {passes} pass)", elems=n_elems * passes)


def add(label: str, rows: int, cols: int, count: int = 1) -> Op:
    return Op(f"{label} [{rows}x{cols}]" + (f" x{count}" if count > 1 else ""),
              adds=count * _t(rows) * _t(cols))


def attn_ops(p: int, n: int, d: int, h: int, m: int) -> list[Op]:
    """Nanda one-layer transformer, logits at "=" only. q is constant across examples (the
    "=" token and position 2 never change), so W_K and q fold into G = W_K Qblk (d x H)."""
    fwd = [
        mm("X0,X1 = E + 1*P_j (row broadcast)", p, 1, d, 2), add("  + E", p, d, 2),
        mm("q = x2 W_Q", 1, d, d), ew("Qblk = blockdiag(q)/sqrt(dh)", d * h),
        mm("G = W_K Qblk", d, d, h),
        mm("s0,s1 = X0 G, X1 G (per-token scores)", p, d, h, 2), mm("s2 = x2 G", 1, d, h),
        mm("V0,V1 = X0 W_V, X1 W_V", p, d, d, 2), mm("v2 = x2 W_V", 1, d, d),
        mm("Sa,Sb = OH_a s0, OH_b s1 (gather)", n, p, h, 2),
        mm("Va,Vb = OH_a V0, OH_b V1 (gather)", n, p, d, 2),
        ew("softmax over 3 positions (exp, normalize)", n * h * 3, 2),
        mm("expand alpha to features: alpha_j R", n, h, d, 3),
        ew("z = Aa*Va + Ab*Vb + A2*v2", n * d, 3),
        mm("z W_O", n, d, d), add("  + x2 (resid)", n, d),
        mm("pre = r W_in", n, d, m), add("  + b_in", n, m),
        ew("h = relu(pre)", n * m),
        mm("h W_out", n, m, d), add("  + r + b_out", n, d, 2),
        mm("logits = r2 W_U", n, d, p),
        ew("dlogits = softmax - onehot (max, exp, normalize)", n * p, 3),
        mm("  row sums via ones", n, p, 1),
    ]
    bwd = [
        mm("dW_U = r2^T dlogits", d, n, p), mm("dr2 = dlogits W_U^T", n, p, d),
        mm("dW_out = h^T dr2", m, n, d), mm("db_out = 1^T dr2", 1, n, d),
        mm("dh = dr2 W_out^T", n, d, m), ew("dpre = dh * [pre>0]", n * m),
        mm("dW_in = r^T dpre", d, n, m), mm("db_in = 1^T dpre", 1, n, m),
        mm("dpre W_in^T", n, m, d), add("  dr = dr2 + ...", n, d),
        mm("dW_O = z^T dr", d, n, d), mm("dz = dr W_O^T", n, d, d),
        ew("dz * V_j for j=a,b,2", n * d, 3), mm("dalpha_j = (dz*V_j) R^T", n, d, h, 3),
        ew("dVa, dVb = Aa*dz, Ab*dz; A2*dz", n * d, 3), mm("dv2 = 1^T (A2*dz)", 1, n, d),
        ew("softmax backward ds_j = a_j(da_j - sum a da)", n * h * 3, 2),
        mm("dV0,dV1 = OH^T dVa, OH^T dVb (scatter)", p, n, d, 2),
        mm("ds0,ds1 = OH^T dSa, OH^T dSb (scatter)", p, n, h, 2),
        mm("ds2, dx2_resid = column sums", 1, n, h + d),
        mm("dG = X0^T ds0 + X1^T ds1", d, p, h, 2), mm("dG += x2^T ds2", d, 1, h),
        mm("dW_K = dG Qblk^T", d, h, d), mm("dQblk = W_K^T dG", d, d, h), ew("dq from dQblk", d * h),
        mm("dW_Q = x2^T dq", d, 1, d), mm("dx2 += dq W_Q^T + dv2 W_V^T + ds2 G^T", 1, d, d, 3),
        mm("dW_V = X0^T dV0 + X1^T dV1", d, p, d, 2), mm("dW_V += x2^T dv2", d, 1, d),
        mm("dX_j = ds_j G^T + dV_j W_V^T", p, d, d, 4), add("  dE = dX0 + dX1", p, d),
        mm("dP_j = column sums of dX_j", 1, p, d, 2),
    ]
    return fwd + bwd


def mlp_ops(arch: str, p: int, n: int, d: int, m: int) -> list[Op]:
    """Attention-free models. concat/sum build two per-token tables T_a, T_b (p x m) from the
    embedding; folded learns them directly. The per-example work is identical."""
    tables: list[Op] = []
    table_bwd: list[Op] = []
    if arch in ("mlp_concat", "mlp_sum"):
        if arch == "mlp_sum":
            tables += [mm("E + 1*P_j", p, 1, d, 2), add("  + E", p, d, 2)]
        tables.append(mm("T_a,T_b = E W_top, E W_bot", p, d, m, 2))
        table_bwd = [mm("dW = E^T dT_j", d, p, m, 2), mm("dE = dT_j W_j^T", p, m, d, 2),
                     add("  dE sum", p, d)]
        if arch == "mlp_sum":
            table_bwd.append(mm("dP_j column sums", 1, p, d, 2))
    elif arch != "mlp_folded":
        raise ValueError(f"unknown MLP arch {arch!r}")
    fwd = [*tables,
           mm("pre = OH_a T_a + OH_b T_b (gather)", n, p, m, 2), add("  + b_in", n, m, 2),
           ew("h = relu(pre)", n * m), mm("logits = h W_U", n, m, p),
           ew("dlogits = softmax - onehot", n * p, 3), mm("  row sums via ones", n, p, 1)]
    bwd = [mm("dW_U = h^T dlogits", m, n, p), mm("dh = dlogits W_U^T", n, p, m),
           ew("dpre = dh * [pre>0]", n * m), mm("dT_a,dT_b = OH^T dpre (scatter)", p, n, m, 2),
           mm("db_in = 1^T dpre", 1, n, m), *table_bwd]
    return fwd + bwd


def param_shapes(arch: str, p: int, d: int, m: int) -> list[tuple[int, int]]:
    if arch == "attn":
        return [(p + 1, d), (3, d), (d, d), (d, d), (d, d), (d, d), (d, m), (1, m), (m, d),
                (1, d), (d, p)]
    if arch == "mlp_concat":
        return [(p, d), (d, m), (d, m), (1, m), (m, p)]
    if arch == "mlp_sum":
        return [(p, d), (2, d), (d, m), (1, m), (m, p)]
    if arch == "mlp_folded":
        return [(p, m), (p, m), (1, m), (m, p)]
    raise ValueError(f"unknown arch {arch!r}")


def optimizer_ops(opt: str, shapes: list[tuple[int, int]], momentum: float) -> list[Op]:
    """SGD runs entirely on native matrix add/scale. AdamW needs g*g, sqrt and a divide,
    which are elementwise, so it pays comprehension cost on every parameter."""
    blocks = sum(_t(r) * _t(c) for r, c in shapes)
    n_params = sum(r * c for r, c in shapes)
    if opt == "sgd":
        per_block = 4 if momentum > 0 else 3
        return [Op(f"SGD{' momentum' if momentum > 0 else ''} update ({blocks} blocks x "
                   f"{per_block} native ops)", adds=blocks * per_block)]
    if opt == "adamw":
        return [Op(f"AdamW m, v, decay, step ({blocks} blocks x 8 native ops)", adds=blocks * 8),
                ew("AdamW g*g, sqrt, divide", n_params, 3)]
    raise ValueError(f"unknown optimizer {opt!r}")


@dataclasses.dataclass(frozen=True)
class StepCost:
    tiles: int
    adds: int
    elems: int
    n_params: int

    def compute_ms(self, add_ms: float = ADD_MS) -> float:
        return self.tiles * MATMUL_MS + self.adds * add_ms + self.elems * ELEM_US / 1000

    def tick_ms(self, overhead_ms: float, add_ms: float = ADD_MS) -> float:
        """One training step per tick, never faster than the ticker's best rate."""
        return max(self.compute_ms(add_ms) + overhead_ms, 1000 / MAX_TICKS_PER_S)


def step_ops(arch: str, p: int, n_train: int, d: int, heads: int, d_mlp: int,
             opt: str, momentum: float) -> list[Op]:
    model = attn_ops(p, n_train, d, heads, d_mlp) if arch == "attn" else mlp_ops(
        arch, p, n_train, d, d_mlp)
    return model + optimizer_ops(opt, param_shapes(arch, p, d, d_mlp), momentum)


def step_cost(arch: str, p: int, n_train: int, d: int = 32, heads: int = 4, d_mlp: int = 128,
              opt: str = "adamw", momentum: float = 0.0) -> StepCost:
    ops = step_ops(arch, p, n_train, d, heads, d_mlp, opt, momentum)
    return StepCost(tiles=sum(o.tiles for o in ops), adds=sum(o.adds for o in ops),
                    elems=sum(o.elems for o in ops),
                    n_params=sum(r * c for r, c in param_shapes(arch, p, d, d_mlp)))


def itemize(ops: list[Op]) -> str:
    lines = ["| op | 32x32 matmuls | native adds | elementwise elems |", "|---|---:|---:|---:|"]
    lines += [f"| {o.label} | {o.tiles} | {o.adds} | {o.elems} |" for o in ops]
    lines.append(f"| **total** | **{sum(o.tiles for o in ops)}** | **{sum(o.adds for o in ops)}**"
                 f" | **{sum(o.elems for o in ops)}** |")
    return "\n".join(lines)


def report(arch: str, p: int, n_train: int, d: int, heads: int, d_mlp: int, opt: str,
           momentum: float, steps: int | None) -> str:
    ops = step_ops(arch, p, n_train, d, heads, d_mlp, opt, momentum)
    c = step_cost(arch, p, n_train, d, heads, d_mlp, opt, momentum)
    mm_ms, add_ms_total, el_ms = c.tiles * MATMUL_MS, c.adds * ADD_MS, c.elems * ELEM_US / 1000
    out = [f"### {arch}, p={p}, N_train={n_train}, d={d}, heads={heads}, d_mlp={d_mlp}, {opt}"
           f"{' (momentum)' if momentum else ''}", "", itemize(ops), "",
           f"- matmuls: {c.tiles} x {MATMUL_MS} ms = {mm_ms:.0f} ms",
           f"- native adds: {c.adds} x {ADD_MS:.3f} ms = {add_ms_total:.0f} ms "
           f"(if adds cost a full matmul: {c.adds * MATMUL_MS:.0f} ms)",
           f"- elementwise: {c.elems} elems x {ELEM_US} us = {el_ms:.0f} ms",
           f"- compute per step: {c.compute_ms():.0f} ms "
           f"= {c.tiles + c.adds / 32 + c.elems * ELEM_US / 1000 / MATMUL_MS:.0f} matmul-equivalents",
           f"- per tick with {TICK_MS[0]:.0f} to {TICK_MS[1]:.0f} ms overhead: "
           f"{c.tick_ms(TICK_MS[0]):.0f} to {c.tick_ms(TICK_MS[1]):.0f} ms"]
    if steps is not None:
        lo, hi = (steps * c.tick_ms(t) / 3.6e6 for t in TICK_MS)
        out.append(f"- {steps} steps to grok: {lo:.2f} to {hi:.2f} hours "
                   f"({steps} x {c.tick_ms(TICK_MS[0]):.0f} ms to {steps} x {c.tick_ms(TICK_MS[1]):.0f} ms)")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arch", required=True, choices=("attn", "mlp_concat", "mlp_sum", "mlp_folded"))
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-mlp", type=int, default=128)
    ap.add_argument("--opt", choices=("adamw", "sgd"), default="adamw")
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args(argv)
    print(report(a.arch, a.p, a.n_train, a.d_model, a.heads, a.d_mlp, a.opt, a.momentum, a.steps))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
