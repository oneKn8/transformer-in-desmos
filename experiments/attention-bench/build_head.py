"""Build the 32x32 causal attention-head benchmark: Desmos graph states plus a numpy reference.

Both sides use the same closed-form weights (scaled sines of the row and column index), so no
weights have to be shipped and the Desmos result can be checked number for number.

The head, T = d = 32, 1-based indices a (row) and b (column):
    X        = sin(0.3a + 0.7b + s)                      input block, depends on the slider s
    W_q..W_o = 0.15 sin(ka*a + kb*b + k0)               fixed projections
    S        = (X W_q)(X W_k)^T / sqrt(d)
    E        = exp(S - max S) where b <= a, else 0      causal mask; a global shift is exact for softmax
    A        = E / rowsum(E)                             attention pattern
    Y        = A (X W_v) W_o

Two layouts of the same math:
    readable  every intermediate is its own named expression, plus a heatmap of A
    fused     the head is a chain of hidden functions and only a checksum is displayed; this is
              the fast layout (the expression list re-renders every displayed matrix on each update)

Desmos flattens `M[a;b] for a=..., b=...` with a varying fastest (column-major), so the reference
is flattened with order="F". The states are written in the format `Calc.setState` takes; plain
`Calc.setExpressions` silently drops the heatmap's per-cell colors (`colorLatex`).

Usage: python3 build_head.py OUT_DIR [--seed 0.05]
"""
import argparse
import json
from pathlib import Path

import numpy as np

T = 32
D = 32
SCALE = 0.15
# (row coefficient, column coefficient, phase) for each projection
PROJ = {
    "q": (1.1, 2.3, 0.1),
    "k": (1.7, 0.9, 0.2),
    "v": (0.5, 1.9, 0.3),
    "o": (2.1, 0.4, 0.4),
}

OVER_A = r"a=\left[1...32\right]"
OVER_B = r"b=\left[1...32\right]"
# `;` between the two ranges builds a 32x32 matrix, `,` builds a flat 1024 list
AS_MATRIX = r"\operatorname{for}" + OVER_A + ";" + OVER_B
AS_LIST = r"\operatorname{for}" + OVER_A + "," + OVER_B
VIEWPORT = {"xmin": -1, "xmax": 58, "ymin": -33.5, "ymax": 1.5}


def reference(seed: float) -> dict:
    a = np.arange(1, T + 1, dtype=np.float64)[:, None]
    b = np.arange(1, D + 1, dtype=np.float64)[None, :]
    x = np.sin(0.3 * a + 0.7 * b + seed)
    w = {k: SCALE * np.sin(ka * a + kb * b + k0) for k, (ka, kb, k0) in PROJ.items()}
    s = ((x @ w["q"]) @ (x @ w["k"]).T) / np.sqrt(D)
    e = np.where(b <= a, np.exp(s - s.max()), 0.0)
    att = e / e.sum(axis=1, keepdims=True)
    y = (att @ (x @ w["v"])) @ w["o"]
    return {
        "seed": seed,
        "Y": y.flatten(order="F").tolist(),
        "A": att.flatten(order="F").tolist(),
    }


def expr(eid: str, latex: str, **style) -> dict:
    return {"type": "expression", "id": eid, "latex": latex, **style}


def header(seed: float, note: str) -> list:
    return [
        {"type": "text", "id": "note", "text": note},
        expr("s", f"s={seed}", slider={"hardMin": True, "hardMax": True, "min": "0", "max": "6.28"}),
    ]


def weights() -> list:
    ex = [
        expr(f"W{k}", rf"W_{{{k}}}={SCALE}\sin\left({ka}a+{kb}b+{k0}\right)" + AS_MATRIX)
        for k, (ka, kb, k0) in PROJ.items()
    ]
    ex.append(expr("U", r"U=1\operatorname{for}" + OVER_A + r";b=\left[1\right]"))
    return ex


def readable(seed: float) -> list:
    note = (
        "One causal attention head, T=32, d=32, on native Desmos matrices. Weights are closed-form "
        "sines so numpy can reproduce them exactly. Heatmap: the attention pattern A (row = query "
        "position, column = key position). Drag s to change the input."
    )
    # the attention math comes first and the big input and weight matrices last, so the list
    # reads top-down next to the heatmap
    return header(seed, note) + [
        expr("Q", r"Q=XW_{q}"),
        expr("K", r"K=XW_{k}"),
        expr("V", r"V=XW_{v}"),
        expr("S", r"S=\frac{QK^{T}}{\sqrt{32}}"),
        expr("m", r"m=\max\left(\left[S\left[a;b\right]" + AS_LIST + r"\right]\right)"),
        expr("E", r"E=\left\{b\le a:\exp\left(S\left[a;b\right]-m\right),0\right\}" + AS_MATRIX),
        expr("Z", r"Z=EU"),
        expr("A", r"A=\frac{E\left[a;b\right]}{Z\left[a;1\right]}" + AS_MATRIX),
        expr("Y", r"Y=AVW_{o}"),
        expr("yflat", r"y_{flat}=\left[Y\left[a;b\right]" + AS_LIST + r"\right]"),
        expr("aflat", r"p_{A}=\left[A\left[a;b\right]" + AS_LIST + r"\right]"),
        # cell (row a, column b) is drawn at x=b, y=-a so the matrix reads like the printed one
        expr("hu", r"h_{u}=\left[b" + AS_LIST + r"\right]"),
        expr("hv", r"h_{v}=\left[-a" + AS_LIST + r"\right]"),
        expr(
            "hc",
            r"h_{c}=\operatorname{rgb}\left(255\left(1-p_{A}^{0.5}\right),"
            r"255\left(1-0.8p_{A}^{0.5}\right),255\right)",
        ),
        expr(
            "heat",
            r"\operatorname{polygon}\left(\left(h_{u},h_{v}\right),\left(h_{u}+1,h_{v}\right),"
            r"\left(h_{u}+1,h_{v}+1\right),\left(h_{u},h_{v}+1\right)\right)",
            colorLatex="h_{c}",
            fillOpacity="1",
            lines=False,
        ),
        expr("X", r"X=\sin\left(0.3a+0.7b+s\right)" + AS_MATRIX),
    ] + weights()


def fused(seed: float) -> list:
    note = (
        "The same attention head as the readable graph, written as a chain of hidden functions so "
        "no intermediate matrix is displayed. c is a checksum (Y[32;32]); y_f is the full output."
    )
    # A comprehension passed straight in as a function argument is misparsed, so each one is
    # wrapped in its own small function (m_x, e_x, n_r).
    fns = [
        ("gx", r"g_{X}\left(u\right)=\sin\left(0.3a+0.7b+u\right)" + AS_MATRIX),
        ("mx", r"m_{x}\left(S\right)=\max\left(\left[S\left[a;b\right]" + AS_LIST + r"\right]\right)"),
        ("ex", r"e_{x}\left(S,m\right)=\left\{b\le a:\exp\left(S\left[a;b\right]-m\right),0\right\}" + AS_MATRIX),
        ("nr", r"n_{r}\left(E,Z\right)=\frac{E\left[a;b\right]}{Z\left[a;1\right]}" + AS_MATRIX),
        ("f1", r"f_{1}\left(X\right)=f_{2}\left(X,\frac{\left(XW_{q}\right)\left(XW_{k}\right)^{T}}{\sqrt{32}}\right)"),
        ("f2", r"f_{2}\left(X,S\right)=f_{3}\left(X,e_{x}\left(S,m_{x}\left(S\right)\right)\right)"),
        ("f3", r"f_{3}\left(X,E\right)=n_{r}\left(E,EU\right)XW_{v}W_{o}"),
    ]
    # one-variable function definitions get plotted as curves unless hidden
    return header(seed, note) + weights() + [expr(eid, latex, hidden=True) for eid, latex in fns] + [
        expr("c", r"c=f_{1}\left(g_{X}\left(s\right)\right)\left[32;32\right]"),
        expr("yf", r"y_{f}=\left[f_{1}\left(g_{X}\left(s\right)\right)\left[a;b\right]" + AS_LIST + r"\right]"),
    ]


def state(expressions: list) -> dict:
    return {"version": 11, "graph": {"viewport": VIEWPORT}, "expressions": {"list": expressions}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--seed", type=float, default=0.05)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "head32_readable.json": state(readable(args.seed)),
        "head32_fused.json": state(fused(args.seed)),
        "head32_ref.json": reference(args.seed),
    }
    for name, payload in outputs.items():
        (args.out_dir / name).write_text(json.dumps(payload))
    print(f"wrote {', '.join(outputs)} to {args.out_dir} (seed {args.seed})")


if __name__ == "__main__":
    main()
