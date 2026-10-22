"""Emit a Desmos graph that TRAINS the one-layer transformer on (a+b) mod 31, one AdamW step
per ticker tick, and shows it grokking.

The math is ref_step.py, expression for expression (that file is checked against torch
autograd). Layout:
  * every parameter and both AdamW moments are flat lists rewritten by the ticker action
    (Desmos actions cannot write matrices larger than 9x9), reshaped to matrices each step;
  * the 31 x 31 pairs (a, b) are a list of 31 matrices, block a holding pair (a, b) in row b;
  * position 3 is always "=", so the query and the third key and value are single rows, and
    positions 1 and 2 use 31-row token tables;
  * the MLP hidden layer is a list of 124 blocks, index 4(a-1)+k for pair block a and
    column block k of the 128 hidden units.
Views: the grid of pairs (grey = training pair, green / red = held-out pair right / wrong),
the number-token embeddings projected onto their strongest Fourier frequency (they form a
circle when the model groks), and the accuracy history.

Usage: python3 gen_grok.py runs/seed0.npz OUT.json [--lr 3e-2] [--wd 1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # language-model/, for the shared passes
from desmos_passes import check_unique, materialize  # noqa: E402 -- needs the path set on the line above

P = 31
FOR_BC = r"\operatorname{for}b=\left[1...31\right];c=\left[1...32\right]"     # 31 x 32 block
FOR_BH = r"\operatorname{for}b=\left[1...31\right];h=\left[1...4\right]"      # 31 x 4 block
FOR_BL = r"\operatorname{for}b=\left[1...31\right];c=\left[1...31\right]"     # 31 x 31 block
OVER_A = r"\operatorname{for}a=\left[1...31\right]"
# parameter key -> (desmos key, rows, cols); flat lists are row-major
def tensors(width: int = 128) -> dict:
    return {
        "W_E": ("e", 32, 32), "W_pos": ("p", 3, 32), "W_Q": ("q", 32, 32), "W_K": ("k", 32, 32),
        "W_V": ("v", 32, 32), "W_O": ("o", 32, 32), "W_in": ("i", 32, width), "b_in": ("b", 1, width),
        "W_out": ("u", width, 32), "b_out": ("c", 1, 32), "W_U": ("y", 32, 31),
    }


TENSORS = tensors()   # key letters and the default (MLP 128) shapes; build() uses the npz's width
# Expressions are written with @K@ (MLP width / 32, the number of hidden blocks) and @W@ (the MLP
# width); build() fills them in.


def num(x: float) -> str:
    """Shortest round-trip decimal, never in exponent form (Desmos LaTeX has no 1e-5)."""
    s = np.format_float_positional(float(x), unique=True, trim="-")
    return "0" if s in ("-0", "0") else s


def lit(values) -> str:
    return r"\left[" + ",".join(num(v) for v in values) + r"\right]"


def bmatrix(m: np.ndarray) -> str:
    return r"\begin{bmatrix}" + r"\\".join("&".join(num(v) for v in row) for row in m) + r"\end{bmatrix}"


def expr(eid: str, latex: str, **extra) -> dict:
    return {"type": "expression", "id": eid, "latex": latex, **extra}


def hid(eid: str, latex: str) -> dict:
    return expr(eid, latex, hidden=True)


def state_lists(init: dict) -> list:
    """Weights w_{key}, AdamW moments m_{key}, v_{key} (literals, so actions can write them)."""
    ex = [expr("ts", r"t_{s}=0")]
    for name, (key, _, _) in TENSORS.items():
        flat = init[name].reshape(-1)
        ex.append(hid(f"w{key}", f"w_{{{key}}}=" + lit(flat)))
        zeros = r"\left[" + ",".join(["0"] * flat.size) + r"\right]"
        ex.append(hid(f"m{key}", f"m_{{{key}}}=" + zeros))
        ex.append(hid(f"v{key}", f"v_{{{key}}}=" + zeros))
    return ex


def helpers() -> list:
    """Per-element functions. Each takes only named, already computed values (Desmos inlines
    function bodies, so heavy arguments would be recomputed)."""
    return [
        hid("fmr", r"m_{r}\left(L,R,C\right)=L\left[C\left(a-1\right)+b\right]\operatorname{for}a=\left[1...R\right];b=\left[1...C\right]"),
        # softmax over the three positions, value for position t, exactly as ref_step
        hid("fsf", r"s_{f}\left(u,v,w,t\right)=\frac{\exp\left(t-\max\left(u,v,w\right)\right)}{\exp\left(u-\max\left(u,v,w\right)\right)"
                   r"+\exp\left(v-\max\left(u,v,w\right)\right)+\exp\left(w-\max\left(u,v,w\right)\right)}"),
        hid("fzf", r"z_{f}\left(X,Y,W,R,S,T\right)=X\left[b;c\right]R\left[1;c\right]+Y\left[b;c\right]S\left[b;c\right]+W\left[b;c\right]T\left[1;c\right]" + FOR_BC),
        hid("fhf", r"h_{f}\left(X,k\right)=\max\left(X\left[b;c\right]+w_{b}\left[32\left(k-1\right)+c\right],0\right)" + FOR_BC),
        hid("fmf", r"m_{f}\left(X\right)=\max\left(\left[X\left[b;c\right]\operatorname{for}c=\left[1...31\right]\right]\right)\operatorname{for}b=\left[1...31\right];j=\left[1\right]"),
        hid("fef", r"e_{f}\left(X,M\right)=\exp\left(X\left[b;c\right]-M\left[b;1\right]\right)" + FOR_BL),
        hid("fnf", r"n_{f}\left(X,S\right)=\frac{X\left[b;c\right]}{S\left[b;1\right]}" + FOR_BL),
        hid("fdf", r"d_{f}\left(X,a\right)=\frac{\left(X\left[b;c\right]-\left\{c=\operatorname{mod}\left(a+b-2,31\right)+1:1,0\right\}\right)T_{r}\left[a;b\right]}{n_{tr}}" + FOR_BL),
        hid("fuf", r"u_{f}\left(X,G\right)=\left\{G\left[b;c\right]>0:X\left[b;c\right],0\right\}" + FOR_BC),
        hid("fxh", r"x_{h}\left(X,Y\right)=X\left[b;c\right]Y\left[b;c\right]" + FOR_BC),
        hid("fxr", r"x_{r}\left(X,R\right)=X\left[b;c\right]R\left[1;c\right]" + FOR_BC),
        hid("fyt", r"y_{t}\left(A,F,B,G,C,J\right)=A\left[b;h\right]F\left[b;h\right]+B\left[b;h\right]G\left[b;h\right]+C\left[b;h\right]J\left[b;h\right]" + FOR_BH),
        hid("fys", r"y_{s}\left(A,F,T\right)=A\left[b;h\right]\left(F\left[b;h\right]-T\left[b;h\right]\right)" + FOR_BH),
        hid("fkq", r"k_{q}\left(X,Q\right)=X\left[a;c\right]Q\left[1;c\right]s_{c}\operatorname{for}a=\left[1...31\right];c=\left[1...32\right]"),
    ]


def constants(train: np.ndarray) -> list:
    return [
        hid("Hs", r"H_{s}=\left\{\operatorname{floor}\left(\frac{a-1}{8}\right)+1=b:1,0\right\}\operatorname{for}a=\left[1...32\right];b=\left[1...4\right]"),
        hid("Ht", r"H_{t}=H_{s}^{T}"),
        hid("On", r"O_{n}=1\operatorname{for}a=\left[1...31\right];b=\left[1\right]"),
        hid("Or", r"O_{r}=O_{n}^{T}"),
        hid("sc", r"s_{c}=\frac{1}{\sqrt{8}}"),
        hid("Tr", r"T_{r}=" + bmatrix(train.astype(int))),
        hid("ntr", f"n_{{tr}}={int(train.sum())}"),
    ]


def params() -> list:
    """Flat state lists reshaped to the matrices the step uses."""
    ex = [hid(f"W{n}", f"W_{{{n}}}=m_{{r}}\\left(w_{{{k}}},{r},{c}\\right)")
          for n, (k, r, c) in (("E", ("e", 32, 32)), ("P", ("p", 3, 32)), ("Q", ("q", 32, 32)),
                               ("K", ("k", 32, 32)), ("V", ("v", 32, 32)), ("O", ("o", 32, 32)),
                               ("Y", ("y", 32, 31)))]
    ex += [
        # W_in column block k: rows a, columns 32(k-1)+b
        hid("WI", r"W_{I}=\left[\left(w_{i}\left[@W@\left(a-1\right)+32\left(k-1\right)+b\right]\operatorname{for}a=\left[1...32\right];b=\left[1...32\right]\right)\operatorname{for}k=\left[1...@K@\right]\right]"),
        # W_out row block k: rows 32(k-1)+a
        hid("WU", r"W_{U}=\left[\left(w_{u}\left[32\left(32\left(k-1\right)+a-1\right)+b\right]\operatorname{for}a=\left[1...32\right];b=\left[1...32\right]\right)\operatorname{for}k=\left[1...@K@\right]\right]"),
        hid("Bo", r"B_{o}=w_{c}\left[b\right]\operatorname{for}a=\left[1\right];b=\left[1...32\right]"),
    ]
    return ex


def forward() -> list:
    return [
        hid("c3", r"c_{3}=W_{E}\left[\left[32\right];\right]+W_{P}\left[\left[3\right];\right]"),
        hid("T1", r"T_{1}=W_{E}\left[a;c\right]+W_{P}\left[1;c\right]\operatorname{for}a=\left[1...31\right];c=\left[1...32\right]"),
        hid("T2", r"T_{2}=W_{E}\left[a;c\right]+W_{P}\left[2;c\right]\operatorname{for}a=\left[1...31\right];c=\left[1...32\right]"),
        hid("qv", r"q_{v}=c_{3}W_{Q}"),
        hid("K1", r"K_{1}=T_{1}W_{K}"), hid("K2", r"K_{2}=T_{2}W_{K}"), hid("k3", r"k_{3}=c_{3}W_{K}"),
        hid("V1", r"V_{1}=T_{1}W_{V}"), hid("V2", r"V_{2}=T_{2}W_{V}"), hid("v3", r"v_{3}=c_{3}W_{V}"),
        # scores per head: (k * q) summed within each head, times 1/sqrt(8)
        hid("S1", r"S_{1}=k_{q}\left(K_{1},q_{v}\right)H_{s}"),
        hid("S2", r"S_{2}=k_{q}\left(K_{2},q_{v}\right)H_{s}"),
        hid("s3", r"s_{3}=\left(k_{3}\left[1;c\right]q_{v}\left[1;c\right]s_{c}\operatorname{for}a=\left[1\right];c=\left[1...32\right]\right)H_{s}"),
        hid("A1", r"A_{1}=\left[\left(s_{f}\left(S_{1}\left[a;h\right],S_{2}\left[b;h\right],s_{3}\left[1;h\right],S_{1}\left[a;h\right]\right)" + FOR_BH + r"\right)" + OVER_A + r"\right]"),
        hid("A2", r"A_{2}=\left[\left(s_{f}\left(S_{1}\left[a;h\right],S_{2}\left[b;h\right],s_{3}\left[1;h\right],S_{2}\left[b;h\right]\right)" + FOR_BH + r"\right)" + OVER_A + r"\right]"),
        hid("A3", r"A_{3}=\left[\left(s_{f}\left(S_{1}\left[a;h\right],S_{2}\left[b;h\right],s_{3}\left[1;h\right],s_{3}\left[1;h\right]\right)" + FOR_BH + r"\right)" + OVER_A + r"\right]"),
        hid("X1", r"X_{1}=A_{1}H_{t}"), hid("X2", r"X_{2}=A_{2}H_{t}"), hid("X3", r"X_{3}=A_{3}H_{t}"),
        hid("Za", r"Z_{a}=\left[z_{f}\left(X_{1}\left[a\right],X_{2}\left[a\right],X_{3}\left[a\right],V_{1}\left[\left[a\right];\right],V_{2},v_{3}\right)" + OVER_A + r"\right]"),
        hid("Ra", r"R_{a}=O_{n}c_{3}+Z_{a}W_{O}"),
        hid("Hd", r"H_{d}=\left[h_{f}\left(R_{a}\left[a\right]W_{I}\left[k\right],k\right)\operatorname{for}k=\left[1...@K@\right],a=\left[1...31\right]\right]"),
        hid("Rb", r"R_{b}=\left[R_{a}\left[a\right]+\operatorname{total}\left(\left[H_{d}\left[@K@\left(a-1\right)+k\right]W_{U}\left[k\right]\operatorname{for}k=\left[1...@K@\right]\right]\right)+O_{n}B_{o}" + OVER_A + r"\right]"),
        hid("Gl", r"G_{l}=R_{b}W_{Y}"),
        hid("Mx", r"M_{x}=\left[m_{f}\left(G_{l}\left[a\right]\right)" + OVER_A + r"\right]"),
        hid("Ex", r"E_{x}=\left[e_{f}\left(G_{l}\left[a\right],M_{x}\left[a\right]\right)" + OVER_A + r"\right]"),
        hid("Sx", r"S_{x}=E_{x}O_{n}"),
        hid("Pb", r"P_{b}=\left[n_{f}\left(E_{x}\left[a\right],S_{x}\left[a\right]\right)" + OVER_A + r"\right]"),
        # per pair: 1 if the label's logit is the row maximum (argmax), as a 31 x 31 matrix [a;b]
        hid("Cg", r"C_{g}=\left\{G_{l}\left[a\right]\left[b;\operatorname{mod}\left(a+b-2,31\right)+1\right]=M_{x}\left[a\right]\left[b;1\right]:1,0\right\}"
                  r"\operatorname{for}a=\left[1...31\right];b=\left[1...31\right]"),
    ]


def backward() -> list:
    return [
        hid("Dl", r"D_{l}=\left[d_{f}\left(P_{b}\left[a\right],a\right)" + OVER_A + r"\right]"),
        hid("Gy", r"G_{y}=\operatorname{total}\left(\left[R_{b}\left[a\right]^{T}D_{l}\left[a\right]" + OVER_A + r"\right]\right)"),
        hid("Db", r"D_{b}=D_{l}W_{Y}^{T}"),
        hid("Gu", r"G_{u}=\left[\operatorname{total}\left(\left[H_{d}\left[@K@\left(a-1\right)+k\right]^{T}D_{b}\left[a\right]" + OVER_A + r"\right]\right)\operatorname{for}k=\left[1...@K@\right]\right]"),
        hid("Gc", r"G_{c}=\operatorname{total}\left(O_{r}D_{b}\right)"),
        hid("Du", r"D_{u}=\left[u_{f}\left(D_{b}\left[a\right]W_{U}\left[k\right]^{T},H_{d}\left[@K@\left(a-1\right)+k\right]\right)\operatorname{for}k=\left[1...@K@\right],a=\left[1...31\right]\right]"),
        hid("Gi", r"G_{i}=\left[\operatorname{total}\left(\left[R_{a}\left[a\right]^{T}D_{u}\left[@K@\left(a-1\right)+k\right]" + OVER_A + r"\right]\right)\operatorname{for}k=\left[1...@K@\right]\right]"),
        hid("Gb", r"G_{b}=\left[\operatorname{total}\left(\left[O_{r}D_{u}\left[@K@\left(a-1\right)+k\right]" + OVER_A + r"\right]\right)\operatorname{for}k=\left[1...@K@\right]\right]"),
        hid("Dr", r"D_{r}=\left[D_{b}\left[a\right]+\operatorname{total}\left(\left[D_{u}\left[@K@\left(a-1\right)+k\right]W_{I}\left[k\right]^{T}\operatorname{for}k=\left[1...@K@\right]\right]\right)" + OVER_A + r"\right]"),
        hid("Go", r"G_{o}=\operatorname{total}\left(\left[Z_{a}\left[a\right]^{T}D_{r}\left[a\right]" + OVER_A + r"\right]\right)"),
        hid("Dz", r"D_{z}=D_{r}W_{O}^{T}"),
        hid("C1", r"C_{1}=\operatorname{total}\left(O_{r}D_{r}\right)"),
        hid("F1", r"F_{1}=\left[x_{r}\left(D_{z}\left[a\right],V_{1}\left[\left[a\right];\right]\right)H_{s}" + OVER_A + r"\right]"),
        hid("F2", r"F_{2}=\left[x_{h}\left(D_{z}\left[a\right],V_{2}\right)H_{s}" + OVER_A + r"\right]"),
        hid("F3", r"F_{3}=\left[x_{r}\left(D_{z}\left[a\right],v_{3}\right)H_{s}" + OVER_A + r"\right]"),
        hid("Q1", r"Q_{1}=\left[O_{r}x_{h}\left(X_{1}\left[a\right],D_{z}\left[a\right]\right)" + OVER_A + r"\right]"),
        hid("J1", r"J_{1}=Q_{1}\left[a\right]\left[1;c\right]\operatorname{for}a=\left[1...31\right];c=\left[1...32\right]"),
        hid("J2", r"J_{2}=\operatorname{total}\left(\left[x_{h}\left(X_{2}\left[a\right],D_{z}\left[a\right]\right)" + OVER_A + r"\right]\right)"),
        hid("j3", r"j_{3}=\operatorname{total}\left(\left[O_{r}x_{h}\left(X_{3}\left[a\right],D_{z}\left[a\right]\right)" + OVER_A + r"\right]\right)"),
        hid("Yt", r"Y_{t}=\left[y_{t}\left(A_{1}\left[a\right],F_{1}\left[a\right],A_{2}\left[a\right],F_{2}\left[a\right],A_{3}\left[a\right],F_{3}\left[a\right]\right)" + OVER_A + r"\right]"),
        hid("Y1", r"Y_{1}=\left[y_{s}\left(A_{1}\left[a\right],F_{1}\left[a\right],Y_{t}\left[a\right]\right)" + OVER_A + r"\right]"),
        hid("Y2", r"Y_{2}=\left[y_{s}\left(A_{2}\left[a\right],F_{2}\left[a\right],Y_{t}\left[a\right]\right)" + OVER_A + r"\right]"),
        hid("Y3", r"Y_{3}=\left[y_{s}\left(A_{3}\left[a\right],F_{3}\left[a\right],Y_{t}\left[a\right]\right)" + OVER_A + r"\right]"),
        hid("W1r", r"W_{1}=O_{r}Y_{1}"),
        hid("U1", r"U_{1}=W_{1}\left[a\right]\left[1;h\right]\operatorname{for}a=\left[1...31\right];h=\left[1...4\right]"),
        hid("U2", r"U_{2}=\operatorname{total}\left(Y_{2}\right)"),
        hid("u3", r"u_{3}=O_{r}\operatorname{total}\left(Y_{3}\right)"),
        hid("L1", r"L_{1}=k_{q}\left(U_{1}H_{t},q_{v}\right)"),
        hid("L2", r"L_{2}=k_{q}\left(U_{2}H_{t},q_{v}\right)"),
        hid("l3", r"l_{3}=\left(\left(u_{3}H_{t}\right)\left[1;c\right]q_{v}\left[1;c\right]s_{c}\operatorname{for}a=\left[1\right];c=\left[1...32\right]\right)"),
        hid("lq", r"l_{q}=O_{r}x_{h}\left(U_{1}H_{t},K_{1}\right)s_{c}+O_{r}x_{h}\left(U_{2}H_{t},K_{2}\right)s_{c}"
                  r"+\left(\left(u_{3}H_{t}\right)\left[1;c\right]k_{3}\left[1;c\right]s_{c}\operatorname{for}a=\left[1\right];c=\left[1...32\right]\right)"),
        hid("Gq", r"G_{q}=c_{3}^{T}l_{q}"),
        hid("Gk", r"G_{k}=T_{1}^{T}L_{1}+T_{2}^{T}L_{2}+c_{3}^{T}l_{3}"),
        hid("Gv", r"G_{v}=T_{1}^{T}J_{1}+T_{2}^{T}J_{2}+c_{3}^{T}j_{3}"),
        hid("I1", r"I_{1}=L_{1}W_{K}^{T}+J_{1}W_{V}^{T}"),
        hid("I2", r"I_{2}=L_{2}W_{K}^{T}+J_{2}W_{V}^{T}"),
        hid("ic", r"i_{c}=C_{1}+l_{q}W_{Q}^{T}+l_{3}W_{K}^{T}+j_{3}W_{V}^{T}"),
        hid("Ge", r"G_{e}=\left\{a\le31:I_{1}\left[a;c\right]+I_{2}\left[a;c\right],i_{c}\left[1;c\right]\right\}\operatorname{for}a=\left[1...32\right];c=\left[1...32\right]"),
        hid("Gp", r"G_{p}=\left\{a=1:\left(O_{r}I_{1}\right)\left[1;c\right],a=2:\left(O_{r}I_{2}\right)\left[1;c\right],i_{c}\left[1;c\right]\right\}\operatorname{for}a=\left[1...3\right];c=\left[1...32\right]"),
    ]


def flat_grads() -> list:
    """Gradients as flat row-major lists matching the state lists (b varies fastest)."""
    def flat(key: str, mat: str, rows: int, cols: int) -> dict:
        return hid(f"g{key}", f"g_{{{key}}}=\\left[{mat}\\left[a;b\\right]\\operatorname{{for}}b=\\left[1...{cols}\\right],a=\\left[1...{rows}\\right]\\right]")
    return [
        flat("e", "G_{e}", 32, 32), flat("p", "G_{p}", 3, 32), flat("q", "G_{q}", 32, 32),
        flat("k", "G_{k}", 32, 32), flat("v", "G_{v}", 32, 32), flat("o", "G_{o}", 32, 32),
        flat("y", "G_{y}", 32, 31), flat("c", "G_{c}", 1, 32),
        # W_in: row a, column 32(k-1)+b lives in block G_i[k]; flat index 128(a-1)+32(k-1)+b
        hid("gi", r"g_{i}=\left[G_{i}\left[k\right]\left[a;b\right]\operatorname{for}b=\left[1...32\right],k=\left[1...@K@\right],a=\left[1...32\right]\right]"),
        hid("gb", r"g_{b}=\left[G_{b}\left[k\right]\left[1;b\right]\operatorname{for}b=\left[1...32\right],k=\left[1...@K@\right]\right]"),
        # W_out: row 32(k-1)+a lives in block G_u[k]
        hid("gu", r"g_{u}=\left[G_{u}\left[k\right]\left[a;b\right]\operatorname{for}b=\left[1...32\right],a=\left[1...32\right],k=\left[1...@K@\right]\right]"),
    ]


def optimizer(lr: float, wd: float) -> list:
    """torch-ordered AdamW on the flat lists, with grok.py's 10-step linear warmup."""
    ex = [
        expr("lr", f"l_{{r}}={num(lr)}"),
        expr("wd", f"w_{{d}}={num(wd)}"),
        hid("lt", r"l_{t}=l_{r}\min\left(\frac{t_{s}+1}{10},1\right)"),
        hid("ss", r"s_{s}=\frac{l_{t}}{1-0.9^{t_{s}+1}}"),
        hid("bc", r"b_{c}=\sqrt{1-0.98^{t_{s}+1}}"),
    ]
    for _, (key, _, _) in TENSORS.items():
        ex += [
            hid(f"nm{key}", f"m_{{n{key}}}=m_{{{key}}}+\\left(1-0.9\\right)\\left(g_{{{key}}}-m_{{{key}}}\\right)"),
            hid(f"nv{key}", f"v_{{n{key}}}=v_{{{key}}}0.98+\\left(1-0.98\\right)g_{{{key}}}g_{{{key}}}"),
            hid(f"nw{key}", f"w_{{n{key}}}=w_{{{key}}}\\left(1-l_{{t}}w_{{d}}\\right)-\\frac{{s_{{s}}m_{{n{key}}}}}{{\\frac{{\\sqrt{{v_{{n{key}}}}}}}{{b_{{c}}}}+0.00000001}}"),
        ]
    assigns = []
    for _, (key, _, _) in TENSORS.items():
        assigns += [f"w_{{{key}}}\\to w_{{n{key}}}", f"m_{{{key}}}\\to m_{{n{key}}}", f"v_{{{key}}}\\to v_{{n{key}}}"]
    assigns.append(r"t_{s}\to t_{s}+1")
    # accuracy history, recorded before the step every 10 steps
    for hist, val in (("h_{t}", "t_{s}"), ("h_{a}", "a_{r}"), ("h_{b}", "a_{e}"), ("h_{k}", "k_{h}"), ("h_{r}", "r_{f}")):
        assigns.append(f"{hist}\\to\\left\\{{\\operatorname{{mod}}\\left(t_{{s}},10\\right)=0:\\operatorname{{join}}\\left({hist},{val}\\right),{hist}\\right\\}}")
    ex.append(hid("As", r"A_{s}=\left(" + ",".join(assigns) + r"\right)"))
    return ex


GRID_X, GRID_Y = 0, 0           # pair grid: cell (a, b) at x = b, y = -a
CIRC_X, CIRC_Y, CIRC_R = 50, -16, 13
SPEC_X, SPEC_Y = 36, -34        # Fourier spectrum bars
HIST_X, HIST_Y, HIST_W, HIST_H = 66, -31, 30, 28
FORM_X, FORM_Y, FORM_W, FORM_H = 0, -43, 96, 5   # formula panel: token c along x, centered logit up
VIEW_W, VIEW_H = 110, 53                          # whole layout, about the 2:1 of a laptop browser


def views() -> list:
    """What the viewer watches: the pair grid, the embedding on its strongest Fourier
    frequency, the spectrum, and the accuracy history."""
    ab = r"\operatorname{for}b=\left[1...31\right],a=\left[1...31\right]"
    return [
        # accuracy: flattened right/wrong grid and train mask, same order as the cells
        hid("cf", r"c_{f}=\left[C_{g}\left[a;b\right]" + ab + r"\right]"),
        hid("tf", r"t_{f}=\left[T_{r}\left[a;b\right]" + ab + r"\right]"),
        hid("ar", r"a_{r}=\frac{\operatorname{total}\left(c_{f}t_{f}\right)}{n_{tr}}"),
        hid("ae", r"a_{e}=\frac{\operatorname{total}\left(c_{f}\left(1-t_{f}\right)\right)}{961-n_{tr}}"),
        hid("qx", r"q_{x}=\left[b" + ab + r"\right]"),
        hid("qy", r"q_{y}=\left[-a" + ab + r"\right]"),
        # cell colours follow the grid view chosen with the buttons (c_z), the readout and history the full network (c_f)
        hid("qc", r"q_{c}=\operatorname{rgb}\left(\left\{t_{f}=1:\left\{c_{z}=1:190,110\right\},\left\{c_{z}=1:46,235\right\}\right\},"
                  r"\left\{t_{f}=1:\left\{c_{z}=1:190,110\right\},\left\{c_{z}=1:160,200\right\}\right\},"
                  r"\left\{t_{f}=1:\left\{c_{z}=1:190,110\right\},\left\{c_{z}=1:67,200\right\}\right\}\right)"),
        expr("grid", r"\operatorname{polygon}\left(\left(q_{x}-1,q_{y}\right),\left(q_{x},q_{y}\right),\left(q_{x},q_{y}+1\right),\left(q_{x}-1,q_{y}+1\right)\right)",
             colorLatex="q_{c}", fillOpacity="1", lines=False,
             clickableInfo={"enabled": True, "latex": r"i_{a}\to-q_{y}\left[\operatorname{index}\right],i_{b}\to q_{x}\left[\operatorname{index}\right]"}),
        # Fourier view: number-token embeddings (rows 1..31 of W_E) against cos / sin of frequency k
        hid("Ck", r"C_{k}=\cos\left(\frac{2\pi a\left(b-1\right)}{31}\right)\operatorname{for}a=\left[1...15\right];b=\left[1...31\right]"),
        hid("Sk", r"S_{k}=\sin\left(\frac{2\pi a\left(b-1\right)}{31}\right)\operatorname{for}a=\left[1...15\right];b=\left[1...31\right]"),
        hid("En", r"E_{n}=W_{E}\left[\left[1...31\right];\right]"),
        hid("Fc", r"F_{c}=C_{k}E_{n}"),
        hid("Fs", r"F_{s}=S_{k}E_{n}"),
        hid("ek", r"e_{k}=\left[\left(F_{c}F_{c}^{T}+F_{s}F_{s}^{T}\right)\left[a;a\right]\operatorname{for}a=\left[1...15\right]\right]"),
        hid("kh", r"k_{h}=\frac{\operatorname{total}\left(\operatorname{sort}\left(e_{k}\right)\left[\left[13,14,15\right]\right]\right)}{\operatorname{total}\left(e_{k}\right)}"),
        hid("kf", r"k_{f}=\left[1...15\right]\left[e_{k}=\max\left(e_{k}\right)\right]\left[1\right]"),
        hid("px", r"p_{x}=\left[\left(E_{n}F_{c}\left[\left[k_{f}\right];\right]^{T}\right)\left[a;1\right]\operatorname{for}a=\left[1...31\right]\right]"),
        hid("py", r"p_{y}=\left[\left(E_{n}F_{s}\left[\left[k_{f}\right];\right]^{T}\right)\left[a;1\right]\operatorname{for}a=\left[1...31\right]\right]"),
        hid("pr", r"p_{r}=\max\left(\sqrt{p_{x}^{2}+p_{y}^{2}}\right)"),
        expr("circ", f"\\left({CIRC_X}+{CIRC_R}\\frac{{p_{{x}}}}{{p_{{r}}}},{CIRC_Y}+{CIRC_R}\\frac{{p_{{y}}}}{{p_{{r}}}}\\right)",
             color="#6042a6", pointSize="9", lines=True, lineOpacity="0.25"),
        expr("spec", f"\\operatorname{{polygon}}\\left(\\left({SPEC_X}+a,{SPEC_Y}\\right),\\left({SPEC_X}+a+0.8,{SPEC_Y}\\right),"
                     f"\\left({SPEC_X}+a+0.8,{SPEC_Y}+6\\frac{{e_{{k}}\\left[a\\right]}}{{\\operatorname{{total}}\\left(e_{{k}}\\right)}}\\right),"
                     f"\\left({SPEC_X}+a,{SPEC_Y}+6\\frac{{e_{{k}}\\left[a\\right]}}{{\\operatorname{{total}}\\left(e_{{k}}\\right)}}\\right)\\right)"
                     r"\operatorname{for}a=\left[1...15\right]", color="#6042a6", fillOpacity="0.6", lines=False),
        # history (state lists written by the step action)
        hid("ht", r"h_{t}=\left[\right]"), hid("ha", r"h_{a}=\left[\right]"),
        hid("hb", r"h_{b}=\left[\right]"), hid("hk", r"h_{k}=\left[\right]"), hid("hr", r"h_{r}=\left[\right]"),
        hid("hs", r"h_{s}=\max\left(\left[200,\max\left(h_{t}\right)\right]\right)"),
        expr("hist_tr", f"\\left({HIST_X}+{HIST_W}\\frac{{h_{{t}}}}{{h_{{s}}}},{HIST_Y}+{HIST_H}h_{{a}}\\right)", color="#888888", pointSize="0", lines=True),
        expr("hist_te", f"\\left({HIST_X}+{HIST_W}\\frac{{h_{{t}}}}{{h_{{s}}}},{HIST_Y}+{HIST_H}h_{{b}}\\right)", color="#2e9e43", pointSize="0", lines=True),
        expr("hist_k", f"\\left({HIST_X}+{HIST_W}\\frac{{h_{{t}}}}{{h_{{s}}}},{HIST_Y}+{HIST_H}h_{{k}}\\right)", color="#6042a6", pointSize="0", lines=True),
        expr("hist_r", f"\\left({HIST_X}+{HIST_W}\\frac{{h_{{t}}}}{{h_{{s}}}},{HIST_Y}+{HIST_H}h_{{r}}\\right)", color="#e07b1a", pointSize="0", lines=True),
        expr("axes", f"\\operatorname{{polygon}}\\left(\\left({HIST_X},{HIST_Y}\\right),\\left({HIST_X}+{HIST_W},{HIST_Y}\\right),"
                     f"\\left({HIST_X}+{HIST_W},{HIST_Y}+{HIST_H}\\right),\\left({HIST_X},{HIST_Y}+{HIST_H}\\right)\\right)",
             color="#cccccc", fillOpacity="0", lineWidth="1"),
        # readout and control
        expr("run", r"r_{n}=0"),
        expr("bgo", f"\\left({HIST_X}+{HIST_W}-6,{HIST_Y}+{HIST_H}+3\\right)", color="#c74440", pointSize="0", pointOpacity="1",
             label="go / stop", showLabel=True, labelOrientation="right",
             clickableInfo={"enabled": True, "latex": r"r_{n}\to1-r_{n}"}),
        # the readout, rounded so it fits above the circle
        hid("oa", r"o_{a}=\operatorname{round}\left(a_{r},3\right)"), hid("oe", r"o_{e}=\operatorname{round}\left(a_{e},3\right)"),
        hid("ok", r"o_{k}=\operatorname{round}\left(k_{h},3\right)"), hid("of", r"o_{f}=\operatorname{round}\left(r_{f},3\right)"),
        expr("rd", f"\\left({CIRC_X}-14,{HIST_Y}+{HIST_H}+3\\right)", color="#000000", pointSize="0", pointOpacity="1",
             label="step ${t_{s}}    train ${o_{a}}    held-out ${o_{e}}    Fourier share ${o_{k}}    formula R² ${o_{f}}", showLabel=True,
             labelOrientation="right"),
    ]


def first_held_out(train: np.ndarray, start: int = 10) -> tuple[int, int]:
    """The first held-out pair (a, b) with a, b >= start in row-major order, 1-based like the
    graph (the formula panel's opening example; a = 0 would make a + b = b, a dull one)."""
    pairs = [(a, b) for a, b in np.argwhere(~train) if a >= start and b >= start]
    a, b = pairs[0] if pairs else np.argwhere(~train)[0]
    return int(a) + 1, int(b) + 1


def formula(pair: tuple[int, int]) -> list:
    """The learned algorithm as a formula. Fit logit(a, b, c) ~ const(a, b) + sum_k alpha_k
    cos(2 pi k (a + b - c) / 31) over all 961 pairs, k the embedding's three key frequencies
    (run_ref.formula_fit), and show one pair's 31 logits as dots on top of the fitted wave.
    The cosine of a + b - c splits into products of cosines and sines of a, b and c, so each
    alpha_k needs only M G M^T per block, M holding cos and sin of the three frequencies."""
    wk = r"\frac{2\pi K_{f}\left[\operatorname{mod}\left(r-1,3\right)+1\right]\left(b-1\right)}{31}"
    at = r"\frac{2\pi w\left(a-1\right)}{31}"
    xs = f"\\frac{{{FORM_W}}}{{30}}"       # graph units per token along the panel
    return [
        hid("Kf", r"K_{f}=\operatorname{sort}\left(\left[1...15\right],-e_{k}\right)\left[1...3\right]"),
        # rows 1-3: cos of the key frequencies at tokens 0..30; rows 4-6: sin, as cos shifted by pi/2
        hid("Mf", r"M_{f}=\cos\left(" + wk + r"-\frac{\pi}{2}\left\{r>3:1,0\right\}\right)\operatorname{for}r=\left[1...6\right];b=\left[1...31\right]"),
        hid("Bf", r"B_{f}=M_{f}G_{l}M_{f}^{T}"),
        hid("fy", r"f_{y}\left(B,k,w\right)=\operatorname{total}\left(\left[\cos\left(" + at + r"\right)\left(B\left[a\right]\left[k;k\right]+B\left[a\right]\left[k+3;k+3\right]\right)"
                  r"-\sin\left(" + at + r"\right)\left(B\left[a\right]\left[k+3;k\right]-B\left[a\right]\left[k;k+3\right]\right)" + OVER_A + r"\right]\right)"),
        # sum over all pairs and c of logit times cos(2 pi k (a + b - c) / 31), per key frequency
        hid("yf", r"y_{f}=\left[f_{y}\left(B_{f},1,K_{f}\left[1\right]\right),f_{y}\left(B_{f},2,K_{f}\left[2\right]\right),f_{y}\left(B_{f},3,K_{f}\left[3\right]\right)\right]"),
        # per-pair-centered sum of squares of the logits: sum of squares minus (row sum)^2 / 31
        hid("Qs", r"Q_{s}=G_{l}O_{n}"),
        hid("st", r"s_{t}=\operatorname{total}\left(\operatorname{trace}\left(G_{l}G_{l}^{T}\right)\right)-\frac{\operatorname{total}\left(\operatorname{trace}\left(Q_{s}^{T}Q_{s}\right)\right)}{31}"),
        # each cosine's sum of squares over all pairs and c is 961 x 31 / 2
        hid("af", r"a_{f}=\frac{y_{f}}{14895.5}"),
        hid("rf", r"r_{f}=\frac{\operatorname{total}\left(y_{f}^{2}\right)}{14895.5s_{t}}"),
        # the pair shown (click a grid cell to change it), its centered logits, and the panel scale
        expr("ia", f"i_{{a}}={pair[0]}"), expr("ib", f"i_{{b}}={pair[1]}"),
        hid("lp", r"l_{p}=\left[G_{l}\left[i_{a}\right]\left[i_{b};c\right]\operatorname{for}c=\left[1...31\right]\right]"),
        hid("lc", r"l_{c}=l_{p}-\operatorname{mean}\left(l_{p}\right)"),
        hid("cs", r"c_{s}=\operatorname{mod}\left(i_{a}+i_{b}-2,31\right)"),
        hid("lm", r"l_{m}=\max\left(\left[\max\left(\left|l_{c}\right|\right),\operatorname{total}\left(\left|a_{f}\right|\right),0.001\right]\right)"),
        expr("fz", f"y={FORM_Y}\\left\\{{{FORM_X}\\le x\\le{FORM_X + FORM_W}\\right\\}}", color="#cccccc", lineWidth="1"),
        expr("fw", f"y={FORM_Y}+\\frac{{{FORM_H}}}{{l_{{m}}}}\\operatorname{{total}}\\left(a_{{f}}\\cos\\left(\\frac{{2\\pi K_{{f}}\\left(i_{{a}}+i_{{b}}-2-\\frac{{x-{FORM_X}}}{{{xs}}}\\right)}}{{31}}\\right)\\right)"
                   f"\\left\\{{{FORM_X}\\le x\\le{FORM_X + FORM_W}\\right\\}}", color="#e07b1a", lineWidth="2.5"),
        expr("fd", f"\\left({FORM_X}+{xs}\\left(\\left[1...31\\right]-1\\right),{FORM_Y}+{FORM_H}\\frac{{l_{{c}}}}{{l_{{m}}}}\\right)", color="#333333", pointSize="7"),
        expr("fc", f"\\left({FORM_X}+{xs}c_{{s}},{FORM_Y}+{FORM_H}\\frac{{l_{{c}}\\left[c_{{s}}+1\\right]}}{{l_{{m}}}}\\right)", color="#2e9e43", pointSize="13"),
        expr("fsel", r"\operatorname{polygon}\left(\left(i_{b}-1,-i_{a}\right),\left(i_{b},-i_{a}\right),\left(i_{b},1-i_{a}\right),\left(i_{b}-1,1-i_{a}\right)\right)",
             color="#e07b1a", fillOpacity="0", lineWidth="3"),
        # plain scalars for the label (labels interpolate names, not expressions)
        hid("fi", r"f_{i}=i_{a}-1"), hid("fj", r"f_{j}=i_{b}-1"),
        *[hid(f"fk{n}", f"f_{{k{n}}}=K_{{f}}\\left[{n}\\right]") for n in (1, 2, 3)],
        *[hid(f"fa{n}", f"f_{{a{n}}}=\\operatorname{{round}}\\left(a_{{f}}\\left[{n}\\right],2\\right)") for n in (1, 2, 3)],
        expr("ft", f"\\left({FORM_X},{FORM_Y + FORM_H + 2}\\right)", color="#000000", pointSize="0", pointOpacity="1", showLabel=True, labelOrientation="right",
             label="pair a = ${f_{i}}, b = ${f_{j}} (click a grid cell to change it): dots = the network's 31 logits over c, green = the answer c = a + b mod 31"),
        expr("ft2", f"\\left({FORM_X},{FORM_Y - FORM_H - 2}\\right)", color="#e07b1a", pointSize="0", pointOpacity="1", showLabel=True, labelOrientation="right",
             label="curve = ${f_{a1}} cos(2π·${f_{k1}}(a+b−c)/31) + ${f_{a2}} cos(2π·${f_{k2}}(a+b−c)/31) + ${f_{a3}} cos(2π·${f_{k3}}(a+b−c)/31)"
                   "   fitted to all 961 pairs, it explains R² = ${r_{f}} of the logits"),
    ]


ABLATION_VIEWS = ("full network", "waves only", "waves removed")


def ablation() -> list:
    """Keep only, or remove, the key-frequency waves in every pair's 31 logits and colour the
    grid by what is left (run_ref.wave_ablation). M_f's six rows are orthogonal with squared
    norm 31 / 2, so (2 / 31) M_f^T M_f projects a row of logits onto the waves."""
    ab = r"\operatorname{for}b=\left[1...31\right],a=\left[1...31\right]"

    def mode(n: int) -> str:
        return f"\\left\\{{z_{{m}}={n}:1,0\\right\\}}"
    ex = [
        hid("Pw", r"P_{w}=\frac{2}{31}M_{f}^{T}M_{f}"),
        hid("Gw", r"G_{w}=G_{l}P_{w}"),
        expr("zm", r"z_{m}=0"),
        # view 0: G_l, 1: G_w, 2: G_l - G_w (0 G + G and G + 0 G are exact, so view 0 is G_l to the bit)
        hid("Gz", f"G_{{z}}=\\left(1-{mode(1)}\\right)G_{{l}}+\\left({mode(1)}-{mode(2)}\\right)G_{{w}}"),
        hid("Mz", r"M_{z}=\left[m_{f}\left(G_{z}\left[a\right]\right)" + OVER_A + r"\right]"),
        hid("Cz", r"C_{z}=\left\{G_{z}\left[a\right]\left[b;\operatorname{mod}\left(a+b-2,31\right)+1\right]=M_{z}\left[a\right]\left[b;1\right]:1,0\right\}"
                  r"\operatorname{for}a=\left[1...31\right];b=\left[1...31\right]"),
        hid("cz", r"c_{z}=\left[C_{z}\left[a;b\right]" + ab + r"\right]"),
        hid("jr", r"j_{r}=\frac{\operatorname{total}\left(c_{z}t_{f}\right)}{n_{tr}}"),
        hid("je", r"j_{e}=\frac{\operatorname{total}\left(c_{z}\left(1-t_{f}\right)\right)}{961-n_{tr}}"),
        hid("ov", r"o_{v}=\operatorname{round}\left(j_{r},3\right)"), hid("ow", r"o_{w}=\operatorname{round}\left(j_{e},3\right)"),
        expr("zrd", f"\\left({GRID_X},{GRID_Y}-32.6\\right)", color="#000000", pointSize="0", pointOpacity="1", showLabel=True,
             labelOrientation="right", label="this grid: train ${o_{v}}   held-out ${o_{w}}"),
    ]
    for n, (name, x) in enumerate(zip(ABLATION_VIEWS, (0, 8.5, 16))):
        ex.append(expr(f"zb{n}", f"\\left({GRID_X + x},{GRID_Y}+1.2\\right)", pointSize="0", pointOpacity="1",
                       colorLatex=f"\\left\\{{z_{{m}}={n}:\\operatorname{{rgb}}\\left(224,123,26\\right),\\operatorname{{rgb}}\\left(130,130,130\\right)\\right\\}}",
                       label=name, showLabel=True, labelOrientation="right",
                       clickableInfo={"enabled": True, "latex": f"z_{{m}}\\to{n}"}))
    return ex


def build(npz: dict, lr: float, wd: float) -> dict:
    init = {k[len("init_"):]: npz[k] for k in npz if k.startswith("init_")}
    width = int(init["W_in"].shape[1]) if "W_in" in init else 0
    if width < 32 or width % 32:
        raise ValueError(f"MLP width must be a positive multiple of 32, got {width}")
    shapes = tensors(width)
    if set(init) != set(shapes):
        raise ValueError(f"init tensors {sorted(init)} do not match {sorted(shapes)}")
    for name, (_, r, c) in shapes.items():
        if init[name].size != r * c:
            raise ValueError(f"{name} has {init[name].size} values, expected {r * c}")
    pairs = npz["pairs"]
    train = np.zeros((P, P), dtype=bool)
    train[pairs[:, 0], pairs[:, 1]] = True
    ex = [{"type": "text", "id": "note", "text": (
        "A one-layer transformer learning (a + b) mod 31 inside Desmos: forward pass, hand-written "
        "backward pass and AdamW, one step per tick.")}]
    ex += state_lists(init) + helpers() + constants(train) + params() + forward() + backward()
    ex += flat_grads() + optimizer(lr, wd) + views() + formula(first_held_out(train)) + ablation()
    for e in ex:
        if "latex" in e:
            e["latex"] = e["latex"].replace("@K@", str(width // 32)).replace("@W@", str(width))
    ex = materialize(ex)
    check_unique(ex)
    return {
        "version": 11,
        "graph": {"viewport": {"xmin": -2, "xmax": VIEW_W - 2, "ymin": 2 - VIEW_H, "ymax": 2},
                  "showGrid": False, "showXAxis": False, "showYAxis": False},
        "expressions": {
            "list": ex,
            # one AdamW step per tick while r_n is 1
            "ticker": {"handlerLatex": r"\left\{r_{n}=1:A_{s}\right\}", "minStepLatex": "0", "open": True, "playing": True},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", type=Path, help="npz written by run_ref.py (initial weights and split)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=1.0)
    args = ap.parse_args()
    with np.load(args.ref) as f:
        npz = {k: f[k] for k in f.files}
    state = build(npz, args.lr, args.wd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB, {len(state['expressions']['list'])} expressions)")


if __name__ == "__main__":
    main()
