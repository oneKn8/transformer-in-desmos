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
from pathlib import Path

import numpy as np

P = 31
FOR_BC = r"\operatorname{for}b=\left[1...31\right];c=\left[1...32\right]"     # 31 x 32 block
FOR_BH = r"\operatorname{for}b=\left[1...31\right];h=\left[1...4\right]"      # 31 x 4 block
FOR_BL = r"\operatorname{for}b=\left[1...31\right];c=\left[1...31\right]"     # 31 x 31 block
OVER_A = r"\operatorname{for}a=\left[1...31\right]"
# parameter key -> (desmos key, rows, cols); flat lists are row-major
TENSORS = {
    "W_E": ("e", 32, 32), "W_pos": ("p", 3, 32), "W_Q": ("q", 32, 32), "W_K": ("k", 32, 32),
    "W_V": ("v", 32, 32), "W_O": ("o", 32, 32), "W_in": ("i", 32, 128), "b_in": ("b", 1, 128),
    "W_out": ("u", 128, 32), "b_out": ("c", 1, 32), "W_U": ("y", 32, 31),
}


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
        hid("fzf", r"z_{f}\left(X,Y,W,a\right)=X\left[b;c\right]V_{1}\left[a;c\right]+Y\left[b;c\right]V_{2}\left[b;c\right]+W\left[b;c\right]v_{3}\left[1;c\right]" + FOR_BC),
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
        hid("fkq", r"k_{q}\left(X\right)=X\left[a;c\right]q_{v}\left[1;c\right]s_{c}\operatorname{for}a=\left[1...31\right];c=\left[1...32\right]"),
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
        hid("WI", r"W_{I}=\left[\left(w_{i}\left[128\left(a-1\right)+32\left(k-1\right)+b\right]\operatorname{for}a=\left[1...32\right];b=\left[1...32\right]\right)\operatorname{for}k=\left[1...4\right]\right]"),
        # W_out row block k: rows 32(k-1)+a
        hid("WU", r"W_{U}=\left[\left(w_{u}\left[32\left(32\left(k-1\right)+a-1\right)+b\right]\operatorname{for}a=\left[1...32\right];b=\left[1...32\right]\right)\operatorname{for}k=\left[1...4\right]\right]"),
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
        hid("S1", r"S_{1}=k_{q}\left(K_{1}\right)H_{s}"),
        hid("S2", r"S_{2}=k_{q}\left(K_{2}\right)H_{s}"),
        hid("s3", r"s_{3}=\left(k_{3}\left[1;c\right]q_{v}\left[1;c\right]s_{c}\operatorname{for}a=\left[1\right];c=\left[1...32\right]\right)H_{s}"),
        hid("A1", r"A_{1}=\left[\left(s_{f}\left(S_{1}\left[a;h\right],S_{2}\left[b;h\right],s_{3}\left[1;h\right],S_{1}\left[a;h\right]\right)" + FOR_BH + r"\right)" + OVER_A + r"\right]"),
        hid("A2", r"A_{2}=\left[\left(s_{f}\left(S_{1}\left[a;h\right],S_{2}\left[b;h\right],s_{3}\left[1;h\right],S_{2}\left[b;h\right]\right)" + FOR_BH + r"\right)" + OVER_A + r"\right]"),
        hid("A3", r"A_{3}=\left[\left(s_{f}\left(S_{1}\left[a;h\right],S_{2}\left[b;h\right],s_{3}\left[1;h\right],s_{3}\left[1;h\right]\right)" + FOR_BH + r"\right)" + OVER_A + r"\right]"),
        hid("X1", r"X_{1}=A_{1}H_{t}"), hid("X2", r"X_{2}=A_{2}H_{t}"), hid("X3", r"X_{3}=A_{3}H_{t}"),
        hid("Za", r"Z_{a}=\left[z_{f}\left(X_{1}\left[a\right],X_{2}\left[a\right],X_{3}\left[a\right],a\right)" + OVER_A + r"\right]"),
        hid("Ra", r"R_{a}=O_{n}c_{3}+Z_{a}W_{O}"),
        hid("Hd", r"H_{d}=\left[h_{f}\left(R_{a}\left[a\right]W_{I}\left[k\right],k\right)\operatorname{for}k=\left[1...4\right],a=\left[1...31\right]\right]"),
        hid("Rb", r"R_{b}=\left[R_{a}\left[a\right]+\operatorname{total}\left(\left[H_{d}\left[4\left(a-1\right)+k\right]W_{U}\left[k\right]\operatorname{for}k=\left[1...4\right]\right]\right)+O_{n}B_{o}" + OVER_A + r"\right]"),
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
        hid("Gu", r"G_{u}=\left[\operatorname{total}\left(\left[H_{d}\left[4\left(a-1\right)+k\right]^{T}D_{b}\left[a\right]" + OVER_A + r"\right]\right)\operatorname{for}k=\left[1...4\right]\right]"),
        hid("Gc", r"G_{c}=\operatorname{total}\left(O_{r}D_{b}\right)"),
        hid("Du", r"D_{u}=\left[u_{f}\left(D_{b}\left[a\right]W_{U}\left[k\right]^{T},H_{d}\left[4\left(a-1\right)+k\right]\right)\operatorname{for}k=\left[1...4\right],a=\left[1...31\right]\right]"),
        hid("Gi", r"G_{i}=\left[\operatorname{total}\left(\left[R_{a}\left[a\right]^{T}D_{u}\left[4\left(a-1\right)+k\right]" + OVER_A + r"\right]\right)\operatorname{for}k=\left[1...4\right]\right]"),
        hid("Gb", r"G_{b}=\left[\operatorname{total}\left(\left[O_{r}D_{u}\left[4\left(a-1\right)+k\right]" + OVER_A + r"\right]\right)\operatorname{for}k=\left[1...4\right]\right]"),
        hid("Dr", r"D_{r}=\left[D_{b}\left[a\right]+\operatorname{total}\left(\left[D_{u}\left[4\left(a-1\right)+k\right]W_{I}\left[k\right]^{T}\operatorname{for}k=\left[1...4\right]\right]\right)" + OVER_A + r"\right]"),
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
        hid("L1", r"L_{1}=k_{q}\left(U_{1}H_{t}\right)"),
        hid("L2", r"L_{2}=k_{q}\left(U_{2}H_{t}\right)"),
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
        hid("gi", r"g_{i}=\left[G_{i}\left[k\right]\left[a;b\right]\operatorname{for}b=\left[1...32\right],k=\left[1...4\right],a=\left[1...32\right]\right]"),
        hid("gb", r"g_{b}=\left[G_{b}\left[k\right]\left[1;b\right]\operatorname{for}b=\left[1...32\right],k=\left[1...4\right]\right]"),
        # W_out: row 32(k-1)+a lives in block G_u[k]
        hid("gu", r"g_{u}=\left[G_{u}\left[k\right]\left[a;b\right]\operatorname{for}b=\left[1...32\right],a=\left[1...32\right],k=\left[1...4\right]\right]"),
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
    ex.append(hid("As", r"A_{s}=\left(" + ",".join(assigns) + r"\right)"))
    return ex


def build(npz: dict, lr: float, wd: float) -> dict:
    init = {k[len("init_"):]: npz[k] for k in npz if k.startswith("init_")}
    if set(init) != set(TENSORS):
        raise ValueError(f"init tensors {sorted(init)} do not match {sorted(TENSORS)}")
    for name, (_, r, c) in TENSORS.items():
        if init[name].size != r * c:
            raise ValueError(f"{name} has {init[name].size} values, expected {r * c}")
    pairs = npz["pairs"]
    train = np.zeros((P, P), dtype=bool)
    train[pairs[:, 0], pairs[:, 1]] = True
    ex = [{"type": "text", "id": "note", "text": (
        "A one-layer transformer learning (a + b) mod 31 inside Desmos: forward pass, hand-written "
        "backward pass and AdamW, one step per tick.")}]
    ex += state_lists(init) + helpers() + constants(train) + params() + forward() + backward()
    ex += flat_grads() + optimizer(lr, wd)
    return {"version": 11, "graph": {"viewport": {"xmin": -2, "xmax": 70, "ymin": -36, "ymax": 2}},
            "expressions": {"list": ex}}


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
