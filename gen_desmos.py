"""Emit a Desmos graph state that runs the trained character-level transformer.

The model (experiments/charlm, config d64_L2_H2_T32) is d_model 64, 2 layers, 2 heads of 32,
MLP 256, a 32-character window and a 32-symbol vocabulary. Desmos matrices stop at 32x32, so
every weight matrix is cut into 32x32 blocks and every activation is a LIST of 32x32 blocks:

    activation (32 positions x 64)   ->  [X1, X2]           column blocks
    weight (in x out), m in-blocks   ->  [W(1,1), W(2,1), ..., W(m,1), W(1,2), ...]
                                          (block (i, k) at list index (k-1) m + i)
    output block k of X W            =   X1 W[(k-1)m+1] + ... + Xm W[(k-1)m+m]

Lists of matrices are never rendered in the expression list, so the forward pass costs the
same with the list open. Comprehensions that would be passed straight into a function call
are wrapped in their own small functions (Desmos misparses the direct form), and every
function is hidden (one-variable functions would otherwise be plotted as curves).

The op-by-op specification is experiments/charlm/reference_numpy.py; the Desmos logits are
checked against experiments/charlm/best/reference.npz.

Usage: python3 gen_desmos.py OUT.json [--prompt "tell me, o muse, of that ingenio"]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BEST = HERE / "experiments" / "charlm" / "best"
BLOCK = 32
GELU_C = "0.7978845608028654"  # sqrt(2/pi) to 16 significant digits
FOR_AB = r"\operatorname{for}a=\left[1...32\right];b=\left[1...32\right]"


def num(x: float) -> str:
    """Shortest decimal that round-trips, never in exponent form (Desmos LaTeX has no 1e-5)."""
    s = np.format_float_positional(float(x), unique=True, trim="-")
    return "0" if s in ("-0", "0") else s


def bmatrix(m: np.ndarray) -> str:
    rows = [r"&".join(num(v) for v in row) for row in m]
    return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"


def blocks(w: np.ndarray) -> list:
    """Cut an (in x out) matrix into 32x32 blocks, ordered by output block then input block."""
    n_in, n_out = w.shape
    if n_in % BLOCK or n_out % BLOCK:
        raise ValueError(f"shape {w.shape} is not a multiple of {BLOCK}")
    return [
        w[i * BLOCK:(i + 1) * BLOCK, k * BLOCK:(k + 1) * BLOCK]
        for k in range(n_out // BLOCK)
        for i in range(n_in // BLOCK)
    ]


def block_list(name: str, w: np.ndarray) -> str:
    return f"{name}=\\left[" + ",".join(bmatrix(b) for b in blocks(w)) + r"\right]"


def expr(eid: str, latex: str, **extra) -> dict:
    return {"type": "expression", "id": eid, "latex": latex, **extra}


def hidden(eid: str, latex: str) -> dict:
    return expr(eid, latex, hidden=True)


def functions(eps: float) -> list:
    """Small per-element helpers. Desmos inlines function bodies, so each one takes only named
    (already computed) values as arguments; heavy intermediates are named expressions instead."""
    return [
        # RMSNorm scale per row from the Gram matrix C = X1 X1^T + X2 X2^T: 1/sqrt(C[a;a]/64 + eps)
        hidden("fq", r"r_{q}\left(C\right)=\left[\frac{1}{\sqrt{\frac{C\left[a;a\right]}{64}+" + num(eps)
               + r"}}\operatorname{for}a=\left[1...32\right]\right]"),
        hidden("fsc", r"s_{c}\left(M,r\right)=M\left[a;b\right]r\left[a\right]" + FOR_AB),
        # output block k of a 64-wide product
        hidden("fpj", r"p_{j}\left(N,W,k\right)=N\left[1\right]W\left[2k-1\right]+N\left[2\right]W\left[2k\right]"),
        # causal softmax without max-subtraction (the reference does the same; scores stay small)
        hidden("fex", r"e_{x}\left(S\right)=\left\{b\le a:\exp\left(S\left[a;b\right]\right),0\right\}" + FOR_AB),
        hidden("fpd", r"p_{d}\left(E,Z\right)=\frac{E\left[a;b\right]}{Z\left[a;1\right]}" + FOR_AB),
        hidden("fgl", r"g_{l}\left(H\right)=0.5H\left[a;b\right]\left(1+\tanh\left(" + GELU_C
               + r"\left(H\left[a;b\right]+0.044715H\left[a;b\right]^{3}\right)\right)\right)" + FOR_AB),
        hidden("fdn", r"d_{n}\left(G,W,k\right)=\operatorname{total}\left(GW\left[\left[8k-7...8k\right]\right]\right)"),
    ]


def two(part) -> str:
    """A list of two blocks, part(1) and part(2)."""
    return r"\left[" + part(1) + "," + part(2) + r"\right]"


def idx(name: str, i) -> str:
    return f"{name}\\left[{i}\\right]"


def call(fn: str, *args) -> str:
    return fn + r"\left(" + ",".join(str(a) for a in args) + r"\right)"


def rmsnorm(x: str, gram: str, scale: str, out: str, eid: str) -> list:
    """out = RMSNorm(x) for a two-block activation x, as three named expressions."""
    return [
        hidden(f"c{eid}", f"{gram}=" + r"\left[" + idx(x, 1) + idx(x, 1) + "^{T}+" + idx(x, 2) + idx(x, 2) + r"^{T}\right]"),
        hidden(f"a{eid}", f"{scale}=" + call("r_{q}", idx(gram, 1))),
        hidden(f"n{eid}", f"{out}=" + two(lambda j: call("s_{c}", idx(x, j), scale))),
    ]


def layer(li: int) -> list:
    """Layer li reads X_{li} and defines X_{li+1}. Every intermediate is a named list of
    32x32 matrices (never rendered), so each one is computed exactly once per update."""
    x, xn = f"X_{{{li}}}", f"X_{{{li + 1}}}"
    w = {k: f"W_{{{k}{li}}}" for k in ("q", "k", "v", "o", "u", "d")}
    c, a, n, q, k, v, s, e, z, pr, o, r, d, bb, m, h, g = (f"{ch}_{{{li}}}" for ch in "CANQKVSEZPORDBMHG")
    return rmsnorm(x, c, a, n, f"1{li}") + [
        hidden(f"q{li}", f"{q}=" + two(lambda j: call("p_{j}", n, w["q"], j))),
        hidden(f"k{li}", f"{k}=" + two(lambda j: call("p_{j}", n, w["k"], j))),
        hidden(f"v{li}", f"{v}=" + two(lambda j: call("p_{j}", n, w["v"], j))),
        # one head per 32-column block: scores, causal exp, row sums, probabilities, output
        hidden(f"s{li}", f"{s}=" + two(lambda j: idx(q, j) + idx(k, j) + "^{T}")),
        hidden(f"e{li}", f"{e}=" + two(lambda j: call("e_{x}", idx(s, j)))),
        hidden(f"z{li}", f"{z}=" + two(lambda j: idx(e, j) + "U")),
        hidden(f"p{li}", f"{pr}=" + two(lambda j: call("p_{d}", idx(e, j), idx(z, j)))),
        hidden(f"o{li}", f"{o}=" + two(lambda j: idx(pr, j) + idx(v, j))),
        hidden(f"r{li}", f"{r}=" + two(lambda j: idx(x, j) + "+" + call("p_{j}", o, w["o"], j))),
    ] + rmsnorm(r, d, bb, m, f"2{li}") + [
        hidden(f"h{li}", f"{h}=" + idx(m, 1) + idx(w["u"], "2j-1") + "+" + idx(m, 2) + idx(w["u"], "2j")
               + r"\operatorname{for}j=\left[1...8\right]"),
        hidden(f"g{li}", f"{g}=" + call("g_{l}", idx(h, "j")) + r"\operatorname{for}j=\left[1...8\right]"),
        hidden(f"x{li + 1}", f"{xn}=" + two(lambda j: idx(r, j) + "+" + call("d_{n}", g, w["d"], j))),
    ]


def weights_exprs(wt: dict, n_layer: int) -> list:
    ex = [
        hidden("we", block_list("E_{b}", wt["tok_emb"])),
        hidden("wp", block_list("P_{b}", wt["pos_emb"])),
        hidden("wy", block_list("W_{y}", wt["w_out"])),
    ]
    names = {"q": "wq", "k": "wk", "v": "wv", "o": "wo", "u": "w1", "d": "w2"}
    for li in range(n_layer):
        for short, key in names.items():
            ex.append(hidden(f"w{short}{li}", block_list(f"W_{{{short}{li}}}", wt[f"l{li}_{key}"])))
    return ex


LINE = 64          # characters per text line
CHAR_W = 0.62      # character advance in graph units
LINE_H = 1.5       # line height in graph units
MAX_CHARS = 480    # generation stops here
KEY_COLS = 8
TEXT_COLOR, GEN_COLOR, KEY_COLOR = "#000000", "#2d70b3", "#000000"


def label_text(sym: str) -> str:
    return "space" if sym == " " else sym


def generation(n_prompt: int) -> list:
    """Sampling and the generation step. p_n is the next-character distribution at the last
    window position; j_n samples it by inverse CDF with a seeded random draw per step."""
    return [
        expr("np", f"n_{{p}}={n_prompt}"),
        expr("tau", r"\tau_{s}=0.8", slider={"hardMin": True, "hardMax": True, "min": "0.05", "max": "2"}),
        expr("sd", r"s_{d}=7", slider={"hardMin": True, "hardMax": True, "min": "1", "max": "100", "step": "1"}),
        expr("ks", r"k_{s}=1"),
        expr("go", r"g_{o}=0"),
        hidden("qe", r"q_{e}=\exp\left(\frac{y_{l}-\max\left(y_{l}\right)}{\tau_{s}}\right)"),
        hidden("pn", r"p_{n}=\frac{q_{e}}{\operatorname{total}\left(q_{e}\right)}"),
        hidden("cs", r"c_{s}=\left[\operatorname{total}\left(p_{n}\left[1...j\right]\right)\operatorname{for}j=\left[1...32\right]\right]"),
        hidden("ur", r"u_{r}=\operatorname{random}\left(1,k_{s}+1000s_{d}\right)\left[1\right]"),
        hidden("jn", r"j_{n}=\min\left(32,1+\operatorname{total}\left(\left\{c_{s}<u_{r}:1,0\right\}\right)\right)"),
        hidden("ag", r"A_{g}=\left(T_{x}\to\operatorname{join}\left(T_{x},j_{n}\right),k_{s}\to k_{s}+1\right)"),
    ]


def text_display(symbols: str) -> list:
    """The text as point labels: one expression per symbol and per colour (prompt vs generated).
    T_c folds the colour into the value so a single comparison picks the points."""
    ex = [
        hidden("ux", f"u_{{x}}={CHAR_W}" + r"\operatorname{mod}\left(\left[1...n_{x}\right]-1," + str(LINE) + r"\right)"),
        hidden("uy", r"u_{y}=-" + str(LINE_H) + r"\operatorname{floor}\left(\frac{\left[1...n_{x}\right]-1}{" + str(LINE) + r"}\right)"),
        hidden("tc", r"T_{c}=T_{x}+100\left\{\left[1...n_{x}\right]>n_{p}:1,0\right\}"),
    ]
    for i, sym in enumerate(symbols):
        if sym == " ":
            continue
        for offset, color, tag in ((0, TEXT_COLOR, "p"), (100, GEN_COLOR, "g")):
            ex.append(expr(
                f"t{tag}{i}",
                r"\left\{T_{c}=" + str(i + 1 + offset) + r":\left(u_{x},u_{y}\right)\right\}",
                color=color, label=sym, showLabel=True, labelOrientation="center",
                pointOpacity="1", pointSize="0",  # a label fades with its point's opacity
            ))
    return ex


def keyboard(symbols: str) -> list:
    """Clickable keys. Each key's square is shaded by the model's probability for that
    character, so the keyboard shows the next-character distribution live."""
    top = -LINE_H * (MAX_CHARS // LINE) - 3
    ex = [
        hidden("kx", r"k_{x}=4+5\operatorname{mod}\left(\left[0...31\right]," + str(KEY_COLS) + r"\right)"),
        hidden("ky", f"k_{{y}}={top}" + r"-2.5\operatorname{floor}\left(\frac{\left[0...31\right]}{" + str(KEY_COLS) + r"}\right)"),
        hidden("kc", r"k_{c}=\operatorname{rgb}\left(255-200p_{n}^{0.5},255-150p_{n}^{0.5},255\right)"),
        expr("keys",
             r"\operatorname{polygon}\left(\left(k_{x}-2,k_{y}-1\right),\left(k_{x}+2,k_{y}-1\right),"
             r"\left(k_{x}+2,k_{y}+1\right),\left(k_{x}-2,k_{y}+1\right)\right)",
             colorLatex="k_{c}", fillOpacity="1", lineOpacity="0.3"),
    ]
    for i, sym in enumerate(symbols):
        col, row = i % KEY_COLS, i // KEY_COLS
        ex.append(expr(
            f"key{i}", f"\\left({4 + 5 * col},{top - 2.5 * row}\\right)",
            color=KEY_COLOR, label=label_text(sym), showLabel=True, labelOrientation="center",
            pointOpacity="1", pointSize="0",
            clickableInfo={"enabled": True,
                           "latex": r"T_{x}\to\operatorname{join}\left(T_{x}," + str(i + 1) + r"\right),n_{p}\to n_{x}+1"},
        ))
    by = top - 2.5 * (len(symbols) // KEY_COLS) - 1
    buttons = [
        ("bgo", "go / stop", 6, r"g_{o}\to1-g_{o}"),
        ("bbk", "back", 20, r"T_{x}\to\left\{n_{x}>1:T_{x}\left[1...n_{x}-1\right],T_{x}\right\},"
                            r"n_{p}\to\min\left(n_{p},n_{x}-1\right)"),
        ("bcl", "clear", 32, r"T_{x}\to\left[1\right],n_{p}\to1,g_{o}\to0"),
    ]
    for eid, text, x, action in buttons:
        ex.append(expr(eid, f"\\left({x},{by}\\right)", color="#c74440", label=text, showLabel=True,
                       labelOrientation="center", pointSize="0", pointOpacity="1",
                       clickableInfo={"enabled": True, "latex": action}))
    return ex


def build(prompt_ids: list, wt: dict, cfg: dict, symbols: str) -> dict:
    n_layer = cfg["n_layer"]
    if (cfg["d_model"], cfg["n_head"], cfg["ctx"], cfg["vocab_size"]) != (64, 2, 32, 32):
        raise ValueError("this generator is written for d_model 64, 2 heads, ctx 32, vocab 32")
    if not 1 <= len(prompt_ids) <= cfg["ctx"]:
        raise ValueError("prompt must be 1 to 32 tokens")
    ids = ",".join(str(i + 1) for i in prompt_ids)  # Desmos rows are 1-based
    ex = [
        {"type": "text", "id": "note", "text": (
            "A character-level transformer (2 layers, d_model 64, 2 heads, 105k parameters) running "
            "entirely in Desmos; the weights are lists of 32x32 matrices. Type on the keyboard, press "
            "go / stop, and it writes one character per tick. Key shading is the model's probability "
            "for the next character. T_x is the text, I the last 32 characters the model sees.")},
        # the whole text so far (written by the keyboard and the generator), and the window the
        # model sees: its last 32 tokens
        expr("Tx", r"T_{x}=\left[" + ids + r"\right]"),
        expr("nx", r"n_{x}=\operatorname{length}\left(T_{x}\right)"),
        hidden("I", r"I=T_{x}\left[\max\left(1,n_{x}-31\right)...n_{x}\right]"),
        hidden("n", r"n_{t}=\operatorname{length}\left(I\right)"),
        # right padding is exact under the causal mask; pad with the space token (row 1). A
        # fixed 32-slot piecewise, because [1...0] counts down to [1,0] instead of being empty.
        hidden("ip", r"I_{p}=\left[\left\{a\le n_{t}:I\left[a\right],1\right\}\operatorname{for}a=\left[1...32\right]\right]"),
        hidden("U", r"U=1\operatorname{for}a=\left[1...32\right];b=\left[1\right]"),
        hidden("x0", r"X_{0}=\left[E_{b}\left[1\right]\left[I_{p};\right]+P_{b}\left[1\right],"
                     r"E_{b}\left[2\right]\left[I_{p};\right]+P_{b}\left[2\right]\right]"),
    ]
    ex += functions(cfg["norm_eps"])
    for li in range(n_layer):
        ex += layer(li)
    xl = f"X_{{{n_layer}}}"
    ex += [
        *rmsnorm(xl, "C_{f}", "A_{f}", "F", "f"),
        hidden("Y", r"Y=\left[F\left[1\right]W_{y}\left[1\right]+F\left[2\right]W_{y}\left[2\right]\right]"),
        hidden("yl", r"y_{l}=\left[Y\left[1\right]\left[n_{t};b\right]\operatorname{for}b=\left[1...32\right]\right]"),
    ]
    ex += generation(len(prompt_ids)) + text_display(symbols) + keyboard(symbols)
    ex += weights_exprs(wt, n_layer)
    bottom = -LINE_H * (MAX_CHARS // LINE) - 3 - 2.5 * (len(symbols) // KEY_COLS) - 3
    return {
        "version": 11,
        "graph": {"viewport": {"xmin": -2, "xmax": 5 * KEY_COLS + 3, "ymin": bottom, "ymax": 2},
                  "showGrid": False, "showXAxis": False, "showYAxis": False},
        "expressions": {
            "list": ex,
            # the generator: one character per tick while g_o is 1, stopping at MAX_CHARS
            "ticker": {"handlerLatex": r"\left\{g_{o}=1:\left\{n_{x}<" + str(MAX_CHARS) + r":A_{g},g_{o}\to0\right\}\right\}",
                       "minStepLatex": "0", "open": True, "playing": True},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--prompt", default=None, help="prompt text; default: the reference input")
    args = ap.parse_args()
    cfg_all = json.loads((BEST / "config.json").read_text())
    cfg, symbols = cfg_all["model_config"], cfg_all["vocab"]["symbols"]
    with np.load(BEST / "weights.npz") as f:
        wt = {k: f[k].astype(np.float64) for k in f.files}
    prompt = args.prompt if args.prompt is not None else cfg_all["reference"]["input_text"]
    unknown = sorted(set(prompt) - set(symbols))
    if unknown:
        raise ValueError(f"prompt has characters outside the 32-symbol vocab: {unknown}")
    state = build([symbols.index(c) for c in prompt], wt, cfg, symbols)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state))
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.0f} KB, {len(state['expressions']['list'])} expressions, prompt {prompt!r})")


if __name__ == "__main__":
    main()
