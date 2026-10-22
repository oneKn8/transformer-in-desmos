"""Passes over a list of Desmos expressions, shared by the graph generators (gen_desmos.py and
experiments/grok-desmos/gen_grok.py).

Desmos LaTeX names a value with one letter or Greek letter and an optional subscript (`I`,
`n_{t}`, `W_{u0}`, `\\tau_{s}`); adjacent letters multiply (`an_{t}` is a times n_t). Other
commands (`\\left`, `\\operatorname{for}`) are not names. The tokenizer below follows those
rules, so a reference to `I` is never confused with `I_{p}`, and `\\le` never reads as the
name `l` times `e`.
"""
import re

GREEK = ("alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|"
         "upsilon|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega")
# a name (group 1: a letter or Greek letter plus an optional subscript), or a command (skipped)
TOKEN = re.compile(r"\\operatorname\{[^}]*\}"
                   r"|((?:\\(?:" + GREEK + r")(?![a-zA-Z])|[a-zA-Z])(?:_\{[^}]*\}|_[a-zA-Z0-9])?)"
                   r"|\\[a-zA-Z]+|\\.")


def names_in(latex: str) -> list:
    """Every name used in a LaTeX string, in order of appearance, with repeats."""
    return [m.group(1) for m in TOKEN.finditer(latex) if m.group(1)]


def rename(latex: str, mapping: dict) -> str:
    """Replace whole names only (commands and longer names that start alike are untouched)."""
    def swap(m: re.Match) -> str:
        return mapping.get(m.group(1), m.group(0)) if m.group(1) else m.group(0)
    return TOKEN.sub(swap, latex)


# equations Desmos plots rather than definitions: y = f(x), x = g(y), polar r = h(theta)
PLOTTED = {"x", "y", "r"}


def defined_name(latex: str):
    """The variable or function an expression defines, e.g. 'g_{c}' or 'm_{r}', else None."""
    head = latex.split("=", 1)[0] if "=" in latex else ""
    if not head or "\\to" in head or head.startswith("\\") or head in PLOTTED:
        return None
    return head.split("\\left(", 1)[0]


def is_value(latex: str) -> bool:
    """True when the expression defines a value (a name), not a function."""
    return bool(defined_name(latex)) and "\\left(" not in latex.split("=", 1)[0]


def check_unique(ex: list) -> None:
    """Desmos silently misbehaves on a repeated id or a name defined twice; refuse to emit."""
    ids = [e["id"] for e in ex]
    dup_ids = {i for i in ids if ids.count(i) > 1}
    names = [n for n in (defined_name(e.get("latex") or "") for e in ex) if n]
    dup_names = {n for n in names if names.count(n) > 1}
    if dup_ids or dup_names:
        raise ValueError(f"duplicate expression ids {sorted(dup_ids)} or names {sorted(dup_names)}")


def materialize(ex: list) -> list:
    """Desmos re-derives a named list each time a comprehension in another expression indexes
    it (L[a] inside [... for a]), which compounds through chains of such expressions. Values
    passed whole into a function are computed once. So every value expression that has a
    comprehension and reads other named values becomes a call to its own hidden function,
    with those values as arguments and all indexing inside the function body."""
    values = {defined_name(e["latex"]) for e in ex if is_value(e.get("latex") or "")}
    out = []
    for e in ex:
        latex = e.get("latex") or ""
        _, _, body = latex.partition("=")
        if not is_value(latex) or "\\operatorname{for}" not in body:
            out.append(e)
            continue
        name = defined_name(latex)
        refs = list(dict.fromkeys(n for n in names_in(body) if n in values and n != name))
        if not refs:
            out.append(e)
            continue
        params = [f"Z_{{p{i + 1}}}" for i in range(len(refs))]
        fname = f"u_{{w{e['id']}}}"
        fbody = rename(body, dict(zip(refs, params)))
        out.append({"type": "expression", "id": "m" + e["id"],
                    "latex": f"{fname}\\left({','.join(params)}\\right)={fbody}", "hidden": True})
        out.append({**e, "latex": f"{name}={fname}\\left({','.join(refs)}\\right)"})
    return out
