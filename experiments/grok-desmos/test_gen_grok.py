"""Tests for gen_grok.py: the emitted graph is well formed for both MLP widths."""
import json

import numpy as np
import pytest

import desmos_passes as dp
import gen_grok as gg
import ref_step as rs

RNG = np.random.default_rng(0)


def fake_npz(width: int) -> dict:
    npz = {f"init_{k}": RNG.standard_normal(shape) for k, shape in rs.param_shapes(width).items()}
    a, b = np.meshgrid(np.arange(31), np.arange(31), indexing="ij")
    pairs = np.stack([a.ravel(), b.ravel()], 1)
    npz["pairs"] = pairs[RNG.permutation(len(pairs))[:480]]
    return npz


@pytest.mark.parametrize("width", [32, 128])
def test_graph_is_well_formed(width):
    state = gg.build(fake_npz(width), lr=3e-2, wd=1.0)
    text = json.dumps(state)
    assert "@K@" not in text and "@W@" not in text
    ex = state["expressions"]["list"]
    ids = [e["id"] for e in ex]
    assert len(ids) == len(set(ids))
    names = [n for n in (dp.defined_name(e.get("latex") or "") for e in ex) if n]
    assert len(names) == len(set(names))
    for needed in ("As", "Gl", "Cg", "ge", "nwe", "grid", "circ", "bgo", "rd"):
        assert needed in ids
    assert "A_{s}" in state["expressions"]["ticker"]["handlerLatex"]
    hd_body = next(e["latex"] for e in ex if e["id"] == "mHd")
    assert rf"k=\left[1...{width // 32}\right]" in hd_body


def test_no_comprehension_reads_named_lists_directly():
    """The materialize pass must leave every comprehension inside a function body, with named
    values passed in whole (Desmos re-derives lists indexed by comprehensions elsewhere)."""
    ex = gg.build(fake_npz(32), lr=3e-2, wd=1.0)["expressions"]["list"]
    defs = set()
    for e in ex:
        name = dp.defined_name(e.get("latex") or "")
        if name and "\\left(" not in e["latex"].split("=", 1)[0]:
            defs.add(name)
    for e in ex:
        latex = e.get("latex") or ""
        head, _, body = latex.partition("=")
        if "\\left(" in head or "\\operatorname{for}" not in body:
            continue
        assert not [n for n in defs if n in body and n != dp.defined_name(latex)], e["id"]


def test_state_lists_hold_the_init_weights_row_major():
    npz = fake_npz(32)
    ex = gg.build(npz, lr=3e-2, wd=1.0)["expressions"]["list"]
    wi = next(e["latex"] for e in ex if e["id"] == "wi")
    values = [float(x) for x in wi[len(r"w_{i}=\left["):-len(r"\right]")].split(",")]
    assert values == npz["init_W_in"].reshape(-1).tolist()


def test_rejects_bad_widths_and_missing_tensors():
    npz = fake_npz(32)
    npz["init_W_in"] = RNG.standard_normal((32, 40))
    with pytest.raises(ValueError):
        gg.build(npz, lr=3e-2, wd=1.0)
    npz = fake_npz(32)
    del npz["init_W_U"]
    with pytest.raises(ValueError):
        gg.build(npz, lr=3e-2, wd=1.0)

