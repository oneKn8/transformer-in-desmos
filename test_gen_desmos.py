"""Tests for gen_desmos.py: number formatting, block layout, and the emitted state.

The Desmos graph itself is checked in the browser against experiments/charlm/best/reference.npz;
these tests pin down the conventions that check depends on.
"""
import json

import numpy as np
import pytest

import desmos_passes as dp
import gen_desmos as g

RNG = np.random.default_rng(0)


def test_num_never_uses_exponent_and_round_trips():
    for x in [1.23e-05, -4.5e-07, 0.000123, 123456.0, -0.0, 0.0, 0.1 + 0.2]:
        s = g.num(x)
        assert "e" not in s.lower()
        assert float(s) == x or (x == 0 and float(s) == 0)


def test_blocks_order_is_output_block_major():
    w = RNG.standard_normal((64, 96))
    b = g.blocks(w)
    assert len(b) == 2 * 3
    # block (i, k) is at list index k*m + i (0-based), m = number of input blocks
    assert np.array_equal(b[1 * 2 + 0], w[0:32, 32:64])
    assert np.array_equal(b[2 * 2 + 1], w[32:64, 64:96])


def test_blocks_rejects_non_multiples():
    with pytest.raises(ValueError):
        g.blocks(np.zeros((30, 32)))


def p_j(n_blocks, w_blocks, k):
    """Python mirror of the Desmos p_j: output block k (1-based) of a 64-wide product."""
    return n_blocks[0] @ w_blocks[2 * k - 2] + n_blocks[1] @ w_blocks[2 * k - 1]


def d_n(g_blocks, w_blocks, k):
    """Python mirror of the Desmos d_n: total of G times W[8k-7 ... 8k] (1-based)."""
    return sum(gb @ wb for gb, wb in zip(g_blocks, w_blocks[8 * k - 8: 8 * k]))


def test_block_products_match_full_matmul():
    x = RNG.standard_normal((32, 64))
    xb = [x[:, :32], x[:, 32:]]
    w = RNG.standard_normal((64, 64))
    full = x @ w
    for k in (1, 2):
        assert np.allclose(p_j(xb, g.blocks(w), k), full[:, 32 * (k - 1): 32 * k])

    h = RNG.standard_normal((32, 256))
    hb = [h[:, 32 * i: 32 * (i + 1)] for i in range(8)]
    w2 = RNG.standard_normal((256, 64))
    full2 = h @ w2
    for k in (1, 2):
        assert np.allclose(d_n(hb, g.blocks(w2), k), full2[:, 32 * (k - 1): 32 * k])

    w1 = RNG.standard_normal((64, 256))
    up = x @ w1
    w1b = g.blocks(w1)
    for j in range(1, 9):
        assert np.allclose(xb[0] @ w1b[2 * j - 2] + xb[1] @ w1b[2 * j - 1], up[:, 32 * (j - 1): 32 * j])


def tiny_weights():
    shapes = {"tok_emb": (32, 64), "pos_emb": (32, 64), "w_out": (64, 32)}
    for li in range(2):
        shapes.update({f"l{li}_wq": (64, 64), f"l{li}_wk": (64, 64), f"l{li}_wv": (64, 64),
                       f"l{li}_wo": (64, 64), f"l{li}_w1": (64, 256), f"l{li}_w2": (256, 64)})
    return {k: np.round(RNG.standard_normal(s), 3) for k, s in shapes.items()}


CFG = {"d_model": 64, "n_head": 2, "ctx": 32, "vocab_size": 32, "n_layer": 2, "norm_eps": 1e-5}
SYMBOLS = " abcdefghijklmnopqrstuvwxyz.,;'\""


def test_build_state_shape_and_ids():
    state = g.build([20, 5, 12], tiny_weights(), CFG, SYMBOLS)
    json.dumps(state)  # must be serialisable as a Desmos state
    ids = [e["id"] for e in state["expressions"]["list"]]
    assert len(ids) == len(set(ids)), "expression ids must be unique"
    for needed in ("Tx", "I", "x0", "x2", "Y", "yl", "jn", "ag", "keys", "bgo"):
        assert needed in ids
    tx = next(e for e in state["expressions"]["list"] if e["id"] == "Tx")
    assert tx["latex"] == r"T_{x}=\left[21,6,13\right]"  # 1-based token rows
    assert "A_{g}" in state["expressions"]["ticker"]["handlerLatex"]


def test_build_rejects_bad_inputs():
    with pytest.raises(ValueError):
        g.build([], tiny_weights(), CFG, SYMBOLS)
    with pytest.raises(ValueError):
        g.build([0] * 33, tiny_weights(), CFG, SYMBOLS)
    with pytest.raises(ValueError):
        g.build([1], tiny_weights(), {**CFG, "d_model": 128}, SYMBOLS)


def test_comprehensions_read_named_values_only_through_function_arguments():
    """Desmos re-derives a named list each time a comprehension in another expression indexes
    it; the materialize pass must leave no value expression doing that."""
    ex = g.build([20, 5, 12], tiny_weights(), CFG, SYMBOLS)["expressions"]["list"]
    values = {dp.defined_name(e["latex"]) for e in ex if dp.is_value(e.get("latex") or "")}
    for e in ex:
        latex = e.get("latex") or ""
        if dp.is_value(latex) and "\\operatorname{for}" in latex:
            body = latex.partition("=")[2]
            assert not set(dp.names_in(body)) & (values - {dp.defined_name(latex)}), e["id"]
    hd = next(e["latex"] for e in ex if e["id"] == "h0")
    assert hd == r"H_{0}=u_{wh0}\left(M_{0},W_{u0}\right)"


def test_random_seed_is_pinned_and_validated():
    assert g.build([1], tiny_weights(), CFG, SYMBOLS)["randomSeed"] == g.DEFAULT_RANDOM_SEED
    assert g.build([1], tiny_weights(), CFG, SYMBOLS, "0" * 32)["randomSeed"] == "0" * 32
    with pytest.raises(ValueError):
        g.build([1], tiny_weights(), CFG, SYMBOLS, "not-a-seed")
