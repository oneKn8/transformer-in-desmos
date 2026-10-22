"""Tests for desmos_passes.py: the LaTeX name tokenizer and the materialize pass."""
import pytest

import desmos_passes as dp


def test_names_skip_commands_and_keep_subscripts():
    latex = r"\left\{a\le n_{t}:I\left[a\right],1\right\}\operatorname{for}a=\left[1...32\right]"
    assert dp.names_in(latex) == ["a", "n_{t}", "I", "a", "a"]


def test_greek_letters_are_names_and_other_commands_are_not():
    assert dp.names_in(r"\tau_{s}\le\theta+\pi x\beta\eta\sqrt{y}") == [r"\tau_{s}", r"\theta", r"\pi", "x", r"\beta", r"\eta", "y"]


def test_rename_touches_whole_names_only():
    # I_{p} starts like I, and an_{t} is a times n_{t}
    assert dp.rename(r"I_{p}+I\left[a\right]+an_{t}", {"I": "Z", "n_{t}": "Q"}) == r"I_{p}+Z\left[a\right]+aQ"


def test_defined_name_and_is_value():
    assert dp.defined_name(r"W_{u0}=\left[1\right]") == "W_{u0}"
    assert dp.defined_name(r"r_{q}\left(C\right)=C") == "r_{q}"
    assert dp.defined_name(r"\left\{T_{c}=3:\left(1,2\right)\right\}") is None
    assert dp.defined_name(r"T_{x}\to\left[1\right]") is None
    assert dp.is_value(r"W_{u0}=\left[1\right]")
    assert not dp.is_value(r"r_{q}\left(C\right)=C")


def ex(eid, latex):
    return {"type": "expression", "id": eid, "latex": latex}


def test_materialize_moves_indexing_into_a_function():
    graph = [
        ex("I", r"I=\left[3,1,2\right]"),
        ex("Ip", r"I_{p}=\left[I\left[a\right]n_{t}\operatorname{for}a=\left[1...3\right]\right]"),
        ex("n", r"n_{t}=3"),
    ]
    out = dp.materialize(graph)
    assert [e["id"] for e in out] == ["I", "mIp", "Ip", "n"]
    fn, call = out[1], out[2]
    assert fn["hidden"] is True
    assert fn["latex"] == r"u_{wIp}\left(Z_{p1},Z_{p2}\right)=\left[Z_{p1}\left[a\right]Z_{p2}\operatorname{for}a=\left[1...3\right]\right]"
    assert call["latex"] == r"I_{p}=u_{wIp}\left(I,n_{t}\right)"
    assert call["id"] == "Ip" and "hidden" not in call


def test_materialize_leaves_functions_and_plain_values_alone():
    graph = [
        ex("f", r"f_{n}\left(M\right)=M\left[a\right]\operatorname{for}a=\left[1...3\right]"),
        ex("U", r"U=1\operatorname{for}a=\left[1...32\right]"),   # comprehension, no named values
        ex("y", r"y=2x"),                                         # named value, no comprehension
        ex("x", r"x=1"),
    ]
    assert dp.materialize(graph) == graph


def test_check_unique_catches_name_and_id_clashes():
    with pytest.raises(ValueError):
        dp.check_unique([ex("x1", "g_{o}=0"), ex("x2", r"g_{o}=\left[1\right]")])
    with pytest.raises(ValueError):
        dp.check_unique([ex("x1", "g_{o}=0"), ex("x1", "h_{o}=0")])
    dp.check_unique([ex("x1", "g_{o}=0"), ex("x2", "g_{p}=0")])


def test_plotted_equations_define_nothing():
    assert dp.defined_name(r"y=-50\left\{0\le x\le96\right\}") is None
    assert dp.defined_name(r"x=2y") is None
    dp.check_unique([ex("z1", "y=1"), ex("z2", "y=2x")])
