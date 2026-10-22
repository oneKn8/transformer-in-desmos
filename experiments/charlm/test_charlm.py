"""Tests for the char-LM study. Run: CHARLM_THREADS=1 python3 -m pytest -q test_charlm.py"""

from __future__ import annotations

import threadcap  # sets OMP_NUM_THREADS; must be imported before torch

import itertools

import numpy as np
import pytest
import torch

from corpus import S32, V32, V64, decode, encode, extract_iliad, extract_odyssey, load_shakespeare, map_text, split_bounds
from cost_model import padded_blocks, token_cost
from model import CharTransformer, ModelConfig, generate
from reference_numpy import fold_weights, forward, round_sig, trained_weights
from train import TrainConfig, lr_at

threadcap.apply_torch_threads()


def _random_model(seed: int = 0, **kw) -> CharTransformer:
    """A model with non-trivial gains/biases so folding is actually exercised."""
    base = {"vocab_size": 32, "ctx": 16, "d_model": 16, "n_layer": 2, "n_head": 4}
    base.update(kw)
    torch.manual_seed(seed)
    model = CharTransformer(ModelConfig(**base)).double()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return model.eval()


# ---------------------------------------------------------------- corpus

def test_v32_mapping_lowercases_and_remaps_punctuation():
    mapped, counts = map_text("Hello—World? “Yes!” (ok) 42", V32)
    assert mapped == 'hello world. "yes." ok'
    assert counts["dropped"] == 2  # the two digits
    assert counts["case_folded"] == 3  # H, W, Y
    assert counts["quote_normalized"] == 2


def test_v64_keeps_case():
    mapped, counts = map_text("Troy—Jove's", V64)
    assert mapped == "Troy-Jove's"
    assert counts["case_folded"] == 0


@pytest.mark.parametrize("vocab", [V32, V64])
def test_unknown_character_raises(vocab):
    with pytest.raises(ValueError, match="no rule"):
        map_text("café", vocab)


def test_s32_keeps_newlines_and_collapses_spaces():
    mapped, counts = map_text("First Citizen:\nBefore we proceed;  hear me!\n\nAll:\nWe know't - 3 times.\n", S32)
    assert mapped == "first citizen:\nbefore we proceed, hear me.\n\nall:\nwe know't times."
    assert counts["dropped"] == 1


def test_shakespeare_hash_is_enforced(tmp_path):
    fake = tmp_path / "input.txt"
    fake.write_text("First Citizen:\nnot the real file\n")
    with pytest.raises(ValueError, match="sha256"):
        load_shakespeare(fake)


def test_encode_decode_roundtrip_and_rejects_out_of_vocab():
    text = "sing, o goddess; the 'anger' of \"achilles\"."
    assert decode(encode(text, V32), V32) == text
    with pytest.raises(ValueError, match="outside v32"):
        encode("Capital", V32)


def test_split_bounds():
    assert split_bounds(1000) == (900, 950)
    with pytest.raises(ValueError, match="bad split"):
        split_bounds(1000, val_frac=0.6, test_frac=0.5)


def test_extractors_fail_loudly_without_markers():
    with pytest.raises(ValueError, match="marker"):
        extract_odyssey("no headings here")
    with pytest.raises(ValueError, match="marker"):
        extract_iliad("BOOK I.\nonly the contents entry")


def test_odyssey_footnote_markers_removed():
    raw = "BOOK I\nhis abode.12 Telemachus's room was high up in a tower13 that\nFOOTNOTES:\n[1] note"
    assert extract_odyssey(raw) == "BOOK I\nhis abode. Telemachus's room was high up in a tower that"


# ---------------------------------------------------------------- model

def test_config_validation():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=32, ctx=32, d_model=30, n_layer=1, n_head=4)
    with pytest.raises(ValueError, match="norm must"):
        ModelConfig(vocab_size=32, ctx=32, d_model=32, n_layer=1, n_head=4, norm="batch")
    with pytest.raises(ValueError, match="n_layer must be positive"):
        ModelConfig(vocab_size=32, ctx=32, d_model=32, n_layer=0, n_head=4)


@pytest.mark.parametrize("pos", ["learned", "alibi"])
def test_fused_attention_matches_explicit_path(pos):
    model = _random_model(pos=pos)
    ids = torch.randint(0, 32, (3, 16), generator=torch.Generator().manual_seed(1))
    fused = model(ids)
    explicit = model(ids, trace={})
    assert torch.max(torch.abs(fused - explicit)).item() < 1e-12


def test_right_padding_is_exact_under_causal_mask():
    """Desmos can always run a full T-row window: rows before the padding are unchanged."""
    model = _random_model()
    ids = torch.randint(0, 32, (1, 7), generator=torch.Generator().manual_seed(2))
    padded = torch.cat([ids, torch.zeros(1, 9, dtype=torch.long)], dim=1)
    diff = torch.max(torch.abs(model(ids) - model(padded)[:, :7])).item()
    assert diff < 1e-12


def test_context_overflow_raises():
    model = _random_model()
    with pytest.raises(ValueError, match="exceeds ctx"):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_generate_is_seeded_and_slides_the_window():
    model = _random_model().float()
    a = generate(model, [1, 2, 3], 40, 0.8, torch.Generator().manual_seed(5))
    b = generate(model, [1, 2, 3], 40, 0.8, torch.Generator().manual_seed(5))
    assert a == b
    assert len(a) == 43  # 43 > ctx 16, so the window slid
    with pytest.raises(ValueError, match="temperature"):
        generate(model, [1], 1, 0.0, torch.Generator())


# ---------------------------------------------------------------- numpy spec

VARIANTS = [
    {"norm": n, "act": a, "tied": t, "bias": b, "pos": p}
    for n, a, t, b, p in itertools.product(("rms", "ln", "none"), ("gelu_tanh", "relu"), (False, True), (False, True), ("learned", "alibi"))
]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(str(x) for x in v.values()))
def test_folded_numpy_forward_equals_torch(variant):
    model = _random_model(**variant)
    ids = np.random.default_rng(3).integers(0, 32, size=(2, 16))
    want = model(torch.from_numpy(ids)).detach().numpy()
    got = forward(fold_weights(trained_weights(model), model.cfg), model.cfg, ids)
    assert np.max(np.abs(got - want)) < 1e-10


def test_numpy_trace_has_probabilities_that_sum_to_one():
    model = _random_model()
    trace: dict = {}
    forward(fold_weights(trained_weights(model), model.cfg), model.cfg, np.arange(16) % 32, trace)
    probs = trace["l0_attn_probs"]
    assert probs.shape == (4, 16, 16)
    assert np.allclose(probs.sum(-1), 1.0)
    assert np.all(np.triu(probs, k=1) == 0.0)


def test_round_sig_roundtrips_through_decimal_text():
    rng = np.random.default_rng(4)
    a = rng.normal(size=500) * 10.0 ** rng.integers(-6, 2, size=500)
    for digits in (3, 5, 7):
        r = round_sig(a, digits)
        assert all(float(format(x, f".{digits}g")) == x for x in r)
        assert np.max(np.abs(r - a) / np.abs(a)) <= 0.5 * 10.0 ** (1 - digits) * 1.0001
    with pytest.raises(ValueError, match="digits"):
        round_sig(a, 0)


# ---------------------------------------------------------------- cost model

def test_padded_blocks():
    assert padded_blocks(32, 32, 32) == 1
    assert padded_blocks(64, 64, 192) == 24
    assert padded_blocks(1, 8, 33) == 2


def test_cost_matches_hand_derivation():
    # d32 L2 H4 T32: per layer QKV 3 + scores 4 + AV 4 + Wo 1 + W1 4 + W2 4 = 20 blocks,
    # x2 layers + 1 unembedding block = 41. Elements: embed 1024, per layer 26624,
    # final norm 64, sampling 128 = 54464.
    cfg = ModelConfig(vocab_size=32, ctx=32, d_model=32, n_layer=2, n_head=4)
    c = token_cost(cfg, "full")
    assert c.padded_blocks == 41
    assert c.elements == 54464
    assert c.ms_padded == pytest.approx(50 + 1.3 * 41 + 0.001 * 54464)
    with pytest.raises(ValueError, match="mode must"):
        token_cost(cfg, "cached")


def test_block_aligned_heads_skip_slicing():
    # d64 H2: head_dim 32, each head is one column block, so no slice/concat passes.
    aligned = token_cost(ModelConfig(vocab_size=32, ctx=32, d_model=64, n_layer=1, n_head=2), "full")
    split = token_cost(ModelConfig(vocab_size=32, ctx=32, d_model=64, n_layer=1, n_head=4), "full")
    assert not any("slices" in line[0] or "concat" in line[0] for line in aligned.lines)
    assert any("slices" in line[0] for line in split.lines)
    softmax_extra = 3 * 2 * 32 * 32  # H=4 has 2 more heads of 32x32 softmax
    assert split.elements - aligned.elements == 4 * 32 * 64 + softmax_extra


def test_kv_mode_is_never_more_matmul_work():
    for d, layers, heads, ctx in itertools.product((32, 64), (1, 2, 4), (2, 4), (32, 64)):
        cfg = ModelConfig(vocab_size=32, ctx=ctx, d_model=d, n_layer=layers, n_head=heads)
        full, kv = token_cost(cfg, "full"), token_cost(cfg, "kv")
        assert kv.padded_blocks <= full.padded_blocks
        assert kv.mac_blocks < full.mac_blocks


# ---------------------------------------------------------------- training

def test_lr_schedule_warmup_peak_and_floor():
    tc = TrainConfig(max_steps=1000, warmup_steps=100, lr=1e-2, min_lr_frac=0.1)
    assert lr_at(0, tc) == pytest.approx(1e-4)
    assert lr_at(100, tc) == pytest.approx(1e-2)
    assert lr_at(1000, tc) == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="max_steps"):
        TrainConfig(max_steps=0)
