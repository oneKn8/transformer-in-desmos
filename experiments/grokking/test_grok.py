"""Correctness checks: the ensemble trainer must reproduce the single-run trainer, the Fourier
and circle metrics must read known inputs correctly, and the cost model's tile arithmetic
must match hand counts. Run: python3 -m pytest -q test_grok.py"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from cost_model import mm, step_cost
from ensemble import train_ensemble
from grok import (
    RunConfig,
    circle_projection,
    circle_quality,
    fourier_energy,
    make_dataset,
    train_run,
)

torch.set_num_threads(1)

SHORT = {"max_steps": 40, "eval_every": 1, "confirm_steps": 0}
CASES = [
    RunConfig(p=23, arch="attn", optimizer="adamw", lr=3e-3, weight_decay=1.0, **SHORT),
    RunConfig(p=23, arch="attn", optimizer="sgd", lr=0.3, weight_decay=0.01, momentum=0.9, **SHORT),
    RunConfig(p=23, arch="mlp_concat", optimizer="sgd", lr=1.0, weight_decay=0.003, **SHORT),
    RunConfig(p=23, arch="mlp_folded", optimizer="adamw", lr=1e-2, weight_decay=1.0,
              batch_size=64, **SHORT),
    RunConfig(p=23, arch="attn", optimizer="adamw", lr=1e-2, weight_decay=1.0,
              batch_size=32, batch_mode="cyclic", **SHORT),
    RunConfig(p=23, arch="mlp_sum", optimizer="adamw", lr=1e-2, weight_decay=3.0,
              train_frac=0.5, act="gelu", **SHORT),
]
METRICS = ("final_train_loss", "final_test_loss", "final_train_acc", "final_test_acc")


@pytest.mark.parametrize("cfg", CASES, ids=lambda c: f"{c.arch}-{c.optimizer}-bs{c.batch_size}-{c.batch_mode}")
def test_ensemble_matches_single_run(cfg: RunConfig) -> None:
    single = train_run(cfg)
    (ens,) = train_ensemble([cfg])
    for m in METRICS:
        assert ens[m] == pytest.approx(single[m], rel=1e-9, abs=1e-12), m
    assert ens["step_train99"] == single["step_train99"]


def test_ensemble_members_do_not_interact() -> None:
    a = CASES[0]
    b = dataclasses.replace(a, seed=3, lr=1e-2, weight_decay=3.0, train_frac=0.5)
    pair = train_ensemble([a, b])
    for cfg, got in zip((a, b), pair, strict=True):
        alone = train_run(cfg)
        for m in METRICS:
            assert got[m] == pytest.approx(alone[m], rel=1e-9, abs=1e-12), m


def test_ensemble_rejects_mixed_shapes() -> None:
    with pytest.raises(ValueError, match="must share"):
        train_ensemble([CASES[0], dataclasses.replace(CASES[0], d_mlp=64)])


def test_fourier_energy_single_frequency() -> None:
    p, k = 31, 7
    t = np.arange(p)
    emb = np.stack([np.cos(2 * np.pi * k * t / p), np.sin(2 * np.pi * k * t / p)], axis=1)
    energy = fourier_energy(emb)
    assert energy[k] == pytest.approx(1.0)
    assert energy.sum() == pytest.approx(1.0)


def test_circle_metrics_separate_circle_from_noise() -> None:
    p, k = 31, 5
    t = np.arange(p)
    rng = np.random.default_rng(0)
    mix = rng.normal(size=(2, 16))
    circle = np.stack([np.cos(2 * np.pi * k * t / p), np.sin(2 * np.pi * k * t / p)], 1) @ mix
    q = circle_quality(circle_projection(circle, k), k)
    assert q["phase_coherence"] > 0.99
    noise = circle_quality(circle_projection(rng.normal(size=(p, 16)), k), k)
    assert noise["phase_coherence"] < 0.6


def test_dataset_split_is_disjoint_and_labelled() -> None:
    d = make_dataset(23, 0.4, seed=1)
    assert len(d.y_train) + len(d.y_test) == 23 * 23
    assert len(d.y_train) == round(0.4 * 23 * 23)
    pairs_tr = {tuple(r) for r in d.x_train[:, :2].tolist()}
    pairs_te = {tuple(r) for r in d.x_test[:, :2].tolist()}
    assert not pairs_tr & pairs_te
    assert torch.equal(d.y_train, (d.x_train[:, 0] + d.x_train[:, 1]) % 23)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="prime"):
        RunConfig(p=33)
    with pytest.raises(ValueError, match="train_frac"):
        RunConfig(p=31, train_frac=1.0)


def test_cost_model_tile_counts() -> None:
    op = mm("x", 64, 32, 96)
    assert (op.tiles, op.adds) == (6, 0)
    op = mm("x", 32, 64, 32)
    assert (op.tiles, op.adds) == (2, 1)
    small = step_cost("mlp_folded", p=31, n_train=64, d_mlp=32, opt="sgd")
    big = step_cost("mlp_folded", p=31, n_train=640, d_mlp=32, opt="sgd")
    assert big.tiles > small.tiles and big.elems > small.elems
