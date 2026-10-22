"""ref_step.py against torch autograd on experiments/grokking/grok.py's own model and data."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ref_step as rs

GROK_PATH = Path(__file__).resolve().parent.parent / "grokking" / "grok.py"


def load_grok():
    spec = importlib.util.spec_from_file_location("grok", GROK_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["grok"] = mod  # dataclasses look their module up while the class is built
    spec.loader.exec_module(mod)
    return mod


grok = load_grok()
CFG = grok.RunConfig(p=31, arch="attn", d_model=32, n_heads=4, d_mlp=128, lr=3e-2,
                     weight_decay=1.0, train_frac=0.5, seed=0, dtype="float64")


def torch_params(model) -> dict:
    flat = model._flat
    return {
        "W_E": model.W_E, "W_pos": model.W_pos,
        "W_Q": flat(model.W_Q), "W_K": flat(model.W_K), "W_V": flat(model.W_V),
        "W_O": model.W_O.reshape(-1, model.W_O.shape[-1]),
        "W_in": model.W_in, "b_in": model.b_in, "W_out": model.W_out, "b_out": model.b_out,
        "W_U": model.W_U,
    }


def to_numpy(params: dict) -> dict:
    return {k: v.detach().numpy().copy() for k, v in params.items()}


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(CFG.seed)
    data = grok.make_dataset(CFG.p, CFG.train_frac, seed=CFG.seed)
    model = grok.build_model(CFG)
    pairs = data.x_train[:, :2].numpy()
    return data, model, rs.make_grid(pairs)


def test_shapes_match(setup):
    _, model, _ = setup
    w = to_numpy(torch_params(model))
    assert {k: v.shape for k, v in w.items()} == rs.PARAM_SHAPES


def test_forward_logits_and_gradients_match_autograd(setup):
    data, model, grid = setup
    model.zero_grad()
    logits_t = model(data.x_train)
    loss_t = F.cross_entropy(logits_t, data.y_train)
    loss_t.backward()

    w = to_numpy(torch_params(model))
    f = rs.forward(w, grid)
    loss, dlogits, _ = rs.loss_and_dlogits(f["logits"], grid)
    a, b = data.x_train[:, 0].numpy(), data.x_train[:, 1].numpy()
    assert np.abs(f["logits"][a, b] - logits_t.detach().numpy()).max() < 1e-12
    assert abs(loss - loss_t.item()) < 1e-13

    g = rs.backward(w, f, dlogits)
    gt = to_numpy({k: t.grad for k, t in (("W_E", model.W_E), ("W_pos", model.W_pos), ("W_in", model.W_in),
                                          ("b_in", model.b_in), ("W_out", model.W_out),
                                          ("b_out", model.b_out), ("W_U", model.W_U))})
    gt["W_Q"] = model._flat(model.W_Q.grad).numpy()
    gt["W_K"] = model._flat(model.W_K.grad).numpy()
    gt["W_V"] = model._flat(model.W_V.grad).numpy()
    gt["W_O"] = model.W_O.grad.reshape(-1, model.W_O.shape[-1]).numpy()
    for k in rs.PARAM_SHAPES:
        scale = max(np.abs(gt[k]).max(), 1e-30)
        assert np.abs(g[k] - gt[k]).max() / scale < 1e-10, k


def test_twenty_adamw_steps_track_torch(setup):
    data, _, grid = setup
    torch.manual_seed(CFG.seed)
    model = grok.build_model(CFG)
    opt = grok.make_optimizer(CFG, model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / max(CFG.warmup_steps, 1), 1.0))

    w = to_numpy(torch_params(model))
    m = {k: np.zeros_like(x) for k, x in w.items()}
    v = {k: np.zeros_like(x) for k, x in w.items()}
    for step in range(20):
        opt.zero_grad()
        F.cross_entropy(model(data.x_train), data.y_train).backward()
        opt.step()
        sched.step()

        f = rs.forward(w, grid)
        _, dlogits, _ = rs.loss_and_dlogits(f["logits"], grid)
        rs.adamw(w, rs.backward(w, f, dlogits), m, v, step, CFG.lr, CFG.weight_decay, CFG.warmup_steps)

    wt = to_numpy(torch_params(model))
    worst = max(np.abs(w[k] - wt[k]).max() for k in w)
    assert worst < 1e-9, worst


def test_correct_grid_matches_argmax(setup):
    data, model, grid = setup
    w = to_numpy(torch_params(model))
    _, _, correct = rs.loss_and_dlogits(rs.forward(w, grid)["logits"], grid)
    with torch.no_grad():
        pred = model(data.x_test).argmax(-1).numpy()
    a, b = data.x_test[:, 0].numpy(), data.x_test[:, 1].numpy()
    assert np.array_equal(correct[a, b], pred == data.y_test.numpy())
    assert math.isclose(grid.n_train, 480)
