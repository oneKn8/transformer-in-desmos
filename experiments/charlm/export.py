"""Export a trained run as the Desmos phase-1 model.

Writes to <out>/:
  model.pt            trained float32 state_dict + model/train configs (as trained)
  weights_trained.npz unfolded float64 weights, descriptive names (inspection)
  weights.npz         CANONICAL Desmos weights: norm gains/betas and 1/sqrt(hd)
                      folded in, rounded to `sig_digits` significant digits.
                      Print each value with format(w, f".{sig_digits}g") (then to
                      positional notation if Desmos needs it) and it parses back to
                      the identical float64.
  reference.npz       fixed inputs and float64 logits (plus every intermediate)
                      computed by reference_numpy.forward over weights.npz: the
                      numbers a Desmos forward pass must reproduce.
  config.json         model config, vocab, forward-pass spec pointer, metrics,
                      Desmos cost estimate with arithmetic, precision study.
"""

from __future__ import annotations

import threadcap  # sets OMP_NUM_THREADS; must be imported before torch

import argparse
import json
import shutil
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch

from corpus import VOCABS, Vocab, decode, encode
from cost_model import token_cost
from model import CharTransformer, ModelConfig
from reference_numpy import fold_weights, forward, round_sig, split_bpc_numpy, trained_weights
from train import SAMPLE_CHARS, SAMPLE_TEMPERATURE, load_splits

HERE = Path(__file__).resolve().parent
REFERENCE_TEXT = "tell me, o muse, of that ingenious hero who travelled far and wide after he had sacked the famous town of troy."
DIGIT_CHOICES = (3, 4, 5, 6, 7, 8)
MAX_BPC_LOSS = 0.001
EXACT_TOL = 1e-9


def load_run(run: str) -> tuple[CharTransformer, Vocab, dict]:
    ckpt = torch.load(HERE / "runs" / run / "model.pt", map_location="cpu", weights_only=True)
    cfg = ModelConfig(**ckpt["model_config"])
    model = CharTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, VOCABS[ckpt["train_config"]["vocab"]], ckpt


def as_text(s: str, vocab: Vocab) -> str:
    return s if vocab.lowercase else s.capitalize()


def sample_numpy(fw: dict, cfg: ModelConfig, prompt: list[int], n: int, temperature: float, rng: np.random.Generator) -> list[int]:
    ids = list(prompt)
    for _ in range(n):
        logits = forward(fw, cfg, np.array(ids[-cfg.ctx:]))[-1] / temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        ids.append(int(rng.choice(len(p), p=p)))
    return ids


def paste_chars(fw: dict, digits: int) -> int:
    """Characters needed to paste every weight in positional decimal, comma separated."""
    total = 0
    for arr in fw.values():
        for w in np.asarray(arr).ravel():
            total += len(format(Decimal(format(w, f".{digits}g")), "f")) + 1
    return total


def export(run: str, out: Path) -> dict:
    model, vocab, ckpt = load_run(run)
    cfg = model.cfg
    model64 = CharTransformer(cfg).double()
    model64.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
    model64.eval()

    splits = {k: v.numpy() for k, v in load_splits(vocab).items()}
    trained = trained_weights(model64)
    exact = fold_weights(trained, cfg)

    # 1. The folded numpy spec must equal the torch model before anything is rounded.
    probe = splits["test"][: cfg.ctx * 8].reshape(8, cfg.ctx)
    with torch.no_grad():
        torch_logits = model64(torch.from_numpy(probe)).numpy()
    spec_err = float(np.max(np.abs(forward(exact, cfg, probe) - torch_logits)))
    if spec_err > EXACT_TOL:
        raise AssertionError(f"numpy spec disagrees with torch: max abs logit diff {spec_err}")

    # 2. Precision study: fewest significant digits that cost <= MAX_BPC_LOSS on val.
    exact_bpc = split_bpc_numpy(exact, cfg, splits["val"])
    study = {"full": round(exact_bpc, 5)}
    chosen = None
    for digits in DIGIT_CHOICES:
        fw = {k: round_sig(v, digits) for k, v in exact.items()}
        bpc = split_bpc_numpy(fw, cfg, splits["val"])
        study[str(digits)] = round(bpc, 5)
        if chosen is None and bpc - exact_bpc <= MAX_BPC_LOSS:
            chosen = digits
    if chosen is None:
        raise RuntimeError(f"no digit count within {MAX_BPC_LOSS} bpc: {study}")
    canonical = {k: round_sig(v, chosen) for k, v in exact.items()}
    roundtrip_ok = all(
        float(format(w, f".{chosen}g")) == w for arr in canonical.values() for w in np.asarray(arr).ravel()
    )
    if not roundtrip_ok:
        raise AssertionError("canonical weights do not round-trip through their decimal text")

    # 3. Reference inputs and logits over the canonical weights.
    main_text = as_text(REFERENCE_TEXT, vocab)[: cfg.ctx]
    main_ids = np.array(encode(main_text, vocab), dtype=np.int64)
    trace: dict = {}
    main_logits = forward(canonical, cfg, main_ids, trace)
    short_text = vocab.sample_prompt
    short_len = len(short_text)
    if short_len > cfg.ctx:
        raise ValueError("short prompt longer than ctx")
    short_ids = np.array(encode(short_text, vocab) + [0] * (cfg.ctx - short_len), dtype=np.int64)
    short_logits = forward(canonical, cfg, short_ids)
    unpadded = forward(canonical, cfg, short_ids[:short_len])
    pad_err = float(np.max(np.abs(short_logits[:short_len] - unpadded)))
    with torch.no_grad():
        f32_logits = model(torch.from_numpy(main_ids[None]))[0].double().numpy()

    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "runs" / run / "model.pt", out / "model.pt")
    np.savez(out / "weights_trained.npz", **trained)
    np.savez(out / "weights.npz", **canonical)
    np.savez(
        out / "reference.npz",
        input_text=np.array(main_text),
        input_ids=main_ids,
        logits=main_logits,
        logits_torch_float32_unrounded=f32_logits,
        short_input_text=np.array(short_text),
        short_input_ids_padded=short_ids,
        short_len=np.array(short_len),
        short_logits=short_logits,
        **{f"trace_{k}": v for k, v in trace.items()},
    )

    rng = np.random.default_rng(7)
    sample = decode(sample_numpy(canonical, cfg, encode(short_text, vocab), SAMPLE_CHARS, SAMPLE_TEMPERATURE, rng), vocab)
    full, kv = token_cost(cfg, "full"), token_cost(cfg, "kv")
    report = {
        "source_run": run,
        "model_config": cfg.to_dict(),
        "train_config": ckpt["train_config"],
        "vocab": {"name": vocab.name, "symbols": vocab.symbols, "index_of_space": vocab.symbols.index(" ")},
        "params": model.num_params(),
        "canonical_numbers": int(sum(np.asarray(v).size for v in canonical.values())),
        "sig_digits": chosen,
        "paste_chars_positional": paste_chars(canonical, chosen),
        "precision_study_val_bpc_ctx": study,
        "metrics_canonical": {
            "val_bpc_ctx": round(split_bpc_numpy(canonical, cfg, splits["val"]), 4),
            "test_bpc_ctx": round(split_bpc_numpy(canonical, cfg, splits["test"]), 4),
        },
        "checks": {
            "numpy_spec_vs_torch_float64_max_abs_logit_diff": spec_err,
            "right_padding_max_abs_logit_diff": pad_err,
            "canonical_vs_torch_float32_unrounded_max_abs_logit_diff": float(np.max(np.abs(main_logits - f32_logits))),
            "decimal_roundtrip": roundtrip_ok,
        },
        "reference": {"input_text": main_text, "short_prompt": short_text, "short_len": short_len},
        "sample_canonical_numpy": {"prompt": short_text, "temperature": SAMPLE_TEMPERATURE, "seed": 7, "text": sample},
        "cost_full": full.summary(),
        "cost_kv": kv.summary(),
        "cost_full_arithmetic": full.arithmetic(),
        "spec": "reference_numpy.py docstring (forward pass over weights.npz)",
        "weights_npz_keys": {k: list(np.shape(v)) for k, v in canonical.items()},
    }
    (out / "config.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    threadcap.apply_torch_threads()
    ap = argparse.ArgumentParser(description="Export a run as the Desmos phase-1 model.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, default=HERE / "best")
    args = ap.parse_args()
    if not (HERE / "runs" / args.run / "model.pt").exists():
        raise FileNotFoundError(f"no checkpoint for run {args.run!r}")
    report = export(args.run, args.out)
    printable = {k: v for k, v in report.items() if k not in ("cost_full_arithmetic", "weights_npz_keys")}
    print(json.dumps(printable, indent=2))
    print(report["cost_full_arithmetic"])


if __name__ == "__main__":
    main()
