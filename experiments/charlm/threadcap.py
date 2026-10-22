"""CPU thread cap. Import this module BEFORE torch in every entry point: OpenMP
reads OMP_NUM_THREADS when torch loads, so setting it later has no effect.

The Desmos browser benchmark runs on the same laptop, so this study may use at
most MAX_TOTAL_THREADS cores in total. A single run uses CHARLM_THREADS
(default 3); the grid launcher runs 3 workers with CHARLM_THREADS=1 each, which
measured about 3x the aggregate throughput of one 3-thread worker for models
this small (44 vs 33 steps/s single-run at d=32).
"""

from __future__ import annotations

import os

MAX_TOTAL_THREADS = 3


def _threads_from_env() -> int:
    raw = os.environ.get("CHARLM_THREADS", str(MAX_TOTAL_THREADS))
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(f"CHARLM_THREADS must be an integer, got {raw!r}") from exc
    if not 1 <= n <= MAX_TOTAL_THREADS:
        raise ValueError(f"CHARLM_THREADS must be in 1..{MAX_TOTAL_THREADS}, got {n}")
    return n


THREADS = _threads_from_env()
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = str(THREADS)


def apply_torch_threads() -> None:
    """Call once after importing torch (intra-op = THREADS, inter-op = 1)."""
    import torch

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
