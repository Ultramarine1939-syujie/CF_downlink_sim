"""MATLAB-facing runtime for strict AP-local GNN power allocation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from train_gnn_local import LocalPowerMLP, build_local_features


@dataclass
class _Runtime:
    model: torch.nn.Module
    K: int


_CACHE: Dict[Tuple[str, int], _Runtime] = {}


def _as_lk(arr, L: int | None = None, K: int | None = None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"Expected a 2D LxK array, got shape {out.shape}")
    if L is not None and K is not None:
        if out.shape == (K, L):
            out = out.T
        return np.ascontiguousarray(out.reshape(L, K))
    return np.ascontiguousarray(out)


def _load_runtime(model_path: str, K: int) -> _Runtime:
    key = (os.path.abspath(model_path), int(K))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    hidden_dim = int(checkpoint.get("hidden_dim", 96)) if isinstance(checkpoint, dict) else 96
    num_layers = int(checkpoint.get("num_layers", 3)) if isinstance(checkpoint, dict) else 3
    dropout = float(checkpoint.get("dropout", 0.10)) if isinstance(checkpoint, dict) else 0.10
    ckpt_k = int(checkpoint.get("K", K)) if isinstance(checkpoint, dict) else K
    if ckpt_k != K:
        raise ValueError(f"Local-GNN checkpoint expects K={ckpt_k}, but input has K={K}")

    model = LocalPowerMLP(K=K, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(state)
    model.eval()

    runtime = _Runtime(model=model, K=K)
    _CACHE[key] = runtime
    return runtime


def infer(model_path, sqrt_gain, D_mask, Pt=1.0, sigma_e=0.3):
    """Run strict AP-local inference.

    This function accepts an LxK matrix for bridge efficiency, but each AP row
    is featurized and inferred independently. No AP row receives another AP's
    gain, association, or embedding.
    """
    total_t0 = time.perf_counter()

    D = _as_lk(D_mask)
    L, K = D.shape
    sqrt_gain = _as_lk(sqrt_gain, L, K)
    Pt = float(Pt)
    sigma_e = float(sigma_e)

    load_t0 = time.perf_counter()
    runtime = _load_runtime(str(model_path), K)
    load_sec = time.perf_counter() - load_t0

    feature_t0 = time.perf_counter()
    sqrt_gain_masked = sqrt_gain * D
    snr_norm = np.log10(max(Pt, 1e-12)) / 3.0
    x = build_local_features(sqrt_gain_masked, D, np.array([snr_norm]), np.array([sigma_e]))
    x_t = torch.from_numpy(x)
    D_t = torch.from_numpy(D)
    feature_sec = time.perf_counter() - feature_t0

    forward_t0 = time.perf_counter()
    with torch.no_grad():
        logits = runtime.model(x_t)
        mask = D_t > 0.5
        logits = logits.masked_fill(~mask, -1e9)
        share = torch.softmax(logits, dim=1).cpu().numpy()
    forward_sec = time.perf_counter() - forward_t0

    post_t0 = time.perf_counter()
    served_count = D.sum(axis=1, keepdims=True)
    valid_rows = served_count[:, 0] > 0
    fallback = Pt * D / np.maximum(served_count, 1.0)

    # share from softmax already sums to 1 per row; scale by Pt
    row_sum = (share * D).sum(axis=1, keepdims=True)
    rho = np.where(row_sum > 0, Pt * share * D / np.maximum(row_sum, 1e-12), fallback)
    rho = rho.astype(np.float64, copy=False)
    post_sec = time.perf_counter() - post_t0

    python_total_sec = time.perf_counter() - total_t0
    return {
        "rho": rho,
        "load_sec": float(load_sec),
        "feature_sec": float(feature_sec),
        "forward_sec": float(forward_sec),
        "post_sec": float(post_sec),
        "python_total_sec": float(python_total_sec),
    }
