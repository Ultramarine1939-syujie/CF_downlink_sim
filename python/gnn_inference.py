"""Fast GNN inference runtime for CF downlink power allocation.

Supports two DCGNN variants:
  - dcgnn (GAT-based): current codebase default (GATConv + supervised)
  - paper_dcgnn:        paper-aligned DCGNN (shared FC + unsupervised)

This module keeps Python-side model state cached and measures the parts of
GNN inference separately.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from train_gnn import PowerGNN_GAT, PowerGNN_MLP, custom_collate, custom_collate_mlp
from dcgnn_paper import PaperDCGNN, load_paper_dcgnn, infer_paper_dcgnn as _infer_paper


@dataclass
class _Runtime:
    model: torch.nn.Module
    model_type: str
    L: int
    K: int
    dcgnn_top_z: int | None = None
    share_temperature: float = 1.0
    mixture_alpha: float = -1.0
    mix_lambda_min: float = 0.05
    mix_lambda_max: float = 0.95


_CACHE: Dict[Tuple[str, int, int], _Runtime] = {}


def _ckpt_arg(checkpoint, name, default):
    if not isinstance(checkpoint, dict):
        return default
    if name in checkpoint:
        return checkpoint[name]
    args = checkpoint.get("args")
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _load_runtime(model_path: str, L: int, K: int) -> _Runtime:
    key = (os.path.abspath(model_path), int(L), int(K))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type", "gat") if isinstance(checkpoint, dict) else "gat"
    output_kind = checkpoint.get("output_kind", "norm_tanh") if isinstance(checkpoint, dict) else "norm_tanh"
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if model_type == "mlp":
        model = PowerGNN_MLP(
            L=L, K=K,
            hidden_dim=int(_ckpt_arg(checkpoint, "hidden_dim", 128)),
            num_layers=int(_ckpt_arg(checkpoint, "num_layers", 3)),
            dropout=float(_ckpt_arg(checkpoint, "dropout", 0.1)),
            output_scale=1.0, output_kind=output_kind
        )
    else:
        model = PowerGNN_GAT(
            L=L, K=K,
            hidden_dim=int(_ckpt_arg(checkpoint, "hidden_dim", 128)),
            num_heads=int(_ckpt_arg(checkpoint, "num_heads", 4)),
            num_layers=int(_ckpt_arg(checkpoint, "num_layers", 3)),
            dropout=float(_ckpt_arg(checkpoint, "dropout", 0.1)),
            output_scale=1.0, output_kind=output_kind
        )
    model.load_state_dict(state)
    model.eval()

    dcgnn_top_z = None
    if isinstance(checkpoint, dict) and model_type == "dcgnn":
        dcgnn_top_z = int(checkpoint.get("dcgnn_top_z") or 15)

    share_temperature = float(checkpoint.get("share_temperature", 1.0)) if isinstance(checkpoint, dict) else 1.0
    mixture_alpha = float(_ckpt_arg(checkpoint, "mixture_alpha", -1.0))
    mix_lambda_min = float(_ckpt_arg(checkpoint, "mix_lambda_min", 0.05))
    mix_lambda_max = float(_ckpt_arg(checkpoint, "mix_lambda_max", 0.95))
    runtime = _Runtime(
        model=model,
        model_type=model_type,
        L=L,
        K=K,
        dcgnn_top_z=dcgnn_top_z,
        share_temperature=max(share_temperature, 1.0e-6),
        mixture_alpha=mixture_alpha,
        mix_lambda_min=mix_lambda_min,
        mix_lambda_max=mix_lambda_max,
    )
    _CACHE[key] = runtime
    return runtime


def _as_lk(arr, L: int, K: int) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.shape == (K, L):
        out = out.T
    return np.ascontiguousarray(out.reshape(L, K))


def infer(model_path, sqrt_gain, D_mask, Pt=1.0, sigma_e=0.3):
    """Run one GNN inference and return rho plus timing diagnostics."""
    total_t0 = time.perf_counter()

    D = _as_lk(D_mask, int(np.asarray(D_mask).shape[0]), int(np.asarray(D_mask).shape[1]))
    L, K = D.shape
    sqrt_gain = _as_lk(sqrt_gain, L, K)
    Pt = float(Pt)
    sigma_e = float(sigma_e)

    runtime_t0 = time.perf_counter()
    runtime = _load_runtime(str(model_path), L, K)
    load_sec = time.perf_counter() - runtime_t0

    feature_t0 = time.perf_counter()
    sqrt_gain_masked = sqrt_gain * D
    snr_norm = np.log10(max(Pt, 1e-12)) / 3.0
    ap_degree = D.sum(axis=1, keepdims=True) / max(K, 1)
    ue_degree = D.sum(axis=0, keepdims=True).T / max(L, 1)
    ap_gain = np.log1p(sqrt_gain_masked.sum(axis=1, keepdims=True))
    ue_gain = np.log1p(sqrt_gain_masked.sum(axis=0, keepdims=True)).T

    x_ap = torch.from_numpy(np.concatenate([
        sqrt_gain_masked,
        np.full((L, 1), snr_norm, dtype=np.float32),
        np.full((L, 1), sigma_e, dtype=np.float32),
        ap_degree.astype(np.float32),
        ap_gain.astype(np.float32),
    ], axis=1))

    x_ue = torch.from_numpy(np.concatenate([
        sqrt_gain_masked.T,
        np.full((K, 1), snr_norm, dtype=np.float32),
        np.full((K, 1), sigma_e, dtype=np.float32),
        ue_degree.astype(np.float32),
        ue_gain.astype(np.float32),
    ], axis=1))

    D_t = torch.from_numpy(D)
    z = torch.zeros(1, dtype=torch.float32)
    sample = {
        "x_ap": x_ap,
        "x_ue": x_ue,
        "D_mask": D_t,
        "rho_is_nonzero": torch.zeros(L, K, dtype=torch.float32),
        "y_share": torch.zeros(L, K, dtype=torch.float32),
        "y": torch.zeros(L, K, dtype=torch.float32),
        "esr": z,
        "snr": z,
        "mode": "DCC",
        "idx": 0,
    }
    feature_sec = time.perf_counter() - feature_t0

    collate_t0 = time.perf_counter()
    if runtime.model_type == "mlp":
        batch = custom_collate_mlp([sample])
    elif runtime.model_type == "dcgnn":
        batch = custom_collate([sample], dynamic_top_z=runtime.dcgnn_top_z)
    else:
        batch = custom_collate([sample])
    collate_sec = time.perf_counter() - collate_t0

    forward_t0 = time.perf_counter()
    with torch.no_grad():
        rho_pred = runtime.model(batch).squeeze(0).detach().cpu().numpy()
    forward_sec = time.perf_counter() - forward_t0

    post_t0 = time.perf_counter()
    served_count = np.maximum(D.sum(axis=1, keepdims=True), 1.0)
    fallback = Pt * D / served_count
    output_kind = getattr(runtime.model, "output_kind", "norm_tanh")
    if output_kind == "share_logits_mix":
        logits = rho_pred[:, :K].copy() / runtime.share_temperature
        logits[D <= 0.5] = -1.0e9
        logits = logits - np.max(logits, axis=1, keepdims=True)
        nn_weights = np.exp(logits) * D
        nn_sum = nn_weights.sum(axis=1, keepdims=True)
        nn_share = np.where(nn_sum > 0, nn_weights / np.maximum(nn_sum, 1e-12), fallback / max(Pt, 1e-12))

        gain = np.maximum(sqrt_gain ** 2, 1e-12)
        prior_weights = np.power(gain, -runtime.mixture_alpha) * D
        prior_sum = prior_weights.sum(axis=1, keepdims=True)
        prior_share = np.where(prior_sum > 0, prior_weights / np.maximum(prior_sum, 1e-12), fallback / max(Pt, 1e-12))

        gate = 1.0 / (1.0 + np.exp(-np.clip(rho_pred[:, K:K + 1], -30.0, 30.0)))
        gate = runtime.mix_lambda_min + (runtime.mix_lambda_max - runtime.mix_lambda_min) * gate
        shares = gate * nn_share + (1.0 - gate) * prior_share
        weights = shares * D
        active_rows = D.sum(axis=1, keepdims=True) > 0
        mix_lambda_mean = float(np.mean(gate[active_rows])) if np.any(active_rows) else 0.0
    elif output_kind == "share_logits":
        logits = rho_pred.copy() / runtime.share_temperature
        logits[D <= 0.5] = -1.0e9
        logits = logits - np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits) * D
        mix_lambda_mean = 1.0
    else:
        weights = np.maximum((rho_pred + 1.0) / 2.0, 0.0) * D
        mix_lambda_mean = 1.0
    row_sum = weights.sum(axis=1, keepdims=True)
    rho = np.where(row_sum > 0, Pt * weights / np.maximum(row_sum, 1e-12), fallback)
    rho = rho.astype(np.float64, copy=False)
    post_sec = time.perf_counter() - post_t0

    python_total_sec = time.perf_counter() - total_t0
    return {
        "rho": rho,
        "load_sec": float(load_sec),
        "feature_sec": float(feature_sec),
        "collate_sec": float(collate_sec),
        "forward_sec": float(forward_sec),
        "post_sec": float(post_sec),
        "mix_lambda_mean": float(mix_lambda_mean),
        "python_total_sec": float(python_total_sec),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Paper DCGNN inference
# ═══════════════════════════════════════════════════════════════════════════

_PAPER_DCGNN_CACHE: Dict[str, PaperDCGNN] = {}


def infer_paper_dcgnn(
    model_path_str: str,
    sqrt_gain: np.ndarray,
    D_mask: np.ndarray,
    Pt: float = 1.0,
    sigma_e: float = 0.3,
) -> dict:
    """Run inference with a paper-aligned DCGNN model.

    Args:
        model_path_str: path to .pt checkpoint (saved by train_dcgnn_paper.py)
        sqrt_gain:      (L, K)  sqrt(β_{k,l}) — input feature from simulator
        D_mask:         (L, K)  DCC AP-UE association mask
        Pt:             per-AP power limit (pmax in paper)
        sigma_e:        CSI error (unused — paper DCGNN only uses large-scale β)

    Returns:
        dict with keys: rho, load_sec, forward_sec, python_total_sec, ...
    """
    total_t0 = time.perf_counter()
    model_path = os.path.abspath(str(model_path_str))

    D = np.asarray(D_mask, dtype=np.float64)
    L, K = D.shape
    sqrt_gain_arr = np.asarray(sqrt_gain, dtype=np.float64)
    if sqrt_gain_arr.shape == (K, L):
        sqrt_gain_arr = sqrt_gain_arr.T
    Pt = float(Pt)

    # β = (sqrt_gain)² , but the simulator's gainOverNoise already includes
    # the noise normalisation.  Use sqrt_gain² as β (linear scale).
    beta = np.maximum(sqrt_gain_arr ** 2, 1e-12).astype(np.float32)

    # Load / cache model
    load_t0 = time.perf_counter()
    if model_path not in _PAPER_DCGNN_CACHE:
        _PAPER_DCGNN_CACHE[model_path] = load_paper_dcgnn(model_path)
    model = _PAPER_DCGNN_CACHE[model_path]
    load_sec = time.perf_counter() - load_t0

    # Forward
    forward_t0 = time.perf_counter()
    rho_raw = _infer_paper(model, beta, pmax=Pt)
    forward_sec = time.perf_counter() - forward_t0

    # Post: apply D mask (only allocate power to served UEs)
    post_t0 = time.perf_counter()
    rho_masked = rho_raw * (D > 0.5)
    # Re-normalise per-AP power after masking
    row_power = rho_masked.sum(axis=1, keepdims=True)
    valid = row_power[:, 0] > 0
    rho = np.where(valid[:, None], Pt * rho_masked / np.maximum(row_power, 1e-12), 0.0)
    rho = rho.astype(np.float64)
    post_sec = time.perf_counter() - post_t0

    python_total_sec = time.perf_counter() - total_t0
    return {
        "rho": rho,
        "load_sec": float(load_sec),
        "feature_sec": 0.0,
        "collate_sec": 0.0,
        "forward_sec": float(forward_sec),
        "post_sec": float(post_sec),
        "mix_lambda_mean": 1.0,
        "python_total_sec": float(python_total_sec),
    }

