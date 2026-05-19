"""Runtime inference for DQN/DDPG power allocation baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Runtime:
    model_path: str | None = None
    checkpoint: dict[str, Any] | None = None
    model: torch.nn.Module | None = None
    device: torch.device = torch.device("cpu")


_RUNTIME = Runtime()


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DDPGActor(nn.Module):
    def __init__(self, state_dim: int, L: int, K: int, hidden_dim: int = 256,
                 base_alpha: float = -1.0, residual_scale: float = 0.75):
        super().__init__()
        self.L = L
        self.K = K
        self.base_alpha = float(base_alpha)
        self.residual_scale = float(residual_scale)
        self.net = MLP(state_dim, L * K, hidden_dim)

    def forward(self, state: torch.Tensor, d_mask: torch.Tensor,
                sqrt_gain: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.residual_scale * torch.tanh(self.net(state).view(-1, self.L, self.K))
        if sqrt_gain is None:
            logits = residual
        else:
            base = fpcp_shares(sqrt_gain, d_mask, self.base_alpha).clamp_min(1.0e-12)
            logits = torch.log(base) + residual
        return masked_row_softmax(logits, d_mask)


def build_state(sqrt_gain: torch.Tensor, d_mask: torch.Tensor, snr_db: torch.Tensor,
                sigma_e: torch.Tensor) -> torch.Tensor:
    masked_gain = torch.log1p(sqrt_gain * d_mask)
    degree_ap = d_mask.mean(dim=2, keepdim=True)
    degree_ue = d_mask.mean(dim=1, keepdim=True)
    context = torch.stack([
        snr_db / 30.0,
        sigma_e,
        d_mask.mean(dim=(1, 2)),
        torch.log1p((sqrt_gain * d_mask).sum(dim=(1, 2))),
    ], dim=1)
    return torch.cat([
        masked_gain.flatten(start_dim=1),
        d_mask.flatten(start_dim=1),
        degree_ap.expand_as(d_mask).flatten(start_dim=1),
        degree_ue.expand_as(d_mask).flatten(start_dim=1),
        context,
    ], dim=1)


def masked_row_softmax(logits: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(d_mask <= 0.5, -1.0e9)
    shares = torch.softmax(masked_logits, dim=2) * d_mask
    row_sum = shares.sum(dim=2, keepdim=True)
    return torch.where(row_sum > 0, shares / row_sum.clamp_min(1.0e-12), shares)


def fpcp_shares(sqrt_gain: torch.Tensor, d_mask: torch.Tensor, alpha: float) -> torch.Tensor:
    gain = (sqrt_gain.square()).clamp_min(1.0e-12)
    weights = torch.pow(gain, -alpha) * d_mask
    row_sum = weights.sum(dim=2, keepdim=True)
    served = d_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    epa = d_mask / served
    return torch.where(row_sum > 0, weights / row_sum.clamp_min(1.0e-12), epa)


def _load(model_path: str):
    if _RUNTIME.model_path == model_path and _RUNTIME.model is not None:
        return 0.0

    t0 = time.perf_counter()
    checkpoint = torch.load(model_path, map_location="cpu")
    method = checkpoint["method"]
    state_dim = int(checkpoint["state_dim"])
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    L = int(checkpoint["L"])
    K = int(checkpoint["K"])

    if method == "dqn":
        action_alphas = checkpoint.get("action_alphas", [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5])
        model = MLP(state_dim, len(action_alphas), hidden_dim)
        model.load_state_dict(checkpoint["state_dict"])
    elif method == "ddpg":
        model = DDPGActor(
            state_dim,
            L,
            K,
            hidden_dim,
            base_alpha=float(checkpoint.get("ddpg_base_alpha", -1.0)),
            residual_scale=float(checkpoint.get("ddpg_residual_scale", 0.75)),
        )
        model.load_state_dict(checkpoint["actor_state_dict"])
    else:
        raise ValueError(f"Unsupported RL method in checkpoint: {method}")

    model.eval()
    _RUNTIME.model_path = model_path
    _RUNTIME.checkpoint = checkpoint
    _RUNTIME.model = model
    _RUNTIME.device = torch.device("cpu")
    return time.perf_counter() - t0


def _to_numpy(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _epa(d_mask: np.ndarray, pt: float) -> np.ndarray:
    row_sum = d_mask.sum(axis=1, keepdims=True)
    shares = np.divide(d_mask, np.maximum(row_sum, 1.0), where=row_sum >= 0)
    return (pt * shares * d_mask).astype(np.float64)


def infer(model_path: str, sqrt_gain, d_mask, pt: float, sigma_e: float = 0.3):
    total_t0 = time.perf_counter()
    load_sec = _load(model_path)

    feature_t0 = time.perf_counter()
    sqrt_gain_np = np.maximum(_to_numpy(sqrt_gain), 0.0)
    d_np = (_to_numpy(d_mask) > 0.5).astype(np.float32)
    if sqrt_gain_np.ndim != 2 or d_np.ndim != 2:
        raise ValueError(f"sqrt_gain and D must be 2D, got {sqrt_gain_np.shape} and {d_np.shape}")
    L, K = d_np.shape
    checkpoint = _RUNTIME.checkpoint
    if int(checkpoint["L"]) != L or int(checkpoint["K"]) != K:
        raise ValueError(f"Model expects L={checkpoint['L']}, K={checkpoint['K']}; got L={L}, K={K}")

    sqrt_gain_t = torch.from_numpy(sqrt_gain_np).unsqueeze(0)
    d_t = torch.from_numpy(d_np).unsqueeze(0)
    snr_db_t = torch.tensor([10.0 * np.log10(max(float(pt), 1.0e-12))], dtype=torch.float32)
    sigma_t = torch.tensor([float(sigma_e)], dtype=torch.float32)
    state = build_state(sqrt_gain_t, d_t, snr_db_t, sigma_t)
    feature_sec = time.perf_counter() - feature_t0

    forward_t0 = time.perf_counter()
    method = checkpoint["method"]
    with torch.no_grad():
        if method == "dqn":
            q = _RUNTIME.model(state)
            action_idx = int(torch.argmax(q, dim=1).item())
            alpha = float(checkpoint.get("action_alphas", [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5])[action_idx])
            shares = fpcp_shares(sqrt_gain_t, d_t, alpha)
        elif method == "ddpg":
            shares = _RUNTIME.model(state, d_t, sqrt_gain_t)
            alpha = np.nan
            action_idx = -1
        else:
            raise ValueError(f"Unsupported RL method: {method}")
    forward_sec = time.perf_counter() - forward_t0

    post_t0 = time.perf_counter()
    shares_np = shares.squeeze(0).cpu().numpy().astype(np.float64)
    rho = float(pt) * shares_np * d_np.astype(np.float64)
    row_sum = rho.sum(axis=1, keepdims=True)
    valid_rows = d_np.sum(axis=1, keepdims=True) > 0
    rho = np.where(valid_rows, rho * (float(pt) / np.maximum(row_sum, 1.0e-12)), 0.0)
    if not np.isfinite(rho).all():
        rho = _epa(d_np, float(pt))
    post_sec = time.perf_counter() - post_t0

    return {
        "rho": rho,
        "load_sec": load_sec,
        "feature_sec": feature_sec,
        "forward_sec": forward_sec,
        "post_sec": post_sec,
        "python_total_sec": time.perf_counter() - total_t0,
        "method": method,
        "action_idx": action_idx,
        "alpha": alpha,
    }
