"""Shared RL model definitions used by both training and inference.

This module avoids code duplication between train_rl.py and rl_inference.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared neural network building blocks
# ---------------------------------------------------------------------------

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


class DDPGCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action.flatten(start_dim=1)], dim=1)).squeeze(1)


# ---------------------------------------------------------------------------
# Shared feature / policy helpers
# ---------------------------------------------------------------------------

def build_state(sqrt_gain: torch.Tensor, d_mask: torch.Tensor, snr_db: torch.Tensor,
                sigma_e: torch.Tensor) -> torch.Tensor:
    """Build a compact state vector from large-scale features."""

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
    """Per-AP masked softmax; rows with no served UE stay zero."""

    masked_logits = logits.masked_fill(d_mask <= 0.5, -1.0e9)
    shares = torch.softmax(masked_logits, dim=2) * d_mask
    row_sum = shares.sum(dim=2, keepdim=True)
    return torch.where(row_sum > 0, shares / row_sum.clamp_min(1.0e-12), shares)


def fpcp_shares(sqrt_gain: torch.Tensor, d_mask: torch.Tensor, alpha: float) -> torch.Tensor:
    """Fractional power-control shares used as DQN discrete actions."""

    gain = (sqrt_gain.square()).clamp_min(1.0e-12)
    weights = torch.pow(gain, -alpha) * d_mask
    row_sum = weights.sum(dim=2, keepdim=True)
    served = d_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    epa = d_mask / served
    return torch.where(row_sum > 0, weights / row_sum.clamp_min(1.0e-12), epa)


def proxy_sum_rate(shares: torch.Tensor, sqrt_gain: torch.Tensor, d_mask: torch.Tensor,
                   pt) -> torch.Tensor:
    """Large-scale reward proxy for one-step RL training.

    The full simulation still evaluates exact SE with the existing routines.
    This proxy gives DQN/DDPG a label-free training signal.

    Args:
        pt: scalar float OR per-sample tensor of shape (batch, 1) or (batch, 1, 1).
    """
    if isinstance(pt, torch.Tensor):
        pt_t = pt.view(-1, 1, 1)
    else:
        pt_t = pt

    rho = pt_t * shares * d_mask
    gain = sqrt_gain.square().clamp_min(0.0)
    desired = (rho * gain).sum(dim=1)
    ap_total_power = rho.sum(dim=2, keepdim=True)
    received_total = (ap_total_power * gain).sum(dim=1)
    interference = (received_total - desired).clamp_min(0.0)
    sinr = desired / (interference + 1.0)
    return torch.log2(1.0 + sinr).sum(dim=1)
