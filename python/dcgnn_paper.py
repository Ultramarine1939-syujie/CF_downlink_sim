#!/usr/bin/env python3
"""
Paper-aligned DCGNN implementation for downlink CF massive MIMO power allocation.

Reference:
  Zhao et al., "A Dynamic Power Allocation Approach for Downlink Cell-Free
  Massive MIMO With Graph Neural Network", IEEE TVT, Vol. 74, No. 4, Apr. 2025.

Architecture (matching the paper):
  - Nodes: L×K AP-UE links, each with scalar feature β_{k,l}
  - Directed graph: top-z dominant interferers per node, selected via sorting
  - Message passing: log₂(β_{neighbor} / β_{self}) for each neighbor
  - GNN layers: shared FC filters (Aggregation + Combination integrated)
    * 2 hidden layers: 32 → 16 neurons
  - Output: FC + Sigmoid → per-AP Softmax → pmax
  - Training: unsupervised, loss = -1/K Σ SE_k (maximize sum-SE)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# Differentiable SINR / SE computation (large-scale proxy)
# ═══════════════════════════════════════════════════════════════════════════


def large_scale_sinr(
    rho: torch.Tensor,
    beta: torch.Tensor,
    sigma2: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute large-scale SINR proxy for gradent-based training.

    Implements a simplified but physically motivated SINR that captures the
    key power-vs-interference trade-off optimised by equation (11) in the
    paper.  The full correlated-fading SINR (eq. 11) can be substituted once
    spatial correlation matrices are integrated into the differentiable path.

    Args:
        rho:   (batch, L, K)  power allocation coefficients  [mW]
        beta:  (batch, L, K)  large-scale fading coefficients (linear)
        sigma2: scalar         noise power (normalised to 1 by default)

    Returns:
        sinr:  (batch, K)     effective SINR per UE
    """
    # Desired signal: Σ_l ρ_{k,l} · β_{k,l}
    desired = (rho * beta).sum(dim=1)  # (batch, K)

    # Total received power at UE k from ALL APs across ALL UE allocations
    # total_rx[b, l, k] = Σ_i ρ_{i,l} · β_{k,l}
    ap_total_tx = rho.sum(dim=2, keepdim=True)  # (batch, L, 1) — total Tx per AP
    total_rx = (ap_total_tx * beta).sum(dim=1)  # (batch, K)

    # Interference = total received − desired
    interference = (total_rx - desired).clamp_min(0.0)

    sinr = desired / (interference + sigma2 + eps)
    return sinr


def spectral_efficiency(
    sinr: torch.Tensor,
    tau_c: float = 200.0,
    tau_p: float = 10.0,
) -> torch.Tensor:
    """SE_k = (1 − τ_p/τ_c) · log₂(1 + SINR_k)  — equation (8) in the paper."""
    prelog = 1.0 - tau_p / tau_c
    return prelog * torch.log2(1.0 + sinr)


def unsupervised_loss(
    rho: torch.Tensor,
    beta: torch.Tensor,
    tau_c: float = 200.0,
    tau_p: float = 10.0,
    sigma2: float = 1.0,
) -> torch.Tensor:
    """Unsupervised loss:  −(1/K) Σ SE_k   (equation 24).

    Minimising this is equivalent to maximising sum-SE.
    """
    sinr = large_scale_sinr(rho, beta, sigma2)
    se = spectral_efficiency(sinr, tau_c, tau_p)
    return -se.mean()


# ═══════════════════════════════════════════════════════════════════════════
# GNN building blocks
# ═══════════════════════════════════════════════════════════════════════════


class DCGNNLayer(nn.Module):
    """One GNN layer as described in the paper: Aggregation + Combination
    integrated into a single shared FC filter.

    For node *v* with neighbour set R'(v):

        c^{u}_v = σ_u( [ c^{u-1}_{n₁} ‖ c^{u-1}_{n₂} ‖ … ‖ c^{u-1}_{n_z} ] )

    where σ_u is a learned FC + activation shared across all nodes.
    """

    def __init__(self, in_features_per_node: int, out_features: int, z: int):
        super().__init__()
        self.z = z
        self.in_features_per_node = in_features_per_node
        self.out_features = out_features

        # FC takes concatenated z neighbour feature vectors
        self.fc = nn.Linear(in_features_per_node * z, out_features, bias=True)
        self.activation = nn.ReLU()

    def forward(
        self,
        node_features: torch.Tensor,
        neighbour_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for one GNN layer.

        Args:
            node_features:  (batch, V, d_in)   features of ALL nodes
            neighbour_idx:  (batch, V, z)      indices of z neighbours per node

        Returns:
            new_features:   (batch, V, d_out)
        """
        batch_size, V, d_in = node_features.shape
        device = node_features.device

        # Gather neighbour features: (batch, V, z, d_in)
        # Expand batch index
        batch_idx = torch.arange(batch_size, device=device)[:, None, None]  # (B, 1, 1)
        batch_idx = batch_idx.expand(batch_size, V, self.z)                # (B, V, z)

        neighbour_feats = node_features[batch_idx, neighbour_idx, :]       # (B, V, z, d_in)

        # Concatenate along last dims → (B, V, z*d_in)
        neighbour_feats = neighbour_feats.reshape(batch_size, V, self.z * d_in)

        # Shared FC + activation
        out = self.fc(neighbour_feats)
        out = self.activation(out)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Paper DCGNN Model
# ═══════════════════════════════════════════════════════════════════════════


class PaperDCGNN(nn.Module):
    """DCGNN power-allocation model matching the IEEE TVT 2025 paper.

    Architecture summary
    --------------------
    * |V| = L × K  link-nodes, each with scalar feature β_{k,l}
    * Dynamic directed graph: top-z interfering APs per victim UE
    * Message passing: log₂(β_{neighbour} / β_{self})  (eqs. 21–22)
    * 2 shared-FC GNN layers (32 → 16 neurons) + output FC + Sigmoid
    * Per-AP Softmax → pmax  power normalisation
    * Unsupervised training: loss = −mean(SE_k)
    """

    def __init__(
        self,
        L: int = 100,
        K: int = 20,
        z: int = 15,
        hidden_dims: list[int] | None = None,
        pmax: float = 1.0,
        tau_c: float = 200.0,
        tau_p: float = 10.0,
        sigma2: float = 1.0,
    ):
        """
        Args:
            L:           number of APs
            K:           number of single-antenna UEs
            z:           max neighbours per node (paper default: 15)
            hidden_dims: output dims of hidden GNN layers (paper: [32, 16])
            pmax:        per-AP transmission power limit
            tau_c:       coherence block length in symbols
            tau_p:       pilot sequence length
            sigma2:      noise power (normalised)
        """
        super().__init__()
        self.L = L
        self.K = K
        self.V = L * K  # total link-nodes
        self.z = min(z, L - 1)  # can't have more neighbours than other APs
        self.pmax = pmax
        self.tau_c = tau_c
        self.tau_p = tau_p
        self.sigma2 = sigma2

        if hidden_dims is None:
            hidden_dims = [32, 16]

        # ── Build GNN layers ──
        self.gnn_layers = nn.ModuleList()
        in_dim = 1  # scalar β per node
        for h_dim in hidden_dims:
            self.gnn_layers.append(DCGNNLayer(in_dim, h_dim, self.z))
            in_dim = h_dim

        # ── Output layer: FC + Sigmoid → 1 scalar per node ──
        self.output_fc = nn.Linear(in_dim, 1, bias=True)
        self.output_sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for all FC layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Dynamic graph construction
    # ------------------------------------------------------------------

    def build_dynamic_neighbours(
        self,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build directed graph: for each link-node (k, l), select the z APs
        (other than l) with the strongest channel to UE k.

        Sorting key for node v = (k, l):  β_{k, l'}  for candidate AP l'.

        Args:
            beta: (batch, L, K) large-scale fading (linear scale)

        Returns:
            neighbour_idx:  (batch, V, z)  neighbour node indices
            log_ratios:     (batch, V, z)  log₂(β_{k, l'} / β_{k, l})
        """
        batch_size, L, K = beta.shape
        device = beta.device

        # beta_t: (batch, K, L) — per-UE view
        beta_t = beta.permute(0, 2, 1)  # (B, K, L)

        # Sort APs by channel strength for each UE (descending)
        sorted_vals, sorted_idx = torch.sort(beta_t, dim=2, descending=True)
        # sorted_idx[b, k, :] = AP indices sorted by β_{k,·} descending

        neighbour_idx = torch.zeros(batch_size, self.V, self.z, dtype=torch.long, device=device)
        log_ratios = torch.zeros(batch_size, self.V, self.z, device=device)

        for b in range(batch_size):
            for k in range(K):
                beta_k = beta[b, :, k]  # (L,)

                # Sorted APs for this UE
                sorted_aps = sorted_idx[b, k, :]  # (L,)

                for l in range(L):
                    v = k * L + l  # node index
                    beta_self = beta_k[l].clamp_min(1e-12)

                    # Select top-(z+1) APs (extra one in case self is in top)
                    top_aps = sorted_aps[:self.z + 1]  # (z+1,)
                    top_vals = sorted_vals[b, k, :self.z + 1]  # (z+1,)

                    # Exclude self
                    mask = top_aps != l
                    selected_aps = top_aps[mask][:self.z]
                    selected_vals = top_vals[mask][:self.z]

                    n_selected = selected_aps.shape[0]
                    neighbour_idx[b, v, :n_selected] = selected_aps + k * L
                    # Selected AP l' → node index = k * L + l' (link (k, l'))

                    # Compute log-ratios
                    if n_selected > 0:
                        log_ratios[b, v, :n_selected] = torch.log2(
                            selected_vals / beta_self
                        )

                    # If fewer than z neighbours (shouldn't happen for L ≫ z),
                    # the remaining entries stay 0 (padding).

                    # Handle the padding: repeat the last valid entry or use
                    # self-reference (log₂(1)=0) for missing neighbours.
                    if n_selected < self.z:
                        # Pad neighbour indices with self
                        neighbour_idx[b, v, n_selected:] = v
                        # log₂(β_self / β_self) = 0  → already zero

        return neighbour_idx, log_ratios

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        beta: torch.Tensor,
        return_se: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: β → power allocation coefficients.

        Args:
            beta:      (batch, L, K)  large-scale fading (linear)
            return_se: if True also return per-UE spectral efficiency

        Returns:
            rho:       (batch, L, K)  power allocation coefficients
            se:        (batch, K)     optional — per-UE SE
        """
        batch_size = beta.shape[0]

        # ── 1. Build dynamic graph & compute log-ratio features ──
        neighbour_idx, log_ratios = self.build_dynamic_neighbours(beta)
        # neighbour_idx: (B, V, z)
        # log_ratios:    (B, V, z)

        # ── 2. Initial node features: scalar β (eq. 18) ──
        # node v = (k,l): feature = log-ratio is used as input to layer 1
        # but the GNN layer expects features of shape (B, V, d_in).
        # For layer 1, each "feature" is the z log-ratios, which are
        # already neighbour-normalised. We store the log-ratio tensor as
        # the per-node representation for the first layer by using a
        # dummy feature per node (the log-ratios ARE the neighbour
        # information that gets processed).
        #
        # The cleanest way: set initial node features as scalar β,
        # and let the first GNN layer's gather step produce the log-ratios.
        beta_flat = beta.permute(0, 2, 1).reshape(batch_size, self.V, 1)
        # (B, V, 1) where V = L*K, node index = k*L + l

        node_feats = beta_flat  # (B, V, 1)

        # ── 3. Precompute log-ratio as the "message" for layer 1 ──
        # The paper preprocesses (eq. 22) and feeds log-ratios directly
        # to the first FC.  We realise this by storing log-ratios as
        # the initial feature so the gather in DCGNNLayer simply
        # concatenates them.
        node_feats = log_ratios.unsqueeze(-1)  # (B, V, z, 1)
        # We need (B, V, d_in) with d_in=z for the first layer.
        # Actually, the DCGNNLayer concatenates z neighbour features of
        # dim d_in each. For layer 1, d_in=1, so it gathers (B, V, z, 1)
        # and reshapes to (B, V, z).
        # The log_ratios are already the neighbour-normalised values!
        #
        # Better approach: the first layer's input IS the raw β scalar.
        # The layer gathers neighbour β's and internally computes the
        # normalisation. But the paper explicitly pre-normalises.
        #
        # Simplest correct approach: use log_ratios directly as the
        # aggregated input to the first FC, skipping the gather step
        # for layer 1. This is mathematically equivalent to the paper's
        # eq. (23).

        # ── Layer 1 (special: input = log-ratios, eq. 23) ──
        # c²_v = σ₁([log₂(β_{n₁}/β_v), …, log₂(β_{n_z}/β_v)]^T)
        h = log_ratios  # (B, V, z) — already the z log-ratios per node
        h = self.gnn_layers[0].activation(
            self.gnn_layers[0].fc(h)
        )  # (B, V, 32)

        # ── Layers 2..U (message passing with GNN layers) ──
        for layer in self.gnn_layers[1:]:
            h = layer(h, neighbour_idx)

        # ── Output: FC + Sigmoid → (B, V, 1) ──
        rho_raw = self.output_sigmoid(self.output_fc(h))  # (B, V, 1)
        rho_raw = rho_raw.view(batch_size, self.L, self.K)  # (B, L, K)
        # node index = k*L + l → reshape to (L, K) with care.
        # rho_raw.view(B, L, K) won't work because reshape goes by k first.
        # Need proper transpose:
        # Current shape after view: rho_raw[b, k*L+l] = rho_raw_from_node[b, v]
        # rho_raw_reshape = rho_raw.reshape(batch_size, self.K, self.L)  # (B, K, L)
        # rho_raw = rho_raw_reshape.permute(0, 2, 1)  # (B, L, K)
        # But rho_raw was already (B, V, 1), V = L*K with node idx = k*L + l.
        # After .view(B, L, K), PyTorch fills row-major: dim0=L, dim1=K.
        # Node (k,l) at index k*L+l → row l, column k → correct!
        # Wait, no. k*L+l ranges 0..KL-1. Reshaping to (B,L,K):
        # index v = l + k*L maps to [b, l, k] in (B, L, K) → correct.

        # ── Per-AP Softmax normalisation ──
        rho_share = torch.softmax(rho_raw, dim=2)  # softmax over K per AP
        rho = rho_share * self.pmax

        if return_se:
            sinr = large_scale_sinr(rho, beta, self.sigma2)
            se = spectral_efficiency(sinr, self.tau_c, self.tau_p)
            return rho, se

        return rho

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Unsupervised training step: maximise sum-SE.

        Returns:
            loss:   scalar loss = −mean(SE_k)
            stats:  dict with mean_SE, max_SE, etc.
        """
        rho, se = self.forward(beta, return_se=True)
        loss = -se.mean()

        with torch.no_grad():
            stats = {
                "loss": float(loss),
                "mean_SE": float(se.mean()),
                "max_SE": float(se.max()),
                "min_SE": float(se.min()),
                "mean_rho": float(rho.mean()),
                "max_rho": float(rho.max()),
            }
        return loss, stats


# ═══════════════════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════════════════


def generate_training_beta(
    L: int,
    K: int,
    square_length: float = 500.0,
    distance_vertical: float = 3.0,
    alpha: float = 36.7,
    constant_term: float = -30.5,
    sigma_sf: float = 8.0,
    decorr: float = 9.0,
    noise_variance_dbm: float = -114.0,
    carrier_freq_hz: float = 2e9,
    bandwidth_hz: float = 20e6,
    batch_size: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate large-scale fading coefficients matching the paper's setup.

    Paper defaults (Section V.A):
      - 500 m × 500 m square, AP height 3 m
      - β_{k,l}[dB] = −30.5 − 36.7·log₁₀(d_{k,l}) + F
      - F ∼ N(0, 8²)  shadow fading σ = 8 dB
      - Noise power −114 dBm

    Returns:
        beta_over_noise: (batch_size, L, K)  β_{k,l} / σ²  (linear scale)
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma2_linear = 10.0 ** (noise_variance_dbm / 10.0)  # mW

    beta_all = np.zeros((batch_size, L, K), dtype=np.float64)

    wrap_locations = np.array([
        -square_length, 0.0, square_length
    ])
    wrap_grid = np.array([
        x + 1j * y
        for x in wrap_locations for y in wrap_locations
    ])

    for b in range(batch_size):
        ap_pos = (rng.random(L) + 1j * rng.random(L)) * square_length
        ue_pos = (rng.random(K) + 1j * rng.random(K)) * square_length

        shadow_realisations = np.zeros((K, L), dtype=np.float64)

        for k in range(K):
            ue = ue_pos[k]
            ap_wrapped = ap_pos[:, None] + wrap_grid[None, :]
            abs_dist = np.abs(ap_wrapped - ue)
            which_pos = np.argmin(abs_dist, axis=1)
            dist_2d = abs_dist[np.arange(L), which_pos]
            distances = np.sqrt(distance_vertical ** 2 + dist_2d ** 2)

            if k > 0:
                shortest_prev = np.array([
                    np.min(np.abs(ue - ue_pos[i] + wrap_grid))
                    for i in range(k)
                ])
                new_col = sigma_sf ** 2 * np.power(2.0, -shortest_prev / decorr)
                # Build correlation matrix incrementally
                corr = np.ones((k, k), dtype=np.float64) * sigma_sf ** 2
                # Shadow correlation between previous UEs
                for i in range(k):
                    for j in range(k):
                        if i != j:
                            d = np.min(np.abs(ue_pos[i] - ue_pos[j] + wrap_grid))
                            corr[i, j] = sigma_sf ** 2 * (2.0 ** (-d / decorr))

                term1 = np.linalg.solve(corr, new_col)
                mean_values = term1 @ shadow_realisations[:k, :]
                variance = sigma_sf ** 2 - term1 @ new_col
                std_value = np.sqrt(max(float(variance), 0.0))
            else:
                mean_values = 0.0
                std_value = sigma_sf

            shadowing = mean_values + std_value * rng.standard_normal(L)
            shadow_realisations[k, :] = shadowing

            beta_db = (
                constant_term
                - alpha * np.log10(distances)
                + shadowing
            )
            beta_all[b, :, k] = 10.0 ** (beta_db / 10.0) / sigma2_linear

    return beta_all.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Model I/O helpers
# ═══════════════════════════════════════════════════════════════════════════


def save_paper_dcgnn(model: PaperDCGNN, path: Path | str, **extra) -> None:
    """Save a PaperDCGNN checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "paper_dcgnn",
        "L": model.L,
        "K": model.K,
        "z": model.z,
        "pmax": model.pmax,
        "tau_c": model.tau_c,
        "tau_p": model.tau_p,
        "sigma2": model.sigma2,
        **extra,
    }
    torch.save(checkpoint, path)


def load_paper_dcgnn(path: Path | str, map_location: str = "cpu") -> PaperDCGNN:
    """Load a PaperDCGNN from checkpoint."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    L = int(ckpt["L"])
    K = int(ckpt["K"])
    z = int(ckpt.get("z", 15))
    pmax = float(ckpt.get("pmax", 1.0))
    tau_c = float(ckpt.get("tau_c", 200.0))
    tau_p = float(ckpt.get("tau_p", 10.0))
    sigma2 = float(ckpt.get("sigma2", 1.0))

    # Infer hidden_dims from state_dict
    hidden_dims = []
    for key in sorted(ckpt["model_state_dict"]):
        if key.startswith("gnn_layers.") and key.endswith(".fc.weight"):
            out_dim = ckpt["model_state_dict"][key].shape[0]
            hidden_dims.append(out_dim)

    model = PaperDCGNN(
        L=L, K=K, z=z,
        hidden_dims=hidden_dims if hidden_dims else [32, 16],
        pmax=pmax, tau_c=tau_c, tau_p=tau_p, sigma2=sigma2,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def infer_paper_dcgnn(
    model: PaperDCGNN,
    beta: np.ndarray,
    pmax: float | None = None,
) -> np.ndarray:
    """Run inference with a trained PaperDCGNN.

    Args:
        model: trained PaperDCGNN
        beta:  (L, K) or (1, L, K) large-scale fading-over-noise (linear)
        pmax:  override per-AP power limit

    Returns:
        rho:   (L, K) power allocation coefficients
    """
    was_training = model.training
    model.eval()

    beta_t = torch.from_numpy(np.asarray(beta, dtype=np.float32))
    if beta_t.ndim == 2:
        beta_t = beta_t.unsqueeze(0)  # (1, L, K)

    if pmax is not None:
        old_pmax = model.pmax
        model.pmax = pmax

    with torch.no_grad():
        rho_t = model.forward(beta_t)

    if pmax is not None:
        model.pmax = old_pmax
    if was_training:
        model.train()

    return rho_t.squeeze(0).cpu().numpy()
