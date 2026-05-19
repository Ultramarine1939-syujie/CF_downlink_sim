#!/usr/bin/env python3
"""Train an AP-local neural power-allocation model.

The model is intentionally strict-local: each training item is one AP row.
Inputs contain only that AP's AP-UE association row, masked large-scale gain
row, SNR, CSI-error scalar, and simple row statistics. The target is the
WMMSE power share inside the same AP row.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

@dataclass
class LocalFeatureConfig:
    """Feature schema shared by training and MATLAB-facing inference."""

    K: int = 20
    eps: float = 1e-8

    @property
    def input_dim(self) -> int:
        return 2 * self.K + 6


class LocalPowerMLP(nn.Module):
    """Shared AP-local MLP that maps one AP feature row to K UE logits."""

    def __init__(self, K: int = 20, hidden_dim: int = 96, num_layers: int = 3, dropout: float = 0.10):
        super().__init__()
        self.K = int(K)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)

        cfg = LocalFeatureConfig(K=self.K)
        layers = []
        in_dim = cfg.input_dim
        for _ in range(self.num_layers):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
            ])
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, self.K))
        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_local_features(
    sqrt_gain_masked: np.ndarray,
    d_mask: np.ndarray,
    snr_norm: np.ndarray,
    sigma_e: np.ndarray,
) -> np.ndarray:
    """Build strict-local AP-row features.

    Args:
        sqrt_gain_masked: (L, K), already multiplied by D.
        d_mask: (L, K), AP-UE association mask.
        snr_norm: scalar or (L, 1), SNR_dB / 30.
        sigma_e: scalar or (L, 1), CSI error std.
    """
    gain = np.asarray(sqrt_gain_masked, dtype=np.float32)
    d = np.asarray(d_mask, dtype=np.float32)
    if gain.shape != d.shape:
        raise ValueError(f"gain and D must have the same shape, got {gain.shape} and {d.shape}")

    L, K = gain.shape
    snr_col = np.full((L, 1), float(np.asarray(snr_norm).reshape(-1)[0]), dtype=np.float32)
    sigma_col = np.full((L, 1), float(np.asarray(sigma_e).reshape(-1)[0]), dtype=np.float32)
    degree = d.sum(axis=1, keepdims=True) / max(K, 1)
    gain_sum = np.log1p(gain.sum(axis=1, keepdims=True))
    denom = np.maximum(d.sum(axis=1, keepdims=True), 1.0)
    gain_mean = np.log1p(gain.sum(axis=1, keepdims=True) / denom)
    gain_max = np.log1p(gain.max(axis=1, keepdims=True))

    return np.concatenate([
        gain,
        d,
        snr_col,
        sigma_col,
        degree.astype(np.float32),
        gain_sum.astype(np.float32),
        gain_mean.astype(np.float32),
        gain_max.astype(np.float32),
    ], axis=1).astype(np.float32)


class LocalAPDataset(Dataset):
    """Flatten a snapshot dataset into independent AP-row samples."""

    def __init__(self, base: Dataset):
        self.base = base
        sample0 = base[0]
        self.L = int(sample0["D_mask"].shape[0])
        self.K = int(sample0["D_mask"].shape[1])
        self.x, self.target, self.mask = self._materialize_rows()

    def _materialize_rows(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs = []
        targets = []
        masks = []

        for snap_idx in range(len(self.base)):
            sample = self.base[snap_idx]
            d = sample["D_mask"].float().numpy()
            x_ap = sample["x_ap"].float().numpy()
            gain = x_ap[:, :self.K]
            snr_norm = x_ap[0, self.K]
            sigma_e = x_ap[0, self.K + 1]

            feat = build_local_features(
                gain,
                d,
                np.array([snr_norm], dtype=np.float32),
                np.array([sigma_e], dtype=np.float32),
            )

            target = sample["y_share"].float().numpy()
            served = d > 0.5
            row_target_sum = (target * served).sum(axis=1, keepdims=True)
            served_count = served.sum(axis=1, keepdims=True)
            fallback_rows = (served_count > 0) & (row_target_sum <= 0)
            fallback = served.astype(np.float32) / np.maximum(served_count, 1.0)
            target = np.where(fallback_rows, fallback, target).astype(np.float32)
            valid = (served & np.isfinite(target)).astype(np.float32)

            xs.append(feat)
            targets.append(target)
            masks.append(valid)

        return (
            torch.from_numpy(np.concatenate(xs, axis=0).astype(np.float32)),
            torch.from_numpy(np.concatenate(targets, axis=0).astype(np.float32)),
            torch.from_numpy(np.concatenate(masks, axis=0).astype(np.float32)),
        )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x[idx],
            "target": self.target[idx],
            "mask": self.mask[idx],
        }


def masked_share_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE between masked softmax shares and WMMSE target shares."""
    valid_rows = mask.sum(dim=1) > 0
    if not torch.any(valid_rows):
        return logits.sum() * 0.0

    logits_v = logits[valid_rows].masked_fill(mask[valid_rows] <= 0.5, -1e9)
    pred = torch.softmax(logits_v, dim=1)
    target_v = target[valid_rows]
    mask_v = mask[valid_rows]
    diff = (pred - target_v) * mask_v
    return (diff.square().sum(dim=1) / mask_v.sum(dim=1).clamp_min(1.0)).mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    all_pred = []
    all_true = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            logits = model(x)
            loss = masked_share_loss(logits, target, mask)
            rows = x.shape[0]
            total_loss += float(loss.item()) * rows
            total_rows += rows

            valid = mask > 0.5
            pred = torch.softmax(logits.masked_fill(~valid, -1e9), dim=1)
            all_pred.append(pred[valid].detach().cpu())
            all_true.append(target[valid].detach().cpu())

    if all_pred:
        pred_cat = torch.cat(all_pred).numpy()
        true_cat = torch.cat(all_true).numpy()
        if pred_cat.size > 1 and np.std(pred_cat) > 0 and np.std(true_cat) > 0:
            corr = float(np.corrcoef(pred_cat, true_cat)[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0
    return total_loss / max(total_rows, 1), corr


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch in loader:
        x = batch["x"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = masked_share_loss(logits, target, mask)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        rows = x.shape[0]
        total_loss += float(loss.item()) * rows
        total_rows += rows
    return total_loss / max(total_rows, 1)


def load_local_dataset(pattern: str, L: int, K: int) -> Dataset:
    from dataset import GNNDataset

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    datasets = [LocalAPDataset(GNNDataset(path, L=L, K=K)) for path in files]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train strict AP-local power allocation MLP")
    parser.add_argument("--data", type=str, default="../data/gnn_training/*.mat")
    parser.add_argument("--output_dir", type=str, default="../models")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--K", type=int, default=20)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading AP-local rows from {args.data}")

    dataset = load_local_dataset(args.data, args.L, args.K)
    n_total = len(dataset)
    val_size = max(1, int(n_total * args.val_split))
    train_size = n_total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Rows: total={n_total}, train={train_size}, val={val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LocalPowerMLP(K=args.K, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: LocalPowerMLP(K={args.K}, params={n_params:,})")

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    best_corr = 0.0
    patience = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_corr = evaluate(model, val_loader, device)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} "
                f"train={train_loss:.6f} val={val_loss:.6f} corr={val_corr:.4f}"
            )

        if val_loss < best_loss and np.isfinite(val_loss):
            best_loss = val_loss
            best_corr = val_corr
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ckpt = {
        "epoch": best_epoch,
        "model_state_dict": best_state,
        "model_type": "local_mlp",
        "K": args.K,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "feature_schema": "ap_local_masked_gain_dmask_snr_sigma_degree_gainstats_v1",
        "target": "per_ap_wmmse_power_share",
        "val_loss": best_loss,
        "val_corr": best_corr,
        "args": vars(args),
        "feature_config": asdict(LocalFeatureConfig(K=args.K)),
    }
    best_path = os.path.join(args.output_dir, "best_local_gnn_power.pt")
    torch.save(ckpt, best_path)
    print(f"Best Local-GNN model saved to {best_path}")
    print(f"Best val loss={best_loss:.6f}, corr={best_corr:.4f}, epoch={best_epoch + 1}")


if __name__ == "__main__":
    main(sys.argv[1:])
