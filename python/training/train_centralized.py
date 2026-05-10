#!/usr/bin/env python3
"""
GNN 功率分配模型训练脚本
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PowerGNN_GAT
from data import GNNDataset, custom_collate


def compute_loss(rho_pred, rho_true, D_mask, rho_is_nonzero):
    nonzero_mask = (rho_is_nonzero > 0.5) & (D_mask > 0.5)
    n_valid = nonzero_mask.sum()

    if n_valid == 0:
        nonzero_mask = D_mask > 0.5
        n_valid = nonzero_mask.sum()

    diff = (rho_pred - rho_true) * nonzero_mask.float()
    diff_valid = diff[nonzero_mask]

    huber_loss = nn.functional.huber_loss(
        diff_valid,
        torch.zeros_like(diff_valid),
        delta=0.5,
        reduction='mean'
    )

    return huber_loss


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        rho_pred = model(batch)
        y_batched = batch.y.view(-1, model.L, model.K)
        D_batched = batch.D_mask.view(-1, model.L, model.K)
        nz_batched = batch.rho_is_nonzero.view(-1, model.L, model.K)

        loss = compute_loss(rho_pred, y_batched, D_batched, nz_batched)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * rho_pred.shape[0]
        n_samples += rho_pred.shape[0]

    return total_loss / max(n_samples, 1)


def evaluate(model, dataloader, device):
    model.eval()
    total_mse = 0
    n_samples = 0

    all_rho_pred = []
    all_rho_true = []
    all_nonzero_mask = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            rho_pred = model(batch)
            y_batched = batch.y.view(-1, model.L, model.K)
            D_batched = batch.D_mask.view(-1, model.L, model.K)
            nz_batched = batch.rho_is_nonzero.view(-1, model.L, model.K)

            mse = compute_loss(rho_pred, y_batched, D_batched, nz_batched).item()
            total_mse += mse * rho_pred.shape[0]

            nonzero_mask = (nz_batched > 0.5) & (D_batched > 0.5)
            all_rho_pred.append(rho_pred.cpu())
            all_rho_true.append(y_batched.cpu())
            all_nonzero_mask.append(nonzero_mask.cpu())

            n_samples += rho_pred.shape[0]

    avg_mse = total_mse / max(n_samples, 1)

    rho_pred_cat = torch.cat(all_rho_pred, dim=0).numpy()
    rho_true_cat = torch.cat(all_rho_true, dim=0).numpy()
    nonzero_cat = torch.cat(all_nonzero_mask, dim=0).numpy()

    pred_nz = rho_pred_cat[nonzero_cat]
    true_nz = rho_true_cat[nonzero_cat]

    valid_mask = np.isfinite(pred_nz) & np.isfinite(true_nz)
    if valid_mask.sum() > 1:
        correlation = np.corrcoef(pred_nz[valid_mask], true_nz[valid_mask])[0, 1]
    else:
        correlation = 0.0

    if valid_mask.sum() > 0:
        nonzero_mse = float(np.mean((pred_nz[valid_mask] - true_nz[valid_mask]) ** 2))
    else:
        nonzero_mse = 0.0

    return avg_mse, correlation, nonzero_mse


def main():
    parser = argparse.ArgumentParser(description='Train GNN Power Allocation Model')
    parser.add_argument('--data', type=str,
                        default='../data/gnn_training/*.mat',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_max', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--output_dir', type=str, default='../models')
    parser.add_argument('--patience', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_files = sorted(glob.glob(args.data))
    if len(data_files) == 0:
        print(f"ERROR: No files matched pattern: {args.data}")
        sys.exit(1)
    print(f"Found {len(data_files)} data file(s)")

    if len(data_files) == 1:
        dataset = GNNDataset(data_files[0])
    else:
        from torch.utils.data import ConcatDataset
        all_datasets = [GNNDataset(f) for f in data_files]
        dataset = ConcatDataset(all_datasets)

    print(f"Total dataset size: {len(dataset)}")

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train size: {train_size}, Val size: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
    )

    if hasattr(dataset, 'L'):
        L, K = dataset.L, dataset.K
    else:
        L, K = 100, 20

    print(f"\nCreating GAT-GNN model: L={L}, K={K}")
    model = PowerGNN_GAT(
        L=L, K=K,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_scale=1.0
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    steps_per_epoch = len(dataset) // args.batch_size + 1
    total_steps = args.epochs * steps_per_epoch
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_max / 10, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr_max,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )

    print("\nStarting training...")
    best_val_loss = float('inf')
    best_state_dict = None
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0
    save_interval = 50

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_mse, val_corr, val_nz_mse = evaluate(model, val_loader, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Val MSE: {val_mse:.6f} - "
                  f"Val Corr: {val_corr:.4f}")

        if val_mse < best_val_loss and not np.isnan(val_mse):
            best_val_loss = val_mse
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = {'val_mse': val_mse, 'val_corr': val_corr, 'val_nz_mse': val_nz_mse}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % save_interval == 0:
            model_path = os.path.join(args.output_dir, 'best_gat_gnn_power.pt')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_state_dict,
                'val_mse': best_val_loss,
                'val_corr': best_metrics.get('val_corr', 0),
            }, model_path)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    model_path = os.path.join(args.output_dir, 'best_gat_gnn_power.pt')
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_state_dict,
        'val_mse': best_val_loss,
        'val_corr': best_metrics.get('val_corr', 0),
    }, model_path)
    print(f"Best model saved to {model_path}")
    print(f"Best val MSE: {best_val_loss:.4f} (epoch {best_epoch+1})")


if __name__ == "__main__":
    main()
