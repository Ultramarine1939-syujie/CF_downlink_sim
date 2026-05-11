#!/usr/bin/env python3
"""
SP-MDGNN 训练脚本

基于论文 "Joint Power Control and Precoding for Cell-Free Massive MIMO Systems
With Sparse Multi-Dimensional Graph Neural Networks" (arXiv:2507.01876)

支持 SP-MDGNN 和 GLP-GNN-Lite 两种模型
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

from models import SP_MDGNN, GLP_GNN_Lite


def custom_collate(batch):
    """自定义 collate 函数 - 支持 SP-MDGNN 批训练"""
    x_ap_list = []
    x_ue_list = []
    D_mask_list = []
    rho_is_nonzero_list = []
    y_list = []
    esr_list = []
    snr_list = []

    for item in batch:
        x_ap_list.append(item['x_ap'])
        x_ue_list.append(item['x_ue'])
        D_mask_list.append(item['D_mask'])
        rho_is_nonzero_list.append(item['rho_is_nonzero'])
        y_list.append(item['y'])
        esr_list.append(item['esr'])
        snr_list.append(item['snr'])

    return {
        'x_ap': torch.stack(x_ap_list),
        'x_ue': torch.stack(x_ue_list),
        'D_mask': torch.stack(D_mask_list),
        'rho_is_nonzero': torch.stack(rho_is_nonzero_list),
        'y': torch.stack(y_list),
        'esr': torch.stack(esr_list),
        'snr': torch.stack(snr_list),
    }


def compute_loss(rho_pred, rho_true, D_mask, rho_is_nonzero):
    """
    功率分配损失函数

    只对 rho_raw > 0 且 D=1 的位置计算损失
    """
    nonzero_mask = (rho_is_nonzero > 0.5) & (D_mask > 0.5)
    n_valid = nonzero_mask.sum()

    if n_valid == 0:
        nonzero_mask = D_mask > 0.5
        n_valid = nonzero_mask.sum()

    diff = (rho_pred - rho_true) * nonzero_mask.float()
    diff_valid = diff[nonzero_mask]

    if len(diff_valid) == 0:
        return torch.tensor(0.0, device=rho_pred.device, requires_grad=True)

    huber_loss = nn.functional.huber_loss(
        diff_valid,
        torch.zeros_like(diff_valid),
        delta=0.5,
        reduction='mean'
    )

    return huber_loss


def compute_esr_loss(esr_pred, esr_true):
    """ESR 预测损失"""
    return nn.functional.mse_loss(esr_pred, esr_true)


def train_epoch(model, dataloader, optimizer, scheduler, device, alpha=0.8):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(batch)
        rho_pred = outputs['power']

        y_batched = batch['y']
        D_batched = batch['D_mask']
        nz_batched = batch['rho_is_nonzero']

        power_loss = compute_loss(rho_pred, y_batched, D_batched, nz_batched)

        if torch.isnan(power_loss) or torch.isinf(power_loss):
            continue

        power_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += power_loss.item() * rho_pred.shape[0]
        n_samples += rho_pred.shape[0]

    return total_loss / max(n_samples, 1)


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_mse = 0
    n_samples = 0

    all_rho_pred = []
    all_rho_true = []
    all_nonzero_mask = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(batch)
            rho_pred = outputs['power']

            y_batched = batch['y']
            D_batched = batch['D_mask']
            nz_batched = batch['rho_is_nonzero']

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
    parser = argparse.ArgumentParser(description='Train SP-MDGNN Power Allocation Model')
    parser.add_argument('--data', type=str,
                        default='../data/gnn_training/*.mat',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr_max', type=float, default=1e-3, help='Max learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--tau', type=float, default=0.5, help='Sparsity threshold')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--output_dir', type=str, default='../models', help='Output directory')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--model_type', type=str, default='sp_mdgnn',
                        choices=['sp_mdgnn', 'glp_lite'],
                        help='Model type: SP-MDGNN or GLP-GNN-Lite')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")

    print(f"\nLoading data from {args.data}...")

    data_files = sorted(glob.glob(args.data))
    if len(data_files) == 0:
        print(f"ERROR: No files matched pattern: {args.data}")
        sys.exit(1)
    print(f"Found {len(data_files)} data file(s):")
    for f in data_files:
        print(f"  - {f}")

    from dataset import GNNDataset
    from torch.utils.data import ConcatDataset

    if len(data_files) == 1:
        dataset = GNNDataset(data_files[0])
    else:
        all_datasets = [GNNDataset(f) for f in data_files]
        dataset = ConcatDataset(all_datasets)

    print(f"Total dataset size: {len(dataset)}")

    L, K = dataset.L, dataset.K
    print(f"System config: L={L} APs, K={K} UEs")

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"\nTrain size: {train_size}, Val size: {val_size}")

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

    print(f"\nCreating {args.model_type} model: L={L}, K={K}")
    print(f"  hidden_dim={args.hidden_dim}, layers={args.num_layers}, "
          f"tau={args.tau}, drop={args.dropout}")

    if args.model_type == 'sp_mdgnn':
        model = SP_MDGNN(
            L=L, K=K,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            tau=args.tau,
            dropout=args.dropout,
            output_scale=1.0
        )
    else:
        model = GLP_GNN_Lite(
            L=L, K=K,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            tau=args.tau,
            dropout=args.dropout,
            output_scale=1.0
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    steps_per_epoch = len(train_dataset) // args.batch_size + 1
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

    print(f"  Optimizer: AdamW, OneCycleLR (max_lr={args.lr_max})")

    print("\nStarting training...")
    best_val_loss = float('inf')
    best_state_dict = None
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = f'best_{args.model_type}_power.pt'

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_mse, val_corr, val_nz_mse = evaluate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Val MSE: {val_mse:.6f} - "
                  f"Val NZ_MSE: {val_nz_mse:.6f} - "
                  f"Val Corr: {val_corr:.4f} - "
                  f"LR: {current_lr:.2e}")

        if val_mse < best_val_loss and not np.isnan(val_mse):
            best_val_loss = val_mse
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = {'val_mse': val_mse, 'val_corr': val_corr, 'val_nz_mse': val_nz_mse}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            model_path = os.path.join(args.output_dir, model_name)
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_state_dict,
                'val_mse': best_val_loss,
                'val_corr': best_metrics.get('val_corr', 0),
                'norm_method': 'per_sample_signedlog_minmax',
                'tau': args.tau,
                'args': args
            }, model_path)
            print(f"  -> Checkpoint saved (best from epoch {best_epoch+1})")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.4f} (epoch {best_epoch+1})")
    print(f"Best val Corr: {best_metrics.get('val_corr', 0):.4f}")

    model_path = os.path.join(args.output_dir, model_name)
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_state_dict,
        'val_mse': best_val_loss,
        'val_corr': best_metrics.get('val_corr', 0),
        'norm_method': 'per_sample_signedlog_minmax',
        'tau': args.tau,
        'args': args
    }, model_path)
    print(f"Best model saved to {model_path}")

    final_path = os.path.join(args.output_dir, f'final_{args.model_type}_power.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_method': 'per_sample_signedlog_minmax',
        'tau': args.tau,
        'args': args
    }, final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
