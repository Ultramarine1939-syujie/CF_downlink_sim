#!/usr/bin/env python3
"""
GLP-GNN Training Script
用于Cell-Free Massive MIMO系统的联合功率控制和预编码优化

基于论文: "Joint Power Control and Precoding for Cell-Free Massive MIMO Systems 
         With Sparse Multi-Dimensional Graph Neural Networks"

用法:
  python train_glp_gnn.py --data "../data/gnn_training/*.mat"
  python train_glp_gnn.py --epochs 300 --batch_size 32 --hidden_dim 128
"""

import os
import sys
import glob
import argparse
import time
import csv
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.optim.lr_scheduler import OneCycleLR

from torch_geometric.data import Batch

from glp_gnn import GLPGNN, GLPGNN_Slim
from dataset import GNNDataset


def custom_collate_glp(batch):
    """
    自定义collate函数 - 支持GLP-GNN批训练
    """
    x_ap_list = []
    x_ue_list = []
    edge_index_ap2ue_list = []
    edge_index_ue2ap_list = []
    D_mask_list = []
    rho_is_nonzero_list = []
    y_list = []
    esr_list = []
    snr_list = []

    L = batch[0]['x_ap'].shape[0]
    K = batch[0]['x_ue'].shape[0]
    num_ap_nodes = L

    for item in batch:
        x_ap_list.append(item['x_ap'])
        x_ue_list.append(item['x_ue'])
        D_mask_list.append(item['D_mask'])
        rho_is_nonzero_list.append(item['rho_is_nonzero'])
        y_list.append(item['y'])
        esr_list.append(item['esr'])
        snr_list.append(item['snr'])

        D = item['D_mask'].numpy()

        ap_src, ue_dst = np.where(D == 1)
        edge_ap2ue = np.array([ap_src, ue_dst + num_ap_nodes])
        edge_index_ap2ue_list.append(torch.LongTensor(edge_ap2ue))

        edge_ue2ap = np.array([ue_dst + num_ap_nodes, ap_src])
        edge_index_ue2ap_list.append(torch.LongTensor(edge_ue2ap))

    data_list = []
    for i in range(len(batch)):
        x_ap = x_ap_list[i]
        x_ue = x_ue_list[i]

        max_feat_dim = max(x_ap.shape[1], x_ue.shape[1])
        x_combined = torch.zeros(L + K, max_feat_dim)
        x_combined[:L, :x_ap.shape[1]] = x_ap
        x_combined[L:, :x_ue.shape[1]] = x_ue

        edge_index = torch.cat([edge_index_ap2ue_list[i], edge_index_ue2ap_list[i]], dim=1)

        data = Batch.from_data_list([{
            'x': x_combined,
            'edge_index': edge_index,
            'D_mask': D_mask_list[i],
            'rho_is_nonzero': rho_is_nonzero_list[i],
            'y': y_list[i],
            'esr': esr_list[i],
            'snr': snr_list[i],
            'num_ap_nodes': L,
            'num_ue_nodes': K
        }])
        data_list.append(data)

    combined_batch = Batch.from_data_list(data_list)
    
    return combined_batch


def compute_loss_glp(rho_pred, rho_true, D_mask, rho_is_nonzero):
    """
    计算GLP-GNN损失函数
    
    只对rho_raw > 0且D=1的位置计算Huber loss
    """
    nonzero_mask = (rho_is_nonzero > 0.5) & (D_mask > 0.5)
    n_valid = nonzero_mask.sum()

    if n_valid == 0:
        nonzero_mask = D_mask > 0.5
        n_valid = nonzero_mask.sum()

    diff = (rho_pred - rho_true) * nonzero_mask.float()
    diff_valid = diff[nonzero_mask]

    if diff_valid.numel() == 0:
        return torch.tensor(0.0, device=rho_pred.device, requires_grad=True)

    huber_loss = nn.functional.huber_loss(
        diff_valid,
        torch.zeros_like(diff_valid),
        delta=0.5,
        reduction='mean'
    )

    return huber_loss


def train_epoch_glp(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
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

        loss = compute_loss_glp(rho_pred, y_batched, D_batched, nz_batched)

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf loss, skipping")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * rho_pred.shape[0]
        n_samples += rho_pred.shape[0]

    return total_loss / max(n_samples, 1)


def evaluate_glp(model, dataloader, device):
    """评估模型"""
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

            mse = compute_loss_glp(rho_pred, y_batched, D_batched, nz_batched).item()
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


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import random


def main():
    parser = argparse.ArgumentParser(
        description='Train GLP-GNN for Cell-Free MIMO Power Allocation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_glp_gnn.py --data "../data/gnn_training/*.mat"
  python train_glp_gnn.py --epochs 300 --batch_size 32 --hidden_dim 128
  python train_glp_gnn.py --model_type slim --epochs 100
        """
    )
    
    parser.add_argument('--data', type=str,
                        default='../data/gnn_training/*.mat',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr_max', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--num_propagations', type=int, default=3,
                        help='Number of label propagation steps')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Label propagation mixing coefficient')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='../models/glp_gnn',
                        help='Output directory')
    parser.add_argument('--model_type', type=str, default='full',
                        choices=['full', 'slim'],
                        help='Model type: full (GLPGNN) or slim (GLPGNN_Slim)')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")

    data_files = sorted(glob.glob(args.data))
    if len(data_files) == 0:
        print(f"ERROR: No files matched pattern: {args.data}")
        sys.exit(1)

    print(f"\nLoading data from {len(data_files)} file(s)...")

    if len(data_files) == 1:
        dataset = GNNDataset(data_files[0])
    else:
        all_datasets = [GNNDataset(f) for f in data_files]
        dataset = ConcatDataset(all_datasets)

    L, K = dataset.L, dataset.K
    print(f"Dataset: {len(dataset)} samples, L={L}, K={K}")

    print("\nData sample stats:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        y = sample['y']
        D = sample['D_mask']
        nz = sample['rho_is_nonzero']
        nz_mask = (nz > 0.5) & (D > 0.5)
        n_nz = int(nz_mask.sum())
        n_total = int(D.sum())
        if n_nz > 0:
            y_nz_min = float(y[nz_mask].min())
            y_nz_max = float(y[nz_mask].max())
        else:
            y_nz_min = y_nz_max = 0.0
        print(f"  Sample {i}: y range=[{y.min():.6f}, {y.max():.6f}], "
              f"D sum={D.sum():.0f}, nonzero rho: {n_nz}/{n_total}")

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
        collate_fn=custom_collate_glp,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_glp,
        num_workers=0
    )

    print(f"\nCreating GLP-GNN model:")
    print(f"  L={L}, K={K}")
    print(f"  hidden_dim={args.hidden_dim}, heads={args.num_heads}")
    print(f"  layers={args.num_layers}, propagations={args.num_propagations}")
    print(f"  alpha={args.alpha}, dropout={args.dropout}")

    if args.model_type == 'full':
        model = GLPGNN(
            L=L, K=K,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_propagations=args.num_propagations,
            alpha=args.alpha,
            dropout=args.dropout
        )
    else:
        model = GLPGNN_Slim(
            L=L, K=K,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    steps_per_epoch = max(1, len(train_dataset) // args.batch_size + 1)
    total_steps = args.epochs * steps_per_epoch

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_max / 10, weight_decay=5e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr_max,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )

    print(f"\nOptimizer: AdamW, OneCycleLR")
    print(f"  max_lr={args.lr_max}, warmup={int(total_steps*0.1)} steps")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nStarting training...")
    best_val_loss = float('inf')
    best_state_dict = None
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0

    training_log = []

    for epoch in range(args.epochs):
        epoch_t0 = time.time()

        train_loss = train_epoch_glp(model, train_loader, optimizer, scheduler, device)
        val_mse, val_corr, val_nz_mse = evaluate_glp(model, val_loader, device)

        epoch_time = time.time() - epoch_t0
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Val MSE: {val_mse:.6f} - "
                  f"Val NZ_MSE: {val_nz_mse:.6f} - "
                  f"Val Corr: {val_corr:.4f} - "
                  f"Time: {epoch_time:.1f}s - "
                  f"LR: {current_lr:.2e}")

        training_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_mse': val_mse,
            'val_nz_mse': val_nz_mse,
            'val_corr': val_corr,
            'lr': current_lr,
            'time': epoch_time
        })

        if val_mse < best_val_loss and np.isfinite(val_mse):
            best_val_loss = val_mse
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = {'val_mse': val_mse, 'val_corr': val_corr, 'val_nz_mse': val_nz_mse}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_every == 0:
            model_path = os.path.join(args.output_dir, f'glp_gnn_checkpoint_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state_dict if best_state_dict else model.state_dict(),
                'val_mse': best_val_loss,
                'val_corr': best_metrics.get('val_corr', 0),
                'args': args
            }, model_path)
            print(f"  -> Checkpoint saved (best from epoch {best_epoch+1})")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete!")
    print(f"Best val MSE: {best_val_loss:.4f} (epoch {best_epoch+1})")
    print(f"Best val Corr: {best_metrics.get('val_corr', 0):.4f}")
    print(f"Best val NZ_MSE: {best_metrics.get('val_nz_mse', 0):.4f}")

    best_model_path = os.path.join(args.output_dir, 'best_glp_gnn.pt')
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_state_dict,
        'val_mse': best_val_loss,
        'val_corr': best_metrics.get('val_corr', 0),
        'val_nz_mse': best_metrics.get('val_nz_mse', 0),
        'norm_method': 'per_sample_signedlog_minmax',
        'args': args
    }, best_model_path)
    print(f"Best model saved to {best_model_path}")

    final_model_path = os.path.join(args.output_dir, 'final_glp_gnn.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_method': 'per_sample_signedlog_minmax',
        'args': args
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    log_csv_path = os.path.join(args.output_dir, 'training_log.csv')
    if training_log:
        fieldnames = list(training_log[0].keys())
        with open(log_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_log)
        print(f"Training log saved to {log_csv_path}")

    summary = {
        'model_type': args.model_type,
        'best_epoch': best_epoch + 1,
        'best_val_mse': best_val_loss,
        'best_val_corr': best_metrics.get('val_corr', 0),
        'best_val_nz_mse': best_metrics.get('val_nz_mse', 0),
        'n_params': n_params,
        'train_size': train_size,
        'val_size': val_size,
        'args': vars(args)
    }

    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
