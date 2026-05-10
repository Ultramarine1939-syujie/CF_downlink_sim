#!/usr/bin/env python3
"""
消融实验主脚本 — Ablation Study for GNN+FL Power Allocation

支持 6 个消融维度:
  1. GNN vs No-GNN (PowerGNN_GAT vs PowerGNN_MLP)
  2. GAT vs MLP (注意力机制 vs 简单MLP)
  3. FedAvg vs 集中式训练
  4. 归一化策略 (Per-sample vs Global)
  5. 数据增强 (With vs Without)
  6. L-MMSE 重构 (全局协方差 vs 本地协方差) — MATLAB端

用法:
  python train_gnn_ablation.py --ablation all --data "../data/gnn_training/*.mat"
  python train_gnn_ablation.py --ablation gnn_vs_nognn --data "../data/gnn_training/*.mat"
  python train_gnn_ablation.py --ablation norm --epochs 100
"""

import argparse
import copy
import csv
import glob
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset

from dataset import GNNDataset, GNNDatasetGlobalNorm
from train_gnn import (
    PowerGNN_GAT, PowerGNN_MLP,
    custom_collate, custom_collate_mlp,
    compute_loss, evaluate, evaluate_mlp,
    train_epoch, train_epoch_mlp,
)
from fedavg import (
    set_seed, split_dataset_to_clients, local_train,
    fedavg_aggregate, save_checkpoint,
)


# ──────────────────────────────────────────────
# 消融变体配置
# ──────────────────────────────────────────────

ABLATION_VARIANTS = {
    # ── 基线 ──
    'full': {
        'model_type': 'gat',
        'training_type': 'centralized',
        'norm_type': 'per_sample',
        'augment': True,
        'description': 'Full model (GAT + centralized + per-sample norm + augment)',
    },
    # ── 维度 1+2: GNN vs No-GNN / GAT vs MLP ──
    'mlp_only': {
        'model_type': 'mlp',
        'training_type': 'centralized',
        'norm_type': 'per_sample',
        'augment': True,
        'description': 'MLP only (no graph, no attention)',
    },
    'mlp_fedavg': {
        'model_type': 'mlp',
        'training_type': 'fedavg',
        'norm_type': 'per_sample',
        'augment': True,
        'description': 'MLP + FedAvg (no graph + FL)',
    },
    # ── 维度 3: FedAvg vs 集中式 ──
    'fedavg_full': {
        'model_type': 'gat',
        'training_type': 'fedavg',
        'norm_type': 'per_sample',
        'augment': True,
        'description': 'GAT + FedAvg (FL vs centralized)',
    },
    # ── 维度 4: 归一化策略 ──
    'global_norm': {
        'model_type': 'gat',
        'training_type': 'centralized',
        'norm_type': 'global',
        'augment': True,
        'description': 'GAT + global min-max normalization',
    },
    'global_norm_fedavg': {
        'model_type': 'gat',
        'training_type': 'fedavg',
        'norm_type': 'global',
        'augment': True,
        'description': 'GAT + FedAvg + global norm',
    },
    # ── 维度 5: 数据增强 ──
    'no_augment': {
        'model_type': 'gat',
        'training_type': 'centralized',
        'norm_type': 'per_sample',
        'augment': False,
        'description': 'GAT + centralized + no augmentation',
    },
}

# 消融维度分组 (用于 --ablation single-dimension 运行)
ABLATION_GROUPS = {
    'gnn_vs_nognn': ['full', 'mlp_only'],
    'gat_vs_mlp': ['full', 'mlp_only'],  # 同上, 名称不同侧重
    'fedavg_vs_central': ['full', 'fedavg_full'],
    'norm': ['full', 'global_norm'],
    'augment': ['full', 'no_augment'],
    'lmmse': [],  # MATLAB端, 打印提示
}


@dataclass
class AblationResult:
    variant: str
    model_type: str
    training_type: str
    norm_type: str
    augment: bool
    best_epoch: int
    val_mse: float
    val_corr: float
    val_nz_mse: float
    train_time_sec: float
    model_params: int
    output_scale: float
    model_path: str


# ──────────────────────────────────────────────
# 集中式训练 (GAT)
# ──────────────────────────────────────────────

def run_centralized_gat(dataset, variant_name, args, device, output_dir):
    """Run centralized training for GAT model."""
    variant = ABLATION_VARIANTS[variant_name]
    save_dir = os.path.join(output_dir, variant_name)
    os.makedirs(save_dir, exist_ok=True)

    L = dataset.L if hasattr(dataset, 'L') else 100
    K = dataset.K if hasattr(dataset, 'K') else 20

    # Split data
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=custom_collate, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=custom_collate, num_workers=0)

    # Build model
    model = PowerGNN_GAT(
        L=L, K=K, hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, dropout=args.dropout, output_scale=1.0
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optimizer
    steps_per_epoch = max(1, len(train_dataset) // args.batch_size + 1)
    total_steps = args.epochs * steps_per_epoch
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_max / 10, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr_max, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100
    )

    # Training loop
    t0 = time.time()
    best_val = float('inf')
    best_state = None
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_mse, val_corr, val_nz_mse = evaluate(model, val_loader, device)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [{variant_name}] Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.6f} - Val MSE: {val_mse:.6f} - "
                  f"Val Corr: {val_corr:.4f}")

        if np.isfinite(val_mse) and val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = {'val_mse': val_mse, 'val_corr': val_corr, 'val_nz_mse': val_nz_mse}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"  [{variant_name}] Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0

    # Save best model
    model_path = os.path.join(save_dir, 'best_model.pt')
    torch.save({
        'model_state_dict': best_state if best_state else model.state_dict(),
        'model_type': 'gat',
        'norm_method': variant['norm_type'],
        'output_scale': float(model.output_scale),
        'epoch': best_epoch,
        'val_mse': best_val,
        'val_corr': best_metrics.get('val_corr', 0),
        'val_nz_mse': best_metrics.get('val_nz_mse', 0),
        'args': args,
    }, model_path)
    print(f"  [{variant_name}] Best model saved: {model_path} "
          f"(epoch={best_epoch+1}, corr={best_metrics.get('val_corr', 0):.4f})")

    return AblationResult(
        variant=variant_name,
        model_type='gat',
        training_type='centralized',
        norm_type=variant['norm_type'],
        augment=variant['augment'],
        best_epoch=best_epoch,
        val_mse=best_val,
        val_corr=best_metrics.get('val_corr', 0),
        val_nz_mse=best_metrics.get('val_nz_mse', 0),
        train_time_sec=train_time,
        model_params=n_params,
        output_scale=float(model.output_scale),
        model_path=model_path,
    )


# ──────────────────────────────────────────────
# 集中式训练 (MLP)
# ──────────────────────────────────────────────

def run_centralized_mlp(dataset, variant_name, args, device, output_dir):
    """Run centralized training for MLP model."""
    variant = ABLATION_VARIANTS[variant_name]
    save_dir = os.path.join(output_dir, variant_name)
    os.makedirs(save_dir, exist_ok=True)

    L = dataset.L if hasattr(dataset, 'L') else 100
    K = dataset.K if hasattr(dataset, 'K') else 20

    # Split data
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=custom_collate_mlp, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=custom_collate_mlp, num_workers=0)

    # Build model
    model = PowerGNN_MLP(
        L=L, K=K, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        dropout=args.dropout, output_scale=1.0
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optimizer
    steps_per_epoch = max(1, len(train_dataset) // args.batch_size + 1)
    total_steps = args.epochs * steps_per_epoch
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_max / 10, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr_max, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100
    )

    # Training loop
    t0 = time.time()
    best_val = float('inf')
    best_state = None
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch_mlp(model, train_loader, optimizer, scheduler, device)
        val_mse, val_corr, val_nz_mse = evaluate_mlp(model, val_loader, device)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [{variant_name}] Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.6f} - Val MSE: {val_mse:.6f} - "
                  f"Val Corr: {val_corr:.4f}")

        if np.isfinite(val_mse) and val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = {'val_mse': val_mse, 'val_corr': val_corr, 'val_nz_mse': val_nz_mse}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"  [{variant_name}] Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0

    # Save best model
    model_path = os.path.join(save_dir, 'best_model.pt')
    torch.save({
        'model_state_dict': best_state if best_state else model.state_dict(),
        'model_type': 'mlp',
        'norm_method': variant['norm_type'],
        'output_scale': float(model.output_scale),
        'epoch': best_epoch,
        'val_mse': best_val,
        'val_corr': best_metrics.get('val_corr', 0),
        'val_nz_mse': best_metrics.get('val_nz_mse', 0),
        'args': args,
    }, model_path)
    print(f"  [{variant_name}] Best model saved: {model_path} "
          f"(epoch={best_epoch+1}, corr={best_metrics.get('val_corr', 0):.4f})")

    return AblationResult(
        variant=variant_name,
        model_type='mlp',
        training_type='centralized',
        norm_type=variant['norm_type'],
        augment=variant['augment'],
        best_epoch=best_epoch,
        val_mse=best_val,
        val_corr=best_metrics.get('val_corr', 0),
        val_nz_mse=best_metrics.get('val_nz_mse', 0),
        train_time_sec=train_time,
        model_params=n_params,
        output_scale=float(model.output_scale),
        model_path=model_path,
    )


# ──────────────────────────────────────────────
# FedAvg 训练 (GAT)
# ──────────────────────────────────────────────

def run_fedavg_gat(dataset, variant_name, args, device, output_dir):
    """Run FedAvg training for GAT model."""
    variant = ABLATION_VARIANTS[variant_name]
    save_dir = os.path.join(output_dir, variant_name)
    os.makedirs(save_dir, exist_ok=True)

    L = dataset.L if hasattr(dataset, 'L') else 100
    K = dataset.K if hasattr(dataset, 'K') else 20

    # Split train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    num_clients = max(1, args.num_clients if args.num_clients > 0 else L)
    client_subsets = split_dataset_to_clients(
        train_dataset, num_clients=num_clients, seed=args.seed, split_by='random'
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=custom_collate, num_workers=0)

    # Build global model
    global_model = PowerGNN_GAT(
        L=L, K=K, hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, dropout=args.dropout, output_scale=1.0
    ).to(device)

    n_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)

    t0 = time.time()
    best_val = float('inf')
    best_round = 0
    best_state = None

    for r in range(args.fedavg_rounds):
        m = max(1, int(args.fedavg_frac * num_clients))
        selected = random.sample(range(num_clients), m)

        local_state_dicts = []
        local_weights = []

        for cid in selected:
            subset = client_subsets[cid]
            if len(subset) == 0:
                continue

            local_loader = DataLoader(
                subset, batch_size=args.batch_size, shuffle=True,
                collate_fn=custom_collate, num_workers=0
            )

            local_model = copy.deepcopy(global_model)
            loss, n_samples = local_train(
                local_model, local_loader, device=device,
                lr=args.fedavg_lr, local_epochs=args.fedavg_local_epochs,
                weight_decay=5e-4, scheduler='onecycle', use_amp=False
            )
            local_state_dicts.append(
                {k: v.detach().to('cpu') for k, v in local_model.state_dict().items()}
            )
            local_weights.append(n_samples)

        if len(local_state_dicts) == 0:
            continue

        agg_state = fedavg_aggregate(local_state_dicts, local_weights)
        global_model.load_state_dict(agg_state)

        val_mse, val_corr, val_nz_mse = evaluate(global_model, val_loader, device=device)

        if (r + 1) % 10 == 0 or r == 0:
            print(f"  [{variant_name}] Round {r+1}/{args.fedavg_rounds} - "
                  f"val_mse={val_mse:.6f} val_corr={val_corr:.4f}")

        if np.isfinite(val_mse) and val_mse < best_val:
            best_val = val_mse
            best_round = r
            best_state = {k: v.detach().to('cpu').clone() for k, v in global_model.state_dict().items()}

    train_time = time.time() - t0

    if best_state is not None:
        global_model.load_state_dict(best_state)

    # Final evaluation
    val_mse, val_corr, val_nz_mse = evaluate(global_model, val_loader, device=device)

    model_path = os.path.join(save_dir, 'best_model.pt')
    save_checkpoint(global_model, model_path, extra={
        'model_type': 'gat',
        'best_round': int(best_round + 1),
        'best_val_mse': float(best_val),
        'val_corr': float(val_corr),
        'val_nz_mse': float(val_nz_mse),
        'clients': int(num_clients),
        'frac': float(args.fedavg_frac),
    })
    print(f"  [{variant_name}] Best model saved: {model_path} "
          f"(round={best_round+1}, corr={val_corr:.4f})")

    return AblationResult(
        variant=variant_name,
        model_type='gat',
        training_type='fedavg',
        norm_type=variant['norm_type'],
        augment=variant['augment'],
        best_epoch=best_round,
        val_mse=best_val,
        val_corr=float(val_corr),
        val_nz_mse=float(val_nz_mse),
        train_time_sec=train_time,
        model_params=n_params,
        output_scale=1.0,
        model_path=model_path,
    )


# ──────────────────────────────────────────────
# FedAvg 训练 (MLP)
# ──────────────────────────────────────────────

def run_fedavg_mlp(dataset, variant_name, args, device, output_dir):
    """Run FedAvg training for MLP model."""
    variant = ABLATION_VARIANTS[variant_name]
    save_dir = os.path.join(output_dir, variant_name)
    os.makedirs(save_dir, exist_ok=True)

    L = dataset.L if hasattr(dataset, 'L') else 100
    K = dataset.K if hasattr(dataset, 'K') else 20

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    num_clients = max(1, args.num_clients if args.num_clients > 0 else L)
    client_subsets = split_dataset_to_clients(
        train_dataset, num_clients=num_clients, seed=args.seed, split_by='random'
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=custom_collate_mlp, num_workers=0)

    global_model = PowerGNN_MLP(
        L=L, K=K, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        dropout=args.dropout, output_scale=1.0
    ).to(device)

    n_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)

    t0 = time.time()
    best_val = float('inf')
    best_round = 0
    best_state = None

    for r in range(args.fedavg_rounds):
        m = max(1, int(args.fedavg_frac * num_clients))
        selected = random.sample(range(num_clients), m)

        local_state_dicts = []
        local_weights = []

        for cid in selected:
            subset = client_subsets[cid]
            if len(subset) == 0:
                continue

            local_loader = DataLoader(
                subset, batch_size=args.batch_size, shuffle=True,
                collate_fn=custom_collate_mlp, num_workers=0
            )

            local_model = copy.deepcopy(global_model)
            loss, n_samples = local_train(
                local_model, local_loader, device=device,
                lr=args.fedavg_lr, local_epochs=args.fedavg_local_epochs,
                weight_decay=5e-4, scheduler='onecycle', use_amp=False
            )
            local_state_dicts.append(
                {k: v.detach().to('cpu') for k, v in local_model.state_dict().items()}
            )
            local_weights.append(n_samples)

        if len(local_state_dicts) == 0:
            continue

        agg_state = fedavg_aggregate(local_state_dicts, local_weights)
        global_model.load_state_dict(agg_state)

        val_mse, val_corr, val_nz_mse = evaluate_mlp(global_model, val_loader, device=device)

        if (r + 1) % 10 == 0 or r == 0:
            print(f"  [{variant_name}] Round {r+1}/{args.fedavg_rounds} - "
                  f"val_mse={val_mse:.6f} val_corr={val_corr:.4f}")

        if np.isfinite(val_mse) and val_mse < best_val:
            best_val = val_mse
            best_round = r
            best_state = {k: v.detach().to('cpu').clone() for k, v in global_model.state_dict().items()}

    train_time = time.time() - t0

    if best_state is not None:
        global_model.load_state_dict(best_state)

    val_mse, val_corr, val_nz_mse = evaluate_mlp(global_model, val_loader, device=device)

    model_path = os.path.join(save_dir, 'best_model.pt')
    save_checkpoint(global_model, model_path, extra={
        'model_type': 'mlp',
        'best_round': int(best_round + 1),
        'best_val_mse': float(best_val),
        'val_corr': float(val_corr),
        'val_nz_mse': float(val_nz_mse),
        'clients': int(num_clients),
        'frac': float(args.fedavg_frac),
    })
    print(f"  [{variant_name}] Best model saved: {model_path} "
          f"(round={best_round+1}, corr={val_corr:.4f})")

    return AblationResult(
        variant=variant_name,
        model_type='mlp',
        training_type='fedavg',
        norm_type=variant['norm_type'],
        augment=variant['augment'],
        best_epoch=best_round,
        val_mse=best_val,
        val_corr=float(val_corr),
        val_nz_mse=float(val_nz_mse),
        train_time_sec=train_time,
        model_params=n_params,
        output_scale=1.0,
        model_path=model_path,
    )


# ──────────────────────────────────────────────
# 调度器
# ──────────────────────────────────────────────

def load_dataset(data_pattern, norm_type='per_sample', seed=42):
    """Load dataset with specified normalization."""
    data_files = sorted(glob.glob(data_pattern))
    if len(data_files) == 0:
        raise SystemExit(f"ERROR: No files matched pattern: {data_pattern}")

    DatasetClass = GNNDatasetGlobalNorm if norm_type == 'global' else GNNDataset

    if len(data_files) == 1:
        dataset = DatasetClass(data_files[0])
    else:
        all_datasets = [DatasetClass(f) for f in data_files]
        dataset = ConcatDataset(all_datasets)

    print(f"Loaded {len(dataset)} samples (norm={norm_type})")
    return dataset


def run_single_variant(variant_name, dataset, args, device, output_dir):
    """Dispatch to the appropriate training function."""
    variant = ABLATION_VARIANTS[variant_name]
    model_type = variant['model_type']
    training_type = variant['training_type']

    print(f"\n{'='*60}")
    print(f"  Variant: {variant_name}")
    print(f"  Desc:    {variant['description']}")
    print(f"  Model:   {model_type} | Training: {training_type}")
    print(f"{'='*60}")

    if model_type == 'gat' and training_type == 'centralized':
        return run_centralized_gat(dataset, variant_name, args, device, output_dir)
    elif model_type == 'mlp' and training_type == 'centralized':
        return run_centralized_mlp(dataset, variant_name, args, device, output_dir)
    elif model_type == 'gat' and training_type == 'fedavg':
        return run_fedavg_gat(dataset, variant_name, args, device, output_dir)
    elif model_type == 'mlp' and training_type == 'fedavg':
        return run_fedavg_mlp(dataset, variant_name, args, device, output_dir)
    else:
        raise ValueError(f"Unknown combination: model={model_type}, training={training_type}")


def generate_summary(results, output_dir):
    """Generate CSV and JSON summary files."""
    csv_path = os.path.join(output_dir, 'ablation_results.csv')
    json_path = os.path.join(output_dir, 'ablation_summary.json')

    # CSV
    fieldnames = list(asdict(results[0]).keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nResults CSV saved: {csv_path}")

    # JSON
    summary = {
        'results': [asdict(r) for r in results],
        'comparison': {},
    }

    # Compute pairwise comparisons for each ablation group
    result_map = {r.variant: asdict(r) for r in results}
    for group_name, variant_list in ABLATION_GROUPS.items():
        if len(variant_list) == 2:
            v1, v2 = variant_list
            if v1 in result_map and v2 in result_map:
                r1, r2 = result_map[v1], result_map[v2]
                corr_diff = r2['val_corr'] - r1['val_corr']
                mse_diff = r2['val_mse'] - r1['val_mse']
                summary['comparison'][group_name] = {
                    'baseline': v1,
                    'ablation': v2,
                    'corr_diff': round(corr_diff, 4),
                    'mse_diff': round(mse_diff, 6),
                    'baseline_corr': round(r1['val_corr'], 4),
                    'ablation_corr': round(r2['val_corr'], 4),
                }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary JSON saved: {json_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  ABLATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Variant':<22} {'Model':<5} {'Train':<12} {'Norm':<11} {'Aug':<4} "
          f"{'Corr':>7} {'MSE':>9} {'Params':>8} {'Time(s)':>8}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r.variant:<22} {r.model_type:<5} {r.training_type:<12} "
              f"{r.norm_type:<11} {str(r.augment):<4} "
              f"{r.val_corr:>7.4f} {r.val_mse:>9.6f} "
              f"{r.model_params:>8,} {r.train_time_sec:>8.1f}")

    # Pairwise comparisons
    print(f"\n{'='*80}")
    print(f"  PAIRWISE COMPARISONS")
    print(f"{'='*80}")
    for group_name, comp in summary['comparison'].items():
        arrow = "↑" if comp['corr_diff'] > 0 else "↓"
        print(f"  {group_name}: {comp['ablation']} vs {comp['baseline']} "
              f"→ Corr {arrow} {abs(comp['corr_diff']):.4f} "
              f"({comp['baseline_corr']:.4f} → {comp['ablation_corr']:.4f})")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ablation Study for GNN+FL Power Allocation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all ablation variants
  python train_gnn_ablation.py --ablation all --data "../data/gnn_training/*.mat"

  # Run single dimension
  python train_gnn_ablation.py --ablation gnn_vs_nognn --data "../data/gnn_training/*.mat"

  # Quick test (10 epochs)
  python train_gnn_ablation.py --ablation all --epochs 10 --fedavg_rounds 3

  # L-MMSE ablation (prints MATLAB instructions)
  python train_gnn_ablation.py --ablation lmmse
"""
    )

    # Ablation control
    parser.add_argument('--ablation', type=str, default='all',
                        choices=['all', 'gnn_vs_nognn', 'gat_vs_mlp', 'fedavg_vs_central',
                                 'norm', 'augment', 'lmmse'],
                        help='Ablation dimension to run (default: all)')
    parser.add_argument('--data', type=str, default='../data/gnn_training/*.mat',
                        help='Path/pattern to training data')
    parser.add_argument('--output_dir', type=str, default='../models/ablation',
                        help='Output directory for ablation results')

    # Shared hyperparams
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_max', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)

    # FedAvg hyperparams
    parser.add_argument('--fedavg_rounds', type=int, default=50)
    parser.add_argument('--fedavg_frac', type=float, default=0.2)
    parser.add_argument('--fedavg_local_epochs', type=int, default=1)
    parser.add_argument('--fedavg_lr', type=float, default=5e-4)
    parser.add_argument('--num_clients', type=int, default=0,
                        help='Number of clients (0 = auto = L from dataset)')

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Handle L-MMSE (MATLAB-only)
    if args.ablation == 'lmmse':
        print("""
╔══════════════════════════════════════════════════════════════════╗
║  消融维度 6: L-MMSE 重构 (全局协方差 vs 本地协方差)             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  此消融维度为 MATLAB 端实验, 需手动操作:                         ║
║                                                                  ║
║  1. 在 Combined_Downlink_Sim.m 中:                               ║
║     - 将 functionPrecoding_LMMSE 替换为                          ║
║       functionPrecoding_LMMSE_global                              ║
║     - 或在 PC 注册表中添加 LMMSE_G 条目                          ║
║                                                                  ║
║  2. 运行仿真:                                                    ║
║     cd('CF_downlink_sim'); run                                   ║
║                                                                  ║
║  3. 对比结果:                                                    ║
║     - functionPrecoding_LMMSE.m  = 本地协方差 (分布式, 本项目方案)║
║     - functionPrecoding_LMMSE_global.m = 全局协方差 (传统方案)    ║
║     - 观察 ESR 差异即为信息损失量                                ║
║                                                                  ║
║  全局协方差版本已保存在:                                          ║
║    precoding/functionPrecoding_LMMSE_global.m                    ║
╚══════════════════════════════════════════════════════════════════╝
""")
        return

    # Determine which variants to run
    if args.ablation == 'all':
        variants_to_run = list(ABLATION_VARIANTS.keys())
    elif args.ablation in ABLATION_GROUPS:
        variants_to_run = ABLATION_GROUPS[args.ablation]
    else:
        variants_to_run = [args.ablation]

    if not variants_to_run:
        print(f"No variants to run for --ablation {args.ablation}")
        return

    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY")
    print(f"  Variants: {variants_to_run}")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets (potentially different norm types)
    norm_types_needed = set()
    for v in variants_to_run:
        norm_types_needed.add(ABLATION_VARIANTS[v]['norm_type'])

    datasets = {}
    for nt in norm_types_needed:
        print(f"\nLoading dataset with norm_type='{nt}'...")
        datasets[nt] = load_dataset(args.data, norm_type=nt, seed=args.seed)

    # Run each variant
    results = []
    for variant_name in variants_to_run:
        variant = ABLATION_VARIANTS[variant_name]
        dataset = datasets[variant['norm_type']]

        # For 'no_augment' variant, we note that augmentation is baked into
        # the MATLAB-side data generation. True ablation requires regenerating
        # data without augmentation. We proceed with the same data but
        # document this limitation.
        if not variant['augment']:
            print(f"  [NOTE] 'no_augment' uses same data (augmentation is in MATLAB export).")
            print(f"  For true ablation, regenerate data with augmentation disabled.")

        result = run_single_variant(variant_name, dataset, args, device, args.output_dir)
        results.append(result)

    # Generate summary
    generate_summary(results, args.output_dir)

    print(f"\n{'='*60}")
    print(f"  All ablation experiments completed!")
    print(f"  Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
