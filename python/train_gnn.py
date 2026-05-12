#!/usr/bin/env python3
"""
GNN 功率分配模型训练脚本
基于 PyTorch Geometric (PyG) 实现，使用 GATConv 层

Phase 2: GNN 模型训练
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

from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

def custom_collate(batch):
    """
    自定义 collate 函数 - 支持 GAT 批训练

    对于 GNN，我们需要为每个样本构建独立的 Data 对象，
    然后用 Batch 打包。edge_index 必须相对于每个样本的节点编号。
    """
    # 分离 AP 特征、UE 特征、D掩码、标签等
    x_ap_list = []
    x_ue_list = []
    edge_index_ap2ue_list = []  # AP -> UE 边
    edge_index_ue2ap_list = []  # UE -> AP 边
    D_mask_list = []
    rho_is_nonzero_list = []
    y_share_list = []
    y_list = []
    esr_list = []
    snr_list = []

    L = batch[0]['x_ap'].shape[0]
    K = batch[0]['x_ue'].shape[0]
    num_ap_nodes = L
    num_ue_nodes = K

    for item in batch:
        # AP 节点特征: (L, K) -> (L, K)
        x_ap_list.append(item['x_ap'])

        # UE 节点特征: (K, L+1) -> (K, L+1)
        x_ue_list.append(item['x_ue'])

        # D mask: (L, K)
        D_mask_list.append(item['D_mask'])

        # rho_is_nonzero: (L, K)
        rho_is_nonzero_list.append(item['rho_is_nonzero'])

        # 标签: (L, K)
        y_share_list.append(item['y_share'])
        y_list.append(item['y'])

        # 辅助标签
        esr_list.append(item['esr'])
        snr_list.append(item['snr'])

        # 为每个样本构建边索引
        # AP 节点: 0 to L-1
        # UE 节点: L to L+K-1
        D = item['D_mask'].numpy()  # (L, K)

        # AP -> UE 边 (正向)
        ap_src, ue_dst = np.where(D == 1)
        edge_ap2ue = np.array([ap_src, ue_dst + num_ap_nodes])  # (2, num_edges)
        edge_index_ap2ue_list.append(torch.LongTensor(edge_ap2ue))

        # UE -> AP 边 (反向)
        edge_ue2ap = np.array([ue_dst + num_ap_nodes, ap_src])  # (2, num_edges)
        edge_index_ue2ap_list.append(torch.LongTensor(edge_ue2ap))

    # 构建 Data 对象列表
    data_list = []
    for i in range(len(batch)):
        # 拼接 AP 和 UE 节点特征
        # x_ap: (L, K), x_ue: (K, L+1)
        # 合并后: (L+K, max(K, L+1)) - 用0填充
        x_ap = x_ap_list[i]  # (L, K)
        x_ue = x_ue_list[i]  # (K, L+1)

        # AP 节点特征维度 K, UE 节点特征维度 L+1
        # 拼接: (L+K, max(K, L+1))
        max_feat_dim = max(x_ap.shape[1], x_ue.shape[1])
        x_combined = torch.zeros(L + K, max_feat_dim)
        x_combined[:L, :x_ap.shape[1]] = x_ap
        x_combined[L:, :x_ue.shape[1]] = x_ue

        # 合并边索引 (AP->UE 和 UE->AP 都包含)
        edge_index = torch.cat([edge_index_ap2ue_list[i], edge_index_ue2ap_list[i]], dim=1)

        data = Data(
            x=x_combined,
            edge_index=edge_index,
            D_mask=D_mask_list[i],
            rho_is_nonzero=rho_is_nonzero_list[i],
            y_share=y_share_list[i],
            y=y_list[i],
            esr=esr_list[i],
            snr=snr_list[i],
            num_ap_nodes=L,
            num_ue_nodes=K
        )
        data_list.append(data)

    # 用 PyG Batch 打包
    batch_data = Batch.from_data_list(data_list)

    return batch_data


class PowerGNN_MLP(nn.Module):
    """
    MLP baseline for ablation — no graph structure, no attention, no message passing.

    Flattens all node features into a single vector, processes through MLP,
    then reshapes back to (batch, L, K) predictions. Parameter count is
    controlled to be comparable to PowerGNN_GAT.
    """

    def __init__(self, L=100, K=20, hidden_dim=128, num_layers=3,
                 dropout=0.15, output_scale=None):
        super().__init__()
        self.L = L
        self.K = K
        self.hidden_dim = hidden_dim

        self.ap_input_dim = K + 4
        self.ue_input_dim = L + 4
        self.input_dim = L * self.ap_input_dim + K * self.ue_input_dim

        # 3-layer MLP with residual connections
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        in_dim = self.input_dim
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        # AP predictor head (with skip connection from raw input)
        self.ap_predictor = nn.Sequential(
            nn.Linear(hidden_dim + self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, L * K)
        )

        self.output_scale = output_scale if output_scale is not None else 1.5
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier uniform initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        # batch is a dict with keys: x_ap (L,K), x_ue (K,L+1), etc.
        # When used with custom_collate_mlp, batch is a dict of batched tensors
        x_ap = batch['x_ap']    # (batch, L, K)
        x_ue = batch['x_ue']    # (batch, K, L+1)

        batch_size = x_ap.size(0)

        # Flatten and concatenate
        x_flat = torch.cat([
            x_ap.view(batch_size, -1),
            x_ue.view(batch_size, -1)
        ], dim=-1)  # (batch, 2*L*K + K)

        x_raw = x_flat.clone()

        # MLP layers with residual + LayerNorm
        x_curr = self.layers[0](x_flat)
        x_curr = self.layer_norms[0](x_curr)
        for i in range(1, len(self.layers)):
            x_next = self.layers[i](x_curr)
            x_next = self.dropout(x_next)
            x_curr = x_next + x_curr  # residual
            x_curr = self.layer_norms[i](x_curr)

        # Predictor head with skip connection
        pred_input = torch.cat([x_curr, x_raw], dim=-1)
        rho_flat = self.ap_predictor(pred_input)  # (batch, K)

        # tanh + reshape to (batch, L, K)
        rho_flat = torch.tanh(rho_flat) * self.output_scale
        rho = rho_flat.view(batch_size, self.L, self.K)

        return rho


def custom_collate_mlp(batch):
    """
    Custom collate function for MLP model — returns flat tensors instead of PyG Batch.

    No edge_index, no graph structure — pure MLP baseline.
    """
    x_ap_list = []
    x_ue_list = []
    D_mask_list = []
    rho_is_nonzero_list = []
    y_share_list = []
    y_list = []
    esr_list = []
    snr_list = []

    for item in batch:
        x_ap_list.append(item['x_ap'])               # (L, K)
        x_ue_list.append(item['x_ue'])               # (K, L+1)
        D_mask_list.append(item['D_mask'])           # (L, K)
        rho_is_nonzero_list.append(item['rho_is_nonzero'])  # (L, K)
        y_share_list.append(item['y_share'])         # (L, K)
        y_list.append(item['y'])                     # (L, K)
        esr_list.append(item['esr'])
        snr_list.append(item['snr'])

    return {
        'x_ap': torch.stack(x_ap_list),             # (batch, L, K)
        'x_ue': torch.stack(x_ue_list),             # (batch, K, L+1)
        'D_mask': torch.stack(D_mask_list),         # (batch, L, K)
        'rho_is_nonzero': torch.stack(rho_is_nonzero_list),  # (batch, L, K)
        'y_share': torch.stack(y_share_list),       # (batch, L, K)
        'y': torch.stack(y_list),                   # (batch, L, K)
        'esr': torch.stack(esr_list),               # (batch, 1)
        'snr': torch.stack(snr_list),               # (batch, 1)
    }


def train_epoch_mlp(model, dataloader, optimizer, scheduler, device):
    """Training one epoch for MLP model (non-graph batch format)"""
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        optimizer.zero_grad()

        rho_pred = model(batch)
        y_batched = batch['y']                  # (batch, L, K)
        D_batched = batch['D_mask']            # (batch, L, K)
        nz_batched = batch['rho_is_nonzero']   # (batch, L, K)

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


def evaluate_mlp(model, dataloader, device):
    """Evaluation for MLP model (non-graph batch format)"""
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

            rho_pred = model(batch)
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


class PowerGNN_GAT(nn.Module):
    """
    GNN 功率分配模型 v3 — 全面优化版

    改进点:
    1. 动态 output_scale（根据实际标签范围自动设置）
    2. AP 预测头带残差连接（保留输入特征的局部信息）
    3. Xavier 初始化（更稳定的训练起点）
    4. SNR 条件嵌入（帮助模型区分不同工作点）
    """

    def __init__(self, L=100, K=20, hidden_dim=128, num_heads=4, num_layers=3,
                 dropout=0.15, edge_dim=1, output_scale=None):
        super().__init__()
        self.L = L
        self.K = K
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 节点特征维度
        self.ap_input_dim = K + 4   # AP 节点: masked sqrt(gain) + SNR/CSI/degree/gain context
        self.ue_input_dim = L + 4   # UE 节点: masked sqrt(gain) + SNR/CSI/degree/gain context

        # 输入特征标准化
        self.input_norm_ap = nn.LayerNorm(self.ap_input_dim)
        self.input_norm_ue = nn.LayerNorm(self.ue_input_dim)

        # 节点嵌入层
        self.node_embedding_ap = nn.Linear(self.ap_input_dim, hidden_dim)
        self.node_embedding_ue = nn.Linear(self.ue_input_dim, hidden_dim)

        # GATConv 层
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            out_dim = hidden_dim // num_heads
            self.gat_layers.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    bias=True
                )
            )

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # ── AP 节点级预测头（带残差连接）──
        self.ap_predictor = nn.Sequential(
            nn.Linear(hidden_dim + self.ap_input_dim, hidden_dim),  # 拼接隐藏态+输入特征
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, K)   # 每个AP输出K个值
        )

        # 动态输出缩放（默认匹配 [-1,+1] 标签）
        self.output_scale = output_scale if output_scale is not None else 1.5

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier 均匀初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        batch_vec = batch.batch
        batch_size = batch.num_graphs

        num_ap = self.L
        num_ue = self.K
        nodes_per_graph = num_ap + num_ue

        # ── 1. 判断 AP / UE 节点位置 ──
        node_pos = torch.arange(x.size(0), device=x.device) % nodes_per_graph
        is_ap_mask = node_pos < num_ap

        # ── 2. 保存原始输入特征（用于残差连接）──
        x_ap_raw = x[is_ap_mask, :self.ap_input_dim].clone()   # (batch*L, K)

        # ── 3. LayerNorm + 分类型嵌入 ──
        x_ue_raw = x[~is_ap_mask, :self.ue_input_dim]
        ap_emb = self.node_embedding_ap(self.input_norm_ap(x_ap_raw))
        ue_emb = self.node_embedding_ue(self.input_norm_ue(x_ue_raw))

        x_emb = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=ap_emb.dtype)
        x_emb[is_ap_mask] = ap_emb
        x_emb[~is_ap_mask] = ue_emb.to(dtype=x_emb.dtype)

        # ── 4. 多层 GATConv 消息传递 ──
        edge_attr = torch.ones(edge_index.size(1), 1, device=x.device, dtype=x_emb.dtype)
        x_curr = x_emb
        for i, gat_conv in enumerate(self.gat_layers):
            x_next = gat_conv(x_curr, edge_index, edge_attr=edge_attr)
            x_next = self.dropout(x_next)
            x_curr = x_next + x_curr                       # 残差
            x_curr = self.layer_norms[i](x_curr)           # LayerNorm

        # ── 5. 提取 AP 节点隐藏状态 + 拼接原始输入 → 预测 ──
        x_ap_hidden = x_curr[is_ap_mask]                   # (batch_size * L, hidden)
        # 残差：拼接 GAT 输出和原始输入特征，让模型同时看到全局消息和局部信息
        ap_pred_input = torch.cat([x_ap_hidden, x_ap_raw], dim=-1)  # (batch*L, hidden+K)
        rho_flat = self.ap_predictor(ap_pred_input)         # (batch*L, K)

        # ── 6. tanh 约束输出范围 + reshape ──
        rho_flat = torch.tanh(rho_flat) * self.output_scale  # (batch*L, K)
        rho = rho_flat.view(batch_size, num_ap, num_ue)       # (batch, L, K)

        return rho


def compute_loss(rho_pred, rho_true, D_mask, rho_is_nonzero, target_share=None):
    """
    只对 rho_raw > 0 的位置计算 loss（方案A：非零 rho mask）

    大部分边的 rho=0（DCC 断开连接或远端 UE），如果全部算 loss，
    模型会学到"输出常数 -1 就行"，无法学到功率分配规律。
    现在只对 rho_is_nonzero=1 的位置算 Huber loss。

    Args:
        rho_pred: (batch, L, K) 预测的功率分配
        rho_true: (batch, L, K) 真实的功率分配（归一化后）
        D_mask: (batch, L, K) D 矩阵掩码
        rho_is_nonzero: (batch, L, K) 原始 rho > 0 的标记
    Returns:
        loss: 标量损失
    """
    # 只在 rho_raw > 0 且 D=1 的位置算 loss
    nonzero_mask = (rho_is_nonzero > 0.5) & (D_mask > 0.5)  # (batch, L, K) bool
    n_valid = nonzero_mask.sum()

    if n_valid == 0:
        # 退回全 mask 模式，避免空梯度
        nonzero_mask = D_mask > 0.5
        n_valid = nonzero_mask.sum()

    diff = (rho_pred - rho_true) * nonzero_mask.float()
    diff_valid = diff[nonzero_mask]

    huber_loss = torch.nn.functional.huber_loss(
        diff_valid,
        torch.zeros_like(diff_valid),
        delta=0.5,
        reduction='mean'
    )

    if target_share is None:
        return huber_loss

    pred_weights = torch.relu((rho_pred + 1.0) / 2.0) * D_mask.float()
    pred_share = pred_weights / (pred_weights.sum(dim=2, keepdim=True) + 1e-8)
    share_mask = D_mask > 0.5
    share_loss = torch.nn.functional.mse_loss(
        pred_share[share_mask],
        target_share[share_mask],
        reduction='mean'
    )

    return huber_loss + 2.0 * share_loss


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()

        rho_pred = model(batch)
        # rho_true 需要按照 batch 中每个图的 D_mask 对应的样本展开
        y_batched = batch.y.view(-1, model.L, model.K)  # (batch, L, K)
        D_batched = batch.D_mask.view(-1, model.L, model.K)  # (batch, L, K)
        nz_batched = batch.rho_is_nonzero.view(-1, model.L, model.K)  # (batch, L, K)
        share_batched = batch.y_share.view(-1, model.L, model.K)

        loss = compute_loss(rho_pred, y_batched, D_batched, nz_batched, share_batched)

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf loss, skipping")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # OneCycleLR: 每 batch 调用一次

        total_loss += loss.item() * rho_pred.shape[0]
        n_samples += rho_pred.shape[0]

    return total_loss / max(n_samples, 1)


def evaluate(model, dataloader, device):
    """评估模型 — 只在非零 rho 位置算 MSE 和 Corr"""
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
            share_batched = batch.y_share.view(-1, model.L, model.K)

            mse = compute_loss(rho_pred, y_batched, D_batched, nz_batched, share_batched).item()
            total_mse += mse * rho_pred.shape[0]

            nonzero_mask = (nz_batched > 0.5) & (D_batched > 0.5)
            all_rho_pred.append(rho_pred.cpu())
            all_rho_true.append(y_batched.cpu())
            all_nonzero_mask.append(nonzero_mask.cpu())

            n_samples += rho_pred.shape[0]

    avg_mse = total_mse / max(n_samples, 1)

    # 只在非零 rho 位置算相关性
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

    # 额外统计：非零位置的 MSE
    if valid_mask.sum() > 0:
        nonzero_mse = float(np.mean((pred_nz[valid_mask] - true_nz[valid_mask]) ** 2))
    else:
        nonzero_mse = 0.0

    return avg_mse, correlation, nonzero_mse


def main():
    from dataset import GNNDataset

    parser = argparse.ArgumentParser(description='Train GNN Power Allocation Model with GATConv')
    parser.add_argument('--data', type=str,
                        default='../data/gnn_training/*.mat',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr_max', type=float, default=1e-3, help='Max learning rate (OneCycleLR)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GAT layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--output_dir', type=str, default='../models', help='Output directory')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (in epochs)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyG version check: torch_geometric available")

    print(f"\nLoading data from {args.data}...")

    # 支持通配符: 自动展开 *.mat 为多文件列表
    data_files = sorted(glob.glob(args.data))
    if len(data_files) == 0:
        print(f"ERROR: No files matched pattern: {args.data}")
        sys.exit(1)
    print(f"Found {len(data_files)} data file(s):")
    for f in data_files:
        print(f"  - {f}")

    # 加载并合并多个数据集
    if len(data_files) == 1:
        dataset = GNNDataset(data_files[0])
    else:
        # 多文件合并: 依次加载，归一化由 __getitem__ 动态完成
        all_datasets = [GNNDataset(f) for f in data_files]
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(all_datasets)
        L, K = all_datasets[0].L, all_datasets[0].K

    print(f"Total dataset size: {len(dataset)}")

    # 检查数据
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
            y_nz_mean = float(y[nz_mask].mean())
        else:
            y_nz_min = y_nz_max = y_nz_mean = 0.0
        print(f"  Sample {i}: y range=[{y.min():.6f}, {y.max():.6f}], "
              f"y mean={y.mean():.6f}, D sum={D.sum():.0f}, "
              f"nonzero rho: {n_nz}/{n_total}, y_nz=[{y_nz_min:.3f}, {y_nz_max:.3f}], "
              f"mean={y_nz_mean:.3f}, share_sum_mean={sample['y_share'].sum(dim=1).mean():.3f}")

    # 划分数据集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"\nTrain size: {train_size}, Val size: {val_size}")

    # 创建数据加载器
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

    # 获取 L, K（样本内归一化不再需要 p1/p99 参数）
    if hasattr(dataset, 'L'):
        L, K = dataset.L, dataset.K
    else:
        L, K = all_datasets[0].L, all_datasets[0].K

    print(f"\nCreating GAT-GNN model v3: L={L}, K={K}")
    print(f"  hidden_dim={args.hidden_dim}, heads={args.num_heads}, "
          f"layers={args.num_layers}, drop={args.dropout}")
    print(f"  Label norm: per-sample (signed-log + nonzero min-max to [-1,1])")

    model = PowerGNN_GAT(
        L=L, K=K,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_scale=1.0   # 样本内归一化后标签严格在 [-1, 1]
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # ── OneCycleLR: 自动 warmup + cosine decay ──
    steps_per_epoch = len(dataset) // args.batch_size + 1
    total_steps = args.epochs * steps_per_epoch
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_max / 10, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr_max,
        total_steps=total_steps,
        pct_start=0.1,       # 前 10% 做 warmup
        anneal_strategy='cos',
        div_factor=10,       # 初始 LR = max_lr/10
        final_div_factor=100  # 最终 LR = max_lr/100
    )
    print(f"  Optimizer: AdamW, OneCycleLR (max_lr={args.lr_max}, "
          f"warmup={int(total_steps*0.1)} steps, total={total_steps} steps)")

    print("\nStarting training...")
    best_val_loss = float('inf')
    best_state_dict = None
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0
    save_interval = 50   # 每 N 轮存一次盘

    os.makedirs(args.output_dir, exist_ok=True)

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

        # 追踪最优模型（仅内存，不写盘）
        if val_mse < best_val_loss and not np.isnan(val_mse):
            best_val_loss = val_mse
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = {'val_mse': val_mse, 'val_corr': val_corr, 'val_nz_mse': val_nz_mse}
            patience_counter = 0
        else:
            patience_counter += 1

        # 每 N 轮存盘一次
        if (epoch + 1) % save_interval == 0:
            model_path = os.path.join(args.output_dir, 'best_gat_gnn_power.pt')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_state_dict,
                'model_type': 'gat',
                'feature_schema': 'masked_gain_snr_csi_degree_sumgain_v2',
                'val_mse': best_val_loss,
                'val_corr': best_metrics.get('val_corr', 0),
                'norm_method': 'per_sample_signedlog_minmax',
                'output_scale': model.output_scale,
                'args': args
            }, model_path)
            print(f"  -> Checkpoint saved (best from epoch {best_epoch+1}, "
                  f"mse={best_val_loss:.4f}, corr={best_metrics.get('val_corr',0):.4f})")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.4f} (epoch {best_epoch+1})")
    print(f"Best val Corr: {best_metrics.get('val_corr', 0):.4f}, "
          f"Best val NZ_MSE: {best_metrics.get('val_nz_mse', 0):.4f}")

    # 最终保存最优模型
    model_path = os.path.join(args.output_dir, 'best_gat_gnn_power.pt')
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_state_dict,
        'model_type': 'gat',
        'feature_schema': 'masked_gain_snr_csi_degree_sumgain_v2',
        'val_mse': best_val_loss,
        'val_corr': best_metrics.get('val_corr', 0),
        'norm_method': 'per_sample_signedlog_minmax',
        'output_scale': model.output_scale,
        'args': args
    }, model_path)
    print(f"Best model saved to {model_path}")

    # 保存最终模型
    final_path = os.path.join(args.output_dir, 'final_gat_gnn_power.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'gat',
        'feature_schema': 'masked_gain_snr_csi_degree_sumgain_v2',
        'norm_method': 'per_sample_signedlog_minmax',
        'output_scale': model.output_scale,
        'args': args
    }, final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
