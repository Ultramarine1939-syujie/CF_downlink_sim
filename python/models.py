#!/usr/bin/env python3
"""
SP-MDGNN (Sparse Multi-Dimensional Graph Neural Network) 模型

基于论文 "Joint Power Control and Precoding for Cell-Free Massive MIMO Systems
With Sparse Multi-Dimensional Graph Neural Networks" (arXiv:2507.01876)

核心特性:
1. 稀疏邻接矩阵: 可学习的 W 参数 + sigmoid 激活 + 阈值 τ
2. 多维消息传递: 4 种聚合操作 (P_1, P_2, P_3, P_4)
3. 联合功率分配与预编码输出头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAdjacency(nn.Module):
    """
    可学习稀疏邻接矩阵

    通过可学习参数 W 和 sigmoid 激活生成邻接矩阵 A，
    然后应用阈值 τ 生成稀疏掩码 M = (A > τ)
    """

    def __init__(self, L, K, tau=0.5):
        super().__init__()
        self.L = L
        self.K = K
        self.tau = tau

        self.W = nn.Parameter(torch.randn(L, K) * 0.01)

    def forward(self, return_sparse_mask=True):
        A = torch.sigmoid(self.W)
        if return_sparse_mask:
            M = (A > self.tau).float()
            return A, M
        return A

    def get_sparse_connections(self):
        A = torch.sigmoid(self.W)
        M = (A > self.tau).float()
        return M


class MDGNNCell(nn.Module):
    """
    多维 GNN 单元 (MDGNN Cell)

    实现论文公式 (10):
    X^{(l)} = φ(Σ_{i=1}^{4} P_i^{(l)} · (A ⊙ X^{(l-1)}))

    使用 4 种不同的聚合操作来捕获多维关系
    """

    def __init__(self, in_dim, out_dim, num_aggregations=4):
        super().__init__()
        self.num_aggregations = num_aggregations

        self.proj = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for _ in range(num_aggregations)
        ])

        self.alpha = nn.Parameter(torch.ones(num_aggregations) / num_aggregations)

    def forward(self, x, adj_mask):
        x_expanded = x.unsqueeze(1)
        outputs = []

        for i in range(self.num_aggregations):
            masked_x = x * adj_mask.unsqueeze(-1)
            aggregated = torch.matmul(masked_x, self.proj[i].weight.T)
            aggregated = aggregated + self.proj[i].bias
            outputs.append(aggregated)

        outputs = torch.stack(outputs, dim=0)
        weights = F.softmax(self.alpha, dim=0)
        weighted_outputs = (outputs * weights.view(-1, 1, 1, 1)).sum(dim=0)

        return F.gelu(weighted_outputs)


class SP_MDGNN(nn.Module):
    """
    SP-MDGNN: Sparse Multi-Dimensional Graph Neural Network

    用于 Cell-Free Massive MIMO 的联合功率分配与预编码优化

    架构:
    1. 输入嵌入层
    2. 稀疏邻接矩阵 (可学习)
    3. L 层 MDGNN 消息传递
    4. 联合输出头 (功率 + 预编码)
    """

    def __init__(
        self,
        L=100,
        K=20,
        N=1,
        hidden_dim=128,
        num_layers=3,
        tau=0.5,
        dropout=0.1,
        output_scale=1.5
    ):
        super().__init__()
        self.L = L
        self.K = K
        self.N = N
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_dim_ap = K
        self.input_dim_ue = L + 1

        self.sparse_adj = SparseAdjacency(L, K, tau=tau)

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim_ap, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.input_proj_ue = nn.Sequential(
            nn.Linear(self.input_dim_ue, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.mdgnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mdgnn_layers.append(
                MDGNNCell(hidden_dim, hidden_dim, num_aggregations=4)
            )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.power_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.precoding_real_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, N)
        )

        self.precoding_imag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, N)
        )

        self.output_scale = output_scale
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        x_ap = batch['x_ap']
        x_ue = batch['x_ue']
        D_mask = batch['D_mask']
        batch_size = x_ap.size(0)

        _, sparse_mask = self.sparse_adj()
        adj_mask = (D_mask * sparse_mask.to(D_mask.device)).clamp(0, 1)

        x_ap_emb = self.input_proj(x_ap)
        x_ue_emb = self.input_proj_ue(x_ue)

        x = torch.cat([x_ap_emb, x_ue_emb], dim=1)

        for i, mdgnn_cell in enumerate(self.mdgnn_layers):
            x_new = mdgnn_cell(x, adj_mask)
            x_new = self.dropout(x_new)
            x = x_new + x
            x = self.layer_norms[i](x)

        x_ap_out = x[:, :self.L, :]

        power = self.power_head(x_ap_out)
        power = torch.tanh(power) * self.output_scale

        precoding_real = self.precoding_real_head(x_ap_out)
        precoding_imag = self.precoding_imag_head(x_ap_out)

        power = power.squeeze(-1)

        return {
            'power': power,
            'precoding_real': precoding_real,
            'precoding_imag': precoding_imag
        }


class GLP_GNN_Lite(nn.Module):
    """
    GLP-GNN Lite: 简化版联合功率与预编码网络

    与 SP-MDGNN 相比，减少了 MDGNN 的聚合操作数量，
    使用更紧凑的消息传递机制
    """

    def __init__(
        self,
        L=100,
        K=20,
        hidden_dim=128,
        num_layers=3,
        tau=0.5,
        dropout=0.1,
        output_scale=1.5
    ):
        super().__init__()
        self.L = L
        self.K = K
        self.hidden_dim = hidden_dim

        self.sparse_adj = SparseAdjacency(L, K, tau=tau)

        self.input_proj = nn.Sequential(
            nn.Linear(K, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.message_passing = nn.ModuleList()
        for _ in range(num_layers):
            self.message_passing.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU()
                )
            )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.output_scale = output_scale
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def message_pass(self, x, adj_mask):
        batch_size, num_nodes, hidden = x.shape
        x_expanded = x.unsqueeze(2)
        x_t = x.unsqueeze(1)

        neighbor_agg = (x_t * adj_mask.unsqueeze(-1)).sum(dim=2, keepdim=True)
        neighbor_count = adj_mask.sum(dim=2, keepdim=True).clamp(min=1)
        neighbor_mean = neighbor_agg / neighbor_count

        combined = torch.cat([x, neighbor_mean.squeeze(2)], dim=-1)

        return combined

    def forward(self, batch):
        x_ap = batch['x_ap']
        D_mask = batch['D_mask']
        batch_size = x_ap.size(0)

        _, sparse_mask = self.sparse_adj()
        adj_mask = (D_mask * sparse_mask.to(D_mask.device)).clamp(0, 1)

        x = self.input_proj(x_ap)

        for layer in self.message_passing:
            msg = self.message_pass(x, adj_mask)
            x_new = layer(msg)
            x = x_new + x
            x = self.dropout(x)

        power = self.output_proj(x)
        power = torch.tanh(power) * self.output_scale
        power = power.squeeze(-1)

        return {
            'power': power,
            'precoding_real': None,
            'precoding_imag': None
        }
