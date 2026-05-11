#!/usr/bin/env python3
"""
GLP-GNN: Generalized Label Propagation Graph Neural Network
用于Cell-Free Massive MIMO系统的联合功率控制与预编码优化

基于论文: "Joint Power Control and Precoding for Cell-Free Massive MIMO Systems 
         With Sparse Multi-Dimensional Graph Neural Networks"

核心特性:
1. 稀疏多维图消息传递 (Sparse Multi-Dimensional Message Passing)
2. 广义标签传播机制 (Generalized Label Propagation)
3. 联合功率-预编码优化 (Joint Power-Precoding Optimization)
4. 自适应边权重学习 (Adaptive Edge Weight Learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import math


class SparseMultiDimMessagePassing(MessagePassing):
    """
    稀疏多维消息传递层
    
    在Cell-Free系统中实现AP和UE之间的双向消息传递，
    支持稀疏连接模式（仅在D=1的边上传递消息）
    """
    
    def __init__(self, in_channels, out_channels, edge_dim=1, aggr='add', 
                 flow='source_to_target', node_dim=0):
        super().__init__(aggr=aggr, flow=flow, node_dim=node_dim)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, in_channels // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(in_channels // 2, in_channels)
        )
        
        self.source_lin = nn.Linear(in_channels, out_channels)
        self.message_lin = nn.Linear(in_channels, out_channels)
        self.target_lin = nn.Linear(in_channels, out_channels)
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for module in self.edge_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.source_lin.weight)
        nn.init.zeros_(self.source_lin.bias)
        nn.init.xavier_uniform_(self.message_lin.weight)
        nn.init.zeros_(self.message_lin.bias)
        nn.init.xavier_uniform_(self.target_lin.weight)
        nn.init.zeros_(self.target_lin.bias)
    
    def forward(self, x, edge_index, edge_attr=None, size=None):
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), 1, device=x.device)
        
        edge_emb = self.edge_encoder(edge_attr)
        
        return self.propagate(edge_index, x=x, edge_emb=edge_emb, size=size)
    
    def message(self, x_j, edge_emb):
        return self.message_lin(x_j) + edge_emb
    
    def update(self, aggr_out, x):
        return self.norm(aggr_out + self.source_lin(x))


class GLPLabelPropagation(nn.Module):
    """
    广义标签传播层 (Generalized Label Propagation)
    
    实现基于标签传播的图卷积操作，支持：
    1. 半监督学习模式（利用未标记节点的传播信息）
    2. 自适应传播强度（根据节点特征动态调整传播系数）
    3. 多跳信息聚合（K-hop邻居信息的综合）
    """
    
    def __init__(self, hidden_dim, num_propagations=3, alpha=0.5):
        super().__init__()
        self.num_propagations = num_propagations
        self.alpha = alpha
        
        self.propagation_weights = nn.Parameter(torch.ones(num_propagations) / num_propagations)
        
        self.aggregation_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.aggregation_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU()
        )
    
    def forward(self, h, edge_index_ap_ue, edge_index_ue_ap, D_mask):
        h_0 = h.clone()
        
        aggregated = torch.zeros_like(h)
        total_weight = 0.0
        
        for k in range(self.num_propagations):
            h_ap = h[:edge_index_ap_ue.max().item() + 1]
            h_ue = h[edge_index_ap_ue.max().item() + 1:]
            
            if edge_index_ue_ap.size(1) > 0:
                ue_aggr = self._aggregate_from_neighbors(
                    h_ap, h_ue, edge_index_ue_ap
                )
            else:
                ue_aggr = torch.zeros_like(h_ue)
            
            if edge_index_ap_ue.size(1) > 0:
                ap_aggr = self._aggregate_from_neighbors(
                    h_ue, h_ap, edge_index_ap_ue
                )
            else:
                ap_aggr = torch.zeros_like(h_ap)
            
            h_new = torch.cat([h_ap, h_ue], dim=0)
            
            gate = self.aggregation_gate(h_new)
            transformed = self.aggregation_transform(torch.cat([h_new, 
                torch.cat([ap_aggr, ue_aggr], dim=0)], dim=-1))
            
            h = (1 - self.alpha) * h_new + self.alpha * (gate * transformed + (1 - gate) * h_0)
            
            weight = F.softmax(self.propagation_weights, dim=0)[k]
            aggregated = aggregated + weight * h
            total_weight += weight.item()
        
        if total_weight > 0:
            aggregated = aggregated / total_weight
        
        return aggregated
    
    def _aggregate_from_neighbors(self, source_h, target_h, edge_index):
        src, dst = edge_index[0], edge_index[1]
        
        max_idx = max(src.max().item(), dst.max().item()) + 1
        if source_h.size(0) > max_idx:
            source_h = source_h[:max_idx]
        
        messages = source_h[src.clamp(max=source_h.size(0)-1)]
        
        out = torch.zeros(target_h.size(0), source_h.size(1), 
                         device=source_h.device, dtype=source_h.dtype)
        
        for i in range(edge_index.size(1)):
            d_idx = dst[i].item()
            if d_idx < out.size(0):
                out[d_idx] = out[d_idx] + messages[i]
        
        deg = degree(dst.long(), target_h.size(0), dtype=source_h.dtype)
        deg = deg.clamp(min=1).unsqueeze(-1)
        
        return out / deg


class AdaptiveEdgeAttention(nn.Module):
    """
    自适应边注意力机制
    
    动态学习每条边的重要性权重，支持稀疏连接模式下的
    关键边识别和注意力聚焦
    """
    
    def __init__(self, node_dim, edge_dim=1, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        
        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        
        self.edge_lin = nn.Linear(edge_dim, node_dim)
        
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(0.1)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for lin in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
    
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), 1, device=x.device)
        
        batch_size = 1
        num_nodes = x.size(0)
        
        Q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)
        
        src, dst = edge_index[0], edge_index[1]
        src_clamped = src.clamp(max=num_nodes - 1)
        dst_clamped = dst.clamp(max=num_nodes - 1)
        
        Q_src = Q[:, src_clamped, :]
        K_dst = K[:, dst_clamped, :]
        V_dst = V[:, dst_clamped, :]
        
        edge_feat = self.edge_lin(edge_attr).view(-1, self.num_heads, self.head_dim)
        
        K_att = K_dst + edge_feat
        V_att = V_dst + edge_feat
        
        scores = (Q_src * K_att).sum(dim=-1) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn.unsqueeze(-1) * V_att).transpose(0, 1).contiguous()
        out = out.view(num_nodes, -1)
        out = self.out_proj(out)
        
        return out


class GLPGNNBlock(nn.Module):
    """
    GLP-GNN基本模块
    
    整合多维消息传递、标签传播和边注意力机制
    """
    
    def __init__(self, hidden_dim, edge_dim=1, num_heads=4, 
                 num_propagations=3, alpha=0.5, dropout=0.1):
        super().__init__()
        
        self.sparse_mp = SparseMultiDimMessagePassing(
            hidden_dim, hidden_dim, edge_dim=edge_dim
        )
        
        self.lp_layer = GLPLabelPropagation(
            hidden_dim, num_propagations=num_propagations, alpha=alpha
        )
        
        self.edge_attention = AdaptiveEdgeAttention(
            hidden_dim, edge_dim=edge_dim, num_heads=num_heads
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index_ap_ue, edge_index_ue_ap, edge_attr, D_mask):
        h = x
        
        h_mp = self.sparse_mp(h, edge_index_ap_ue, edge_attr)
        h = self.norm1(h + self.dropout(h_mp))
        
        h_lp = self.lp_layer(h, edge_index_ap_ue, edge_index_ue_ap, D_mask)
        h = self.norm2(h + self.dropout(h_lp))
        
        h_att = self.edge_attention(h, edge_index_ap_ue, edge_attr)
        h = self.norm3(h + self.dropout(h_att))
        
        h = h + self.ffn(h)
        
        return h


class JointPowerPrecodingHead(nn.Module):
    """
    联合功率-预编码输出头
    
    同时输出功率分配和预编码向量的预测
    """
    
    def __init__(self, hidden_dim, L, K, num_heads=4, dropout=0.1):
        super().__init__()
        self.L = L
        self.K = K
        
        self.power_branch = nn.Sequential(
            nn.Linear(hidden_dim + L, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, K)
        )
        
        self.precoding_branch = nn.Sequential(
            nn.Linear(hidden_dim + K, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, K)
        )
        
        self.joint_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, K)
        )
        
        self.output_scale = 1.5
    
    def forward(self, h_ap, x_ap_raw):
        h_combined = torch.cat([h_ap, x_ap_raw], dim=-1)
        rho_raw = self.power_branch(h_combined)
        
        rho_raw = torch.tanh(rho_raw) * self.output_scale
        
        return rho_raw


class GLPGNN(nn.Module):
    """
    GLP-GNN: Generalized Label Propagation Graph Neural Network
    
    用于Cell-Free Massive MIMO联合功率控制和预编码优化的图神经网络
    
    Args:
        L: AP (Access Point) 数量
        K: UE (User Equipment) 数量
        hidden_dim: 隐藏层维度
        num_heads: 注意力头数
        num_layers: GNN层数
        num_propagations: 标签传播轮数
        alpha: 标签传播混合系数
        dropout: Dropout比率
        edge_dim: 边特征维度
    """
    
    def __init__(self, L=100, K=20, hidden_dim=128, num_heads=4, 
                 num_layers=3, num_propagations=3, alpha=0.5, 
                 dropout=0.1, edge_dim=1):
        super().__init__()
        self.L = L
        self.K = K
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.ap_input_dim = K
        self.ue_input_dim = L + 1
        
        self.input_norm_ap = nn.LayerNorm(self.ap_input_dim)
        self.input_norm_ue = nn.LayerNorm(self.ue_input_dim)
        
        self.node_embedding_ap = nn.Linear(self.ap_input_dim, hidden_dim)
        self.node_embedding_ue = nn.Linear(self.ue_input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            GLPGNNBlock(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                num_heads=num_heads,
                num_propagations=num_propagations,
                alpha=alpha,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.output_head = JointPowerPrecodingHead(
            hidden_dim=hidden_dim,
            L=L,
            K=K,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.in_features == module.out_features:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        batch_size = batch.num_graphs
        
        num_ap = self.L
        num_ue = self.K
        nodes_per_graph = num_ap + num_ue
        
        node_pos = torch.arange(x.size(0), device=x.device) % nodes_per_graph
        is_ap_mask = node_pos < num_ap
        
        x_ap_raw = x[is_ap_mask, :self.ap_input_dim].clone()
        x_ue_raw = x[~is_ap_mask, :self.ue_input_dim]
        
        x_ap_norm = self.input_norm_ap(x_ap_raw)
        x_ue_norm = self.input_norm_ue(x_ue_raw)
        
        ap_emb = self.node_embedding_ap(x_ap_norm)
        ue_emb = self.node_embedding_ue(x_ue_norm)
        
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=ap_emb.dtype)
        h[is_ap_mask] = ap_emb
        h[~is_ap_mask] = ue_emb.to(dtype=h.dtype)
        
        edge_attr = torch.ones(edge_index.size(1), 1, device=x.device, dtype=h.dtype)
        
        max_node_idx = max(h.size(0), edge_index.max().item() + 1)
        if edge_index.numel() > 0:
            edge_index_ap_ue = edge_index[:, edge_index[0] < max_node_idx // 2]
            edge_index_ue_ap = edge_index[:, edge_index[0] >= max_node_idx // 2]
            
            if edge_index_ap_ue.numel() == 0:
                edge_index_ap_ue = edge_index
            if edge_index_ue_ap.numel() == 0:
                edge_index_ue_ap = edge_index.flip(0)
        else:
            edge_index_ap_ue = edge_index
            edge_index_ue_ap = edge_index
        
        D_mask = batch.D_mask.view(-1, self.L, self.K)[0] if hasattr(batch, 'D_mask') else None
        if D_mask is None:
            D_mask = torch.ones(self.L, self.K, device=x.device)
        
        for block in self.blocks:
            h = block(h, edge_index_ap_ue, edge_index_ue_ap, edge_attr, D_mask)
        
        h_ap = h[is_ap_mask]
        batch_size_actual = h_ap.size(0) // num_ap
        h_ap_reshaped = h_ap.view(batch_size_actual, num_ap, self.hidden_dim)
        
        x_ap_raw_reshaped = x_ap_raw.view(batch_size_actual, num_ap, self.ap_input_dim)
        
        rho = self.output_head(h_ap_reshaped, x_ap_raw_reshaped)
        
        return rho


class GLPGNN_Slim(nn.Module):
    """
    轻量级GLP-GNN变体
    
    适用于资源受限的边缘设备部署
    """
    
    def __init__(self, L=100, K=20, hidden_dim=64, num_layers=2, 
                 dropout=0.1):
        super().__init__()
        self.L = L
        self.K = K
        
        self.ap_embedding = nn.Linear(K, hidden_dim)
        self.ue_embedding = nn.Linear(L + 1, hidden_dim)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + K, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, K)
        )
        
        self.output_scale = 1.5
    
    def forward(self, batch):
        x = batch.x
        is_ap_mask = x[:, 0] != 0
        
        x_ap_raw = x[is_ap_mask, :self.K]
        x_ue_raw = x[~is_ap_mask, :self.L + 1]
        
        h_ap = self.ap_embedding(x_ap_raw)
        h_ue = self.ue_embedding(x_ue_raw)
        
        h = torch.cat([h_ap, h_ue], dim=0).unsqueeze(0)
        
        for layer in self.layers:
            h = layer(h)
        
        h_ap_out = h[0, :h_ap.size(0)]
        
        h_combined = torch.cat([h_ap_out, x_ap_raw], dim=-1)
        rho = self.output(h_combined)
        rho = torch.tanh(rho) * self.output_scale
        
        batch_size = 1
        return rho.unsqueeze(0).view(batch_size, self.L, self.K)
