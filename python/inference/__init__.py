"""
推理模块

提供模型推理功能，用于 MATLAB 调用
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PowerGNN_GAT, PowerGNN_MLP
from data import GNNDataset, custom_collate


class GNNInferrer:
    """GNN 模型推理器"""

    def __init__(self, model_path, L=100, K=20, hidden_dim=128, num_heads=4, num_layers=3):
        self.L = L
        self.K = K
        self.device = torch.device('cpu')

        self.torch = torch
        self.np = np

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model_type = checkpoint.get('model_type', 'gat')

        if model_type == 'mlp':
            self.model = PowerGNN_MLP(
                L=L, K=K, hidden_dim=hidden_dim,
                num_layers=num_layers, dropout=0.1, output_scale=1.0
            )
        else:
            self.model = PowerGNN_GAT(
                L=L, K=K, hidden_dim=hidden_dim,
                num_heads=num_heads, num_layers=num_layers,
                dropout=0.1, output_scale=1.0
            )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def infer(self, sqrt_gain, D_mask, sigma_e=0.0, Pt=1.0):
        """
        执行推理

        Args:
            sqrt_gain: (L, K) sqrt(gainOverNoise)
            D_mask: (L, K) AP-UE 关联矩阵
            sigma_e: CSI 误差标准差
            Pt: 总发射功率

        Returns:
            rho: (L, K) 功率分配系数
        """
        sqrt_gain_masked = sqrt_gain * D_mask
        snr_norm = np.log10(max(Pt, 1e-12)) / 3.0
        ap_degree = D_mask.sum(axis=1, keepdims=True) / max(self.K, 1)
        ue_degree = D_mask.sum(axis=0, keepdims=True).T / max(self.L, 1)
        ap_gain = np.log1p(sqrt_gain_masked.sum(axis=1, keepdims=True))
        ue_gain = np.log1p(sqrt_gain_masked.sum(axis=0, keepdims=True)).T

        x_ap = torch.FloatTensor(np.concatenate([
            sqrt_gain_masked,
            np.full((self.L, 1), snr_norm, dtype=np.float32),
            np.full((self.L, 1), sigma_e, dtype=np.float32),
            ap_degree.astype(np.float32),
            ap_gain.astype(np.float32),
        ], axis=1))

        x_ue = torch.FloatTensor(np.concatenate([
            sqrt_gain_masked.T,
            np.full((self.K, 1), snr_norm, dtype=np.float32),
            np.full((self.K, 1), sigma_e, dtype=np.float32),
            ue_degree.astype(np.float32),
            ue_gain.astype(np.float32),
        ], axis=1))

        D_mask_t = torch.FloatTensor(D_mask)
        y_dummy = torch.zeros(self.L, self.K)
        rho_is_nonzero_dummy = torch.zeros(self.L, self.K)

        max_feat_dim = max(x_ap.shape[1], x_ue.shape[1])
        x_combined = torch.zeros(self.L + self.K, max_feat_dim)
        x_combined[:self.L, :x_ap.shape[1]] = x_ap
        x_combined[self.L:, :x_ue.shape[1]] = x_ue

        D = D_mask.numpy() if isinstance(D_mask, torch.Tensor) else D_mask
        src_ap, ue_dst = np.where(D == 1)
        edge_ap2ue = np.array([src_ap, ue_dst + self.L])
        edge_ue2ap = np.array([ue_dst + self.L, src_ap])
        edge_index = np.concatenate([edge_ap2ue, edge_ue2ap], axis=1)

        from torch_geometric.data import Data, Batch
        data = Data(
            x=x_combined,
            edge_index=torch.LongTensor(edge_index),
            D_mask=D_mask_t,
            rho_is_nonzero=rho_is_nonzero_dummy,
            y=y_dummy,
            esr=torch.FloatTensor([0.0]),
            snr=torch.FloatTensor([10.0])
        )
        batch = Batch.from_data_list([data])

        with torch.no_grad():
            rho_pred = self.model(batch)
            rho_np = rho_pred.squeeze(0).numpy()

        weights = np.maximum((rho_np + 1.0) / 2.0, 0.0)
        weights = weights * D

        rho = np.zeros((self.L, self.K))
        for l in range(self.L):
            served = np.where(D[l, :] > 0.5)[0]
            if len(served) == 0:
                continue
            w = weights[l, served]
            s = np.sum(w)
            if s <= 0:
                rho[l, served] = Pt / len(served)
            else:
                rho[l, served] = Pt * w / s

        return rho


def create_inferrer(model_path, L=100, K=20):
    """工厂函数：创建推理器"""
    return GNNInferrer(model_path, L=L, K=K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GNN inference')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--L', type=int, default=100)
    parser.add_argument('--K', type=int, default=20)
    args = parser.parse_args()

    inferrer = GNNInferrer(args.model, L=args.L, K=args.K)

    sqrt_gain = np.random.randn(args.L, args.K)
    D_mask = (np.random.rand(args.L, args.K) > 0.8).astype(float)

    rho = inferrer.infer(sqrt_gain, D_mask)

    print(f"Input shape: {sqrt_gain.shape}")
    print(f"D shape: {D_mask.shape}")
    print(f"Output rho shape: {rho.shape}")
    print(f"rho sum: {rho.sum():.4f}")
