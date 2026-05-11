"""
推理模块

提供GLP-GNN模型推理功能，用于MATLAB调用
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glp_gnn import GLPGNN, GLPGNN_Slim


class GLPInferrer:
    """GLP-GNN模型推理器"""

    def __init__(self, model_path, L=100, K=20, hidden_dim=128, num_heads=4, 
                 num_layers=3, num_propagations=3, alpha=0.5, model_type='full'):
        self.L = L
        self.K = K
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_propagations = num_propagations
        self.alpha = alpha
        self.model_type = model_type
        self.device = torch.device('cpu')

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        if model_type == 'slim':
            self.model = GLPGNN_Slim(
                L=L, K=K, hidden_dim=hidden_dim,
                num_layers=num_layers, dropout=0.1
            )
        else:
            self.model = GLPGNN(
                L=L, K=K, hidden_dim=hidden_dim,
                num_heads=num_heads, num_layers=num_layers,
                num_propagations=num_propagations, alpha=alpha,
                dropout=0.1
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
        x_ap = torch.FloatTensor(sqrt_gain).unsqueeze(0)

        sqrt_gain_ue = sqrt_gain.T
        x_ue = torch.cat([
            torch.FloatTensor(sqrt_gain_ue),
            torch.full((self.K, 1), sigma_e)
        ], dim=1).unsqueeze(0)

        D_mask_t = torch.FloatTensor(D_mask).unsqueeze(0)
        y_dummy = torch.zeros(1, self.L, self.K)
        rho_is_nonzero_dummy = torch.zeros(1, self.L, self.K)

        D = D_mask.numpy() if isinstance(D_mask, torch.Tensor) else D_mask
        src_ap, ue_dst = np.where(D == 1)
        edge_index = np.array([
            src_ap,
            ue_dst + self.L
        ])

        from torch_geometric.data import Data, Batch
        data = Data(
            x=torch.cat([x_ap, x_ue], dim=1).squeeze(0),
            edge_index=torch.LongTensor(edge_index),
            D_mask=D_mask_t.squeeze(0),
            rho_is_nonzero=rho_is_nonzero_dummy.squeeze(0),
            y=y_dummy.squeeze(0),
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


def create_inferrer(model_path, L=100, K=20, hidden_dim=128, num_heads=4,
                    num_layers=3, num_propagations=3, alpha=0.5, model_type='full'):
    """工厂函数：创建GLP-GNN推理器"""
    return GLPInferrer(
        model_path, L=L, K=K, hidden_dim=hidden_dim,
        num_heads=num_heads, num_layers=num_layers,
        num_propagations=num_propagations, alpha=alpha,
        model_type=model_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GLP-GNN inference')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--L', type=int, default=100)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--model_type', type=str, default='full',
                       choices=['full', 'slim'])
    args = parser.parse_args()

    inferrer = GLPInferrer(args.model, L=args.L, K=args.K, model_type=args.model_type)

    sqrt_gain = np.random.randn(args.L, args.K)
    D_mask = (np.random.rand(args.L, args.K) > 0.8).astype(float)

    rho = inferrer.infer(sqrt_gain, D_mask)

    print(f"Input shape: {sqrt_gain.shape}")
    print(f"D shape: {D_mask.shape}")
    print(f"Output rho shape: {rho.shape}")
    print(f"rho sum: {rho.sum():.4f}")
