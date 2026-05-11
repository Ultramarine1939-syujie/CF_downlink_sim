"""
推理模块

提供 SP-MDGNN 模型推理功能，用于 MATLAB 调用
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SP_MDGNN, GLP_GNN_Lite


class SPMDGNNInferrer:
    """SP-MDGNN 模型推理器"""

    def __init__(self, model_path, L=100, K=20, hidden_dim=128, num_layers=3, tau=0.5):
        self.L = L
        self.K = K
        self.device = torch.device('cpu')
        self.tau = tau

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model_type = checkpoint.get('model_type', 'sp_mdgnn')

        if model_type == 'glp_lite':
            self.model = GLP_GNN_Lite(
                L=L, K=K, hidden_dim=hidden_dim,
                num_layers=num_layers, tau=tau, dropout=0.1, output_scale=1.0
            )
        else:
            self.model = SP_MDGNN(
                L=L, K=K, hidden_dim=hidden_dim,
                num_layers=num_layers, tau=tau, dropout=0.1, output_scale=1.0
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

        batch = {
            'x_ap': x_ap,
            'x_ue': x_ue,
            'D_mask': D_mask_t,
            'rho_is_nonzero': rho_is_nonzero_dummy,
            'y': y_dummy,
            'esr': torch.FloatTensor([[0.0]]),
            'snr': torch.FloatTensor([[10.0]])
        }

        with torch.no_grad():
            outputs = self.model(batch)
            rho_np = outputs['power'].squeeze(0).numpy()

        weights = np.maximum((rho_np + 1.0) / 2.0, 0.0)
        weights = weights * D_mask

        rho = np.zeros((self.L, self.K))
        for l in range(self.L):
            served = np.where(D_mask[l, :] > 0.5)[0]
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
    return SPMDGNNInferrer(model_path, L=L, K=K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SP-MDGNN inference')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--L', type=int, default=100)
    parser.add_argument('--K', type=int, default=20)
    args = parser.parse_args()

    inferrer = SPMDGNNInferrer(args.model, L=args.L, K=args.K)

    sqrt_gain = np.random.randn(args.L, args.K)
    D_mask = (np.random.rand(args.L, args.K) > 0.8).astype(float)

    rho = inferrer.infer(sqrt_gain, D_mask)

    print(f"Input shape: {sqrt_gain.shape}")
    print(f"D shape: {D_mask.shape}")
    print(f"Output rho shape: {rho.shape}")
    print(f"rho sum: {rho.sum():.4f}")
