#!/usr/bin/env python3
"""
GNN 训练数据集加载模块
处理 MATLAB v7.3 .mat 文件 (HDF5 格式)
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

class GNNDataset(TorchDataset):
    """GNN 功率分配数据集

    数据维度说明 (MATLAB v7.3 存储特点):
    - MATLAB 保存时维度会被转置: 原始 (L, K, N) → HDF5 (N, K, L)
    - 加载后需要转置回来: HDF5 (N, K, L) → (L, K, N)

    Attributes:
        x_ap: AP 节点特征 (N_snaps, K, L) → 转置后 (N_snaps, L, K)
        x_ue: UE 节点特征
        edge_index: 边索引
        y: 标签 (rho_WMMSE)
        esr_labels: ESR 标签
    """

    def __init__(self, mat_path, L=100, K=20, snr_dB_range=None):
        """
        Args:
            mat_path: .mat 文件路径
            L: AP 数量 (默认 100)
            K: UE 数量 (默认 20)
            snr_dB_range: 可选，筛选特定 SNR 的数据
        """
        super().__init__()
        self.mat_path = mat_path
        self.L = L
        self.K = K

        # 加载数据
        with h5py.File(mat_path, 'r') as f:
            # 读取 features
            features = f['features']
            sqrtGain = np.array(features['sqrtGain'])  # (N, K, L) in HDF5
            D = np.array(features['D'])                # (N, K, L) in HDF5
            sigma_e = np.array(features['sigma_e'])    # (N, 1, 1) in HDF5

            # 读取 labels
            labels = f['labels']
            rho_WMMSE = np.array(labels['rho_WMMSE'])  # (N, K, L) in HDF5
            ESR_WMMSE = np.array(labels['ESR_WMMSE']).flatten()  # (N,)

            # 读取 meta 信息 (结构体数组)
            meta = f['meta']
            meta_raw = np.array(meta)  # (1, N) array of references

            snrs = []
            modes = []
            for i in range(meta_raw.shape[1]):
                ref = meta_raw[0, i]
                struct = f[ref]
                snr = int(np.array(struct['SNR_dB'])[0, 0])
                # mode is stored as (n_chars, 1) uint16 array
                mode_arr = np.array(struct['mode']).flatten()
                mode = ''.join([chr(c) for c in mode_arr])
                snrs.append(snr)
                modes.append(mode.strip())

        # 转置: HDF5 (N, K, L) → 标准 (L, K, N)
        self.sqrtGain = np.transpose(sqrtGain, (2, 1, 0))  # (L, K, N)
        self.D = np.transpose(D, (2, 1, 0))                # (L, K, N)
        # sigma_e: 每个快照一个标量，MATLAB 存储形状不确定
        # 统一处理为 (N,) 的 1D 数组
        sigma_e_flat = np.array(sigma_e).flatten()
        n_snaps_raw = self.sqrtGain.shape[2]  # 从 sqrtGain 获取真实快照数
        if len(sigma_e_flat) == n_snaps_raw:
            self.sigma_e = sigma_e_flat  # (N,)
        elif len(sigma_e_flat) == 1:
            # 所有快照共用同一个 sigma_e（广播）
            self.sigma_e = np.full(n_snaps_raw, sigma_e_flat[0])
        else:
            print(f"WARNING: sigma_e shape {sigma_e.shape} unexpected, "
                  f"got {len(sigma_e_flat)} values for {n_snaps_raw} snapshots. Using first value.")
            self.sigma_e = np.full(n_snaps_raw, sigma_e_flat[0])
        self.rho_raw = np.transpose(rho_WMMSE, (2, 1, 0))   # (L, K, N) 原始标签
        self.ESR_WMMSE = ESR_WMMSE                          # (N,)

        self.snrs = np.array(snrs)
        self.modes = modes

        # 如果指定了 SNR 范围，筛选数据
        if snr_dB_range is not None:
            mask = np.isin(self.snrs, snr_dB_range)
            self.sqrtGain = self.sqrtGain[:, :, mask]
            self.D = self.D[:, :, mask]
            self.sigma_e = self.sigma_e[mask]
            self.rho_raw = self.rho_raw[:, :, mask]
            self.ESR_WMMSE = self.ESR_WMMSE[mask]
            self.snrs = self.snrs[mask]
            self.modes = [self.modes[i] for i in range(len(mask)) if mask[i]]

        self.n_snaps = self.sqrtGain.shape[2]
        print(f"Loaded dataset: {self.n_snaps} snapshots")
        print(f"  Shape: sqrtGain={self.sqrtGain.shape}, D={self.D.shape}")
        print(f"  SNR range: {np.unique(self.snrs)}")
        print(f"  Modes: {set(self.modes)}")

        # ── 方案B: 样本内归一化 ──
        # 不再在 __init__ 做全局归一化，改为在 __getitem__ 中对每个快照独立归一化
        # 这样每个样本内非零 rho 的差异会被充分利用，不会因全局长尾分布被压缩
        self.rho_WMMSE = None   # 不再预计算，改为动态生成

        # 全局统计信息（仅用于打印）
        n_neg = int((self.rho_raw < 0).sum())
        n_zero = int((self.rho_raw == 0).sum())
        n_nonzero = int((self.rho_raw > 0).sum())
        print(f"  Raw rho: min={self.rho_raw.min():.2f}, max={self.rho_raw.max():.2f}, "
              f"negatives={n_neg} ({n_neg/self.rho_raw.size*100:.1f}%)")
        print(f"  Rho sparsity: zero={n_zero} ({n_zero/self.rho_raw.size*100:.1f}%), "
              f"nonzero={n_nonzero} ({n_nonzero/self.rho_raw.size*100:.1f}%)")
        print(f"  Normalization: per-sample (signed-log + nonzero min-max to [-1,1])")

        # 构建固定的边索引 (基于 All 模式的完整二部图: L*K 条边)
        # AP 索引: 0 to L-1, UE 索引: L to L+K-1
        # 使用 All 模式的边来保证所有样本边数一致
        D_all = self.D[:, :, 0] if self.modes[0] == 'All' else np.ones((self.L, self.K))
        src_ap, dst_ue = np.where(D_all == 1)
        # UE 节点索引从 L 开始 (AP 节点是 0 to L-1, UE 节点是 L to L+K-1)
        dst_ue = dst_ue + self.L
        self.edge_index_fixed = np.vstack([src_ap, dst_ue])  # (2, num_edges)
        self.num_edges = self.edge_index_fixed.shape[1]
        print(f"  Fixed edge_index: {self.num_edges} edges (L={self.L}, K={self.K})")

    def get_fixed_edge_index(self):
        """返回固定的边索引 (用于批训练)"""
        return self.edge_index_fixed

    def __len__(self):
        return self.n_snaps

    def __getitem__(self, idx):
        """
        返回一个快照的图数据

        Returns:
            Data: PyTorch Geometric Data 对象
                x_ap: AP 节点特征 (L, K) - 每个 AP 到所有 UE 的 sqrt(gain)
                x_ue: UE 节点特征 (K, L+1) - 每个 UE 来自所有 AP 的 sqrt(gain) + sigma_e
                edge_index: 边索引 (2, num_edges) - 固定的全连接边
                D_mask: D 矩阵掩码 (L, K) - 指示哪些边有效
                y: 标签 rho_WMMSE (L, K)
                esr: ESR 标签 (标量)
                snr: SNR 值 (标量)
                mode: 接入模式 (字符串)
        """
        # 获取第 idx 个快照
        sqrt_g = self.sqrtGain[:, :, idx]  # (L, K)
        D = self.D[:, :, idx]               # (L, K)
        sigma_e = self.sigma_e[idx]  # 标量 (1D array)
        rho_raw_snap = self.rho_raw[:, :, idx]  # (L, K) 原始 rho
        esr = self.ESR_WMMSE[idx]
        snr = self.snrs[idx]
        mode = self.modes[idx]

        # ── 方案B: 样本内归一化 ──
        # 1. signed-log 变换: sign(rho) * log1p(|rho|)
        rho_transformed = np.sign(rho_raw_snap) * np.log1p(np.abs(rho_raw_snap))  # (L, K)

        # 2. 只取 rho_raw > 0 的位置做 min-max 缩放
        nonzero_mask = rho_raw_snap > 0  # (L, K) bool
        rho_nz = rho_transformed[nonzero_mask]

        if len(rho_nz) > 0:
            nz_min = float(rho_nz.min())
            nz_max = float(rho_nz.max())
            nz_range = nz_max - nz_min
            if nz_range < 1e-8:
                # 所有非零值相同 → 映射到 +1
                rho_normed = np.where(nonzero_mask, 1.0, -1.0).astype(np.float32)
            else:
                # 非零: 线性映射到 [-1, +1]; 零值: 固定 -1
                rho_normed = np.full_like(rho_transformed, -1.0, dtype=np.float32)
                rho_normed[nonzero_mask] = 2.0 * (rho_transformed[nonzero_mask] - nz_min) / nz_range - 1.0
        else:
            # 全零快照（极端情况）
            rho_normed = np.full_like(rho_transformed, -1.0, dtype=np.float32)

        # 裁剪到 [-1, 1]
        rho_normed = np.clip(rho_normed, -1.0, 1.0)

        # 使用固定的边索引
        edge_index = self.edge_index_fixed

        # AP 节点特征: sqrt(gain) 每一行是一个 AP 的特征向量 (K,)
        x_ap = torch.FloatTensor(sqrt_g)  # (L, K)

        # UE 节点特征: [sqrt(gain) from all APs, sigma_e] (L+1,)
        sqrt_g_ue = sqrt_g.T  # (K, L)
        sigma_e_vec = torch.full((self.K, 1), sigma_e)  # (K, 1)
        x_ue = torch.cat([torch.FloatTensor(sqrt_g_ue), sigma_e_vec], dim=1)  # (K, L+1)

        # D 掩码作为额外输入
        D_mask = torch.FloatTensor(D)  # (L, K)

        # 标签
        y = torch.FloatTensor(rho_normed)  # (L, K)

        # 非零 rho 标记（基于原始 rho_raw，不受归一化影响）
        rho_is_nonzero = torch.FloatTensor(nonzero_mask.astype(np.float32))  # (L, K)

        # ESR 作为辅助标签
        esr_label = torch.FloatTensor([esr])

        # SNR 作为辅助特征
        snr_label = torch.FloatTensor([snr])

        data = {
            'x_ap': x_ap,
            'x_ue': x_ue,
            'edge_index': torch.LongTensor(edge_index),
            'D_mask': D_mask,
            'rho_is_nonzero': rho_is_nonzero,
            'y': y,
            'esr': esr_label,
            'snr': snr_label,
            'mode': mode,
            'idx': idx
        }

        return data


class GNNDatasetGlobalNorm(TorchDataset):
    """GNN 训练数据集 — 全局 min-max 归一化变体（消融实验用）

    与 GNNDataset 的唯一区别：归一化统计量在 __init__ 中跨所有样本预计算，
    __getitem__ 使用全局 min/max 进行缩放，而非每个样本独立计算。

    用于消融实验维度4: 样本内归一化 vs 全局归一化
    """

    def __init__(self, mat_path, L=100, K=20, snr_dB_range=None):
        super().__init__(mat_path, L, K, snr_dB_range)

        # signed-log transform all samples
        rho_transformed = np.sign(self.rho_raw) * np.log1p(np.abs(self.rho_raw))

        nonzero_mask = self.rho_raw > 0

        # Global min/max across ALL nonzero positions in ALL snapshots
        nz_values = rho_transformed[nonzero_mask]
        if len(nz_values) > 0:
            self.global_min = float(nz_values.min())
            self.global_max = float(nz_values.max())
            self.global_range = self.global_max - self.global_min
        else:
            self.global_min = 0.0
            self.global_max = 1.0
            self.global_range = 1.0

        print(f"  [GlobalNorm] global min={self.global_min:.4f}, max={self.global_max:.4f}, range={self.global_range:.4f}")

    def __getitem__(self, idx):
        """返回一个快照的图数据，使用全局归一化"""
        sqrt_g = self.sqrtGain[:, :, idx]
        D = self.D[:, :, idx]
        sigma_e = self.sigma_e[idx]
        rho_raw_snap = self.rho_raw[:, :, idx]
        esr = self.ESR_WMMSE[idx]
        snr = self.snrs[idx]
        mode = self.modes[idx]

        # signed-log transform
        rho_transformed = np.sign(rho_raw_snap) * np.log1p(np.abs(rho_raw_snap))

        nonzero_mask = rho_raw_snap > 0

        if self.global_range < 1e-8:
            rho_normed = np.where(nonzero_mask, 1.0, -1.0).astype(np.float32)
        else:
            rho_normed = np.full_like(rho_transformed, -1.0, dtype=np.float32)
            rho_normed[nonzero_mask] = 2.0 * (rho_transformed[nonzero_mask] - self.global_min) / self.global_range - 1.0

        rho_normed = np.clip(rho_normed, -1.0, 1.0)

        edge_index = self.edge_index_fixed
        x_ap = torch.FloatTensor(sqrt_g)
        sqrt_g_ue = sqrt_g.T
        sigma_e_vec = torch.full((self.K, 1), sigma_e)
        x_ue = torch.cat([torch.FloatTensor(sqrt_g_ue), sigma_e_vec], dim=1)

        D_mask = torch.FloatTensor(D)
        y = torch.FloatTensor(rho_normed)
        rho_is_nonzero = torch.FloatTensor(nonzero_mask.astype(np.float32))
        esr_label = torch.FloatTensor([esr])
        snr_label = torch.FloatTensor([snr])

        data = {
            'x_ap': x_ap,
            'x_ue': x_ue,
            'edge_index': torch.LongTensor(edge_index),
            'D_mask': D_mask,
            'rho_is_nonzero': rho_is_nonzero,
            'y': y,
            'esr': esr_label,
            'snr': snr_label,
            'mode': mode,
            'idx': idx
        }

        return data


def load_and_explore_dataset(mat_path):
    """加载并探索数据集"""
    with h5py.File(mat_path, 'r') as f:
        print("Top-level keys:", list(f.keys()))

        features = f['features']
        print("\nFeatures keys:", list(features.keys()))

        labels = f['labels']
        print("\nLabels keys:", list(labels.keys()))

        sysConfig = f['sysConfig']
        print("\nSysConfig keys:", list(sysConfig.keys()))

        # 打印配置
        L = int(np.array(sysConfig['L'])[0,0])
        K = int(np.array(sysConfig['K'])[0,0])
        N = int(np.array(sysConfig['N'])[0,0])
        print(f"\nSystem config: L={L}, K={K}, N={N}")

        # 打印各变量维度
        print("\nFeatures shapes:")
        for key in features.keys():
            arr = np.array(features[key])
            print(f"  {key}: {arr.shape}")

        print("\nLabels shapes:")
        for key in labels.keys():
            arr = np.array(labels[key])
            print(f"  {key}: {arr.shape}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        mat_path = sys.argv[1]
    else:
        mat_path = "../data/gnn_training/gnn_training_data_20260426_230559.mat"

    print("="*60)
    print("Exploring dataset structure...")
    print("="*60)
    load_and_explore_dataset(mat_path)

    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    dataset = GNNDataset(mat_path)
    print(f"\nDataset length: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample data:")
        print(f"  x_ap shape: {sample['x_ap'].shape}")
        print(f"  x_ue shape: {sample['x_ue'].shape}")
        print(f"  edge_index shape: {sample['edge_index'].shape}")
        print(f"  y shape: {sample['y'].shape}")
        print(f"  esr: {sample['esr'].item():.2f}")
        print(f"  snr: {sample['snr'].item()}")
        print(f"  mode: {sample['mode']}")