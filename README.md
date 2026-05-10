# CF-Downlink-Sim

Cell-Free 大规模 MIMO 下行链路仿真系统 - 结合图神经网络与联邦学习的分布式功率分配研究平台

[![版本](https://img.shields.io/badge/version-v3.0-blue.svg)]()
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)]()

---

## 📖 项目简介

本项目研究 **Cell-Free 分布式 MIMO** 下行链路中的**低复杂度、低同步时延功率分配**问题。核心创新在于：

- **GNN 本地推理**：用图神经网络学习 WMMSE 的隐式映射，每个 AP 仅用本地 CSI 即可推理功率分配
- **联邦学习协作**：通过 FedAvg 在 AP 间协作优化，弥补分布式推理的信息损失
- **混合架构**：MATLAB 处理信道仿真与矩阵运算，Python 处理 GNN 训练与推理

```
传统方案:                    本项目方案:
CPU 收集全部 CSI    →      GNN 本地推理 (零通信)
集中计算 ρ 矩阵     →      FedAvg 协作聚合
O(L×K) 同步代价    →      O(K) 模型参数交换
```

---

## 🚀 快速开始

### 环境要求

- **MATLAB** R2020b+ (Parallel Computing Toolbox)
- **Python** 3.10+ (torch, torch-geometric, numpy, h5py)

### 安装依赖

```bash
cd python
pip install -r requirements.txt
```

### 运行仿真

```matlab
cd CF_downlink_sim
run
```

### 完整流程

```bash
# 1. 导出训练数据
matlab -batch "exportTrainingData"

# 2. 训练 GNN
python train_gnn.py --data "../data/gnn_training/*.mat" --epochs 300

# 3. 联邦学习微调 (可选)
python fedavg.py --data "../data/gnn_training/*.mat" --rounds 50

# 4. 运行仿真
matlab -batch "run"
```

详细步骤见 [docs/08. 项目运行指南.md](docs/08.%20项目运行指南.md)

---

## 📁 项目结构

```
CF_downlink_sim/
├── run.m                              # MATLAB 入口脚本
├── main/
│   └── Combined_Downlink_Sim.m        # v3.0 主仿真引擎
├── channel/                           # 信道模型
├── precoding/                         # 预编码 (已向量化优化)
├── power_allocation/                   # 功率分配 (已并行化)
│   ├── computeRhoDist.m               # Baseline
│   ├── computeRhoWMMSE.m              # WMMSE
│   ├── computeRhoGNN.m                # GNN 推理接口
│   └── exportTrainingData.m            # 训练数据导出 (parfor)
├── se_calculation/                    # SE 计算 (已优化)
├── visualization/                     # 可视化
├── config/                           # 配置管理 (v3.0)
│   ├── getDefaultParams.m           # 统一参数
│   └── SimulationLogger.m            # 日志系统
├── python/                           # Python 智能模块
│   ├── models/gnn.py                 # GAT/MLP 模型
│   ├── data/dataset.py              # 数据集加载
│   ├── training/train_centralized.py # 训练脚本
│   ├── inference/                    # 推理封装
│   ├── utils/                       # 工具函数
│   ├── train_gnn.py                # 主训练
│   ├── fedavg.py                   # 联邦学习
│   └── requirements.txt              # 依赖
├── models/                           # 训练好的 .pt 模型
├── data/gnn_training/               # 训练数据
└── docs/                            # 文档
```

---

## 🎯 核心功能

### 算法组合 (42+ 种)

| 功率分配 | 预编码 | 接入模式 |
|---------|--------|---------|
| Baseline, EPA, WMMSE | MR, L-MMSE, R-MMSE | All, DCC |
| PSO, Random, **GNN**, **GNN+FL** | L-MMSE-G (消融) | |

### 性能对比

| 方法 | 频谱效率 | 延迟 | 通信量 |
|------|---------|------|--------|
| WMMSE | ⭐⭐⭐⭐⭐ | 高 | O(L×K) CSI |
| GNN | ⭐⭐⭐⭐ | 低 | 零 |
| GNN+FL | ⭐⭐⭐⭐⭐ | 低 | O(K) 模型 |

---

## 📊 v3.0 性能优化

| 优化项 | 性能提升 | 说明 |
|--------|---------|------|
| L-MMSE 向量化 | 2-3x | 消除双层循环 |
| 并行数据导出 | 4-8x | parfor 并行 |
| Kronecker 消除 | 微优化 | N=1 时直接 reshape |
| 分层缓存 | 显著加速 | 避免重复计算 |

---

## 📚 文档导航

| 文档 | 内容 |
|------|------|
| [00. 项目概览与导航.md](docs/00.%20项目概览与导航.md) | **入口**，项目总览与快速定位 |
| [07. 项目架构与创新点.md](docs/07.%20项目架构与创新点.md) | 系统架构、模块分工、5大创新点 |
| [05. GNN与联邦学习混合功率分配方案.md](docs/05.%20GNN与联邦学习混合功率分配方案.md) | **核心方案**，GNN+FL 详细设计 |
| [08. 项目运行指南.md](docs/08.%20项目运行指南.md) | 环境配置、完整流程、常见问题 |
| [09. 消融实验设计.md](docs/09.%20消融实验设计.md) | 6维度消融实验设计 |

---

## 🔧 配置管理

使用统一参数配置模块：

```matlab
% 获取默认参数
params = getDefaultParams();

% 自定义参数
params = loadParams('simulation.nbrOfRealizations', 100, ...
                    'gnn.hidden_dim', 256);

% 获取仿真配置
config = getSimulationConfig();
```

---

## 📦 模型文件

训练好的模型应放在 `models/` 目录：

- `best_gat_gnn_power.pt` - 集中式 GNN 最优模型
- `gnn_power_fedavg.pt` - 联邦学习聚合模型
- `ablation/` - 消融实验模型

> ⚠️ 模型文件通过 `.gitignore` 忽略，不会上传到 GitHub

---

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 许可证

本项目仅供学术研究使用。

---

## 📎 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{CF_downlink_sim,
  title = {CF-Downlink-Sim: Cell-Free MIMO Downlink Simulation with GNN and Federated Learning},
  author = {Your Team},
  year = {2026},
  note = {https://github.com/your-repo/CF_downlink_sim}
}
```

---

## 👥 团队

- **核心开发**: 研究团队
- **技术支持**: MATLAB + Python 混合架构

---

<p align="center">
  <strong>CF-Downlink-Sim</strong> - 面向 6G 的 Cell-Free 分布式 MIMO 研究平台
</p>
