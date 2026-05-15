# CF-Downlink-Sim

Cell-Free 大规模 MIMO 下行链路仿真系统 - 结合图神经网络的分布式功率分配研究平台

[![版本](https://img.shields.io/badge/version-v3.0-blue.svg)]()
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)]()

---

## 📖 项目简介

本项目研究 **Cell-Free 分布式 MIMO** 下行链路中的**低复杂度、低同步时延功率分配**问题。核心创新在于：

- **GNN 本地推理**：用图神经网络学习 WMMSE 的隐式映射，每个 AP 仅用本地 CSI 即可推理功率分配
- **混合架构**：MATLAB 处理信道仿真与矩阵运算，Python 处理 GNN 训练与推理

```
传统方案:                    本项目方案:
CPU 收集全部 CSI    →      GNN 本地推理 (零通信)
集中计算 ρ 矩阵     →      分布式推理
O(L×K) 同步代价    →      O(1) 无需迭代
```

---

## 🚀 快速开始（5分钟上手）

### 前置条件检查清单

在开始之前，请确保你的系统已安装：

- [x] **MATLAB R2020b 或更高版本**（含 Parallel Computing Toolbox）
- [x] **Python 3.10 或更高版本**
- [x] **CUDA GPU**（可选，用于加速 GNN 训练）

### 一键运行流程概览

```bash
# Step 1: 安装 Python 依赖
cd python && pip install -r requirements.txt

# Step 2: 在 MATLAB 中导出训练数据
matlab -batch "addpath(genpath(pwd)); exportTrainingData()"

# Step 3: 训练 GNN 模型
python train_gnn.py --data "../data/gnn_training/*.mat" --epochs 300

# Step 4: 运行完整仿真
matlab -batch "run"
```

> 💡 **提示**：如果只想快速体验传统算法（不含 GNN），可以直接跳到 Step 4，GNN 相关算法会自动回退到 EPA。

---

## 📋 详细运行指南

### 第一步：环境配置

#### 1.1 MATLAB 环境

| 组件 | 要求 | 检查命令 |
|------|------|---------|
| MATLAB 版本 | R2020b+ | `version` |
| Parallel Computing Toolbox | 必需 | `ver('parallel')` |
| Python 接口 | 用于 GNN 推理 | `pyenv` |

**配置 Python 解释器**（MATLAB 调用 Python 时需要）：

```matlab
% 检查当前 Python 配置
pyenv

% 设置 Python 路径（根据实际安装路径修改）
pyenv('Version', 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe');

% 验证配置
pyversion
```

#### 1.2 Python 环境

```bash
# 进入 Python 模块目录
cd CF_downlink_sim/python

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**核心依赖说明**：

| 包名 | 用途 | 最低版本 |
|------|------|---------|
| `torch` | PyTorch 深度学习框架 | >= 2.0.0 |
| `torch-geometric` | 图神经网络库 (PyTorch Geometric) | >= 2.5.0 |
| `numpy` | 数值计算 | >= 1.24.0 |
| `h5py` | 读取 MATLAB v7.3 格式的 .mat 文件 | >= 3.8.0 |

**验证安装**：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import h5py; print(f'h5py: {h5py.__version__}')"
```

---

### 第二步：导出训练数据（MATLAB）

训练 GNN 需要先用 MATLAB 生成带 WMMSE 标签的训练数据。

#### 2.1 标准导出

**方式 A：MATLAB GUI**

```matlab
% 切换到项目根目录
cd 'C:\Users\Admin\Documents\个人资料\CF_downlink_sim'

% 添加所有子目录到路径
addpath(genpath(pwd));

% 导出训练数据（默认参数）
exportTrainingData()
```

**方式 B：命令行批处理**

```powershell
matlab -batch "cd('C:\Users\Admin\Documents\个人资料\CF_downlink_sim'); addpath(genpath(pwd)); exportTrainingData(); exit"
```

#### 2.2 自定义导出参数

```matlab
% 自定义输出路径和每 SNR 点的快照数
exportTrainingData(fullfile(pwd, 'data', 'gnn_training'), 100)

% 快速测试模式（仅 2 个快照/SNR，用于验证流程）
exportTrainingData(fullfile(pwd, 'data', 'gnn_training'), 2)
```

#### 2.3 输出说明

导出的数据会保存在 `data/gnn_training/` 目录：

| 属性 | 默认值 | 说明 |
|------|--------|------|
| 文件名格式 | `gnn_training_data_YYYYMMDD_HHMMSS.mat` | 含时间戳 |
| SNR 范围 | 10:5:30 dB | 共 5 个 SNR 点 |
| 接入模式 | All + DCC | 两种模式 |
| 每 SNR 快照数 | 500 | 可自定义 |
| 总数据量 | 5 × 500 × 2 = 5000 条 | 取决于参数 |
| 数据增强 | 随机丢弃 AP + CSI 误差扰动 | 提升泛化能力 |

---

### 第三步：验证数据集（可选但推荐）

在训练前验证导出的数据是否正常：

```bash
cd CF_downlink_sim
python validate_dataset.py "data/gnn_training/gnn_training_data_XXXXXXXX_XXXXXX.mat"
```

> 将 `XXXXXXXX_XXXXXX` 替换为实际的文件名时间戳。

该脚本会检查：
- [ ] 数据维度是否正确
- [ ] 是否存在异常值（负数、NaN、Inf）
- [ ] 特征和标签的统计分布

---

### 第四步：训练 GNN 模型（Python）

#### 4.1 基本训练命令

```bash
cd CF_downlink_sim/python

# 使用单个数据文件训练
python train_gnn.py --data "../data/gnn_training/gnn_training_data_20260428_191512.mat" --epochs 200

# 使用 glob 匹配多个数据文件（推荐）
python train_gnn.py --data "../data/gnn_training/*.mat" --epochs 300

# 训练严格分布式的 AP-local GNN
python train_gnn_local.py --data "../data/gnn_training/*.mat" --epochs 120
```

#### 4.2 快速测试（验证流程是否通畅）

```bash
python train_gnn.py --data "../data/gnn_training/*.mat" --epochs 10 --batch_size 64
python train_gnn_local.py --data "../data/gnn_training/*.mat" --epochs 10
```

#### 4.3 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `../data/gnn_training/*.mat` | 训练数据路径，支持通配符 |
| `--epochs` | 300 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr_max` | 1e-3 | OneCycleLR 最大学习率 |
| `--hidden_dim` | 128 | GNN 隐藏层维度 |
| `--num_heads` | 4 | GAT 注意力头数 |
| `--num_layers` | 3 | GAT 层数 |
| `--dropout` | 0.1 | Dropout 比率 |
| `--val_split` | 0.15 | 验证集比例 |
| `--output_dir` | `../models` | 模型保存目录 |
| `--patience` | 50 | Early stopping 耐心值 |

#### 4.4 输出文件

训练完成后会在 `models/` 目录生成：

| 文件名 | 说明 |
|--------|------|
| `best_gat_gnn_power.pt` | **验证集最优模型**（推荐使用） |
| `best_local_gnn_power.pt` | **Local-GNN 最优模型**（严格分布式 GNN 候选） |
| `final_gat_gnn_power.pt` | 最后一轮模型 |
| `training_log.csv` | 训练过程指标记录 |

---

### 第五步：运行完整仿真（MATLAB）

将训练好的模型放入正确位置后，运行完整的 Cell-Free 下行链路仿真。

#### 5.1 模型文件放置

确保以下模型文件存在于 `models/` 目录：

```
CF_downlink_sim/
└── models/
    ├── best_gat_gnn_power.pt      # 全图 GNN 推理模型（Step 4 产出）
    └── best_local_gnn_power.pt    # Local-GNN 推理模型（可选，缺失则回退 EPA）
```

> ⚠️ **重要**：如果缺少这些文件，对应 GNN 算法会自动回退到 EPA（等功率分配），不影响其他算法运行。

`run_test.bat` 在训练 AP-local GNN 前会自动检测
`models/best_local_gnn_power.pt`。如果旧模型已存在，脚本会询问是否重新训练并替换；选择否时会复用旧模型继续后续推理测试和仿真。

若完整仿真已经跑完，只想根据已有 `Simulation_Results_v2.mat` 和
`Sync_Ablation_Results.mat` 重新绘图，可执行 `run_test.bat 6 /y`。

#### 5.2 启动仿真

**方式 A：MATLAB GUI（推荐新手）**

```matlab
% 切换到项目根目录
cd 'C:\Users\Admin\Documents\个人资料\CF_downlink_sim'

% 运行入口脚本
run
```

**方式 B：命令行批处理（适合自动化）**

```powershell
matlab -batch "cd('C:\Users\Admin\Documents\个人资料\CF_downlink_sim'); run"
```

#### 6.3 仿真输出

运行完成后，结果保存在：

| 目录 | 内容 | 清理策略 |
|------|------|---------|
| `main/Imgs/` | ESR 性能曲线图、对比柱状图 | **每次清空** |
| `main/SimulationData/` | 数值数据、缓存文件 | **持久保留** |

主要输出图表：
- ESR vs SNR 性能曲线（所有算法对比）
- 各 SNR 点最佳算法排名
- 功率分配计算时间对比
- 通信开销对比
- 预编码 × 功率分配的同步时延消融图（Pareto、热力图、R-MMSE 固定对比）

同步时延消融结果还会导出到：
- `main/SimulationData/Sync_Ablation_Table.csv`
- `main/SimulationData/Sync_Ablation_Results.mat`

---

## 🎛️ 高级配置与调优

### 修改仿真参数

所有实验参数统一在 [config/getDefaultParams.m](config/getDefaultParams.m) 中配置。主仿真和训练数据导出都会读取该文件，因此修改实验条件时优先只改这一处。

```matlab
params.system.L = 100;                     % AP 数量
params.system.K = 20;                      % UE 数量
params.power.SNR_dB = 10:5:30;             % 主仿真 SNR 扫描范围
params.csi.sigma_e = 0.3;                  % CSI 误差强度
params.simulation.numScenarios = 10;       % 场景数
params.simulation.nbrOfRealizations = 200; % 每场景信道实现数
params.runtime.runStage = 3;               % 1=传统/参考, 2=仅GNN, 3=全部
params.runtime.useCache = true;            % 是否启用缓存
params.training.nSnapshotsPerSNR = 500;    % 每个 SNR/模式的训练快照数
```

**常用参数调整建议**：

| 目标 | 参数调整 | 说明 |
|------|---------|------|
| 快速验证 | `params.simulation.nbrOfRealizations=20`, `params.simulation.numScenarios=1` | 减少计算量 |
| 更高精度 | `params.simulation.nbrOfRealizations=200`, `params.simulation.numScenarios=10` | 结果更稳定 |
| 更宽 SNR 范围 | `params.power.SNR_dB=-10:2:30` | 更细粒度 |
| 仅跑传统算法 | `params.runtime.runStage=1` | 跳过 GNN 部分 |
| 仅重跑 GNN | `params.runtime.runStage=2` | 传统方法从缓存加载 |
| 关闭同步时延消融 | `params.syncAblation.enable=false` | 不生成 FigA5-FigA7 |

同步时延模型可通过 `params.syncAblation` 配置块调节，关键参数包括
`fronthaulMbps`、`syncRttMs`、`dccPayloadRatio` 和 `includeComputeTime`。

### v3.0 缓存机制详解

v3.0 引入了智能缓存系统，大幅减少重复计算时间：

#### 工作原理

```
第一次运行:
  参数指纹生成 → 全部计算 → 结果写入缓存文件

后续运行:
  参数比对 → 匹配则直接读取缓存 → 秒级完成
            ↓ 不匹配
         自动重新计算并更新缓存
```

#### 缓存文件结构

```
main/SimulationData/
├── cache_scenario_s1.mat      # 场景1: 信道矩阵、拓扑、D矩阵
├── cache_scenario_s2.mat      # 场景2: ...
├── cache_snr_s1_sn10.mat      # 场景1, SNR=10dB: 预编码、rho、ESR
├── cache_snr_s1_sn15.mat      # 场景1, SNR=15dB: ...
└── ...
```

#### 缓存有效性判定规则

| 触发条件 | 行为 |
|---------|------|
| 修改 L/K/N/sigma_e 等 | **全部缓存失效**，重新计算 |
| 修改 SNR_dB 范围 | **对应 SNR 点失效**，其他保留 |
| 替换 .pt 模型文件 | **仅 GNN 缓存失效**，传统方法复用 |
| 手动删除 cache_*.mat | **强制全部重算** |
| 设置 `params.runtime.useCache=false` | **禁用缓存**，等同 v2.0 |

#### 典型使用场景

| 场景 | 操作 | 耗时 |
|------|------|------|
| 首次运行 | 直接运行 | ~30分钟（取决于配置） |
| 参数不变重跑 | 直接运行 | **<10秒**（纯读缓存） |
| 换了新 GNN 模型 | 替换 .pt 文件，设置 `params.runtime.runStage=2` | ~2分钟（仅 GNN） |
| 调了 sigma_e | 修改参数，直接运行 | ~30分钟（自动检测变更） |
| 强制全重算 | 删除 `SimulationData/cache_*.mat` | ~30分钟 |

---

## 🔬 算法组合说明

仿真系统自动遍历 **56 种算法组合**（7 PA × 4 PC × 2 Mode）：

### 功率分配方法 (PA)

| 方法 | 类型 | 复杂度 | 说明 |
|------|------|--------|------|
| **Baseline** | 传统 | O(LK) | 基于大尺度衰落距离比 |
| **EPA** | 传统 | O(LK) | 等功率分配 |
| **D-WMMSE** | 分布式 | O(固定轮次×LK) | 固定消息交换轮次的分布式 WMMSE 近似 |
| **Local-GNN** | 分布式智能 | O(L×前向传播) | 每个 AP 仅用本地 AP-UE 行独立推理 |
| **WMMSE** | 传统 | O(迭代×LK²) | 加权最小均方误差（最慢但最优） |
| **Random** | 传统 | O(LK) | 随机分配基线 |
| **GNN** | 智能参考 | O(前向传播) | 全图 GAT-GNN 低时延参考 |

主排名只统计 PC 和 PA 均满足分布式信息约束的组合。Local-GNN、D-WMMSE、EPA、Baseline、Random 可作为分布式 PA 候选；WMMSE、L-MMSE-G 和当前全图 GNN 保留为集中式参考。

### 预编码方法 (PC)

| 方法 | 说明 |
|------|------|
| **MR** | 最大比传输（最简单） |
| **L-MMSE** | 线性最小均方误差（分布式） |
| **R-MMSE** | 鲁棒 MMSE（抗 CSI 误差） |
| **L-MMSE-G** | 全局协方差 MMSE（集中式基准） |

### AP 接入模式

| 模式 | 说明 |
|------|------|
| **All** | 所有 AP 服务所有 UE |
| **DCC** | 基于大尺度衰落的动态关联 |

---

## 📊 性能预期

### 算法性能对比

| 方法 | 频谱效率 | 计算延迟 | 通信开销 | 适用场景 |
|------|---------|---------|---------|---------|
| WMMSE | ⭐⭐⭐⭐⭐ | 高 | O(LK) CSI | 集中式基准 |
| GNN | ⭐⭐⭐⭐ | **极低** | **零** | 本地实时 |
| EPA | ⭐⭐ | 极低 | 零 | 简单基线 |
| Baseline | ⭐⭐ | 极低 | 零 | 距离基线 |

### v3.0 性能优化亮点

| 优化项 | 提升倍数 | 技术细节 |
|--------|---------|---------|
| L-MMSE 向量化 | 2-3x | 消除双层 for 循环 |
| 并行数据导出 | 4-8x | parfor 并行化 |
| Kronecker 优化 | 微优化 | N=1 时跳过冗余计算 |
| 分层缓存 | 显著 | 避免重复信道生成和预编码 |
| GNN 推理加速 | 10x+ | 相比 WMMSE 迭代 |

---

## 🛠️ 故障排除

### 常见问题及解决方案

#### Q1: 运行 `run.m` 时 GNN 算法没有结果或报错？

**原因**：缺少训练好的模型文件

**解决**：
```bash
# 检查 models 目录
ls CF_downlink_sim/models/

# 应包含：
# - best_gat_gnn_power.pt

# 如果缺失，完成 Step 2-4 训练模型
```

> 模型缺失时 GNN 会**自动回退到 EPA**，不会导致程序崩溃。

#### Q2: MATLAB 调用 Python 失败？

**错误信息示例**：
```
Undefined variable "py" or class "py.torch"
```

**解决方案**：
```matlab
% 1. 检查 Python 配置
pyenv

% 2. 设置正确的 Python 路径
pyenv('Version', '你的Python路径\python.exe');

% 3. 测试导入
py.importlib.import_module('torch')
```

#### Q3: WMMSE 计算非常慢？

**优化建议**：
1. **启用缓存**：确保 `params.runtime.useCache = true`（默认开启）
2. **减少实现数**：临时设 `params.simulation.nbrOfRealizations = 20`
3. **减少 SNR 点**：改为 `params.power.SNR_dB = [10, 20, 30]`
4. **分阶段运行**：先用 `params.runtime.runStage=1` 跑传统方法，再用 `params.runtime.runStage=2` 跑 GNN

#### Q4: 如何清除缓存强制重算？

**方法 A**：删除缓存文件
```powershell
del CF_downlink_sim\main\SimulationData\cache_*.mat
```

**方法 B**：关闭缓存
```matlab
% 在 config/getDefaultParams.m 中修改
params.runtime.useCache = false;
```

#### Q5: 更换了 GNN 模型后需要全部重跑吗？

**不需要！** 这是 v3.0 缓存的核心优势：

1. 将新 `.pt` 文件放到 `models/` 目录
2. 设置 `params.runtime.runStage = 2`
3. 运行仿真
4. 系统会**自动检测模型变更**，仅重跑 GNN 推理，传统方法从缓存加载

耗时对比：
- 全部重算：~30 分钟
- 仅重跑 GNN：~2 分钟

#### Q6: 导出训练数据时报错 "Undefined function or variable"？

**原因**：未添加路径

**解决**：
```matlab
% 确保先执行这两步
cd('CF_downlink_sim根目录');
addpath(genpath(pwd));
exportTrainingData()
```

#### Q7: 内存不足 (Out of Memory)？

**解决方案**：
1. 减小系统规模：`params.system.L=50, params.system.K=10`，并重新训练模型
2. 减少实现数：`params.simulation.nbrOfRealizations = 20`
3. 减少场景数：`params.simulation.numScenarios = 1`
4. 关闭不必要的 MATLAB 窗口/变量

#### Q8: CUDA/GPU 相关错误？

**检查**：
```bash
# 检查 PyTorch CUDA 支持
python -c "import torch; print(f'CUDA available: {torch.is_cuda_available()}')"

# 如果为 False，安装 GPU 版 PyTorch
# https://pytorch.org/get-started/locally/
```

> 即使没有 GPU，GNN 训练也可以在 CPU 上运行，只是速度较慢。

---

## 📁 项目结构详解

```
CF_downlink_sim/
│
├── run.m                              # 🚀 MATLAB 入口脚本（双击即运行）
├── README.md                          # 📖 本文档
├── test_export.m                      # 🧪 测试脚本
├── validate_dataset.py                # ✅ 数据验证工具
│
├── main/                              # 🔬 主仿真引擎
│   └── Combined_Downlink_Sim.m        # v3.0 核心仿真脚本（含缓存）
│   ├── Imgs/                          # 📊 输出图表（每次清空）
│   └── SimulationData/                # 💾 仿真数据 + 缓存（持久）
│
├── channel/                           # 📡 信道模型模块
│   ├── generateSetup.m                # 场景拓扑生成
│   ├── functionChannelEstimates.m     # 信道估计
│   └── functionRlocalscattering.m     # 局部散射
│
├── precoding/                         # 🎯 预编码方案
│   ├── functionPrecoding_MR.m         # 最大比传输
│   ├── functionPrecoding_LMMSE.m      # 分布式 L-MMSE
│   ├── functionPrecoding_RobustMMSE.m # 鲁棒 MMSE
│   └── functionPrecoding_LMMSE_global.m # 全局 L-MMSE
│
├── power_allocation/                  # ⚡ 功率分配算法
│   ├── computeRhoDist.m              # Baseline（距离基线）
│   ├── computeRhoEPA.m               # 等功率分配
│   ├── computeRhoDistributedWMMSE.m   # 固定轮次分布式 WMMSE 近似
│   ├── computeRhoWMMSE.m             # WMMSE（迭代优化）
│   ├── computeRhoGNN.m               # GNN 推理接口
│   ├── computeRhoLocalGNN.m          # AP-local GNN 分布式推理接口
│   └── exportTrainingData.m           # 训练数据导出器
│
├── se_calculation/                    # 📈 频谱效率计算
│   ├── functionComputeSE_downlink_MR.m
│   ├── functionComputeSE_downlink_LMMSE.m
│   └── functionComputeSE_downlink_RobustMMSE.m
│
├── visualization/                     # 📊 可视化模块
│   ├── plotESRResults_v2.m           # ESR 性能曲线
│   ├── plotScenarioSetup.m           # 场景拓扑图
│   ├── plotAblationResults.m         # 消融实验图
│   ├── printFinalResults_v2.m        # 结果排名打印
│   └── printSimConfig.m              # 配置信息打印
│
├── config/                            # ⚙️ 配置管理
│   ├── getDefaultParams.m            # 统一参数定义
│   └── SimulationLogger.m            # 日志系统
│
├── python/                            # 🤖 Python 智能模块
│   ├── train_gnn.py                   # GAT-GNN 训练主脚本
│   ├── train_gnn_local.py             # AP-local GNN 训练脚本
│   ├── gnn_runtime_local.py           # AP-local MATLAB 推理运行时
│   ├── dataset.py                    # 数据集加载器
│   ├── requirements.txt              # Python 依赖清单
│   ├── inference/                    # 推理封装
│   ├── training/                     # 训练逻辑
│   │   └── train_centralized.py
│   └── utils/                        # 工具函数
│
├── data/                              # 📦 数据目录
│   └── gnn_training/                 # GNN 训练数据 (.mat)
│
├── models/                            # 🧠 模型仓库
│   └── best_gat_gnn_power.pt         # 最优 GNN 模型
│
└── docs/                              # 📚 详细文档
    ├── 00. 项目概览与导航.md          # 📍 文档入口
    ├── 07. 项目架构与创新点.md        # 🏗️ 系统架构设计
    └── 08. 项目运行指南.md            # 📘 完整运行手册
```

---

## 📚 扩展阅读

| 文档 | 适用人群 | 链接 |
|------|---------|------|
| **项目运行指南** | 所有用户 | [docs/08. 项目运行指南.md](docs/08.%20项目运行指南.md) |
| **架构与创新点** | 开发者/研究者 | [docs/07. 项目架构与创新点.md](docs/07.%20项目架构与创新点.md) |
| **Cell-Free 技术现状** | 背景调研 | [docs/01. Cell-Free技术研究现状与路径.md](docs/01.%20Cell-Free技术研究现状与路径.md) |

---

## 🤝 贡献指南

欢迎贡献代码、报告 Bug 或提出改进建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 许可证

本项目仅供学术研究使用。如需用于商业用途，请联系作者团队。

---

## 📎 学术引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{CF_downlink_sim,
  title = {CF-Downlink-Sim: Cell-Free MIMO Downlink Simulation with GNN Power Allocation},
  author = {Research Team},
  year = {2026},
  note = {Cell-Free Massive MIMO Downlink Simulation Platform},
  url = {https://github.com/your-repo/CF_downlink_sim}
}
```

---

## 👥 团队与支持

- **核心开发**: Cell-Free MIMO 研究团队
- **技术栈**: MATLAB R2020b + Python 3.10 + PyTorch 2.0 + PyTorch Geometric
- **应用领域**: 6G 无线通信、分布式 MIMO、边缘智能

---

<p align="center">
  <strong>🌟 CF-Downlink-Sim - 面向 6G 的 Cell-Free 分布式 MIMO 研究平台 🌟</strong>
</p>

---

## 📞 获取帮助

如果遇到问题：

1. 📖 先查阅本文档的 [故障排除](#-故障排除) 章节
2. 🔍 查看 `docs/08. 项目运行指南.md` 获取更详细说明
3. 🐛 如遇 Bug，请提交 Issue 并附上错误信息和配置

**最后更新时间**: 2026-05-10 | **当前版本**: v3.0 (含缓存与分阶段运行)
