# CF_downlink_sim 项目 Code Wiki

> Cell-Free Massive MIMO 下行链路仿真平台 — 基于 GNN+联邦学习的功率分配优化

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 项目整体架构](#2-项目整体架构)
- [3. 目录结构](#3-目录结构)
- [4. 主要模块职责](#4-主要模块职责)
  - [4.1 config — 配置与日志](#41-config--配置与日志)
  - [4.2 channel — 信道建模](#42-channel--信道建模)
  - [4.3 precoding — 预编码](#43-precoding--预编码)
  - [4.4 se_calculation — 频谱效率计算](#44-se_calculation--频谱效率计算)
  - [4.5 power_allocation — 功率分配](#45-power_allocation--功率分配)
  - [4.6 visualization — 可视化](#46-visualization--可视化)
  - [4.7 main — 仿真主逻辑](#47-main--仿真主逻辑)
  - [4.8 python — GNN/FL 训练框架](#48-python--gnnfl-训练框架)
- [5. 关键类与函数说明](#5-关键类与函数说明)
- [6. 数据流与依赖关系](#6-数据流与依赖关系)
- [7. 项目运行方式](#7-项目运行方式)
- [8. 配置参数说明](#8-配置参数说明)
- [9. 算法对比体系](#9-算法对比体系)
- [10. 开发约定与注意事项](#10-开发约定与注意事项)

---

## 1. 项目概述

本项目是一个 **Cell-Free Massive MIMO 下行链路仿真平台**，核心研究目标是：

- 在 **分布式** 架构下（各 AP 仅需本地 CSI），设计低开销、高效率的下行链路预编码与功率分配方案
- 提出 **GNN+FL（图神经网络+联邦学习）** 闭环功率分配方案，替代传统迭代优化算法（WMMSE、PSO 等）
- 通过完整的消融实验验证各设计选择的合理性

### 研究成果

- **低开销分布式预编码**：仅需本地 CSI 即可逼近全局 CSI 性能
- **GNN+FL 功率分配**：推理速度接近 EPA，性能接近 WMMSE
- **同步/通信开销双降**：消除 WMMSE/PSO 的 AP 间迭代同步

### 技术栈

| 层级 | 技术 |
|------|------|
| 仿真核心 | MATLAB (R2024a+) |
| 信道模型 | 3GPP TR 38.901 Indoor-Hotspot |
| 深度学习 | Python 3.10+ / PyTorch 2.0+ / PyTorch Geometric 2.5+ |
| 联邦学习 | FedAvg (自实现) |
| GNN 模型 | GAT (Graph Attention Network) |

---

## 2. 项目整体架构

项目采用 **MATLAB 仿真 + Python 训练** 的混合架构，通过 `.mat` 文件进行数据交换：

```
┌──────────────────────────────────────────────────────────────────────┐
│                        MATLAB 仿真层                                │
│                                                                      │
│  run.m ──► Combined_Downlink_Sim.m (仿真主循环)                      │
│              │                                                       │
│              ├──► channel/       信道建模与估计                        │
│              │     ├── generateSetup.m          随机拓扑生成           │
│              │     ├── functionChannelEstimates.m  信道估计            │
│              │     └── functionRlocalscattering.m  散射建模            │
│              │                                                       │
│              ├──► precoding/     预编码方案                            │
│              │     ├── functionPrecoding_MR.m           MR 预编码     │
│              │     ├── functionPrecoding_LMMSE.m        L-MMSE       │
│              │     ├── functionPrecoding_LMMSE_global.m 全局 L-MMSE  │
│              │     └── functionPrecoding_RobustMMSE.m   Robust-MMSE  │
│              │                                                       │
│              ├──► se_calculation/  频谱效率计算                        │
│              │     ├── functionComputeSE_downlink_MR.m               │
│              │     ├── functionComputeSE_downlink_LMMSE.m            │
│              │     └── functionComputeSE_downlink_RobustMMSE.m       │
│              │                                                       │
│              ├──► power_allocation/  功率分配                          │
│              │     ├── computeRhoEPA.m             均匀功率分配        │
│              │     ├── computeRhoDist.m            理想 DPC 功率分配  │
│              │     ├── computeRhoWMMSE.m           WMMSE 优化        │
│              │     ├── computeRhoGNN.m             GNN 推理           │
│              │     ├── functionOptimize_d_PSO.m    PSO 优化           │
│              │     ├── functionComputeRho_all_dcc_pso.m  PSO 全流程  │
│              │     └── exportTrainingData.m        数据导出           │
│              │                                                       │
│              └──► visualization/  结果可视化                           │
│                    ├── plotESRResults.m / plotESRResults_v2.m        │
│                    ├── plotScenarioSetup.m                            │
│                    ├── plotAblationResults.m                          │
│                    ├── printSimConfig.m                               │
│                    └── printFinalResults.m / printFinalResults_v2.m  │
│                                                                      │
│  数据交换: data/*.mat ◄──► exportTrainingData                         │
└──────────────────────────────────────────────────────────────────────┘
                              │
                         .mat 文件
                              │
┌──────────────────────────────────────────────────────────────────────┐
│                       Python 训练层                                   │
│                                                                      │
│  python/                                                             │
│    ├── train_gnn.py              GNN 训练主脚本 (含模型定义)          │
│    ├── train_gnn_ablation.py     消融实验主脚本                       │
│    ├── dataset.py                数据集加载 (GNNDataset)              │
│    ├── fedavg.py                 联邦学习 FedAvg 实现                 │
│    ├── requirements.txt          Python 依赖                         │
│    │                                                                    │
│    ├── training/                  训练模块                             │
│    │     └── train_centralized.py  集中式训练 (旧版)                   │
│    ├── inference/                 推理模块                             │
│    │     └── __init__.py          GNNInferrer 推理器                  │
│    └── utils/                     工具函数                             │
│          └── __init__.py          MetricTracker 等工具                │
│                                                                      │
│  模型输出: models/*.pt                                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 目录结构

```
CF_downlink_sim/
├── run.m                              # 仿真启动入口
├── test_export.m                      # 数据导出启动脚本
├── README.md                          # 项目说明文档
├── .gitignore                         # Git 忽略规则
│
├── config/                            # 配置与日志
│   ├── getDefaultParams.m             # 默认仿真参数
│   └── SimulationLogger.m            # 日志管理类
│
├── channel/                           # 信道建模
│   ├── generateSetup.m               # AP/UE 拓扑生成
│   ├── functionChannelEstimates.m    # 信道估计
│   └── functionRlocalscattering.m    # 散射系数生成
│
├── precoding/                         # 预编码方案
│   ├── functionPrecoding_MR.m        # Maximum Ratio
│   ├── functionPrecoding_LMMSE.m     # L-MMSE (本地协方差)
│   ├── functionPrecoding_LMMSE_global.m  # L-MMSE (全局协方差)
│   └── functionPrecoding_RobustMMSE.m    # Robust-MMSE
│
├── se_calculation/                    # 频谱效率计算
│   ├── functionComputeSE_downlink_MR.m
│   ├── functionComputeSE_downlink_LMMSE.m
│   └── functionComputeSE_downlink_RobustMMSE.m
│
├── power_allocation/                  # 功率分配算法
│   ├── computeRhoEPA.m               # 等功率分配 (EPA)
│   ├── computeRhoDist.m              # 理想 DPC (上界)
│   ├── computeRhoWMMSE.m            # WMMSE 优化
│   ├── computeRhoGNN.m              # GNN 推理
│   ├── functionOptimize_d_PSO.m     # PSO 粒子群优化
│   ├── functionComputeRho_all_dcc_pso.m  # PSO 全流程封装
│   └── exportTrainingData.m         # GNN 训练数据导出
│
├── main/                              # 仿真主逻辑
│   └── Combined_Downlink_Sim.m       # 核心仿真循环
│
├── visualization/                     # 可视化
│   ├── plotESRResults.m              # ESR 结果图 (v1)
│   ├── plotESRResults_v2.m           # ESR 结果图 (v2)
│   ├── plotScenarioSetup.m           # 场景拓扑图
│   ├── plotAblationResults.m         # 消融实验图
│   ├── printSimConfig.m              # 配置信息打印
│   ├── printFinalResults.m           # 结果汇总打印 (v1)
│   └── printFinalResults_v2.m        # 结果汇总打印 (v2)
│
├── python/                            # Python 训练框架
│   ├── train_gnn.py                  # GNN 训练脚本 (含模型定义)
│   ├── train_gnn_ablation.py         # 消融实验脚本
│   ├── dataset.py                    # 数据集加载
│   ├── fedavg.py                     # 联邦学习实现
│   ├── requirements.txt              # Python 依赖
│   ├── training/                     # 训练模块
│   │   ├── __init__.py
│   │   └── train_centralized.py      # 集中式训练
│   ├── inference/                    # 推理模块
│   │   └── __init__.py              # GNNInferrer 推理器
│   └── utils/                        # 工具模块
│       └── __init__.py              # 工具函数
│
├── data/                              # 数据目录
│   └── gnn_training/                 # GNN 训练数据 (.mat)
│
├── models/                            # 模型输出
│   ├── best_gat_gnn_power.pt         # 集中式 GAT 最佳模型
│   ├── final_gat_gnn_power.pt        # 集中式 GAT 最终模型
│   ├── gnn_power_fedavg.pt           # 联邦学习最终模型
│   ├── gnn_power_fedavg_round*.pt    # 联邦学习各轮次模型
│   ├── fedavg_summary.json           # 联邦学习训练摘要
│   └── fedavg_log.csv               # 联邦学习训练日志
│
└── figures/                           # 输出图表
```

---

## 4. 主要模块职责

### 4.1 config — 配置与日志

#### [getDefaultParams.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/config/getDefaultParams.m)

集中管理所有仿真参数，返回一个 `params` 结构体。其他模块通过 `params.fieldName` 统一访问参数。

**核心参数组：**

| 参数组 | 字段 | 默认值 | 说明 |
|--------|------|--------|------|
| 信道 | `L` | 100 | AP 数量 |
| 信道 | `N` | 4 | 每 AP 天线数 |
| 信道 | `K` | 20 | UE 数量 |
| 信道 | `tau_c` | 200 | 相干长度 (符号) |
| 信道 | `tau_p` | 10 | 导频长度 |
| 信道 | `pilots_same_or_not` | 1 | 1=导频复用, 2=正交 |
| 信道 | `nbrOfRealizations` | 10 | 信道实现数 |
| 信道 | `numScenarios` | 2 | 随机拓扑场景数 |
| 信道 | `seed` | 42 | 随机种子 |
| 功控 | `power_control` | 1 | 0=全功率, 1=DPC, 2=EPA |
| 功控 | `pilot_power_control` | 0 | 导频功率控制 |
| 功控 | `Pt` | 200 mW | 总下行发射功率 |
| 噪声 | `noiseFigure` | 7 dB | 噪声系数 |
| 噪声 | `bandwidth` | 20 MHz | 系统带宽 |
| 预编码 | `preLogFactor` | 1 | 预对数因子 |
| DCC | `dcc_factor` | 0.1 | DCC 比例 (10% UE 保持连接) |
| 鲁棒 | `sigma_e` | 0.1 | CSI 估计误差标准差 |
| 鲁棒 | `nIter` | 10 | Robust-MMSE 迭代次数 |
| 输出 | `isSaveFig` | 0 | 是否保存图表 |
| 输出 | `isSaveData` | 1 | 是否保存数据 |
| PSO | `pso.nParticles` | 20 | 粒子数 |
| PSO | `pso.maxIter` | 50 | 最大迭代 |
| PSO | `pso.w` | 0.7 | 惯性权重 |
| PSO | `pso.c1` | 1.5 | 认知系数 |
| PSO | `pso.c2` | 1.5 | 社会系数 |

#### [SimulationLogger.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/config/SimulationLogger.m)

日志管理类，支持控制台输出和文件日志双通道。

**方法：**

| 方法 | 说明 |
|------|------|
| `SimulationLogger(logFilePath)` | 构造函数，创建日志文件 |
| `logInfo(msg)` | 记录 INFO 级别日志 |
| `logWarning(msg)` | 记录 WARNING 级别日志 |
| `logError(msg)` | 记录 ERROR 级别日志 |
| `logConfig(params)` | 记录仿真配置 |
| `logProgress(current, total, msg)` | 记录进度信息 |
| `logResult(name, value, unit)` | 记录结果 |
| `logSeparator()` | 记录分隔线 |
| `close()` | 关闭日志文件 |

---

### 4.2 channel — 信道建模

#### [generateSetup.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/channel/generateSetup.m)

在 500m × 500m 正方形区域内随机生成 AP 和 UE 的位置拓扑。

**算法：**
1. 使用复数表示二维坐标：`APpositions(l) = APdistanceUL(l) + 1i * APdistanceVL(l)`
2. AP 位置在 `[0, areaSide]` 均匀分布
3. UE 位置在 `[min(AP)-margin, max(AP)+margin]` 范围均匀分布
4. 限制 UE 在 `[0, areaSide]` 范围内

**输出：** `APpositions` (L×1), `UEpositions` (K×1) 复数向量

#### [functionChannelEstimates.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/channel/functionChannelEstimates.m)

基于 3GPP TR 38.901 Indoor-Hotspot 模型生成信道矩阵并执行最小二乘信道估计。

**算法流程：**
1. **大尺度衰落**：`gainOverNoise = 10^(0.1*(APgain-PL-sigma_sf*Zlf))`，其中路径损耗 PL = `32.4 + 17.3*log10(d3D) + 20*log10(fc)`
2. **小尺度衰落**：基于散射系数生成 `channelGain = sqrt(gainOverNoise) * sum_r(alpha_r * exp(j*theta_r))`
3. **导频复用**：按 `reuseDistance > 100m` 判断是否复用导频
4. **LS 信道估计**：`Hhat = sqrt(tau_p*Pt_pilot) * H * Fpilot'`
5. **MMSE 去导频污染**：使用本地协方差矩阵 `Rhat = (Hhat*Hhat')/nbrOfRealizations`，计算 `inv(Rhat + noise)`

**输出变量：**

| 变量 | 维度 | 说明 |
|------|------|------|
| `Hhat_all` | (N, K, tau_p, L, R) | 所有导频的信道估计 |
| `Hhat` | (N, tau_p, L, R) | 选定导频的信道估计 |
| `gainOverNoise` | (L, K, R) | 大尺度增益 |
| `HhatD` | (N, K, L, R) | AP-UE 关联后的信道估计 |
| `D` | (L, K) | AP-UE 关联矩阵 |
| `Fpilot` | (tau_p, K) | 导频矩阵 |

#### [functionRlocalscattering.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/channel/functionRlocalscattering.m)

生成局部散射系数 `alpha_r` 和到达角 `theta_r`，用于构造小尺度衰落。

**算法：**
- 使用 3GPP TR 38.901 中的簇延迟线 (CDL) 模型
- 8 个散射体均匀分布在 AP-UE 连线两侧 ±60° 范围
- 每个散射体的增益由路径损耗和角度衰减加权

---

### 4.3 precoding — 预编码

#### [functionPrecoding_MR.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/precoding/functionPrecoding_MR.m)

**Maximum Ratio (MR) 预编码** — 最简单的线性预编码。

**算法：** `V = sqrt(rho) * Hhat / norm(Hhat, 'fro')`

- 每个 AP 仅使用本地 CSI
- 无需 AP 间协调
- 性能最低但实现最简单

#### [functionPrecoding_LMMSE.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/precoding/functionPrecoding_LMMSE.m)

**L-MMSE 预编码（本地协方差版本）** — 本项目提出的分布式预编码方案。

**核心思想：** 将传统的全局 MMSE 预编码解耦为本地 L-MMSE 滤波 + 分布式功率分配。

**算法：**
1. 每个 AP l 计算本地协方差矩阵：`Rhat_l = (1/R) * sum_r Hhat_l * Hhat_l'`
2. 计算逆矩阵：`invRhat_l = inv(Rhat_l + sigma2 * eye(N))`
3. L-MMSE 滤波矩阵：`Vhat_l = invRhat_l * Hhat_l * inv(RhatD_l + invRhat_l * sigma2 / rho_l)`
4. 功率分配矩阵：`rho_l = diag(1 ./ diag(inv(RhatD_l + invRhat_l * sigma2 / rho_l)))`
5. 归一化：`V_l = Vhat_l * rho_l^(1/2) / norm(Vhat_l * rho_l^(1/2), 'fro')`

**关键创新：** 仅需本地 CSI（`Hhat_l` 和 `Rhat_l`），无需全局协方差矩阵。

#### [functionPrecoding_LMMSE_global.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/precoding/functionPrecoding_LMMSE_global.m)

**L-MMSE 预编码（全局协方差版本）** — 传统方案，用于对比。

与本地版本的区别：使用全局协方差矩阵 `Rhat`（所有 AP 的 CSI 拼接），需要 AP 间完全协调。

#### [functionPrecoding_RobustMMSE.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/precoding/functionPrecoding_RobustMMSE.m)

**Robust-MMSE 预编码** — 考虑 CSI 误差的鲁棒预编码。

**算法：** 迭代优化，每次迭代包含三个步骤：
1. **预编码更新**：基于当前 `W` 和 `mu` 计算 `V`
2. **辅助变量更新**：`mu` 和 `W` 闭式更新
3. **收敛检查**：当 `norm(W_new - W_old) < threshold` 时停止

**参数：** `nIter` 控制最大迭代次数，`sigma_e` 控制 CSI 误差大小。

---

### 4.4 se_calculation — 频谱效率计算

#### [functionComputeSE_downlink_MR.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/se_calculation/functionComputeSE_downlink_MR.m)

**MR 预编码的频谱效率计算。**

**算法：**
1. 计算每个 AP 的发射信号功率：`signal_power = abs(Hhat_l' * v_l)^2`
2. 计算干扰功率（来自其他 AP 和其他 UE）
3. 计算噪声功率
4. SE = `preLogFactor * log2(1 + SINR)`

#### [functionComputeSE_downlink_LMMSE.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/se_calculation/functionComputeSE_downlink_LMMSE.m)

**L-MMSE 预编码的频谱效率计算。**

与 MR 版本的主要区别：
- L-MMSE 的协方差矩阵 `Rhat` 比 MR 更精确
- 需要额外计算 `invRhat` 和 `RhatD` 矩阵

#### [functionComputeSE_downlink_RobustMMSE.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/se_calculation/functionComputeSE_downlink_RobustMMSE.m)

**Robust-MMSE 预编码的频谱效率计算。**

在 L-MMSE 基础上增加了 CSI 误差的鲁棒性处理。

---

### 4.5 power_allocation — 功率分配

#### [computeRhoEPA.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/power_allocation/computeRhoEPA.m)

**等功率分配 (Equal Power Allocation)。**

**算法：** 对每个 AP l，均匀分配功率给其服务的 UE：
```
rho(l, served_k) = Pt / num_served_UEs
```

#### [computeRhoDist.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/power_allocation/computeRhoDist.m)

**理想 DPC (Dirty Paper Coding) 功率分配** — 性能上界。

基于 D 矩阵中各 UE 的信道增益进行排序和功率分配，理论上可达到最大 ESR。

#### [computeRhoWMMSE.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/power_allocation/computeRhoWMMSE.m)

**WMMSE (Weighted Minimum Mean Square Error) 功率分配。**

迭代优化算法，每次迭代包含三个步骤：
1. **E 步**：更新误差权重
2. **W 步**：更新 WMMSE 权重
3. **A 步**：更新功率分配

**特点：** 性能最优但需要 AP 间迭代同步，通信开销大。

#### [computeRhoGNN.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/power_allocation/computeRhoGNN.m)

**GNN 功率分配推理。**

通过 `pyrunfile` 调用 Python 脚本进行 GNN 推理，输入为信道增益和 D 矩阵，输出为功率分配系数。

#### [functionOptimize_d_PSO.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/power_allocation/functionOptimize_d_PSO.m)

**PSO (Particle Swarm Optimization) 粒子群优化。**

**算法：**
1. 初始化粒子位置和速度
2. 迭代更新：
   - 速度更新：`v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)`
   - 位置更新：`x = x + v`
   - 边界约束：`x = max(0, min(1, x))`
3. 评估适应度（ESR）
4. 更新个体最优和全局最优

**适应度函数：** `fitness = -ESR(rho)` (取负号因为 PSO 默认最大化)

#### [functionComputeRho_all_dcc_pso.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/power_allocation/functionComputeRho_all_dcc_pso.m)

**PSO 全流程封装。**

集成 PSO 优化、结果评估和可视化，是 PSO 功率分配的完整实现。

#### [exportTrainingData.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/power_allocation/exportTrainingData.m)

**GNN 训练数据导出。**

将仿真产生的信道数据和功率分配标签导出为 `.mat` 文件，供 Python 训练使用。

**数据增强：** 通过 `numAugment` 参数控制数据增强倍数。

**导出数据字段：**

| 字段 | 维度 | 说明 |
|------|------|------|
| `Hhat_all` | (N, K, tau_p, L) | 所有导频的信道估计 |
| `gainOverNoise` | (L, K) | 大尺度增益 |
| `D` | (L, K) | AP-UE 关联矩阵 |
| `Fpilot` | (tau_p, K) | 导频矩阵 |
| `HhatD` | (N, K, L) | AP-UE 关联后的信道估计 |
| `L` | 标量 | AP 数量 |
| `K` | 标量 | UE 数量 |
| `N` | 标量 | 每 AP 天线数 |
| `tau_p` | 标量 | 导频长度 |
| `Pt` | 标量 | 总发射功率 |
| `sigma2` | 标量 | 噪声功率 |

---

### 4.6 visualization — 可视化

#### [plotESRResults.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/visualization/plotESRResults.m) / [plotESRResults_v2.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/visualization/plotESRResults_v2.m)

**ESR 结果可视化。** 生成 10 张对比图表：
- 图 1-2：PSO 预编码方案对比 (All/DCC)
- 图 3-5：各预编码方案下的功率分配方法对比
- 图 6-8：功率分配方法综合对比 (MR/L-MMSE/R-MMSE)
- 图 9-10：所有方法最优性能汇总

v2 版本改进：
- 增加 GNN+FL 和 GNN 方法的对比
- 增加性能增益分析图
- 增加综合性能摘要表
- 改进颜色方案和图例

#### [plotScenarioSetup.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/visualization/plotScenarioSetup.m)

**场景拓扑可视化。** 绘制 AP（蓝色三角形）和 UE（红色圆形）的位置分布。

#### [plotAblationResults.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/visualization/plotAblationResults.m)

**消融实验结果可视化。** 生成 4 张图表：
- 图 A1：验证集相关性对比
- 图 A2：验证集 NZ-MSE 对比
- 图 A3：多指标雷达图
- 图 A4：消融维度成对对比

#### [printSimConfig.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/visualization/printSimConfig.m)

**仿真配置信息打印。** 以格式化表格形式输出仿真参数。

#### [printFinalResults.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/visualization/printFinalResults.m) / [printFinalResults_v2.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/visualization/printFinalResults_v2.m)

**仿真结果汇总打印。** 输出完整的算法排名表、功率分配方法对比表、PSO 优化详情等。

v2 版本改进：
- 聚焦排名与增益分析
- 增加 GNN+FL vs WMMSE 的 ESR loss 分析
- 增加同步延迟和通信开销减少比例统计

---

### 4.7 main — 仿真主逻辑

#### [Combined_Downlink_Sim.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/main/Combined_Downlink_Sim.m)

**核心仿真循环。** 这是整个 MATLAB 仿真层的中枢，协调所有模块的执行。

**执行流程：**

```
1. 参数初始化 (getDefaultParams)
2. 日志初始化 (SimulationLogger)
3. 配置打印 (printSimConfig)
4. 外层循环: 遍历 SNR 值
   5. 中层循环: 遍历场景 (numScenarios)
      6. 生成拓扑 (generateSetup)
      7. 生成信道 (functionChannelEstimates)
      8. 内层循环: 遍历信道实现 (nbrOfRealizations)
         9. 预编码计算 (MR/L-MMSE/Robust-MMSE)
         10. 功率分配 (Baseline/Random/PSO/EPA/WMMSE/GNN)
         11. 频谱效率计算 (SE_downlink)
         12. 结果累加
      13. 场景级平均
   14. SNR 级平均
15. 结果打印 (printFinalResults)
16. 结果可视化 (plotESRResults)
17. 日志关闭
```

**算法组合矩阵：** 3 种预编码 × 5 种功率分配 × 2 种接入模式 = 30 种算法组合

| 预编码 | 功率分配 | 接入模式 |
|--------|----------|----------|
| MR | Baseline (理想 DPC) | All (全连接) |
| L-MMSE | Random | DCC (动态协作簇) |
| Robust-MMSE | PSO | |
| | EPA | |
| | WMMSE | |

---

### 4.8 python — GNN/FL 训练框架

#### [train_gnn.py](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/python/train_gnn.py)

**GNN 训练主脚本。** 包含模型定义和训练逻辑。

**模型架构：**

```python
class PowerGNN_GAT(nn.Module):
    """
    GAT-GNN 功率分配模型 v3

    架构:
    - 输入: AP 特征 (L×1) + UE 特征 (K×3) + 边特征 (4)
    - GAT 层: 3 层 GATConv, 每层 4 头注意力
    - 输出: AP 预测头 → (L×K) 功率分配矩阵

    关键设计:
    - 动态 output_scale (根据实际标签范围自动设置)
    - AP 预测头带残差连接
    - Xavier 初始化
    - SNR 条件嵌入
    """

class PowerGNN_MLP(nn.Module):
    """
    MLP baseline — 无图结构, 无注意力机制

    用于消融实验, 与 PowerGNN_GAT 参数量可比
    """
```

**图构建：**
- **节点类型**：AP 节点 (L 个) + UE 节点 (K 个)
- **AP 节点特征**：`sqrt(gainOverNoise)` (1 维)
- **UE 节点特征**：`sqrt(gainOverNoise)^T` + `sigma_e` (3 维)
- **边连接**：AP-UE 关联矩阵 D 中的非零元素
- **边特征**：`(gain_ap, gain_ue, D_mask, sigma_e)` (4 维)

**训练配置：**
- 优化器：AdamW (lr=1e-4, weight_decay=5e-4)
- 学习率调度：OneCycleLR (max_lr=1e-3, warmup=10%)
- 损失函数：Huber Loss (delta=0.5)，仅在 `rho_is_nonzero=1` 的位置计算
- 早停：patience=50 epochs
- 评估指标：Val MSE, Val Correlation, Val NZ-MSE

#### [dataset.py](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/python/dataset.py)

**数据集加载模块。**

```python
class GNNDataset(Dataset):
    """
    GNN 训练数据集

    从 .mat 文件加载数据, 对每个 SNR 值生成一个样本:
    - 图结构: AP-UE 二部图
    - 节点特征: 信道增益
    - 标签: 功率分配系数 (归一化到 [-1, 1])

    标签归一化策略:
    - 非零值: signed-log + min-max → [-1, 1]
    - 零值: 保持为 0
    """

class GNNDatasetGlobalNorm(Dataset):
    """全局归一化版本 (用于消融实验)"""
```

**数据增强：**
- 随机 UE 索引排列（保持信道结构不变）
- 随机 SNR 选择
- 通过 `numAugment` 参数控制增强倍数

#### [fedavg.py](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/python/fedavg.py)

**联邦学习 FedAvg 实现。**

```python
def split_dataset_to_clients(dataset, num_clients, seed, split_by):
    """将数据集按 AP 索引分割到各客户端"""

def local_train(model, loader, device, lr, local_epochs, ...):
    """本地训练: 在客户端数据上训练模型"""

def fedavg_aggregate(state_dicts, weights):
    """FedAvg 聚合: 加权平均各客户端模型参数"""
```

**联邦学习流程：**
1. 初始化全局模型
2. 每轮随机选择 `frac` 比例的客户端
3. 各客户端在本地数据上训练
4. 加权聚合各客户端模型
5. 在验证集上评估全局模型
6. 重复直到收敛

#### [train_gnn_ablation.py](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/python/train_gnn_ablation.py)

**消融实验主脚本。** 支持 6 个消融维度：

| 维度 | 变体 | 说明 |
|------|------|------|
| 1. GNN vs No-GNN | `full` vs `mlp_only` | 图结构的作用 |
| 2. GAT vs MLP | `full` vs `mlp_only` | 注意力机制的作用 |
| 3. FedAvg vs Central | `full` vs `fedavg_full` | 联邦学习 vs 集中式 |
| 4. 归一化策略 | `full` vs `global_norm` | Per-sample vs Global |
| 5. 数据增强 | `full` vs `no_augment` | 增强的作用 |
| 6. L-MMSE 重构 | MATLAB 端 | 全局协方差 vs 本地协方差 |

#### [__init__.py (inference)](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/python/inference/__init__.py)

**GNN 推理器。** 供 MATLAB 调用的推理接口。

```python
class GNNInferrer:
    def __init__(self, model_path, L=100, K=20, ...):
        """加载模型权重"""

    def infer(self, sqrt_gain, D_mask, sigma_e, Pt):
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
```

---

## 5. 关键类与函数说明

### 5.1 MATLAB 核心函数

| 函数 | 文件 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| `getDefaultParams` | config/getDefaultParams.m | 无 | params 结构体 | 获取默认仿真参数 |
| `generateSetup` | channel/generateSetup.m | areaSide, L, K | APpositions, UEpositions | 生成 AP/UE 拓扑 |
| `functionChannelEstimates` | channel/functionChannelEstimates.m | params, APpos, UEpos | Hhat, gainOverNoise, D, ... | 信道估计 |
| `functionPrecoding_MR` | precoding/functionPrecoding_MR.m | HhatD, rho, params | V_MR | MR 预编码 |
| `functionPrecoding_LMMSE` | precoding/functionPrecoding_LMMSE.m | HhatD, rho, params | V_L | L-MMSE 预编码 |
| `functionPrecoding_RobustMMSE` | precoding/functionPrecoding_RobustMMSE.m | HhatD, rho, params | V_R | Robust-MMSE 预编码 |
| `functionComputeSE_downlink_MR` | se_calculation/functionComputeSE_downlink_MR.m | HhatD, V, params | SE_MR | MR 频谱效率 |
| `functionComputeSE_downlink_LMMSE` | se_calculation/functionComputeSE_downlink_LMMSE.m | HhatD, V, RhatD, params | SE_L | L-MMSE 频谱效率 |
| `functionComputeSE_downlink_RobustMMSE` | se_calculation/functionComputeSE_downlink_RobustMMSE.m | HhatD, V, params | SE_R | Robust-MMSE 频谱效率 |
| `computeRhoEPA` | power_allocation/computeRhoEPA.m | D, Pt | rho_EPA | 等功率分配 |
| `computeRhoDist` | power_allocation/computeRhoDist.m | D, gainOverNoise, Pt | rho_Dist | 理想 DPC |
| `computeRhoWMMSE` | power_allocation/computeRhoWMMSE.m | D, gainOverNoise, V, params | rho_WMMSE | WMMSE 优化 |
| `computeRhoGNN` | power_allocation/computeRhoGNN.m | gainOverNoise, D, sigma_e, Pt | rho_GNN | GNN 推理 |
| `functionOptimize_d_PSO` | power_allocation/functionOptimize_d_PSO.m | ESR_func, params | d_opt, iterUsed, bestFitness | PSO 优化 |
| `exportTrainingData` | power_allocation/exportTrainingData.m | saveDir, numAugment | .mat 文件 | 数据导出 |
| `Combined_Downlink_Sim` | main/Combined_Downlink_Sim.m | params | SE 结果, 图表 | 仿真主循环 |

### 5.2 Python 核心类/函数

| 类/函数 | 文件 | 说明 |
|---------|------|------|
| `PowerGNN_GAT` | python/train_gnn.py | GAT-GNN 功率分配模型 |
| `PowerGNN_MLP` | python/train_gnn.py | MLP baseline 模型 |
| `GNNDataset` | python/dataset.py | GNN 训练数据集 |
| `GNNDatasetGlobalNorm` | python/dataset.py | 全局归一化数据集 |
| `custom_collate` | python/train_gnn.py | GAT 批处理 collate 函数 |
| `custom_collate_mlp` | python/train_gnn.py | MLP 批处理 collate 函数 |
| `compute_loss` | python/train_gnn.py | Huber Loss (非零 rho mask) |
| `train_epoch` | python/train_gnn.py | GAT 训练一个 epoch |
| `evaluate` | python/train_gnn.py | GAT 模型评估 |
| `GNNInferrer` | python/inference/__init__.py | GNN 推理器 |
| `set_seed` | python/fedavg.py | 随机种子设置 |
| `split_dataset_to_clients` | python/fedavg.py | 数据集分割 |
| `local_train` | python/fedavg.py | 本地训练 |
| `fedavg_aggregate` | python/fedavg.py | FedAvg 参数聚合 |
| `MetricTracker` | python/utils/__init__.py | 指标追踪器 |

---

## 6. 数据流与依赖关系

### 6.1 MATLAB 内部数据流

```
getDefaultParams
    │
    ▼
Combined_Downlink_Sim
    │
    ├──► generateSetup ──► APpositions, UEpositions
    │
    ├──► functionChannelEstimates ──► Hhat, gainOverNoise, D, Fpilot
    │       │
    │       └──► functionRlocalscattering ──► alpha_r, theta_r
    │
    ├──► [功率分配]
    │       ├── computeRhoDist ──► rho_Dist (理想上界)
    │       ├── computeRhoEPA ──► rho_EPA (均匀分配)
    │       ├── computeRhoWMMSE ──► rho_WMMSE (迭代优化)
    │       ├── computeRhoGNN ──► rho_GNN (GNN 推理)
    │       └── functionOptimize_d_PSO ──► rho_PSO (PSO 优化)
    │
    ├──► [预编码]
    │       ├── functionPrecoding_MR ──► V_MR
    │       ├── functionPrecoding_LMMSE ──► V_L
    │       └── functionPrecoding_RobustMMSE ──► V_R
    │
    ├──► [频谱效率计算]
    │       ├── functionComputeSE_downlink_MR ──► SE_MR
    │       ├── functionComputeSE_downlink_LMMSE ──► SE_L
    │       └── functionComputeSE_downlink_RobustMMSE ──► SE_R
    │
    └──► [可视化]
            ├── printFinalResults_v2
            ├── plotESRResults_v2
            └── plotScenarioSetup
```

### 6.2 MATLAB ↔ Python 数据流

```
MATLAB 仿真层                        Python 训练层
─────────────                        ──────────────

exportTrainingData.m
    │
    ├── 生成训练样本 (SNR随机, 数据增强)
    │
    ▼
data/gnn_training/*.mat
    │
    ├── GNNDataset 加载
    │   ├── 图构建 (AP-UE 二部图)
    │   ├── 标签归一化 (signed-log + min-max)
    │   └── 数据增强 (UE排列)
    │
    ▼
PowerGNN_GAT 训练
    │
    ├── FedAvg 联邦学习
    │   ├── split_dataset_to_clients
    │   ├── local_train
    │   └── fedavg_aggregate
    │
    ▼
models/*.pt (训练好的模型)
    │
    ├── GNNInferrer 加载
    │
    ▼
computeRhoGNN.m (通过 pyrunfile 调用)
    │
    ├── 输出 rho_GNN (L×K)
    │
    ▼
SE 计算 → ESR 结果
```

### 6.3 模块依赖关系图

```
config/getDefaultParams
    │
    ▼
main/Combined_Downlink_Sim
    ├──► config/SimulationLogger
    ├──► channel/generateSetup
    ├──► channel/functionChannelEstimates
    │       └──► channel/functionRlocalscattering
    ├──► precoding/functionPrecoding_MR
    ├──► precoding/functionPrecoding_LMMSE
    ├──► precoding/functionPrecoding_RobustMMSE
    ├──► se_calculation/functionComputeSE_downlink_MR
    ├──► se_calculation/functionComputeSE_downlink_LMMSE
    ├──► se_calculation/functionComputeSE_downlink_RobustMMSE
    ├──► power_allocation/computeRhoEPA
    ├──► power_allocation/computeRhoDist
    ├──► power_allocation/computeRhoWMMSE
    ├──► power_allocation/computeRhoGNN
    ├──► power_allocation/functionOptimize_d_PSO
    ├──► visualization/printSimConfig
    ├──► visualization/printFinalResults_v2
    ├──► visualization/plotESRResults_v2
    └──► visualization/plotScenarioSetup
```

---

## 7. 项目运行方式

### 7.1 环境要求

**MATLAB：**
- MATLAB R2024a 或更高版本
- 需要 Statistics and Machine Learning Toolbox

**Python：**
- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.5+
- NumPy 1.24+
- h5py 3.8+ (读取 MATLAB v7.3 .mat 文件)

### 7.2 安装依赖

```bash
cd python
pip install -r requirements.txt
```

### 7.3 运行仿真

**方式一：交互式运行（推荐）**

```matlab
% 在 MATLAB 中打开项目根目录
cd('CF_downlink_sim')

% 添加所有子目录到路径
addpath(genpath(pwd))

% 运行仿真
run('run.m')
```

**方式二：命令行运行**

```bash
cd CF_downlink_sim
matlab -batch "addpath(genpath(pwd)); run('run.m')"
```

### 7.4 导出 GNN 训练数据

```matlab
% 方法 1: 使用 test_export.m
run('test_export.m')

% 方法 2: 手动调用
cd('CF_downlink_sim')
addpath(genpath(pwd))
exportTrainingData(fullfile(pwd, 'data', 'gnn_training'), 2)
```

### 7.5 训练 GNN 模型

```bash
cd python

# 集中式训练 (GAT)
python train_gnn.py --data "../data/gnn_training/*.mat" --epochs 300

# 联邦学习训练
python fedavg.py --data "../data/gnn_training/*.mat" --epochs 300

# 消融实验
python train_gnn_ablation.py --ablation all --data "../data/gnn_training/*.mat"

# 快速测试
python train_gnn_ablation.py --ablation gnn_vs_nognn --epochs 10
```

### 7.6 运行 GNN 推理

```bash
cd python

# 测试推理
python inference/__init__.py --model ../models/best_gat_gnn_power.pt --L 100 --K 20
```

### 7.7 自定义参数运行

```matlab
cd('CF_downlink_sim')
addpath(genpath(pwd))

% 修改参数
params = getDefaultParams();
params.L = 50;           % 减少 AP 数量
params.K = 10;           % 减少 UE 数量
params.numScenarios = 1; % 减少场景数
params.isSaveFig = 1;    % 保存图表

% 运行仿真
Combined_Downlink_Sim(params)
```

---

## 8. 配置参数说明

### 8.1 仿真参数分组

| 分组 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| **信道** | `L` | 100 | AP 数量 |
| | `N` | 4 | 每 AP 天线数 |
| | `K` | 20 | UE 数量 |
| | `tau_c` | 200 | 相干长度 (符号) |
| | `tau_p` | 10 | 导频长度 |
| | `nbrOfRealizations` | 10 | 信道实现数 |
| | `numScenarios` | 2 | 随机拓扑场景数 |
| | `seed` | 42 | 随机种子 |
| **功率控制** | `power_control` | 1 | 0=全功率, 1=DPC, 2=EPA |
| | `pilot_power_control` | 0 | 导频功率控制 |
| | `Pt` | 200e-3 | 总下行发射功率 (W) |
| **噪声** | `noiseFigure` | 7 | 噪声系数 (dB) |
| | `bandwidth` | 20e6 | 系统带宽 (Hz) |
| **预编码** | `preLogFactor` | 1 | 预对数因子 |
| **DCC** | `dcc_factor` | 0.1 | DCC 比例 |
| **鲁棒** | `sigma_e` | 0.1 | CSI 误差标准差 |
| | `nIter` | 10 | Robust-MMSE 迭代次数 |
| **PSO** | `pso.nParticles` | 20 | 粒子数 |
| | `pso.maxIter` | 50 | 最大迭代 |
| | `pso.w` | 0.7 | 惯性权重 |
| | `pso.c1` | 1.5 | 认知系数 |
| | `pso.c2` | 1.5 | 社会系数 |
| **输出** | `isSaveFig` | 0 | 是否保存图表 |
| | `isSaveData` | 1 | 是否保存数据 |

### 8.2 Python 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 300 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr_max` | 1e-3 | 最大学习率 (OneCycleLR) |
| `--hidden_dim` | 128 | 隐藏层维度 |
| `--num_heads` | 4 | GAT 注意力头数 |
| `--num_layers` | 3 | GAT 层数 |
| `--dropout` | 0.1 | Dropout 比例 |
| `--val_split` | 0.15 | 验证集比例 |
| `--patience` | 50 | 早停耐心值 |
| `--fedavg_rounds` | 50 | FedAvg 轮数 |
| `--fedavg_frac` | 0.2 | FedAvg 客户端选择比例 |
| `--fedavg_local_epochs` | 1 | FedAvg 本地训练轮数 |
| `--fedavg_lr` | 5e-4 | FedAvg 学习率 |

---

## 9. 算法对比体系

### 9.1 预编码方案

| 方案 | 协调需求 | CSI 需求 | 性能 |
|------|----------|----------|------|
| MR | 无 | 本地 CSI | 基线 |
| L-MMSE (本地) | 低 | 本地 CSI | 接近全局 |
| L-MMSE (全局) | 高 | 全局 CSI | 最优 |
| Robust-MMSE | 高 | 全局 CSI + 误差 | 鲁棒最优 |

### 9.2 功率分配方案

| 方案 | 复杂度 | 通信开销 | 性能 |
|------|--------|----------|------|
| Baseline (DPC) | O(LK) | 无 | 理想上界 |
| Random | O(LK) | 无 | 随机基线 |
| EPA | O(LK) | 无 | 均匀分配 |
| PSO | O(nParticles × iter) | AP 间同步 | 迭代优化 |
| WMMSE | O(iter × LK²) | AP 间同步 | 迭代优化 |
| GNN | O(forward) | 无 | 推理快速 |
| GNN+FL | O(forward) | 无 | 联邦学习 |

### 9.3 接入模式

| 模式 | 说明 |
|------|------|
| All | 所有 AP 服务所有 UE (全连接) |
| DCC | 动态协作簇，仅部分 AP-UE 保持连接 |

---

## 10. 开发约定与注意事项

### 10.1 代码风格

- **MATLAB**：函数命名采用 `camelCase`，变量命名采用 `camelCase`
- **Python**：遵循 PEP 8，类命名采用 `PascalCase`，函数命名采用 `snake_case`

### 10.2 文件组织

- 每个模块放在独立目录中
- 函数与文件名保持一致
- 配置参数集中管理在 `getDefaultParams.m`

### 10.3 数据交换

- MATLAB 与 Python 通过 `.mat` 文件交换数据
- 使用 `h5py` 读取 MATLAB v7.3 格式的 `.mat` 文件
- 训练数据通过 `exportTrainingData.m` 导出

### 10.4 模型保存

- MATLAB 仿真结果保存为 `.mat` 文件
- Python 模型保存为 `.pt` 文件 (PyTorch checkpoint)
- 包含模型状态字典、训练配置、评估指标

### 10.5 版本控制

- `.gitignore` 忽略 `*.asv`, `*.m~`, `slprj/`, `*.slxc` 等 MATLAB 临时文件
- 忽略 `*.mat` 数据文件和 `models/*.pt` 模型文件
- 忽略 `figures/` 输出目录

### 10.6 已知问题

- **inference 模块导入**：[inference/__init__.py](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/python/inference/__init__.py) 中 `from models import ...` 和 `from data import ...` 引用了尚未拆分为独立包的模块（模型定义仍在 `train_gnn.py` 中，数据集在 `dataset.py` 中）。MATLAB 端的 [computeRhoGNN.m](file:///c:/Users/Admin/Documents/%E4%B8%AA%E4%BA%BA%E8%B5%84%E6%96%99/CF_downlink_sim/power_allocation/computeRhoGNN.m) 通过 `py.importlib.import_module('train_gnn')` 直接加载，绕过了此问题。

### 10.7 注意事项

1. **信道模型**：基于 3GPP TR 38.901 Indoor-Hotspot，修改路径损耗模型需同步更新
2. **导频复用**：默认导频复用距离为 100m，修改 `reuseDistance` 需重新评估导频污染
3. **DCC 因子**：`dcc_factor=0.1` 表示仅 10% UE 保持连接，修改此参数会显著影响性能
4. **PSO 收敛**：默认 50 次迭代，对于大规模场景可能需要增加
5. **GNN 推理**：通过 `pyrunfile` 调用 Python，需确保 MATLAB 和 Python 环境正确配置
6. **数据增强**：`exportTrainingData` 中的 `numAugment` 参数控制增强倍数，增加可提升训练效果但会增加数据量
7. **随机种子**：`seed=42` 用于保证结果可复现，修改种子会影响仿真结果
