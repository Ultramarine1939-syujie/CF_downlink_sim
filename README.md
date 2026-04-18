# Cell-Free 下行链路仿真 (CF Downlink Simulation)

比较 L-MMSE 和 Robust MMSE 两种预编码在 All-UE 和 DCC-UE 两种接入模式下下行链路频谱效率（ESR）的仿真项目。

## 项目结构

```
CF_downlink_sim/
├── Combined_Downlink_Sim.m              主仿真程序：参数配置 + 仿真循环 + 调用输出函数
├── generateSetup.m                      生成 AP/UE 分布、大尺度衰落、相关矩阵、导频分配
├── functionChannelEstimates.m           信道估计（MMSE 估计 + 误差协方差矩阵 C）
├── functionComputeSE_downlink_LMMSE.m   L-MMSE 预编码 SE 计算
├── functionComputeSE_downlink_RobustMMSE.m  Robust MMSE 预编码 SE 计算
├── functionRlocalscattering.m           局部散射模型空间相关矩阵
├── computeRhoDist.m                     功率分配系数计算
├── printSimConfig.m                    打印仿真配置信息表
├── printFinalResults.m                 打印最终结果汇总表
├── plotESRResults.m                    绘制 ESR 曲线并保存
├── plotScenarioSetup.m                 绘制场景布局图（AP/UE 分布）
├── Imgs/                               图像输出目录（自动创建）
└── SimulationData/                      数据输出目录（自动创建）
```

## 函数概览

### 核心仿真函数

| 文件 | 描述 |
|------|------|
| `Combined_Downlink_Sim.m` | 主脚本。配置参数、运行场景×SNR 双重循环、引入 CSI 误差、累计结果、调用输出函数 |
| `generateSetup.m` | 生成 AP/UE 随机分布、路径损耗、阴影衰落、接入矩阵 D、导频分配 |
| `functionChannelEstimates.m` | 根据相关矩阵生成真实信道，执行上行导频估计，输出信道估计 Hhat 和误差协方差 C |
| `functionComputeSE_downlink_LMMSE.m` | 经典 L-MMSE 预编码，计算各 UE 的 SE |
| `functionComputeSE_downlink_RobustMMSE.m` | Robust MMSE 预编码，考虑信道估计误差 σ_e，迭代 nIter 次优化 |

### 辅助输出函数

| 文件 | 描述 |
|------|------|
| `computeRhoDist.m` | 根据接入矩阵 D、大尺度增益、总功率 Pt 计算功率分配系数 ρ_dist |
| `printSimConfig.m` | 打印仿真参数配置信息表（表格形式） |
| `printFinalResults.m` | 打印仿真完成后的性能汇总表（Min/Max ESR + 统计信息） |
| `plotESRResults.m` | 绘制 4 条 ESR 曲线并保存为 .fig / .png，数据保存为 .mat |
| `plotScenarioSetup.m` | 绘制 AP/UE 场景布局图并保存 |

### 底层工具函数

| 文件 | 描述 |
|------|------|
| `functionRlocalscattering.m` | 基于局部散射模型生成空间相关矩阵 R |

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `L` | 100 | 接入点（AP）数量 |
| `K` | 20 | 用户（UE）数量 |
| `N` | 1 | 每个 AP 的天线数 |
| `tau_c` | 200 | 相干时间符号数 |
| `tau_p` | 10 | 导频长度 |
| `sigma_e` | 0.3 | CSI 误差标准差 |
| `nIter` | 5 | Robust MMSE 迭代次数 |
| `numScenarios` | 10 | 仿真场景数（独立信道实现次数） |
| `nbrOfRealizations` | 200 | 每场景的信道样本数 |
| `SNR_dB` | 5:5:30 | SNR 扫描范围（dB） |
| `numScenariosToPlot` | 1 | 需要绘制布局的场景数 |

## 使用方法

### 1. 环境要求
- MATLAB R2020b 或更高版本

### 2. 运行仿真
```matlab
Combined_Downlink_Sim
```

### 3. 配置说明
在 `Combined_Downlink_Sim.m` 的 `Initialization` 部分修改参数：
- 仿真规模：`numScenarios`、`nbrOfRealizations`、`SNR_dB`
- 系统配置：`L`、`K`、`N`、`tau_c`、`tau_p`
- Robust 参数：`sigma_e`、`nIter`
- 输出控制：`isSaveFig`、`isSaveData`、`numScenariosToPlot`

### 4. 查看输出

**图像**（`Imgs/`）：
- `Scenario_X_Layout.png/.fig` — 第 X 个场景的 AP/UE 布局
- `Averaged_ESR_Results.png/.fig` — 4 种算法 ESR 对比曲线

**数据**（`SimulationData/`）：
- `Scenario_X_Positions.mat` — 各场景 AP/UE 坐标
- `Simulation_Results_Data.mat` — 完整 ESR 数据

## 状态显示

仿真运行时显示详细进度信息：

```
╔══════════════════════════════════════════════════════════════╗
║            Downlink Simulation - Status Display              ║
╠══════════════════════════════════════════════════════════════╣
║  Simulation Configuration:                                   ║
║    • Number of APs (L):         100                          ║
║    • Number of UEs (K):         20                           ║
║    • Scenarios:                 10                           ║
║    • SNR range (dB):            [5 10 15 20 25 30]            ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  SCENARIO  1 / 10  [  0.0% ]                              ║
╚══════════════════════════════════════════════════════════════╝
  ┌─ [1/5] Generating scenario setup (AP/UE distribution)...
  │   └─ Done
  ┌─ [2/5] Performing channel estimation...
  │   └─ Realizations: 200, Pilots: 10
  │   └─ Done
  ┌─ [3/5] Introducing CSI error (sigma_e = 0.30)...
  │   └─ Done
  ┌─ [4/5] Running SNR sweep (6 points)...
  │   ┌─ SNR  1/ 6 (5.0 dB) - All-UE
  │   │   ├─ L-MMSE:      SE=12.8593
  │   │   └─ Robust-MMSE: SE=14.6987
  │   └─ SNR  1/ 6 (5.0 dB) - DCC-UE
  │       ├─ L-MMSE:      SE=14.0303
  │       └─ Robust-MMSE: SE=15.9874
  │  [============                    ]  8.3%
  └─ SNR sweep completed
```

设置 `SHOW_DETAILED_STATUS = false` 可关闭详细步骤输出。

## 算法说明

- **L-MMSE**：经典线性最小均方误差预编码，不考虑信道估计误差
- **Robust MMSE**：考虑信道估计误差的鲁棒预编码，通过迭代优化提升性能

接入模式：
- **All-UE**：所有 AP 为所有 UE 服务（每个 AP 连接所有 UE）
- **DCC**：分布式协作聚类接入（每个 UE 仅被部分 AP 服务，根据信道强度选择）

## 注意事项

- 每次运行会自动清空 `Imgs/` 和 `SimulationData/` 文件夹，重要结果请预先备份
- `plotScenarioSetup.m` 使用非交互式 figure，适合批量保存
