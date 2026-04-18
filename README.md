# 下行链路仿真项目 (Downlink Simulation Project)

本项目用于多天线系统（Cell-free Massive MIMO）下行链路的频谱效率（SE）仿真，支持多种算法（L-MMSE, Robust MMSE）以及不同接入场景（All-AP, DCC）的对比分析。

## 项目结构

```text
.
├── Combined_Downlink_Sim.m           % 主仿真程序：负责流程控制、参数初始化和结果绘制
├── generateSetup.m                   % 功能函数：随机生成 AP 和 UE 的位置及大尺度增益
├── plotScenarioSetup.m               % 功能函数：可视化场景布局，并保存图像和位置数据
├── functionChannelEstimates.m        % 功能函数：生成信道实现及其 MMSE 估计
├── functionComputeSE_downlink_LMMSE.m % 功能函数：计算 L-MMSE 算法下的下行链路频谱效率
├── functionComputeSE_downlink_RobustMMSE.m % 功能函数：计算鲁棒 MMSE 算法下的频谱效率
├── functionRlocalscattering.m        % 功能函数：生成局部散射模型下的空间相关矩阵
├── Figures/                          % 输出目录：存放仿真的图像结果（自动生成）
└── SimulationData/                   % 输出目录：存放仿真的 .mat 数据结果（自动生成）
```

## 函数功能简介

1.  **Combined_Downlink_Sim.m**:
    -   设置系统参数（AP数量、UE数量、相干时间等）。
    -   管理仿真循环（Scenario Loop）和 SNR 扫描。
    -   自动清理并创建结果存放文件夹。
    -   绘制最终的平均可达和速率（ESR）曲线。

2.  **plotScenarioSetup.m**:
    -   输入 AP 和 UE 的坐标。
    -   在二维平面上绘制分布图。
    -   根据配置自动将布局图保存为 `.png` 和 `.fig`，并将坐标保存为 `.mat`。

3.  **generateSetup.m**:
    -   模拟方型区域内的随机部署。
    -   计算路径损耗和阴影衰落。
    -   生成导频分配和接入矩阵。

4.  **functionChannelEstimates.m**:
    -   根据空间相关矩阵生成真实信道。
    -   模拟上行导频传输并获取信道估计值。

5.  **算法函数**:
    -   `functionComputeSE_downlink_LMMSE`: 实现经典的 L-MMSE 合并/预编码方案。
    -   `functionComputeSE_downlink_RobustMMSE`: 实现考虑信道估计误差的鲁棒性优化方案。

## 如何使用

### 1. 准备环境
确保您的电脑已安装 MATLAB (建议 2020b 或更高版本)。

### 2. 配置参数
打开 `Combined_Downlink_Sim.m`，在 `Initialization` 部分根据需要修改参数：
-   `L`: 接入点 (AP) 数量。
-   `K`: 用户 (UE) 数量。
-   `numScenarios`: 仿真场景的实验次数。
-   `numScenariosToPlot`: 需要可视化并保存布局的场景数量。
-   `isSaveFig` / `isSaveData`: 是否开启自动保存功能。

### 3. 运行仿真
在 MATLAB 命令行窗口运行：
```matlab
Combined_Downlink_Sim
```

### 4. 查看结果
-   **图像**: 检查 `Figures/` 文件夹，包含 `Scenario_X_Layout` (布局图) 和 `Averaged_ESR_Results` (最终 ESR 曲线)。
-   **数据**: 检查 `SimulationData/` 文件夹，包含位置数据和所有 SNR 点下的数值结果，方便后续分析。

## 注意事项
-   代码在每次运行开始前会**自动清空** `Figures/` 和 `SimulationData/` 文件夹，请注意备份重要的仿真结果。
-   本项目已实现模块化，您可以根据需要替换或增加新的算法函数。
