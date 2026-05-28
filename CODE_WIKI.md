# CF_downlink_sim 代码百科

本文档描述当前代码状态。项目已经整理为 `matlab/` 和 `python/` 两个源码区，根目录只保留启动入口、说明文档和少量工具脚本。

## 1. 总体架构

项目分为四层：

1. Windows 启动层：`start_project.bat`。
2. MATLAB 仿真层：`matlab/`。
3. Python 学习层：`python/`。
4. 生成物层：`data/`、`models/`、`main/SimulationData/`、`main/Imgs/`。

数据流如下：

```text
MATLAB exportTrainingData
        |
        v
data/gnn_training/*.mat
        |
        v
Python train_*.py
        |
        v
models/*.pt
        |
        v
MATLAB computeRhoGNN / computeRhoLocalGNN / computeRhoRL
        |
        v
main/Combined_Downlink_Sim.m
        |
        v
main/SimulationData + main/Imgs
```

## 2. 启动入口

### start_project.bat

唯一 Windows `.bat` 启动脚本。菜单能力包括：

- 快速完整流程。
- 完整实验流程。
- 仅主仿真。
- 仅重绘图。
- 仅烟测。
- 验证训练数据。
- 快速导出训练数据。
- 打开 MATLAB GUI。
- 环境检查。

### run.m

MATLAB 主入口。它只做三件事：

1. 找到项目根目录。
2. 将 `matlab/` 加入 MATLAB 路径。
3. 进入 `matlab/main/` 并运行 `Combined_Downlink_Sim`。

### test_export.m

用于快速检查数据导出链路。它调用：

```matlab
exportTrainingData(paths.trainingData, 2)
```

因此它只生成很小的数据集，不用于正式实验。

## 3. MATLAB 代码

### matlab/config

- `getDefaultParams.m`：统一参数入口。主仿真和数据导出都会读取它。
- `getProjectPaths.m`：统一路径入口。MATLAB 代码不要重复硬编码目录。
- `SimulationLogger.m`：日志辅助类。

### matlab/main

- `Combined_Downlink_Sim.m`：主仿真脚本，负责完整实验循环。
- `runUGNNGuardExperiments.m`：U-GNN 熵保护消融实验。

`Combined_Downlink_Sim.m` 的主流程：

1. 读取参数和路径。
2. 注册功率分配方法和预编码方法。
3. 生成或读取场景缓存。
4. 按 SNR 扫描。
5. 计算传统方法结果。
6. 调用 Python 推理学习类方法。
7. 计算 SE/ESR。
8. 保存缓存和结果。
9. 绘图并输出表格。

### matlab/channel

- `generateSetup.m`：生成 AP/UE 拓扑、大尺度衰落、DCC 连接矩阵等。
- `functionChannelEstimates.m`：生成信道估计。
- `functionRlocalscattering.m`：局部散射模型。

### matlab/precoding

- `functionPrecoding_MR.m`
- `functionPrecoding_LMMSE.m`
- `functionPrecoding_LMMSE_global.m`
- `functionPrecoding_RobustMMSE.m`

### matlab/power_allocation

传统与学习方法都集中在这里：

- `computeRhoEPA.m`
- `computeRhoFPCP.m`
- `computeRhoDist.m`
- `computeRhoDistributedWMMSE.m`
- `computeRhoWMMSE.m`
- `computeRhoGNN.m`
- `computeRhoLocalGNN.m`
- `computeRhoUGNN.m`
- `computeRhoRL.m`
- `exportTrainingData.m`

其中 `computeRhoGNN.m`、`computeRhoLocalGNN.m` 和 `computeRhoRL.m` 会把 `python/` 加入 Python 路径，并调用对应运行时模块。

### matlab/se_calculation

负责不同预编码方案下的频谱效率计算。

### matlab/visualization

负责最终结果打印、ESR 曲线、同步时延消融图和场景图。

## 4. Python 代码

### 训练入口

- `train_gnn.py`：GNN 和 DCGNN。
- `train_gnn_unsup.py`：U-GNN。
- `train_gnn_local.py`：Local-GNN。
- `train_rl_power.py`：DQN/DDPG。

### 推理运行时

- `gnn_runtime.py`：集中式 GNN/DCGNN/U-GNN 推理。
- `gnn_runtime_local.py`：AP-local 推理。
- `rl_runtime.py`：DQN/DDPG 推理。

这些模块会缓存已加载模型，减少 MATLAB 多次调用时的重复开销。

### 数据工具

- `dataset.py`：读取 MATLAB v7.3 `.mat`，统一维度为 `(L, K, snapshots)`。
- `project_paths.py`：Python 侧路径中心。

## 5. 生成物

- `data/gnn_training/*.mat`：训练数据。
- `models/*.pt`：训练出的模型。
- `main/SimulationData/*.mat`、`*.csv`：仿真缓存和结果。
- `main/Imgs/*.png`、`*.fig`：图像。

## 6. 维护原则

- 新 MATLAB 函数放入 `matlab/` 下对应模块。
- 新 Python 训练或推理代码放入 `python/`。
- 修改实验规模优先改 `matlab/config/getDefaultParams.m`。
- 修改目录结构优先改 `matlab/config/getProjectPaths.m` 和 `python/project_paths.py`。
- 不要在算法代码中散落绝对路径。
- 大文件生成物不要混入源码维护。
