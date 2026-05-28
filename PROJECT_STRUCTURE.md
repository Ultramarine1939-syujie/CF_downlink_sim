# 项目结构说明

本项目采用“MATLAB 仿真核心 + Python 学习模型”的混合架构。当前结构的目标是让入口、源码、数据、模型和结果各自归位，便于在 Windows 下启动、调试和复现实验。

## 根目录入口

- `start_project.bat`：唯一 Windows 启动脚本。提供菜单，可执行环境检查、快速完整流程、完整实验流程、仅仿真、仅重绘图、烟测、数据验证、快速导出和打开 MATLAB GUI。
- `run.m`：MATLAB 主入口。只负责加入 `matlab/` 源码路径并启动主仿真。
- `test_export.m`：MATLAB 数据导出快速检查脚本，调用 `exportTrainingData(..., 2)` 生成很小的数据集，用于验证导出链路。
- `validate_dataset.py`：Python 数据集验证工具。无参数运行时会自动检查 `data/gnn_training/` 下最新的训练数据。
- `README.md`：快速使用说明。
- `CODE_WIKI.md`：代码级说明和模块关系。

## MATLAB 源码

MATLAB 代码统一放在 `matlab/` 下。

- `matlab/config/`
  - `getDefaultParams.m`：实验参数中心，包括系统规模、SNR、CSI 误差、缓存、输出、GNN/RL 模型文件名等。
  - `getProjectPaths.m`：项目路径中心。目录调整时优先修改这里。
  - `SimulationLogger.m`：日志辅助类。
- `matlab/main/`
  - `Combined_Downlink_Sim.m`：主仿真流程，负责场景生成、信道估计、预编码、功率分配、SE 计算、缓存和绘图。
  - `runUGNNGuardExperiments.m`：U-GNN 熵保护相关消融实验。
- `matlab/channel/`：拓扑生成、信道估计和局部散射模型。
- `matlab/precoding/`：MR、L-MMSE、全局 L-MMSE、Robust-MMSE 预编码。
- `matlab/power_allocation/`：Baseline、EPA、FPCP、D-WMMSE、WMMSE、GNN、Local-GNN、U-GNN、DQN/DDPG 等功率分配方法。
- `matlab/se_calculation/`：下行频谱效率计算。
- `matlab/visualization/`：结果打印、ESR 图、同步时延消融图和场景图。

## Python 源码

Python 代码统一放在 `python/` 下。

- `python/project_paths.py`：Python 侧路径中心。
- `python/dataset.py`：读取 MATLAB v7.3 `.mat` 训练数据并构造 GNN 样本。
- `python/train_gnn.py`：监督式 GNN/DCGNN 训练入口。
- `python/train_gnn_unsup.py`：无监督 U-GNN 训练入口。
- `python/train_gnn_local.py`：AP-local GNN/MLP 训练入口。
- `python/train_rl_power.py`：DQN/DDPG 强化学习功率分配基线。
- `python/gnn_runtime.py`、`python/gnn_runtime_local.py`、`python/rl_runtime.py`：供 MATLAB 调用的缓存推理运行时。
- `python/models/`、`python/data/`、`python/training/`、`python/utils/`：兼容性封装和辅助模块。

## 数据、模型和结果

- `data/gnn_training/`：MATLAB 导出的 GNN 训练数据。
- `models/`：PyTorch 模型权重和训练输出。
- `main/SimulationData/`：仿真结果、缓存、CSV 表格。
- `main/Imgs/`：生成的 `.png` 和 `.fig` 图像。

这些目录通常包含大文件，默认应视为生成物。日常维护时优先修改源码、配置和文档，不要手动搬动缓存和模型，除非明确要归档某次实验结果。
