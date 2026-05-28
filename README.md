# CF_downlink_sim

Cell-Free Massive MIMO 下行链路仿真项目，用 MATLAB 完成无线链路仿真与性能评估，用 Python 训练和推理 GNN/RL 功率分配模型。

当前项目已经按语言分层整理：

- `matlab/`：MATLAB 仿真源码。
- `python/`：Python 训练和推理源码。
- `data/`：训练数据。
- `models/`：模型权重。
- `main/SimulationData/`：仿真缓存和数值结果。
- `main/Imgs/`：仿真图像输出。

## 快速启动

Windows 下只保留一个启动脚本：

```bat
start_project.bat
```

双击或在命令行运行后，会看到菜单：

- `1` 快速完整流程：小数据导出、训练、烟测、仿真。
- `2` 完整实验流程：按 `matlab/config/getDefaultParams.m` 的配置运行。
- `3` 仅运行主仿真。
- `4` 仅重绘图。
- `5` 仅烟测。
- `6` 验证最新训练数据集。
- `7` 快速导出训练数据。
- `8` 打开 MATLAB GUI 并自动加入项目源码路径。
- `C` 环境检查。

也可以直接带参数运行：

```bat
start_project.bat C /y
start_project.bat 3 /y
start_project.bat 6 /y
start_project.bat 1 /y
```

其中 `/y` 表示运行结束后不暂停，适合命令行或自动化调用。重新训练已有模型时可加 `/retrain`。

## 环境要求

- Windows。
- MATLAB，命令行中能执行 `matlab`。
- Python 3.10 或更高版本。
- Python 依赖见 `python/requirements.txt`。
- 训练 GNN/RL 模型时建议有 CUDA GPU；仅仿真和烟测可以不用 GPU。

安装 Python 依赖：

```bat
cd python
pip install -r requirements.txt
```

## 常用流程

环境检查：

```bat
start_project.bat C /y
```

只运行主仿真：

```bat
start_project.bat 3 /y
```

验证最新训练数据：

```bat
start_project.bat 6 /y
```

快速导出很小的数据集：

```bat
start_project.bat 7 /y
```

快速完整流程：

```bat
start_project.bat 1 /y
```

完整实验流程：

```bat
start_project.bat 2 /y
```

## 核心参数在哪里改

优先修改：

```text
matlab/config/getDefaultParams.m
```

常用参数包括：

- `params.system.L`：AP 数量。
- `params.system.K`：UE 数量。
- `params.system.N`：每个 AP 的天线数。
- `params.power.SNR_dB`：主仿真的 SNR 扫描范围。
- `params.csi.sigma_e`：CSI 误差强度。
- `params.simulation.numScenarios`：随机场景数。
- `params.simulation.nbrOfRealizations`：每个场景的信道实现数。
- `params.runtime.runStage`：运行阶段，`1` 传统方法，`2` 仅学习方法，`3` 全部。
- `params.runtime.useCache`：是否启用缓存。
- `params.training.nSnapshotsPerSNR`：训练数据导出规模。

路径统一由下面文件管理：

```text
matlab/config/getProjectPaths.m
python/project_paths.py
```

## test_export.m 的作用

`test_export.m` 是 MATLAB 侧的快速数据导出脚本。它会：

1. 进入项目根目录。
2. 加入 `matlab/` 源码路径。
3. 调用 `exportTrainingData(paths.trainingData, 2)`。

也就是说，它只导出每个 SNR 的极小样本，用来检查数据导出链路、路径配置、MATLAB 函数依赖是否正常。它不是正式训练数据生成脚本，也不会运行完整仿真。

等价菜单入口是：

```bat
start_project.bat 7 /y
```

## 输出位置

- 图像：`main/Imgs/`
- 仿真数据和缓存：`main/SimulationData/`
- 训练数据：`data/gnn_training/`
- 模型权重：`models/`

## 进一步阅读

- `PROJECT_STRUCTURE.md`：当前目录结构。
- `CODE_WIKI.md`：模块职责和代码关系。
- `docs/`：按当前代码重写后的运行、模块、数据和维护文档。
