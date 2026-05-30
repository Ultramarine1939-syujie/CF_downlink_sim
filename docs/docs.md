我静态检查了当前代码，没有跑完整仿真。结论是：这个项目主要实现的是 **Cell-Free Massive MIMO 下行链路** 场景下的功率分配和预编码对比实验。

**场景基础**

默认系统是 `L=100` 个 AP、`K=20` 个 UE、每 AP `N=1` 根天线， coherence block `tau_c=200`、pilot 长度 `tau_p=10`，SNR 扫描 `5:5:30 dB`，每个拓扑场景 200 个信道 realization；主接入模式固定为 `DCC` 动态 AP-UE 关联。配置在 [getDefaultParams.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/config/getDefaultParams.m:22)。

信道/拓扑上，AP 和 UE 随机撒在 `1000m x 1000m` 区域，使用 wrap-around 边界；包含路径损耗、相关阴影衰落、噪声、AP-UE 垂直距离、本地散射空间相关矩阵；导频分配考虑 pilot contamination，DCC 里先给每个 UE 选 master AP，再让每个 AP 对每个 pilot 选择最强 UE 服务。关键逻辑在 [generateSetup.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/channel/generateSetup.m:26)、[generateSetup.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/channel/generateSetup.m:59)、[generateSetup.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/channel/generateSetup.m:82)。

信道估计是 TDD 上行导频 MMSE 估计，真实信道由协方差矩阵 `R` 生成，并输出估计误差协方差 `C` 给 L-MMSE 预编码使用，见 [functionChannelEstimates.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/channel/functionChannelEstimates.m:2)。

**功率分配方法**

主仿真注册了这些 PA 方法，见 [Combined_Downlink_Sim.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/main/Combined_Downlink_Sim.m:89)：

| 方法 | 类型 | 大意 |
|---|---|---|
| `Baseline` / `computeRhoDist` | 分布式传统 | 按大尺度增益平方根在每个 AP 内分功率 |
| `Random` | 分布式基线 | 随机功率分配 |
| `EPA` | 分布式传统 | Equal Power Allocation，每个 AP 平分给服务 UE |
| `FPCP` | 分布式传统 | Fractional Power Control，默认 `alpha=-1`，偏向强链路 |
| `D-WMMSE` | 分布式近似 | 固定轮数、基于大尺度增益和 UE 侧标量聚合的 WMMSE-style 更新，不是集中式精确 WMMSE |
| `WMMSE` | 集中式参考 | 基于等效下行信道、每 AP 功率约束的标准交替 WMMSE |
| `GNN` | 低时延集中式学习 | GAT/PyG 图神经网络，学习 WMMSE per-AP power share |
| `Local-GNN` | 分布式学习 | 每个 AP 只用本地 DCC 行和本地增益特征推理 |
| `DCGNN` | 低时延集中式学习 | 动态图/Top-Z dominant AP-UE 边的 GNN 变体 |
| `U-GNN` | 无教师学习 | 不用 WMMSE 标签，使用可微大尺度 SE/rate proxy 训练 |
| `DQN`, `DDPG` | RL 学习基线 | 使用大尺度 sum-rate reward proxy，不加载 WMMSE 标签 |

几个实现细节也很明确：`WMMSE` 的注释说明它在 deterministic equivalent channel 上做标准 WMMSE，并受每 AP 功率约束约束，见 [computeRhoWMMSE.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/power_allocation/computeRhoWMMSE.m:1)；`D-WMMSE` 明确是 fixed-round distributed WMMSE-style，不是 centralized solver，见 [computeRhoDistributedWMMSE.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/power_allocation/computeRhoDistributedWMMSE.m:1)。GNN 的监督目标是 `rho_WMMSE`，见 [dataset.py](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/python/dataset.py:63)。

**预编码方法**

主仿真注册了 4 类 PC 方法，见 [Combined_Downlink_Sim.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/main/Combined_Downlink_Sim.m:104)：

| 方法 | 类型 | 大意 |
|---|---|---|
| `MR` | 分布式 | Maximum Ratio，直接用 `Hhat` 作为预编码方向 |
| `L-MMSE` | 分布式 | 本地 L-MMSE；每个 AP 只用本 AP 的信道估计和本地误差协方差 |
| `R-MMSE` | 分布式鲁棒 | 考虑 CSI 误差 `sigma_e`，迭代式 robust MMSE 预编码 |
| `L-MMSE-G` | 集中式参考/消融 | global L-MMSE，用所有 AP 的协方差信息作集中式参考 |

对应实现分别在 [functionPrecoding_MR.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/precoding/functionPrecoding_MR.m:1)、[functionPrecoding_LMMSE.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/precoding/functionPrecoding_LMMSE.m:1)、[functionPrecoding_RobustMMSE.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/precoding/functionPrecoding_RobustMMSE.m:1)、[functionPrecoding_LMMSE_global.m](c:/Users/Admin/Documents/个人资料/CF_downlink_sim/matlab/precoding/functionPrecoding_LMMSE_global.m:1)。

一句话概括：这个项目是在 **DCC Cell-Free Massive MIMO 下行链路** 中，把 **MR/L-MMSE/R-MMSE/全局 L-MMSE 预编码** 和 **传统、WMMSE、分布式 WMMSE、GNN/Local-GNN/DCGNN/U-GNN/RL 功率分配** 做系统组合对比，性能指标主要是下行 SE/ESR，并额外统计控制/同步时延。