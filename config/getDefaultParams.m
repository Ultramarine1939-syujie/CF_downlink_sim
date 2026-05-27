function params = getDefaultParams()
%GETDEFAULTPARAMS 项目统一实验参数配置文件。
%
% 修改实验条件时，优先只改本文件。主仿真
% main/Combined_Downlink_Sim.m 与训练数据导出
% power_allocation/exportTrainingData.m 都会从这里读取参数。
%
% 推荐修改入口：
%   1. params.simulation.numScenarios
%   2. params.simulation.nbrOfRealizations
%   3. params.power.SNR_dB
%   4. params.csi.sigma_e
%   5. params.training.nSnapshotsPerSNR

params = struct();

%% ================= 1. 系统规模参数 =================
% L: AP 数量；K: UE 数量；N: 每个 AP 的天线数。
% 注意：当前 Python GNN/Local-GNN 默认按 L=100, K=20 训练。
% 若修改 L/K，需要重新导出训练数据并重新训练模型。
params.system = struct();
params.system.L = 100;
params.system.K = 20;
params.system.N = 1;

% tau_c: 相干块长度；tau_p: 导频长度。
% tau_p 不能大于 tau_c。
params.system.tau_c = 200;
params.system.tau_p = 10;

%% ================= 2. 信道与网络拓扑参数 =================
% ASD_varphi / ASD_theta: 方位角/俯仰角角度扩展。
% generateSetup.m 当前主要显式使用这两个参数。
params.channel = struct();
params.channel.ASD_varphi = deg2rad(15);
params.channel.ASD_theta = deg2rad(15);

% 以下参数作为集中配置保留，便于后续扩展 generateSetup.m 时统一读取。
params.channel.squareLength = 1000;      % 仿真区域边长，单位 m
params.channel.B = 20e6;                 % 系统带宽，单位 Hz
params.channel.noiseFigure = 7;          % 噪声系数，单位 dB
params.channel.alpha = 36.7;             % 路损斜率参数
params.channel.constantTerm = -30.5;     % 路损常数项
params.channel.sigma_sf = 4;             % 阴影衰落标准差，单位 dB
params.channel.decorr = 9;               % 阴影衰落去相关距离，单位 m
params.channel.distanceVertical = 10;    % AP 与 UE 的垂直距离，单位 m
params.channel.antennaSpacing = 0.5;     % 阵元间距，单位波长

%% ================= 3. 功率与 SNR 扫描参数 =================
params.power = struct();
params.power.p = 100;                    % 上行导频发射功率/归一化功率参数
params.power.SNR_dB = 5:5:30;           % 主仿真 SNR 扫描范围

% 训练数据默认使用同一 SNR 范围，保证训练/测试操作点一致。
% 若要测试外推能力，可只改 params.power.SNR_dB 后重新跑主仿真，
% 不重新训练模型，用于观察模型在训练范围外的泛化表现。
params.power.trainingSNR_dB = params.power.SNR_dB;

%% ================= 4. CSI 误差与鲁棒预编码参数 =================
params.csi = struct();
params.csi.sigma_e = 0.3;                % 信道估计误差标准差
params.csi.nIter = 5;                    % Robust MMSE 预编码迭代次数

%% ================= 5. 最终仿真评估规模 =================
% numScenarios: 独立拓扑场景数。
% nbrOfRealizations: 每个拓扑场景下的信道实现数。
% 这两个参数越大，结果越稳定，但运行时间和缓存体积也会显著增加。
params.simulation = struct();
params.simulation.numScenarios = 10;
params.simulation.nbrOfRealizations = 200;
params.simulation.seed = 42;

% AP 接入模式。当前项目统一使用 DCC 动态 AP-UE 关联。
params.simulation.accessModes = {'DCC'};

%% ================= 6. 主仿真运行控制 =================
params.runtime = struct();
params.runtime.runStage = 3;             % 1=传统/参考方法, 2=仅 GNN, 3=全部
params.runtime.useCache = true;          % 是否启用缓存
params.runtime.verbose = true;           % 输出主要进度与每轮结果
params.runtime.verboseAlgo = false;      % 是否输出逐算法 ESR 细节

%% ================= 7. 输出文件控制 =================
params.output = struct();
params.output.isSaveFig = true;          % 是否保存图像
params.output.isSaveData = true;         % 是否保存 .mat/.csv 数据
params.output.savePath = 'Imgs';         % 相对 main/ 的图像目录
params.output.dataPath = 'SimulationData'; % 相对 main/ 的数据与缓存目录
params.output.cleanOldFigures = true;    % 每次主仿真开始前清空旧图

%% ================= 8. 功率分配与 WMMSE 参数 =================
params.wmmse = struct();
params.wmmse.maxIter = 30;               % 训练数据标签生成时的 WMMSE 最大迭代
params.wmmse.simMaxIter = 20;            % 主仿真 WMMSE 最大迭代
params.wmmse.tol = 1e-4;                 % WMMSE 收敛阈值

params.dwmmse = struct();
params.dwmmse.rounds = 5;                % D-WMMSE 固定消息交换轮次
params.dwmmse.damping = 0.6;             % D-WMMSE 阻尼系数

params.fpcp = struct();
params.fpcp.alpha = -1.0;                % FPCP 分数补偿指数；0=EPA，负值在下行 sum-rate 中偏向强链路

%% ================= 9. 同步时延消融参数 =================
params.syncAblation = struct();
params.syncAblation.enable = true;       % 是否输出 PA/PC 同步时延消融图
params.syncAblation.fronthaulMbps = 1000; % 控制/前传链路速率，单位 Mbps
params.syncAblation.syncRttMs = 0.05;    % 单轮同步控制往返时延，单位 ms
params.syncAblation.dccPayloadRatio = 0.35; % DCC 同步载荷折算比例
params.syncAblation.includeComputeTime = true; % 控制时延是否叠加 PA 计算时间

%% ================= 10. GNN 模型路径与训练超参数 =================
params.gnn = struct();
params.gnn.fullModelFile = fullfile('models', 'best_gat_gnn_power.pt');
params.gnn.localModelFile = fullfile('models', 'best_local_gnn_power.pt');
params.gnn.dcgnnModelFile = fullfile('models', 'best_dcgnn_power.pt');
params.gnn.ugnnModelFile = fullfile('models', 'best_ugnn_power.pt');
params.gnn.hidden_dim = 128;
params.gnn.num_heads = 4;
params.gnn.num_layers = 3;
params.gnn.dropout = 0.1;
params.gnn.batch_size = 32;
params.gnn.lr_max = 1e-3;
params.gnn.val_split = 0.15;
params.gnn.patience = 50;

params.rl = struct();
params.rl.dqnModelFile = fullfile('models', 'best_dqn_power.pt');
params.rl.ddpgModelFile = fullfile('models', 'best_ddpg_power.pt');

%% ================= 11. 训练数据导出参数 =================
params.training = struct();
params.training.nSnapshotsPerSNR = 500;  % 每个 SNR、每个接入模式导出的训练快照数
params.training.nbrOfRealizations = params.simulation.nbrOfRealizations;
params.training.nbrOfSetups = 2;         % 每个训练快照内部 generateSetup 的 setup 数
params.training.useParallel = true;      % 导出训练数据时是否尝试 parfor
params.training.SNR_dB = params.power.trainingSNR_dB;
params.training.accessModes = params.simulation.accessModes;

% 数据增强：随机丢弃部分 AP-UE 连接，并扰动 sigma_e。
params.training.dataAug_dropRate_min = 0.1;
params.training.dataAug_dropRate_max = 0.3;
params.training.dataAug_sigma_e_var = 0.2;

%% ================= 12. 自动校验 =================
validateDefaultParams(params);
end

function validateDefaultParams(params)
%VALIDATEDEFAULTPARAMS 对统一配置做基本合法性检查。

assert(params.system.L > 0, 'params.system.L must be positive');
assert(params.system.K > 0, 'params.system.K must be positive');
assert(params.system.N > 0, 'params.system.N must be positive');
assert(params.system.tau_c >= params.system.tau_p, 'tau_c must be >= tau_p');
assert(params.system.tau_p > 0, 'tau_p must be positive');

assert(params.csi.sigma_e >= 0 && params.csi.sigma_e <= 1, ...
    'params.csi.sigma_e must be in [0, 1]');
assert(params.simulation.numScenarios >= 1, ...
    'params.simulation.numScenarios must be >= 1');
assert(params.simulation.nbrOfRealizations >= 1, ...
    'params.simulation.nbrOfRealizations must be >= 1');
assert(params.training.nSnapshotsPerSNR >= 1, ...
    'params.training.nSnapshotsPerSNR must be >= 1');
assert(params.training.nbrOfRealizations >= 1, ...
    'params.training.nbrOfRealizations must be >= 1');
assert(params.training.dataAug_dropRate_min >= 0 && ...
    params.training.dataAug_dropRate_max <= 1 && ...
    params.training.dataAug_dropRate_min <= params.training.dataAug_dropRate_max, ...
    'training drop-rate range must be inside [0, 1]');
assert(params.syncAblation.fronthaulMbps > 0, ...
    'params.syncAblation.fronthaulMbps must be positive');
end
