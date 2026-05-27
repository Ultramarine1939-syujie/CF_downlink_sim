%% ================= CF_downlink_sim - 主仿真引擎 (v3.0) =================
%% 功能: Cell-Free Massive MIMO 下行链路仿真
%% 核心目标: 通过系统性地对比预编码×功率分配组合, 找出最优方案
%% v3.0 新增: 分阶段运行 + 结果缓存 (避免重复运行 WMMSE 等耗时计算)
% ==========================================================================

close all; clear;

%% ================= 统一参数入口 =================
% 所有实验条件优先在 config/getDefaultParams.m 中修改。
scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);
addpath(genpath(rootDir));
params = getDefaultParams();

% ---------- 系统参数 ----------
L = params.system.L;
K = params.system.K;
N = params.system.N;
tau_c = params.system.tau_c;
tau_p = params.system.tau_p;
ASD_varphi = params.channel.ASD_varphi;
ASD_theta  = params.channel.ASD_theta;
p = params.power.p;
SNR_dB = params.power.SNR_dB;
num_snr = length(SNR_dB);

% ---------- CSI误差参数 ----------
sigma_e = params.csi.sigma_e;
nIter = params.csi.nIter;

% ---------- 仿真规模 ----------
numScenarios = params.simulation.numScenarios;
nbrOfRealizations = params.simulation.nbrOfRealizations;

% ---------- 输出控制 ----------
isSaveFig = params.output.isSaveFig;
isSaveData = params.output.isSaveData;

% ---------- 显示模式 ----------
VERBOSE = params.runtime.verbose;
VERBOSE_ALGO = params.runtime.verboseAlgo;

% ---------- 缓存 & 阶段控制 (v3.0 新增) ----------
runStage = params.runtime.runStage; % 1=传统/参考方法, 2=仅GNN推理, 3=全部
useCache = params.runtime.useCache;

% ---------- 预编码 x 功率分配同步时延消融 ----------
enableSyncAblation = params.syncAblation.enable;
syncAblation = params.syncAblation;
syncAblation.dwmmseRounds = params.dwmmse.rounds;
syncAblation.wmmseRounds = params.wmmse.simMaxIter;

%% ================= 路径配置 =================
cd(scriptDir);
savePath = fullfile(scriptDir, params.output.savePath);
dataPath = fullfile(scriptDir, params.output.dataPath);
gnnLocalModelPath = fullfile(rootDir, params.gnn.fullModelFile);
localGnnModelPath = fullfile(rootDir, params.gnn.localModelFile);
dcgnnModelPath = fullfile(rootDir, params.gnn.dcgnnModelFile);
ugnnModelPath = fullfile(rootDir, params.gnn.ugnnModelFile);
dqnModelPath = fullfile(rootDir, params.rl.dqnModelFile);
ddpgModelPath = fullfile(rootDir, params.rl.ddpgModelFile);

% 消融实验模型路径
gnnAblationDir = fullfile(rootDir, 'models', 'ablation');
gnnMLPModelPath = fullfile(gnnAblationDir, 'mlp_only', 'best_model.pt');
gnnGlobalNormModelPath = fullfile(gnnAblationDir, 'global_norm', 'best_model.pt');

% 所有 GNN 模型路径 (用于缓存有效性检测)
allModelPaths = {gnnLocalModelPath, localGnnModelPath, dcgnnModelPath, ugnnModelPath, dqnModelPath, ddpgModelPath, gnnMLPModelPath, gnnGlobalNormModelPath};
allModelNames = {'gnnLocalModelPath', 'localGnnModelPath', 'dcgnnModelPath', 'ugnnModelPath', 'dqnModelPath', 'ddpgModelPath', 'gnnMLPModelPath', 'gnnGlobalNormModelPath'};

if ~exist(savePath, 'dir'); mkdir(savePath); end
if ~exist(dataPath, 'dir'); mkdir(dataPath); end

if isSaveFig && params.output.cleanOldFigures && exist(savePath, 'dir')
    fprintf('>>> Cleaning old figures...\n');
    rmdir(savePath, 's'); mkdir(savePath);
end
% 注意: 不再清空 dataPath — 缓存文件需持久保留

%% ================= 算法注册表 =================
% 每个算法组合定义: {名称, 预编码函数, 功率分配函数, 模式}

% 功率分配方法 (Power Allocation)
PA = struct();
PA.baseline.name = 'Baseline';       PA.baseline.fcn = @computeRhoDist;              PA.baseline.arch = 'distributed';
PA.random.name   = 'Random';         PA.random.fcn   = [];                           PA.random.arch   = 'distributed';
PA.EPA.name      = 'EPA';            PA.EPA.fcn      = @computeRhoEPA;               PA.EPA.arch      = 'distributed';
PA.FPCP.name     = 'FPCP';           PA.FPCP.fcn     = @computeRhoFPCP;              PA.FPCP.arch     = 'distributed';
PA.DWMMSE.name   = 'D-WMMSE';        PA.DWMMSE.fcn   = @computeRhoDistributedWMMSE;  PA.DWMMSE.arch   = 'distributed';
PA.WMMSE.name    = 'WMMSE';          PA.WMMSE.fcn    = @computeRhoWMMSE;             PA.WMMSE.arch    = 'centralized_reference';
PA.GNN.name      = 'GNN';            PA.GNN.fcn      = @computeRhoGNN;               PA.GNN.arch      = 'low_latency_centralized';
PA.LocalGNN.name = 'Local-GNN';      PA.LocalGNN.fcn = @computeRhoLocalGNN;          PA.LocalGNN.arch = 'distributed';
PA.DCGNN.name    = 'DCGNN';          PA.DCGNN.fcn    = @computeRhoGNN;               PA.DCGNN.arch    = 'low_latency_centralized';
PA.UGNN.name     = 'U-GNN';          PA.UGNN.fcn     = @computeRhoUGNN;              PA.UGNN.arch     = 'low_latency_centralized';
PA.DQN.name      = 'DQN';            PA.DQN.fcn      = @computeRhoRL;                PA.DQN.arch      = 'low_latency_centralized';
PA.DDPG.name     = 'DDPG';           PA.DDPG.fcn     = @computeRhoRL;                PA.DDPG.arch     = 'low_latency_centralized';

% 预编码方法 (Precoding)
PC = struct();
PC.MR.name     = 'MR';              PC.MR.fcn       = @functionPrecoding_MR;             PC.MR.arch      = 'distributed';
PC.LMMSE.name  = 'L-MMSE';          PC.LMMSE.fcn    = @functionPrecoding_LMMSE;          PC.LMMSE.arch   = 'distributed';
PC.RMMSE.name  = 'R-MMSE';          PC.RMMSE.fcn    = @functionPrecoding_RobustMMSE;     PC.RMMSE.arch   = 'distributed';
PC.LMMSE_G.name = 'L-MMSE-G';       PC.LMMSE_G.fcn  = @functionPrecoding_LMMSE_global;   PC.LMMSE_G.arch = 'centralized_reference';

% SE计算器
SE = struct();
SE.MR.fcn    = @functionComputeSE_downlink_MR;
SE.LMMSE.fcn = @functionComputeSE_downlink_LMMSE;
SE.RMMSE.fcn = @functionComputeSE_downlink_RobustMMSE;
SE.LMMSE_G.fcn = @functionComputeSE_downlink_LMMSE;

% 接入模式
AP_MODE = params.simulation.accessModes;

% Stage 划分: 传统/参考方法索引 vs GNN 索引
paList = fieldnames(PA);
pcList = fieldnames(PC);
numPA_trad = 6;   % baseline, random, EPA, FPCP, D-WMMSE, WMMSE
numPA_gnn  = numel(paList) - numPA_trad;  % learning/reference family: GNN / Local-GNN / DCGNN / U-GNN / DQN / DDPG
numPC = length(pcList);  % 4
numMode = length(AP_MODE);

%% ================= 构建算法组合表 =================
algoTable = struct('id',{},'pa',{},'pc',{},'mode',{},'name',{}, ...
    'pcArch',{},'paArch',{},'isDistributed',{},'ESR',{});

for pi = 1:length(paList)
    paName = paList{pi};
    for ci = 1:numPC
        pcName = pcList{ci};
        for mi = 1:numMode
            modeName = AP_MODE{mi};
            pcArch = PC.(pcName).arch;
            paArch = PA.(paName).arch;
            algoTable(end+1) = struct('id',0,'pa',paName,'pc',pcName, ...
                'mode',modeName,'name','','pcArch',pcArch,'paArch',paArch, ...
                'isDistributed',strcmp(pcArch, 'distributed') && strcmp(paArch, 'distributed'), ...
                'ESR',[]);
        end
    end
end

for i = 1:numel(algoTable)
    paName   = algoTable(i).pa;
    pcName   = algoTable(i).pc;
    modeName = algoTable(i).mode;
    paShort  = PA.(paName).name;
    pcShort  = PC.(pcName).name;
    algoTable(i).id   = i;
    algoTable(i).name = sprintf('%s+%s (%s)', paShort, pcShort, modeName);
end

numAlgos = numel(algoTable);

%% ================= 打印配置 =================
fprintf('\n');
fprintf('=====================================================================\n');
fprintf('       Cell-Free Downlink Simulation  v3.0  (with cache)\n');
fprintf('=====================================================================\n');
fprintf('  System:  L=%d APs  N=%d Antennas  K=%d UEs\n', L, N, K);
fprintf('  Channel: tau_c=%d  tau_p=%d  ASD=%.0f deg\n', tau_c, tau_p, rad2deg(ASD_varphi));
fprintf('  CSI Error: sigma_e=%.2f  nIter=%d\n', sigma_e, nIter);
fprintf('  SNR: %s dB\n', mat2str(SNR_dB));
fprintf('  Scenarios: %d x %d realizations\n', numScenarios, nbrOfRealizations);
fprintf('  Algorithms: %d  (Trad: %d, Learning/ref: %d)\n', numAlgos, numPA_trad*numPC*numMode, numPA_gnn*numPC*numMode);
fprintf('  Stage: %d  |  Cache: %s\n', runStage, mat2str(useCache));
fprintf('  Output: fig=%s  data=%s\n', mat2str(isSaveFig), mat2str(isSaveData));
fprintf('  Sync ablation: %s  |  RTT=%.3f ms  fronthaul=%.0f Mbps\n', ...
    mat2str(enableSyncAblation), syncAblation.syncRttMs, syncAblation.fronthaulMbps);
fprintf('=====================================================================\n');

%% ================= 初始化结果存储 =================
if (runStage == 2 || runStage == 3) && isfile(gnnLocalModelPath)
    try
        fprintf('  Warming up GNN runtime (excluded from timing metrics)...\n');
        computeRhoGNN([], ones(L, K), ones(L, K), 10^(SNR_dB(1)/10), gnnLocalModelPath, sigma_e);
    catch ME
        warning('GNN warm-up failed; timed inference may include Python initialization. Reason: %s', ME.message);
    end
end
if (runStage == 2 || runStage == 3) && isfile(localGnnModelPath)
    try
        fprintf('  Warming up Local-GNN runtime (excluded from timing metrics)...\n');
        computeRhoLocalGNN([], ones(L, K), ones(L, K), 10^(SNR_dB(1)/10), localGnnModelPath, sigma_e);
    catch ME
        warning('Local-GNN warm-up failed; timed inference may include Python initialization. Reason: %s', ME.message);
    end
end
if (runStage == 2 || runStage == 3) && isfile(dcgnnModelPath)
    try
        fprintf('  Warming up DCGNN runtime (excluded from timing metrics)...\n');
        computeRhoGNN([], ones(L, K), ones(L, K), 10^(SNR_dB(1)/10), dcgnnModelPath, sigma_e);
    catch ME
        warning('DCGNN warm-up failed; timed inference may include Python initialization. Reason: %s', ME.message);
    end
end
if (runStage == 2 || runStage == 3) && isfile(ugnnModelPath)
    try
        fprintf('  Warming up U-GNN runtime (excluded from timing metrics)...\n');
        computeRhoUGNN([], ones(L, K), ones(L, K), 10^(SNR_dB(1)/10), ugnnModelPath, sigma_e);
    catch ME
        warning('U-GNN warm-up failed; timed inference may include Python initialization. Reason: %s', ME.message);
    end
end
if (runStage == 2 || runStage == 3) && isfile(dqnModelPath)
    try
        fprintf('  Warming up DQN runtime (excluded from timing metrics)...\n');
        computeRhoRL([], ones(L, K), ones(L, K), 10^(SNR_dB(1)/10), dqnModelPath, sigma_e);
    catch ME
        warning('DQN warm-up failed; timed inference may include Python initialization. Reason: %s', ME.message);
    end
end
if (runStage == 2 || runStage == 3) && isfile(ddpgModelPath)
    try
        fprintf('  Warming up DDPG runtime (excluded from timing metrics)...\n');
        computeRhoRL([], ones(L, K), ones(L, K), 10^(SNR_dB(1)/10), ddpgModelPath, sigma_e);
    catch ME
        warning('DDPG warm-up failed; timed inference may include Python initialization. Reason: %s', ME.message);
    end
end

ESR_acc = zeros(numAlgos, num_snr);
ESR_best = -inf(1, num_snr);
ESR_best_algo = strings(1, num_snr);
Perf = struct();
Perf.methodNames = {'Baseline', 'FPCP', 'D-WMMSE', 'WMMSE', 'GNN', 'Local-GNN', 'DCGNN', 'U-GNN', 'DQN', 'DDPG'};
Perf.modeNames = AP_MODE;
Perf.SNR_dB = SNR_dB;
Perf.time_pa_sec = zeros(numel(Perf.methodNames), num_snr, numMode);
Perf.time_core_sec = zeros(numel(Perf.methodNames), num_snr, numMode);
Perf.comm_bytes = zeros(numel(Perf.methodNames), num_snr, numMode);
Perf.time_pc_sec = zeros(numPC, num_snr, numMode);
Perf.modelBytes_GNN = 0;
if isfile(gnnLocalModelPath)
    d1 = dir(gnnLocalModelPath);
    Perf.modelBytes_GNN = double(d1.bytes);
end
Perf.modelBytes_LocalGNN = 0;
if isfile(localGnnModelPath)
    d1 = dir(localGnnModelPath);
    Perf.modelBytes_LocalGNN = double(d1.bytes);
end
Perf.modelBytes_DCGNN = 0;
if isfile(dcgnnModelPath)
    d1 = dir(dcgnnModelPath);
    Perf.modelBytes_DCGNN = double(d1.bytes);
end
Perf.modelBytes_UGNN = 0;
if isfile(ugnnModelPath)
    d1 = dir(ugnnModelPath);
    Perf.modelBytes_UGNN = double(d1.bytes);
end
Perf.modelBytes_DQN = 0;
if isfile(dqnModelPath)
    d1 = dir(dqnModelPath);
    Perf.modelBytes_DQN = double(d1.bytes);
end
Perf.modelBytes_DDPG = 0;
if isfile(ddpgModelPath)
    d1 = dir(ddpgModelPath);
    Perf.modelBytes_DDPG = double(d1.bytes);
end

% 计算参数指纹 (缓存有效性判定)
scenarioFP = buildParamFingerprint(L, K, N, tau_p, tau_c, ASD_varphi, ASD_theta, ...
    nbrOfRealizations, sigma_e, p, nIter, NaN);
snrFPs = buildSNRFingerprints(L, K, N, tau_p, tau_c, ASD_varphi, ASD_theta, ...
    nbrOfRealizations, sigma_e, p, nIter, SNR_dB);
snrCacheVersion = '_v19_latency_timing';
for fpIdx = 1:numel(snrFPs)
    snrFPs(fpIdx).fp = [snrFPs(fpIdx).fp snrCacheVersion];
end

% 统计缓存命中情况
cacheStats = struct('scenario_hits', 0, 'snr_hits', 0, 'snr_misses', 0);

%% ================= 主循环 =================
totalIters = numScenarios * num_snr * numAlgos;
completedIters = 0;

fprintf('\n');
fprintf('---------------------------------------------------------------------\n');
fprintf('  STARTING MAIN LOOP  |  %d scenarios x %d SNR x %d algos = %d iters\n', ...
    numScenarios, num_snr, numAlgos, totalIters);
fprintf('  runStage=%d  useCache=%s\n', runStage, mat2str(useCache));
fprintf('---------------------------------------------------------------------\n');

for s = 1:numScenarios
    fprintf('\n');
    fprintf('  === Scenario %d/%d ===\n', s, numScenarios);

    % ================================================================
    % Step 1: 场景生成 + 信道估计 (带缓存)
    % ================================================================
    scenarioCacheFile = fullfile(dataPath, sprintf('cache_scenario_s%d.mat', s));
    scenarioLoaded = false;

    if useCache && isfile(scenarioCacheFile)
        try
            tmp = load(scenarioCacheFile, 'paramFingerprint', 'gainOverNoisedB', ...
                'gainOverNoise', 'R', 'pilotIndex', 'D_dcc', 'APpositions', ...
                'UEpositions', 'Hhat', 'H_ideal', 'C', 'H');
            if strcmp(tmp.paramFingerprint, scenarioFP)
                gainOverNoisedB = tmp.gainOverNoisedB;
                gainOverNoise = tmp.gainOverNoise;
                R = tmp.R;
                pilotIndex = tmp.pilotIndex;
                D_dcc = tmp.D_dcc;
                APpositions = tmp.APpositions;
                UEpositions = tmp.UEpositions;
                Hhat = tmp.Hhat;
                H_ideal = tmp.H_ideal;
                C = tmp.C;
                H = tmp.H;
                scenarioLoaded = true;
                cacheStats.scenario_hits = cacheStats.scenario_hits + 1;
                if VERBOSE; fprintf('  [CACHE HIT] Scenario %d loaded\n', s); end
            else
                if VERBOSE; fprintf('  [CACHE MISS] Scenario %d param mismatch, regenerating\n', s); end
            end
        catch ME
            if VERBOSE; fprintf('  [CACHE MISS] Scenario %d load error: %s\n', s, ME.message); end
        end
    end

    if ~scenarioLoaded
        if runStage == 2
            error('Stage 1 cache not found for scenario %d. Run with runStage=1 or 3 first.', s);
        end
        if VERBOSE; fprintf('  [1/4] Generating scenario layout...\n'); end
        [gainOverNoisedB, R, pilotIndex, D_dcc, ~, APpositions, UEpositions] = ...
            generateSetup(L, K, N, tau_p, 1, s, ASD_varphi, ASD_theta);
        gainOverNoise = db2pow(gainOverNoisedB(:, :, 1));
        if s <= 1
            plotScenarioSetup(APpositions, UEpositions, s, isSaveFig, savePath, isSaveData, dataPath);
        end

        if VERBOSE; fprintf('  [2/4] Channel estimation (%d realizations)...\n', nbrOfRealizations); end
        [Hhat, H_ideal, ~, C] = functionChannelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p);
        if VERBOSE; fprintf('  [3/4] Injecting CSI error (sigma_e=%.2f)...\n', sigma_e); end
        H = Hhat + sqrt(sigma_e^2/2) * (randn(size(Hhat)) + 1i*randn(size(Hhat)));

        % 保存场景缓存
        if useCache
            if VERBOSE; fprintf('  [CACHE] Saving scenario cache...\n'); end
            paramFingerprint = scenarioFP;  % save 要求变量名与字符串一致
            save(scenarioCacheFile, 'paramFingerprint', 'gainOverNoisedB', ...
                'gainOverNoise', 'R', 'pilotIndex', 'D_dcc', 'APpositions', ...
                'UEpositions', 'Hhat', 'H_ideal', 'C', 'H', '-v7.3');
        end
    end

    % ================================================================
    % Step 2: SNR 扫描 (带缓存 + 分阶段)
    % ================================================================
    fprintf('  [4/4] SNR scan (%d points x %d algos)...\n', num_snr, numAlgos);

    for snr_idx = 1:num_snr
        Pt = db2pow(SNR_dB(snr_idx));
        snrCacheFile = fullfile(dataPath, sprintf('cache_snr_s%d_sn%.0f.mat', s, SNR_dB(snr_idx)));
        perfTimePA_snr = zeros(numel(Perf.methodNames), numMode);
        perfTimeCore_snr = zeros(numel(Perf.methodNames), numMode);
        perfCommBytes_snr = zeros(numel(Perf.methodNames), numMode);
        perfTimePC_snr = zeros(numPC, numMode);

        % --- 检查 SNR 缓存有效性 ---
        stage1_ok = false;
        stage2_ok = false;
        snrCacheData = [];

        if useCache && isfile(snrCacheFile)
            try
                tmp = load(snrCacheFile, 'paramFingerprint', 'stage1_valid', ...
                    'stage2_valid', 'modelModTimes');
                if strcmp(tmp.paramFingerprint, snrFPs(snr_idx).fp)
                    stage1_ok = tmp.stage1_valid;
                    stage2_ok = tmp.stage2_valid && ~gnnModelChanged(tmp.modelModTimes, allModelPaths, allModelNames);
                    if ~stage1_ok; stage2_ok = false; end  % 级联失效
                    % 指纹匹配 → 加载完整缓存数据
                    snrCacheData = load(snrCacheFile);
                    cacheStats.snr_hits = cacheStats.snr_hits + 1;
                else
                    cacheStats.snr_misses = cacheStats.snr_misses + 1;
                end
            catch
                cacheStats.snr_misses = cacheStats.snr_misses + 1;
            end
        else
            cacheStats.snr_misses = cacheStats.snr_misses + 1;
        end

        % runStage=2 需要 stage1 缓存
        if runStage == 2 && ~stage1_ok
            error('Stage 1 cache not valid for scenario %d, SNR %d dB. Run with runStage=1 or 3 first.', s, SNR_dB(snr_idx));
        end

        fprintf('  |   SNR %2d/%2d (%3.0f dB)  cache: stage1=%s stage2=%s\n', ...
            snr_idx, num_snr, SNR_dB(snr_idx), ...
            mat2str(stage1_ok), mat2str(stage2_ok));

        % 需要重新计算stage1时, 先构建 V_cache 和 pa_rhos_trad
        if (runStage == 1 || runStage == 3) && ~stage1_ok
            % ========== Stage 1: 传统方法 ==========
            V_cache_all = struct();  % V_cache_all.DCC.MR.V, V_cache_all.DCC.LMMSE.scaling, ...
            pa_rhos_all = struct();  % pa_rhos_all.DCC.baseline, pa_rhos_all.DCC.WMMSE, ...
            ESR_cell_all = cell(numMode, numPC, numPA_trad);  % (mode, pc, pa_trad)

            time_sec_s1 = zeros(numel(Perf.methodNames), numMode);
            time_core_s1 = zeros(numel(Perf.methodNames), numMode);
            comm_bytes_s1 = zeros(numel(Perf.methodNames), numMode);
            time_pc_s1 = zeros(numPC, numMode);

            for mi = 1:numMode
                modeName = AP_MODE{mi};
                D = D_dcc;

                if VERBOSE; fprintf('  |   |  [%s] Precoding...', modeName); end

                % 预编码向量
                V_cache_mode = struct();
                for ci = 1:numPC
                    pcName = pcList{ci};
                    pc_tic = tic;
                    switch pcName
                        case 'MR'
                            [V, sc] = functionPrecoding_MR(Hhat, nbrOfRealizations, N, K, L);
                        case 'LMMSE'
                            [V, sc] = functionPrecoding_LMMSE(Hhat, D, C, nbrOfRealizations, N, K, L, p);
                        case 'LMMSE_G'
                            [V, sc] = functionPrecoding_LMMSE_global(Hhat, D, C, nbrOfRealizations, N, K, L, p);
                        case 'RMMSE'
                            [V, sc] = functionPrecoding_RobustMMSE(Hhat, D, nbrOfRealizations, N, K, L, 1, sigma_e, nIter);
                    end
                    time_pc_s1(ci, mi) = toc(pc_tic);
                    V_cache_mode.(pcName).V = V;
                    V_cache_mode.(pcName).scaling = sc;
                end
                V_cache_all.(modeName) = V_cache_mode;

                % 传统功率分配
                rho_baseline_t = tic;
                rho_baseline = computeRhoDist(D, gainOverNoise, Pt, L, K);
                rho_baseline_sec = toc(rho_baseline_t);

                rho_EPA = computeRhoEPA(D, Pt, L, K);

                rho_FPCP_t = tic;
                rho_FPCP = computeRhoFPCP(D, gainOverNoise, Pt, L, K, params.fpcp.alpha);
                rho_FPCP_sec = toc(rho_FPCP_t);

                rho_DWMMSE_t = tic;
                [rho_DWMMSE, dWMMSE_info] = computeRhoDistributedWMMSE(D, gainOverNoise, Pt, L, K, ...
                    syncAblation.dwmmseRounds, params.dwmmse.damping);
                rho_DWMMSE_sec = toc(rho_DWMMSE_t);

                if VERBOSE; fprintf(' WMMSE...'); end
                rho_WMMSE_t = tic;
                [rho_WMMSE, ~] = computeRhoWMMSE(Hhat, D, Pt, N, K, L, ...
                    nbrOfRealizations, params.wmmse.simMaxIter, params.wmmse.tol);
                rho_WMMSE_sec = toc(rho_WMMSE_t);

                rho_Random = computeRhoRandom(V_cache_mode.MR.scaling, D, Pt, K, L);

                pa_rhos_mode = struct();
                pa_rhos_mode.baseline = rho_baseline;
                pa_rhos_mode.EPA      = rho_EPA;
                pa_rhos_mode.FPCP     = rho_FPCP;
                pa_rhos_mode.DWMMSE   = rho_DWMMSE;
                pa_rhos_mode.WMMSE    = rho_WMMSE;
                pa_rhos_mode.random   = rho_Random;
                pa_rhos_all.(modeName) = pa_rhos_mode;

                % 性能统计
                idxBL = find(strcmp(Perf.methodNames, 'Baseline'), 1);
                idxFP = find(strcmp(Perf.methodNames, 'FPCP'), 1);
                idxDW = find(strcmp(Perf.methodNames, 'D-WMMSE'), 1);
                idxW = find(strcmp(Perf.methodNames, 'WMMSE'), 1);
                time_sec_s1(idxBL, mi) = rho_baseline_sec;
                time_sec_s1(idxFP, mi) = rho_FPCP_sec;
                time_sec_s1(idxDW, mi) = rho_DWMMSE_sec;
                time_sec_s1(idxW, mi) = rho_WMMSE_sec;
                time_core_s1(idxBL, mi) = rho_baseline_sec;
                time_core_s1(idxFP, mi) = rho_FPCP_sec;
                time_core_s1(idxDW, mi) = rho_DWMMSE_sec;
                time_core_s1(idxW, mi) = rho_WMMSE_sec;
                comm_bytes_s1(idxBL, mi) = 0;
                comm_bytes_s1(idxFP, mi) = 0;
                comm_bytes_s1(idxDW, mi) = dWMMSE_info.messageBytes;
                comm_bytes_s1(idxW, mi) = double(numel(Hhat) * 16);

                % Parfor: 传统 PA x 全部 PC
                % 预提取 V/sc 到 cell 数组 (parfor 不支持动态字段索引)
                V_cells = cell(numPC, 1);
                sc_cells = cell(numPC, 1);
                for ci = 1:numPC
                    pcName = pcList{ci};
                    V_cells{ci} = V_cache_mode.(pcName).V;
                    sc_cells{ci} = V_cache_mode.(pcName).scaling;
                end
                % Must match paList order: baseline, random, EPA, FPCP, D-WMMSE, WMMSE.
                pa_trad_rhos = {rho_baseline, rho_Random, rho_EPA, rho_FPCP, rho_DWMMSE, rho_WMMSE};
                ESR_cell_mode = cell(numPC, numPA_trad);
                parfor ci = 1:numPC
                    pcName = pcList{ci};
                    V  = V_cells{ci};
                    sc = sc_cells{ci};
                    SE_k = zeros(K, 1);
                    for pi = 1:numPA_trad
                        rho = pa_trad_rhos{pi};
                        switch pcName
                            case 'MR'
                                SE_k = functionComputeSE_downlink_MR(H, V, sc, D, tau_c, tau_p, nbrOfRealizations, N, K, L, rho);
                            case {'LMMSE', 'LMMSE_G'}
                                SE_k = functionComputeSE_downlink_LMMSE(Hhat, H, V, sc, D, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho);
                            case 'RMMSE'
                                SE_k = functionComputeSE_downlink_RobustMMSE(Hhat, H, V, sc, D, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho, sigma_e, Pt, nIter);
                        end
                        ESR_cell_mode{ci, pi} = sum(SE_k);
                    end
                end
                ESR_cell_all(mi, :, :) = ESR_cell_mode;

                if VERBOSE; fprintf(' done\n'); end
            end

            % 汇总 stage1 ESR 到 ESR_acc
            for mi = 1:numMode
                for ci = 1:numPC
                    for pi = 1:numPA_trad
                        % algoTable 索引: pa 优先, pc 次之, mode 最后
                        algoIdx = (pi - 1) * numPC * numMode + (ci - 1) * numMode + mi;
                        ESR_k = ESR_cell_all{mi, ci, pi};
                        ESR_acc(algoIdx, snr_idx) = ESR_acc(algoIdx, snr_idx) + ESR_k;

                        if VERBOSE_ALGO
                            fprintf('  |   |  |  %-24s ESR=%8.3f\n', algoTable(algoIdx).name, ESR_k);
                        end
                        if ESR_k > ESR_best(snr_idx)
                            ESR_best(snr_idx) = ESR_k;
                            ESR_best_algo(snr_idx) = algoTable(algoIdx).name;
                        end
                    end
                end
            end

            perfTimePA_snr = time_sec_s1;
            perfTimeCore_snr = time_core_s1;
            perfCommBytes_snr = comm_bytes_s1;
            perfTimePC_snr = time_pc_s1;

            % 保存 stage1 到 SNR 缓存
            if useCache
                saveSNRCache(snrCacheFile, snrFPs(snr_idx).fp, s, SNR_dB(snr_idx), Pt, ...
                    V_cache_all, pa_rhos_all, ESR_cell_all, ...
                    struct(), cell(numMode, numPC, numPA_gnn), ...  % 空 GNN 结果
                    perfTimePA_snr, perfTimeCore_snr, perfCommBytes_snr, perfTimePC_snr, ...
                    allModelPaths, allModelNames, true, false);
            end

        elseif stage1_ok
            % Stage 1 从缓存加载
            ESR_cell_all = snrCacheData.ESR_cell_all;
            pa_rhos_all = snrCacheData.pa_rhos_all;
            V_cache_all = snrCacheData.V_cache_all;

            % 恢复 stage1 ESR
            for mi = 1:numMode
                for ci = 1:numPC
                    for pi = 1:numPA_trad
                        algoIdx = (pi - 1) * numPC * numMode + (ci - 1) * numMode + mi;
                        ESR_k = ESR_cell_all{mi, ci, pi};
                        ESR_acc(algoIdx, snr_idx) = ESR_acc(algoIdx, snr_idx) + ESR_k;
                    end
                end
            end

            % 恢复性能数据 (仅 stage1 部分)
            if isfield(snrCacheData, 'time_pa_sec')
                perfTimePA_snr = snrCacheData.time_pa_sec;
            end
            if isfield(snrCacheData, 'time_core_sec')
                perfTimeCore_snr = snrCacheData.time_core_sec;
            elseif isfield(snrCacheData, 'time_pa_sec')
                perfTimeCore_snr = snrCacheData.time_pa_sec;
            end
            if isfield(snrCacheData, 'comm_bytes')
                perfCommBytes_snr = snrCacheData.comm_bytes;
            end
            if isfield(snrCacheData, 'time_pc_sec')
                perfTimePC_snr = snrCacheData.time_pc_sec;
            end

            if VERBOSE; fprintf('  |   |  [Stage 1 loaded from cache]\n'); end
        end

        % ========== Stage 2: GNN 推理 ==========
        if (runStage == 2 || runStage == 3)
            if ~stage2_ok
                % 从缓存或 stage1 结果加载 V_cache_all
                if stage1_ok && ~isempty(snrCacheData) && isfield(snrCacheData, 'V_cache_all')
                    V_cache_all = snrCacheData.V_cache_all;
                end

                ESR_cell_gnn = cell(numMode, numPC, numPA_gnn);
                pa_rhos_gnn_all = struct();
                time_sec_s2 = zeros(numPA_gnn, numMode);
                time_core_s2 = zeros(numPA_gnn, numMode);
                comm_bytes_s2 = zeros(numPA_gnn, numMode);
                gnnPAList = paList(numPA_trad+1:end);

                for mi = 1:numMode
                    modeName = AP_MODE{mi};
                    D = D_dcc;

                    V_cache_mode = V_cache_all.(modeName);
                    % 回退 rho 用于 GNN 模型缺失时
                    if isfield(pa_rhos_all, modeName)
                        rho_fallback = pa_rhos_all.(modeName).EPA;
                    else
                        rho_fallback = computeRhoEPA(D, Pt, L, K);
                    end

                    if VERBOSE; fprintf('  |   |  [GNN family] Inference (%s)...', modeName); end

                    % Full-graph GNN (low-latency centralized reference)
                    rho_GNN_sec = nan;
                    rho_GNN_core_sec = nan;
                    try
                        rho_GNN_t = tic;
                        [rho_GNN, rho_GNN_timing] = computeRhoGNN(Hhat, D, gainOverNoise, Pt, gnnLocalModelPath, sigma_e);
                        rho_GNN_sec = toc(rho_GNN_t);
                        rho_GNN_core_sec = rho_GNN_timing.forward_sec;
                    catch ME
                        warning('GNN inference failed for %s at SNR %.0f dB; falling back to EPA. Reason: %s', ...
                            modeName, SNR_dB(snr_idx), ME.message);
                        rho_GNN = rho_fallback;
                        rho_GNN_core_sec = rho_GNN_sec;
                    end

                    % AP-local GNN (strict distributed candidate)
                    rho_LocalGNN_sec = nan;
                    rho_LocalGNN_core_sec = nan;
                    try
                        rho_LocalGNN_t = tic;
                        [rho_LocalGNN, rho_LocalGNN_timing] = computeRhoLocalGNN(Hhat, D, gainOverNoise, Pt, localGnnModelPath, sigma_e);
                        rho_LocalGNN_sec = toc(rho_LocalGNN_t);
                        rho_LocalGNN_core_sec = rho_LocalGNN_timing.forward_sec;
                    catch ME
                        warning('Local-GNN inference failed for %s at SNR %.0f dB; falling back to EPA. Reason: %s', ...
                            modeName, SNR_dB(snr_idx), ME.message);
                        rho_LocalGNN = rho_fallback;
                        rho_LocalGNN_core_sec = rho_LocalGNN_sec;
                    end

                    % 消融模型
                    % Dynamically-connected GNN candidate
                    rho_DCGNN_sec = nan;
                    rho_DCGNN_core_sec = nan;
                    try
                        rho_DCGNN_t = tic;
                        [rho_DCGNN, rho_DCGNN_timing] = computeRhoGNN(Hhat, D, gainOverNoise, Pt, dcgnnModelPath, sigma_e);
                        rho_DCGNN_sec = toc(rho_DCGNN_t);
                        rho_DCGNN_core_sec = rho_DCGNN_timing.forward_sec;
                    catch ME
                        warning('DCGNN inference failed for %s at SNR %.0f dB; falling back to EPA. Reason: %s', ...
                            modeName, SNR_dB(snr_idx), ME.message);
                        rho_DCGNN = rho_fallback;
                        rho_DCGNN_core_sec = rho_DCGNN_sec;
                    end

                    % U-GNN: teacher-free unsupervised GNN trained from SE proxy
                    rho_UGNN_sec = nan;
                    rho_UGNN_core_sec = nan;
                    try
                        rho_UGNN_t = tic;
                        [rho_UGNN, rho_UGNN_timing] = computeRhoUGNN(Hhat, D, gainOverNoise, Pt, ugnnModelPath, sigma_e);
                        rho_UGNN_sec = toc(rho_UGNN_t);
                        rho_UGNN_core_sec = rho_UGNN_timing.forward_sec;
                    catch ME
                        warning('U-GNN inference failed for %s at SNR %.0f dB; falling back to EPA. Reason: %s', ...
                            modeName, SNR_dB(snr_idx), ME.message);
                        rho_UGNN = rho_fallback;
                        rho_UGNN_core_sec = rho_UGNN_sec;
                    end

                    % DQN and DDPG RL baselines (one-step reward-trained policies)
                    rho_DQN_sec = nan;
                    rho_DQN_core_sec = nan;
                    try
                        rho_DQN_t = tic;
                        [rho_DQN, rho_DQN_timing] = computeRhoRL(Hhat, D, gainOverNoise, Pt, dqnModelPath, sigma_e);
                        rho_DQN_sec = toc(rho_DQN_t);
                        rho_DQN_core_sec = rho_DQN_timing.forward_sec;
                    catch ME
                        warning('DQN inference failed for %s at SNR %.0f dB; falling back to EPA. Reason: %s', ...
                            modeName, SNR_dB(snr_idx), ME.message);
                        rho_DQN = rho_fallback;
                        rho_DQN_core_sec = rho_DQN_sec;
                    end

                    rho_DDPG_sec = nan;
                    rho_DDPG_core_sec = nan;
                    try
                        rho_DDPG_t = tic;
                        [rho_DDPG, rho_DDPG_timing] = computeRhoRL(Hhat, D, gainOverNoise, Pt, ddpgModelPath, sigma_e);
                        rho_DDPG_sec = toc(rho_DDPG_t);
                        rho_DDPG_core_sec = rho_DDPG_timing.forward_sec;
                    catch ME
                        warning('DDPG inference failed for %s at SNR %.0f dB; falling back to EPA. Reason: %s', ...
                            modeName, SNR_dB(snr_idx), ME.message);
                        rho_DDPG = rho_fallback;
                        rho_DDPG_core_sec = rho_DDPG_sec;
                    end

                    rho_GNNMLP = rho_fallback;
                    rho_GNNGN = rho_fallback;
                    if isfile(gnnMLPModelPath)
                        try
                            rho_GNNMLP = computeRhoGNN(Hhat, D, gainOverNoise, Pt, gnnMLPModelPath, sigma_e);
                        catch
                        end
                    end
                    if isfile(gnnGlobalNormModelPath)
                        try
                            rho_GNNGN = computeRhoGNN(Hhat, D, gainOverNoise, Pt, gnnGlobalNormModelPath, sigma_e);
                        catch
                        end
                    end

                    pa_rhos_gnn_mode = struct();
                    pa_rhos_gnn_mode.GNN    = rho_GNN;
                    pa_rhos_gnn_mode.LocalGNN = rho_LocalGNN;
                    pa_rhos_gnn_mode.DCGNN = rho_DCGNN;
                    pa_rhos_gnn_mode.UGNN = rho_UGNN;
                    pa_rhos_gnn_mode.DQN = rho_DQN;
                    pa_rhos_gnn_mode.DDPG = rho_DDPG;
                    pa_rhos_gnn_mode.GNNMLP = rho_GNNMLP;
                    pa_rhos_gnn_mode.GNNGN  = rho_GNNGN;
                    pa_rhos_gnn_all.(modeName) = pa_rhos_gnn_mode;

                    featBytes = double(numel(gainOverNoise) * 8 + numel(D) * 8);
                    for pi = 1:numPA_gnn
                        switch gnnPAList{pi}
                            case 'GNN'
                                if isfinite(rho_GNN_sec); time_sec_s2(pi, mi) = rho_GNN_sec; end
                                if isfinite(rho_GNN_core_sec); time_core_s2(pi, mi) = rho_GNN_core_sec; end
                                comm_bytes_s2(pi, mi) = featBytes;
                            case 'LocalGNN'
                                if isfinite(rho_LocalGNN_sec); time_sec_s2(pi, mi) = rho_LocalGNN_sec; end
                                if isfinite(rho_LocalGNN_core_sec); time_core_s2(pi, mi) = rho_LocalGNN_core_sec; end
                                comm_bytes_s2(pi, mi) = 0;
                            case 'DCGNN'
                                if isfinite(rho_DCGNN_sec); time_sec_s2(pi, mi) = rho_DCGNN_sec; end
                                if isfinite(rho_DCGNN_core_sec); time_core_s2(pi, mi) = rho_DCGNN_core_sec; end
                                comm_bytes_s2(pi, mi) = featBytes;
                            case 'UGNN'
                                if isfinite(rho_UGNN_sec); time_sec_s2(pi, mi) = rho_UGNN_sec; end
                                if isfinite(rho_UGNN_core_sec); time_core_s2(pi, mi) = rho_UGNN_core_sec; end
                                comm_bytes_s2(pi, mi) = featBytes;
                            case 'DQN'
                                if isfinite(rho_DQN_sec); time_sec_s2(pi, mi) = rho_DQN_sec; end
                                if isfinite(rho_DQN_core_sec); time_core_s2(pi, mi) = rho_DQN_core_sec; end
                                comm_bytes_s2(pi, mi) = featBytes;
                            case 'DDPG'
                                if isfinite(rho_DDPG_sec); time_sec_s2(pi, mi) = rho_DDPG_sec; end
                                if isfinite(rho_DDPG_core_sec); time_core_s2(pi, mi) = rho_DDPG_core_sec; end
                                comm_bytes_s2(pi, mi) = featBytes;
                        end
                    end

                    % 预提取 V/sc 到 cell (parfor 不支持动态字段索引)
                    V_cells_gnn = cell(numPC, 1);
                    sc_cells_gnn = cell(numPC, 1);
                    for ci = 1:numPC
                        pcN = pcList{ci};
                        V_cells_gnn{ci} = V_cache_mode.(pcN).V;
                        sc_cells_gnn{ci} = V_cache_mode.(pcN).scaling;
                    end
                    % Parfor: GNN PA x 全部 PC
                    gnn_rhos = cell(1, numPA_gnn);
                    for pi = 1:numPA_gnn
                        switch gnnPAList{pi}
                            case 'GNN'
                                gnn_rhos{pi} = rho_GNN;
                            case 'LocalGNN'
                                gnn_rhos{pi} = rho_LocalGNN;
                            case 'DCGNN'
                                gnn_rhos{pi} = rho_DCGNN;
                            case 'UGNN'
                                gnn_rhos{pi} = rho_UGNN;
                            case 'DQN'
                                gnn_rhos{pi} = rho_DQN;
                            case 'DDPG'
                                gnn_rhos{pi} = rho_DDPG;
                            otherwise
                                gnn_rhos{pi} = rho_fallback;
                        end
                    end
                    ESR_cell_mode_gnn = cell(numPC, numPA_gnn);
                    parfor ci = 1:numPC
                        pcName = pcList{ci};
                        V  = V_cells_gnn{ci};
                        sc = sc_cells_gnn{ci};
                        SE_k = zeros(K, 1);
                        for pi = 1:numPA_gnn
                            rho = gnn_rhos{pi};
                            switch pcName
                                case 'MR'
                                    SE_k = functionComputeSE_downlink_MR(H, V, sc, D, tau_c, tau_p, nbrOfRealizations, N, K, L, rho);
                                case {'LMMSE', 'LMMSE_G'}
                                    SE_k = functionComputeSE_downlink_LMMSE(Hhat, H, V, sc, D, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho);
                                case 'RMMSE'
                                    SE_k = functionComputeSE_downlink_RobustMMSE(Hhat, H, V, sc, D, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho, sigma_e, Pt, nIter);
                            end
                            ESR_cell_mode_gnn{ci, pi} = sum(SE_k);
                        end
                    end
                    ESR_cell_gnn(mi, :, :) = ESR_cell_mode_gnn;

                    if VERBOSE; fprintf(' done\n'); end
                end

                % 汇总 stage2 ESR
                for mi = 1:numMode
                    for ci = 1:numPC
                        for pi = 1:numPA_gnn
                            algoIdx = (numPA_trad + pi - 1) * numPC * numMode + (ci - 1) * numMode + mi;
                            ESR_k = ESR_cell_gnn{mi, ci, pi};
                            ESR_acc(algoIdx, snr_idx) = ESR_acc(algoIdx, snr_idx) + ESR_k;

                            if VERBOSE_ALGO
                                fprintf('  |   |  |  %-24s ESR=%8.3f\n', algoTable(algoIdx).name, ESR_k);
                            end
                            if ESR_k > ESR_best(snr_idx)
                                ESR_best(snr_idx) = ESR_k;
                                ESR_best_algo(snr_idx) = algoTable(algoIdx).name;
                            end
                        end
                    end
                end

                % 更新 GNN 性能数据
                for pi = 1:numPA_gnn
                    switch gnnPAList{pi}
                        case 'GNN'
                            methodName = 'GNN';
                        case 'LocalGNN'
                            methodName = 'Local-GNN';
                        case 'DCGNN'
                            methodName = 'DCGNN';
                        case 'UGNN'
                            methodName = 'U-GNN';
                        case 'DQN'
                            methodName = 'DQN';
                        case 'DDPG'
                            methodName = 'DDPG';
                        otherwise
                            methodName = '';
                    end
                    perfIdx = find(strcmp(Perf.methodNames, methodName), 1);
                    if ~isempty(perfIdx)
                        perfTimePA_snr(perfIdx, :) = time_sec_s2(pi, :);
                        perfTimeCore_snr(perfIdx, :) = time_core_s2(pi, :);
                        perfCommBytes_snr(perfIdx, :) = comm_bytes_s2(pi, :);
                    end
                end

                % 合并保存 (stage1 + stage2)
                if useCache
                    % 需要同时保存 stage1 结果
                    if stage1_ok && ~isempty(snrCacheData)
                        ESR_cell_all_save = snrCacheData.ESR_cell_all;
                        pa_rhos_all_save = snrCacheData.pa_rhos_all;
                        V_cache_all_save = snrCacheData.V_cache_all;
                    else
                        ESR_cell_all_save = ESR_cell_all;
                        pa_rhos_all_save = pa_rhos_all;
                        V_cache_all_save = V_cache_all;
                    end
                    saveSNRCache(snrCacheFile, snrFPs(snr_idx).fp, s, SNR_dB(snr_idx), Pt, ...
                        V_cache_all_save, pa_rhos_all_save, ESR_cell_all_save, ...
                        pa_rhos_gnn_all, ESR_cell_gnn, ...
                        perfTimePA_snr, perfTimeCore_snr, perfCommBytes_snr, perfTimePC_snr, ...
                        allModelPaths, allModelNames, true, true);
                end

            else
                % Stage 2 从缓存加载
                ESR_cell_gnn = snrCacheData.ESR_cell_gnn;
                for mi = 1:numMode
                    for ci = 1:numPC
                        for pi = 1:numPA_gnn
                            algoIdx = (numPA_trad + pi - 1) * numPC * numMode + (ci - 1) * numMode + mi;
                            ESR_k = ESR_cell_gnn{mi, ci, pi};
                            ESR_acc(algoIdx, snr_idx) = ESR_acc(algoIdx, snr_idx) + ESR_k;
                        end
                    end
                end
                if VERBOSE; fprintf('  |   |  [Stage 2 loaded from cache]\n'); end
            end
        end

        Perf.time_pa_sec(:, snr_idx, :) = reshape(squeeze(Perf.time_pa_sec(:, snr_idx, :)) + perfTimePA_snr, ...
            [numel(Perf.methodNames), 1, numMode]);
        Perf.time_core_sec(:, snr_idx, :) = reshape(squeeze(Perf.time_core_sec(:, snr_idx, :)) + perfTimeCore_snr, ...
            [numel(Perf.methodNames), 1, numMode]);
        Perf.comm_bytes(:, snr_idx, :) = reshape(squeeze(Perf.comm_bytes(:, snr_idx, :)) + perfCommBytes_snr, ...
            [numel(Perf.methodNames), 1, numMode]);
        Perf.time_pc_sec(:, snr_idx, :) = reshape(squeeze(Perf.time_pc_sec(:, snr_idx, :)) + perfTimePC_snr, ...
            [numPC, 1, numMode]);

        completedIters = completedIters + numAlgos;
        fprintf('  |   SNR %d done\n', snr_idx);
    end  % end SNR loop

    fprintf('  Scenario %d completed\n', s);
end  % end scenario loop

%% ================= 缓存统计 =================
if useCache
    fprintf('\n');
    fprintf('  Cache stats: scenario_hit=%d, snr_hit=%d, snr_miss=%d\n', ...
        cacheStats.scenario_hits, cacheStats.snr_hits, cacheStats.snr_misses);
end

%% ================= 结果汇总 =================
ESR_mean = ESR_acc / numScenarios;
[ESR_best, ESR_best_idx] = max(ESR_mean, [], 1);
ESR_best_algo = strings(1, numel(SNR_dB));
for snr_idx = 1:numel(SNR_dB)
    ESR_best_algo(snr_idx) = string(algoTable(ESR_best_idx(snr_idx)).name);
end

if exist('Perf', 'var') && isstruct(Perf)
    Perf.time_pa_sec = Perf.time_pa_sec / numScenarios;
    if isfield(Perf, 'time_core_sec')
        Perf.time_core_sec = Perf.time_core_sec / numScenarios;
    end
    Perf.comm_bytes = Perf.comm_bytes / numScenarios;
    if isfield(Perf, 'time_pc_sec')
        Perf.time_pc_sec = Perf.time_pc_sec / numScenarios;
    end
end

fprintf('\n');
fprintf('=====================================================================\n');
fprintf('  SIMULATION COMPLETED  |  Total: %d iterations\n', completedIters);
fprintf('=====================================================================\n');

%% ================= 打印最终排名 =================
printFinalResults_v2(ESR_mean, algoTable, SNR_dB, ESR_best, ESR_best_algo, ...
    numScenarios, nbrOfRealizations, isSaveFig, isSaveData, savePath, dataPath, Perf);

%% ================= 绘图 =================
plotESRResults_v2(ESR_mean, ESR_best, ESR_best_algo, algoTable, SNR_dB, ...
    numScenarios, isSaveFig, savePath, isSaveData, dataPath, Perf);

if enableSyncAblation
    Ablation = buildSyncAblationMetrics(ESR_mean, algoTable, SNR_dB, Perf, ...
        L, K, N, nbrOfRealizations, nIter, syncAblation);
    plotLatencyAblationResults(Ablation, savePath, isSaveFig, isSaveData, dataPath);
end

fprintf('\n>>> All done. Figures saved to: %s\n', savePath);
fprintf('>>> Data saved to: %s\n', dataPath);

%% ========================================================================
%% 辅助函数
%% ========================================================================

%% ------------------------------------------------------------------------
function fp = buildParamFingerprint(L, K, N, tau_p, tau_c, ASD_varphi, ASD_theta, ...
    nbrOfRealizations, sigma_e, p_val, nIter, snr_dB)
% 构建参数指纹 (MD5)
% snr_dB: NaN 表示场景级指纹 (不含SNR), 数值表示SNR级指纹
    if isnan(snr_dB)
        keyStr = sprintf('L%d_K%d_N%d_tp%d_tc%d_asd%.6f_nrt%d_se%.4f_p%d_ni%d', ...
            L, K, N, tau_p, tau_c, ASD_varphi, nbrOfRealizations, sigma_e, p_val, nIter);
    else
        keyStr = sprintf('L%d_K%d_N%d_tp%d_tc%d_asd%.6f_nrt%d_se%.4f_p%d_ni%d_sn%.1f', ...
            L, K, N, tau_p, tau_c, ASD_varphi, nbrOfRealizations, sigma_e, p_val, nIter, snr_dB);
    end
    % 使用 Java MD5 (兼容所有 MATLAB 版本, 不依赖 hash 函数)
    md5Obj = java.security.MessageDigest.getInstance('MD5');
    md5Obj.update(uint8(keyStr(:)));
    fp = char(reshape(sprintf('%02x', typecast(md5Obj.digest, 'uint8')), 1, []));
end

%% ------------------------------------------------------------------------
function fps = buildSNRFingerprints(L, K, N, tau_p, tau_c, ASD_varphi, ASD_theta, ...
    nbrOfRealizations, sigma_e, p_val, nIter, SNR_dB_vec)
% 批量构建每个 SNR 点的指纹
    fps = struct('fp', cell(numel(SNR_dB_vec), 1));
    for i = 1:numel(SNR_dB_vec)
        fps(i).fp = buildParamFingerprint(L, K, N, tau_p, tau_c, ASD_varphi, ASD_theta, ...
            nbrOfRealizations, sigma_e, p_val, nIter, SNR_dB_vec(i));
    end
end

%% ------------------------------------------------------------------------
function changed = gnnModelChanged(modelModTimes, modelPaths, modelNames)
% 检测 GNN 模型文件是否自缓存以来被修改过
    changed = false;
    for i = 1:numel(modelPaths)
        p = modelPaths{i};
        name = modelNames{i};
        storedTime = modelModTimes.(name);
        if ~isfile(p)
            if storedTime > 0
                changed = true;  % 模型文件被删除
            end
            continue;
        end
        currentModTime = dir(p).datenum;
        if currentModTime > storedTime + 1e-6  % ~0.1秒容差
            changed = true;
        end
    end
end

%% ------------------------------------------------------------------------
function saveSNRCache(filePath, paramFingerprint, scenarioIdx, snr_dB, Pt, ...
        V_cache_all, pa_rhos_all, ESR_cell_all, ...
        pa_rhos_gnn_all, ESR_cell_gnn, ...
        time_pa_sec, time_core_sec, comm_bytes, time_pc_sec, allModelPaths, allModelNames, ...
        stage1_valid, stage2_valid)
% 保存 SNR 级缓存文件
    modelModTimes = struct();
    for i = 1:numel(allModelPaths)
        name = allModelNames{i};
        if isfile(allModelPaths{i})
            modelModTimes.(name) = dir(allModelPaths{i}).datenum;
        else
            modelModTimes.(name) = 0;
        end
    end
    save(filePath, 'paramFingerprint', 'scenarioIdx', 'snr_dB', 'Pt', ...
        'V_cache_all', 'pa_rhos_all', 'ESR_cell_all', ...
        'pa_rhos_gnn_all', 'ESR_cell_gnn', ...
        'time_pa_sec', 'time_core_sec', 'comm_bytes', 'time_pc_sec', ...
        'modelModTimes', 'stage1_valid', 'stage2_valid', '-v7.3');
end

%% ------------------------------------------------------------------------
function rho_random = computeRhoRandom(scaling, D, Pt, K, L)
% 随机功率分配
    d_random = rand(K, 1);
    d_random = d_random / sqrt(sum(d_random.^2));
    d2 = d_random.^2;
    powerPerAP = Pt / L;
    apPower = (scaling .* D) * d2;
    eta = zeros(size(apPower));
    nonzeroAP = apPower > 0;
    eta(nonzeroAP) = sqrt(powerPerAP ./ apPower(nonzeroAP));
    rho_random = (eta.^2) .* (scaling .* D) .* (d2.');
end

%% ------------------------------------------------------------------------
function Ablation = buildSyncAblationMetrics(ESR_mean, algoTable, SNR_dB, Perf, ...
        L, K, N, nbrOfRealizations, nIter, cfg)
% 构建预编码+功率分配组合的同步时延消融指标。
% 该模型刻画控制面同步代价: 迭代方法按同步轮次累积 RTT, 集中式/全局预编码按
% CSI/统计量载荷折算传输时延。GNN/EPA/Baseline 不需要迭代同步。
    numAlgos = numel(algoTable);
    numSNR = numel(SNR_dB);
    syncDelayMs = zeros(numAlgos, numSNR);
    computeDelayMs = nan(numAlgos, numSNR);
    pcComputeDelayMs = nan(numAlgos, numSNR);
    controlDelayMs = zeros(numAlgos, numSNR);
    syncBytes = zeros(numAlgos, numSNR);
    syncRounds = zeros(numAlgos, numSNR);

    avgESR = mean(ESR_mean, 2);
    for ai = 1:numAlgos
        payloadRatio = cfg.dccPayloadRatio;

        [pcRounds, pcBytes] = estimatePCSync(algoTable(ai).pc, L, K, N, ...
            nbrOfRealizations, nIter, payloadRatio);
        [paRounds, paBytes] = estimatePASync(algoTable(ai).pa, L, K, cfg, payloadRatio);

        totalRounds = pcRounds + paRounds;
        baseBytes = pcBytes + paBytes;

        for si = 1:numSNR
            [featureBytes, featureRounds] = lookupFeatureCollectionBytes(algoTable(ai).pa, ...
                algoTable(ai).mode, si, L, K, Perf);
            totalBytes = baseBytes + featureBytes;
            totalRoundsSi = totalRounds + featureRounds;
            syncDelayMs(ai, si) = totalRoundsSi * cfg.syncRttMs + ...
                bytesToMs(totalBytes, cfg.fronthaulMbps);
            syncBytes(ai, si) = totalBytes;
            syncRounds(ai, si) = totalRoundsSi;
            computeDelayMs(ai, si) = lookupComputeDelayMs(algoTable(ai).pa, ...
                algoTable(ai).mode, si, Perf);
            pcComputeDelayMs(ai, si) = lookupPCComputeDelayMs(algoTable(ai).pc, ...
                algoTable(ai).mode, si, Perf);
            totalComputeMs = sumFiniteDelays([computeDelayMs(ai, si), pcComputeDelayMs(ai, si)]);
            if cfg.includeComputeTime
                controlDelayMs(ai, si) = syncDelayMs(ai, si) + totalComputeMs;
            else
                controlDelayMs(ai, si) = syncDelayMs(ai, si);
            end
        end
    end

    Ablation = struct();
    Ablation.SNR_dB = SNR_dB;
    Ablation.algoTable = algoTable;
    Ablation.ESR_mean = ESR_mean;
    Ablation.avgESR = avgESR;
    Ablation.sync_delay_ms = syncDelayMs;
    Ablation.compute_delay_ms = computeDelayMs;
    Ablation.pc_compute_delay_ms = pcComputeDelayMs;
    Ablation.control_delay_ms = controlDelayMs;
    Ablation.sync_bytes = syncBytes;
    Ablation.sync_rounds = syncRounds;
    Ablation.config = cfg;
end

%% ------------------------------------------------------------------------
function [rounds, bytes] = estimatePCSync(pcName, L, K, N, nbrOfRealizations, nIter, payloadRatio)
    lkBytes = L * K * 8 * payloadRatio;
    switch pcName
        case 'MR'
            rounds = 0;
            bytes = 0;
        case 'LMMSE'
            rounds = 1;
            bytes = lkBytes;
        case 'RMMSE'
            rounds = max(nIter, 1);
            bytes = lkBytes * max(nIter, 1);
        case 'LMMSE_G'
            rounds = 1;
            bytes = L * N * K * nbrOfRealizations * 16 * payloadRatio;
        otherwise
            rounds = 0;
            bytes = 0;
    end
end

%% ------------------------------------------------------------------------
function [rounds, bytes] = estimatePASync(paName, L, K, cfg, payloadRatio)
    lkComplexBytes = L * K * 16 * payloadRatio;
    switch paName
        case 'WMMSE'
            rounds = cfg.wmmseRounds;
            bytes = lkComplexBytes * cfg.wmmseRounds;
        case 'DWMMSE'
            rounds = cfg.dwmmseRounds;
            bytes = K * 8 * cfg.dwmmseRounds;
        otherwise
            rounds = 0;
            bytes = 0;
    end
end

%% ------------------------------------------------------------------------
function delayMs = bytesToMs(numBytes, fronthaulMbps)
    delayMs = double(numBytes) * 8 / (fronthaulMbps * 1e6) * 1e3;
end

%% ------------------------------------------------------------------------
function [bytes, rounds] = lookupFeatureCollectionBytes(paName, modeName, snrIdx, L, K, Perf)
% Centralized learned PA methods avoid iterative PA synchronization, but
% still need one-shot feature collection at the central inference point.
    bytes = 0;
    rounds = 0;
    switch paName
        case 'GNN'
            methodName = 'GNN';
        case 'DCGNN'
            methodName = 'DCGNN';
        case 'UGNN'
            methodName = 'U-GNN';
        case 'DQN'
            methodName = 'DQN';
        case 'DDPG'
            methodName = 'DDPG';
        otherwise
            return;
    end

    bytes = double(L * K * 16);
    rounds = 1;

    if isempty(Perf) || ~isstruct(Perf) || ~isfield(Perf, 'comm_bytes') || ...
            ~isfield(Perf, 'modeNames') || ~isfield(Perf, 'methodNames')
        return;
    end

    mi = find(strcmp(Perf.modeNames, modeName), 1);
    pi = find(strcmp(Perf.methodNames, methodName), 1);
    if isempty(mi) || isempty(pi); return; end

    value = Perf.comm_bytes(pi, snrIdx, mi);
    if isfinite(value) && value > 0
        bytes = max(bytes, value);
    end
end

%% ------------------------------------------------------------------------
function delayMs = lookupPCComputeDelayMs(pcName, modeName, snrIdx, Perf)
    delayMs = 0;
    if isempty(Perf) || ~isstruct(Perf) || ~isfield(Perf, 'time_pc_sec') || ...
            ~isfield(Perf, 'modeNames')
        return;
    end
    pcOrder = {'MR', 'LMMSE', 'RMMSE', 'LMMSE_G'};
    ci = find(strcmp(pcOrder, pcName), 1);
    mi = find(strcmp(Perf.modeNames, modeName), 1);
    if isempty(ci) || isempty(mi); return; end
    delayMs = Perf.time_pc_sec(ci, snrIdx, mi) * 1000;
end

%% ------------------------------------------------------------------------
function totalMs = sumFiniteDelays(values)
    values = values(isfinite(values));
    if isempty(values)
        totalMs = 0;
    else
        totalMs = sum(values);
    end
end

%% ------------------------------------------------------------------------
function delayMs = lookupComputeDelayMs(paName, modeName, snrIdx, Perf)
    delayMs = nan;
    if isempty(Perf) || ~isstruct(Perf) || ~isfield(Perf, 'time_pa_sec') || ...
            ~isfield(Perf, 'modeNames') || ~isfield(Perf, 'methodNames')
        return;
    end
    switch paName
        case 'baseline'
            methodName = 'Baseline';
        case 'FPCP'
            methodName = 'FPCP';
        case 'DWMMSE'
            methodName = 'D-WMMSE';
        case 'WMMSE'
            methodName = 'WMMSE';
        case 'GNN'
            methodName = 'GNN';
        case 'LocalGNN'
            methodName = 'Local-GNN';
        case 'DCGNN'
            methodName = 'DCGNN';
        case 'UGNN'
            methodName = 'U-GNN';
        case 'DQN'
            methodName = 'DQN';
        case 'DDPG'
            methodName = 'DDPG';
        case {'EPA', 'random'}
            delayMs = 0;
            return;
        otherwise
            return;
    end
    mi = find(strcmp(Perf.modeNames, modeName), 1);
    pi = find(strcmp(Perf.methodNames, methodName), 1);
    if isempty(mi) || isempty(pi); return; end
    delayMs = Perf.time_pa_sec(pi, snrIdx, mi) * 1000;
end
