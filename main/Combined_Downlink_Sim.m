%% ================= CF_downlink_sim - 主仿真引擎 (v3.0) =================
%% 功能: Cell-Free Massive MIMO 下行链路仿真
%% 核心目标: 通过系统性地对比预编码×功率分配组合, 找出最优方案
%% v3.0 新增: 分阶段运行 + 结果缓存 (避免重复运行 WMMSE 等耗时计算)
% ==========================================================================

close all; clear;

%% ================= 用户配置区 =================
% ---------- 系统参数 ----------
L = 100; N = 1; K = 20;
tau_c = 200; tau_p = 10;
ASD_varphi = deg2rad(15);
ASD_theta  = deg2rad(15);
p = 100;                          % 导频功率
SNR_dB = 10:5:30;                 % SNR 扫描范围
num_snr = length(SNR_dB);

% ---------- CSI误差参数 ----------
sigma_e = 0.3;                    % 信道估计误差标准差
nIter = 5;                        % Robust MMSE 迭代次数

% ---------- 仿真规模 ----------
numScenarios = 2;                 % 场景数
nbrOfRealizations = 50;         % 每个场景的信道实现数

% ---------- 输出控制 ----------
isSaveFig = true;                 % 是否保存图像
isSaveData = true;               % 是否保存数据

% ---------- 显示模式 ----------
VERBOSE = true;                   % 详细输出(算法名称+ESR)
VERBOSE_ALGO = false;             % 逐算法 ESR 详情输出

% ---------- 缓存 & 阶段控制 (v3.0 新增) ----------
runStage = 2;     % 1=仅传统方法(Baseline/EPA/PSO/WMMSE/Random)
                   % 2=仅GNN推理(需已有stage1缓存)
                   % 3=全部(默认, 等价原版行为)
useCache = true;   % 缓存总开关; false 则每次全部重算(等同v2.0)

%% ================= 路径配置 =================
scriptDir = fileparts(mfilename('fullpath'));
cd(scriptDir);
rootDir = fileparts(scriptDir);
savePath = fullfile(scriptDir, 'Imgs');
dataPath = fullfile(scriptDir, 'SimulationData');
gnnLocalModelPath = fullfile(rootDir, 'models', 'best_gat_gnn_power.pt');
gnnFLModelPath = fullfile(rootDir, 'models', 'gnn_power_fedavg.pt');

% 消融实验模型路径
gnnAblationDir = fullfile(rootDir, 'models', 'ablation');
gnnMLPModelPath = fullfile(gnnAblationDir, 'mlp_only', 'best_model.pt');
gnnGlobalNormModelPath = fullfile(gnnAblationDir, 'global_norm', 'best_model.pt');

% 所有 GNN 模型路径 (用于缓存有效性检测)
allModelPaths = {gnnLocalModelPath, gnnFLModelPath, gnnMLPModelPath, gnnGlobalNormModelPath};
allModelNames = {'gnnLocalModelPath', 'gnnFLModelPath', 'gnnMLPModelPath', 'gnnGlobalNormModelPath'};

if ~exist(savePath, 'dir'); mkdir(savePath); end
if ~exist(dataPath, 'dir'); mkdir(dataPath); end

if isSaveFig && exist(savePath, 'dir')
    fprintf('>>> Cleaning old figures...\n');
    rmdir(savePath, 's'); mkdir(savePath);
end
% 注意: 不再清空 dataPath — 缓存文件需持久保留

%% ================= 算法注册表 =================
% 每个算法组合定义: {名称, 预编码函数, 功率分配函数, 模式}

% 功率分配方法 (Power Allocation)
PA = struct();
PA.baseline.name = 'Baseline';       PA.baseline.fcn = @computeRhoDist;
PA.random.name   = 'Random';         PA.random.fcn   = [];
PA.EPA.name      = 'EPA';            PA.EPA.fcn      = @computeRhoEPA;
PA.WMMSE.name    = 'WMMSE';          PA.WMMSE.fcn    = @computeRhoWMMSE;
PA.PSO.name      = 'PSO';            PA.PSO.fcn      = [];
PA.GNN.name      = 'GNN';            PA.GNN.fcn      = @computeRhoGNN;
PA.GNNFL.name    = 'GNN+FL';         PA.GNNFL.fcn    = @computeRhoGNN;

% 预编码方法 (Precoding)
PC = struct();
PC.MR.name     = 'MR';              PC.MR.fcn       = @functionPrecoding_MR;
PC.LMMSE.name  = 'L-MMSE';          PC.LMMSE.fcn    = @functionPrecoding_LMMSE;
PC.RMMSE.name  = 'R-MMSE';          PC.RMMSE.fcn    = @functionPrecoding_RobustMMSE;
PC.LMMSE_G.name = 'L-MMSE-G';       PC.LMMSE_G.fcn  = @functionPrecoding_LMMSE_global;

% SE计算器
SE = struct();
SE.MR.fcn    = @functionComputeSE_downlink_MR;
SE.LMMSE.fcn = @functionComputeSE_downlink_LMMSE;
SE.RMMSE.fcn = @functionComputeSE_downlink_RobustMMSE;
SE.LMMSE_G.fcn = @functionComputeSE_downlink_LMMSE;

% 接入模式
AP_MODE = {'All', 'DCC'};

% Stage 划分: 传统方法索引 vs GNN 索引
paList = fieldnames(PA);
pcList = fieldnames(PC);
numPA_trad = 5;   % baseline, random, EPA, WMMSE, PSO
numPA_gnn  = numel(paList) - numPA_trad;  % GNN, GNN+FL
numPC = length(pcList);  % 4
numMode = length(AP_MODE);       % 2

%% ================= 构建算法组合表 =================
algoTable = struct('id',{},'pa',{},'pc',{},'mode',{},'name',{},'ESR',{});

for pi = 1:length(paList)
    paName = paList{pi};
    for ci = 1:numPC
        pcName = pcList{ci};
        for mi = 1:numMode
            modeName = AP_MODE{mi};
            algoTable(end+1) = struct('id',0,'pa',paName,'pc',pcName,'mode',modeName,'name','','ESR',[]);
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
fprintf('  Algorithms: %d  (Trad: %d, GNN: %d)\n', numAlgos, numPA_trad*numPC*numMode, numPA_gnn*numPC*numMode);
fprintf('  Stage: %d  |  Cache: %s\n', runStage, mat2str(useCache));
fprintf('  Output: fig=%s  data=%s\n', mat2str(isSaveFig), mat2str(isSaveData));
fprintf('=====================================================================\n');

%% ================= 初始化结果存储 =================
ESR_acc = zeros(numAlgos, num_snr);
ESR_best = -inf(1, num_snr);
ESR_best_algo = strings(1, num_snr);
PSO_info = struct('iterUsed',[], 'bestFitness',[], 'd_opt',[]);

Perf = struct();
Perf.methodNames = {'Baseline', 'WMMSE', 'GNN', 'GNN+FL'};
Perf.modeNames = AP_MODE;
Perf.SNR_dB = SNR_dB;
Perf.time_pa_sec = zeros(numel(Perf.methodNames), num_snr, numMode);
Perf.comm_bytes = zeros(numel(Perf.methodNames), num_snr, numMode);
Perf.modelBytes_GNN = 0;
Perf.modelBytes_GNNFL = 0;
if isfile(gnnLocalModelPath)
    d1 = dir(gnnLocalModelPath);
    Perf.modelBytes_GNN = double(d1.bytes);
end
if isfile(gnnFLModelPath)
    d2 = dir(gnnFLModelPath);
    Perf.modelBytes_GNNFL = double(d2.bytes);
end

D_all = ones(L, K);

% 计算参数指纹 (缓存有效性判定)
scenarioFP = buildParamFingerprint(L, K, N, tau_p, tau_c, ASD_varphi, ASD_theta, ...
    nbrOfRealizations, sigma_e, p, nIter, NaN);
snrFPs = buildSNRFingerprints(L, K, N, tau_p, tau_c, ASD_varphi, ASD_theta, ...
    nbrOfRealizations, sigma_e, p, nIter, SNR_dB);

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
            V_cache_all = struct();  % V_cache_all.All.MR.V, V_cache_all.DCC.LMMSE.scaling, ...
            pa_rhos_all = struct();  % pa_rhos_all.All.baseline, pa_rhos_all.DCC.WMMSE, ...
            ESR_cell_all = cell(numMode, numPC, numPA_trad);  % (mode, pc, pa_trad)

            time_sec_s1 = zeros(4, numMode);  % {Baseline, WMMSE, GNN, GNN+FL} x mode
            comm_bytes_s1 = zeros(4, numMode);

            for mi = 1:numMode
                modeName = AP_MODE{mi};
                D = D_all;
                if strcmp(modeName, 'DCC'); D = D_dcc; end

                if VERBOSE; fprintf('  |   |  [%s] Precoding...', modeName); end

                % 预编码向量
                V_cache_mode = struct();
                for ci = 1:numPC
                    pcName = pcList{ci};
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
                    V_cache_mode.(pcName).V = V;
                    V_cache_mode.(pcName).scaling = sc;
                end
                V_cache_all.(modeName) = V_cache_mode;

                % 传统功率分配
                rho_baseline_t = tic;
                rho_baseline = computeRhoDist(D, gainOverNoise, Pt, L, K);
                rho_baseline_sec = toc(rho_baseline_t);

                rho_EPA = computeRhoEPA(D, Pt, L, K);

                gainWithD = gainOverNoise .* D;
                [~, bestAP] = max(gainWithD, [], 1);
                d_baseline = sqrt(gainWithD(bestAP + (0:K-1)*L)).';
                d_baseline = d_baseline / sqrt(sum(d_baseline.^2)) * sqrt(Pt);

                if VERBOSE; fprintf(' PSO...'); end
                [d_opt, iterUsed, bestFitness] = functionOptimize_d_PSO( ...
                    Hhat, H, D, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, Pt, d_baseline);
                rho_PSO = computeRhoFromD(d_opt, V_cache_mode.MR.scaling, D, Pt, K, L);

                % PSO 记录
                if snr_idx == num_snr
                    PSO_info.iterUsed(end+1) = iterUsed;
                    PSO_info.bestFitness(end+1) = bestFitness;
                    if strcmp(modeName, 'All'); PSO_info.d_opt = d_opt; end
                end

                if VERBOSE; fprintf(' WMMSE...'); end
                rho_WMMSE_t = tic;
                [rho_WMMSE, ~] = computeRhoWMMSE(Hhat, D, Pt, N, K, L, nbrOfRealizations, 20, 1e-4);
                rho_WMMSE_sec = toc(rho_WMMSE_t);

                rho_Random = computeRhoRandom(V_cache_mode.MR.scaling, D, Pt, K, L);

                pa_rhos_mode = struct();
                pa_rhos_mode.baseline = rho_baseline;
                pa_rhos_mode.EPA      = rho_EPA;
                pa_rhos_mode.PSO      = rho_PSO;
                pa_rhos_mode.WMMSE    = rho_WMMSE;
                pa_rhos_mode.random   = rho_Random;
                pa_rhos_all.(modeName) = pa_rhos_mode;

                % 性能统计
                idxBL = 1; idxW = 2;
                time_sec_s1(idxBL, mi) = rho_baseline_sec;
                time_sec_s1(idxW, mi) = rho_WMMSE_sec;
                comm_bytes_s1(idxBL, mi) = 0;
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
                pa_trad_rhos = {rho_baseline, rho_EPA, rho_PSO, rho_WMMSE, rho_Random};
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

            Perf.time_pa_sec(1:2, snr_idx, :) = time_sec_s1(1:2, :);
            Perf.comm_bytes(1:2, snr_idx, :) = comm_bytes_s1(1:2, :);

            % 保存 stage1 到 SNR 缓存
            if useCache
                saveSNRCache(snrCacheFile, snrFPs(snr_idx).fp, s, SNR_dB(snr_idx), Pt, ...
                    V_cache_all, pa_rhos_all, ESR_cell_all, ...
                    struct(), cell(numMode, numPC, numPA_gnn), ...  % 空 GNN 结果
                    Perf.time_pa_sec(:, snr_idx, :), Perf.comm_bytes(:, snr_idx, :), ...
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
                Perf.time_pa_sec(:, snr_idx, :) = snrCacheData.time_pa_sec;
            end
            if isfield(snrCacheData, 'comm_bytes')
                Perf.comm_bytes(:, snr_idx, :) = snrCacheData.comm_bytes;
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
                time_sec_s2 = zeros(2, numMode);   % GNN, GNN+FL x mode
                comm_bytes_s2 = zeros(2, numMode);

                for mi = 1:numMode
                    modeName = AP_MODE{mi};
                    D = D_all;
                    if strcmp(modeName, 'DCC'); D = D_dcc; end

                    V_cache_mode = V_cache_all.(modeName);
                    % 回退 rho 用于 GNN 模型缺失时
                    if isfield(pa_rhos_all, modeName)
                        rho_fallback = pa_rhos_all.(modeName).EPA;
                    else
                        rho_fallback = computeRhoEPA(D, Pt, L, K);
                    end

                    if VERBOSE; fprintf('  |   |  [GNN] Inference (%s)...', modeName); end

                    % GNN
                    rho_GNN_sec = nan;
                    try
                        rho_GNN_t = tic;
                        rho_GNN = computeRhoGNN(Hhat, D, gainOverNoise, Pt, gnnLocalModelPath, sigma_e);
                        rho_GNN_sec = toc(rho_GNN_t);
                    catch
                        rho_GNN = rho_fallback;
                    end

                    % GNN+FL
                    rho_GNNFL_sec = nan;
                    try
                        rho_GNNFL_t = tic;
                        rho_GNNFL = computeRhoGNN(Hhat, D, gainOverNoise, Pt, gnnFLModelPath, sigma_e);
                        rho_GNNFL_sec = toc(rho_GNNFL_t);
                    catch
                        rho_GNNFL = rho_fallback;
                    end

                    % 消融模型
                    rho_GNNMLP = rho_fallback;
                    rho_GNNGN = rho_fallback;
                    if isfile(gnnMLPModelPath)
                        try
                            rho_GNNMLP = computeRhoGNN(Hhat, D, gainOverNoise, Pt, gnnMLPModelPath, sigma_e);
                        catch; end
                    end
                    if isfile(gnnGlobalNormModelPath)
                        try
                            rho_GNNGN = computeRhoGNN(Hhat, D, gainOverNoise, Pt, gnnGlobalNormModelPath, sigma_e);
                        catch; end
                    end

                    pa_rhos_gnn_mode = struct();
                    pa_rhos_gnn_mode.GNN    = rho_GNN;
                    pa_rhos_gnn_mode.GNNFL  = rho_GNNFL;
                    pa_rhos_gnn_mode.GNNMLP = rho_GNNMLP;
                    pa_rhos_gnn_mode.GNNGN  = rho_GNNGN;
                    pa_rhos_gnn_all.(modeName) = pa_rhos_gnn_mode;

                    idxG = 1; idxGF = 2;
                    if isfinite(rho_GNN_sec);   time_sec_s2(idxG, mi)  = rho_GNN_sec;   end
                    if isfinite(rho_GNNFL_sec); time_sec_s2(idxGF, mi) = rho_GNNFL_sec; end

                    featBytes = double(numel(gainOverNoise) * 8 + numel(D) * 8);
                    comm_bytes_s2(idxG, mi)  = featBytes;
                    comm_bytes_s2(idxGF, mi) = featBytes;

                    % 预提取 V/sc 到 cell (parfor 不支持动态字段索引)
                    V_cells_gnn = cell(numPC, 1);
                    sc_cells_gnn = cell(numPC, 1);
                    for ci = 1:numPC
                        pcN = pcList{ci};
                        V_cells_gnn{ci} = V_cache_mode.(pcN).V;
                        sc_cells_gnn{ci} = V_cache_mode.(pcN).scaling;
                    end
                    % Parfor: GNN PA x 全部 PC
                    gnn_rhos = {rho_GNN, rho_GNNFL};
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
                Perf.time_pa_sec(3:4, snr_idx, :) = time_sec_s2;
                Perf.comm_bytes(3:4, snr_idx, :) = comm_bytes_s2;

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
                        Perf.time_pa_sec(:, snr_idx, :), Perf.comm_bytes(:, snr_idx, :), ...
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

if exist('Perf', 'var') && isstruct(Perf)
    Perf.time_pa_sec = Perf.time_pa_sec / numScenarios;
    Perf.comm_bytes = Perf.comm_bytes / numScenarios;
end

fprintf('\n');
fprintf('=====================================================================\n');
fprintf('  SIMULATION COMPLETED  |  Total: %d iterations\n', completedIters);
fprintf('=====================================================================\n');

%% ================= 打印最终排名 =================
printFinalResults_v2(ESR_mean, algoTable, SNR_dB, ESR_best, ESR_best_algo, ...
    PSO_info, numScenarios, nbrOfRealizations, isSaveFig, isSaveData, savePath, dataPath, Perf);

%% ================= 绘图 =================
plotESRResults_v2(ESR_mean, ESR_best, ESR_best_algo, algoTable, SNR_dB, ...
    numScenarios, isSaveFig, savePath, isSaveData, dataPath, Perf);

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
        time_pa_sec, comm_bytes, allModelPaths, allModelNames, ...
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
        'time_pa_sec', 'comm_bytes', ...
        'modelModTimes', 'stage1_valid', 'stage2_valid', '-v7.3');
end

%% ------------------------------------------------------------------------
function rho = computeRhoFromD(d_opt, scaling, D, Pt, K, L)
% 从PSO优化得到的d向量计算rho
    d2 = d_opt(:).^2;
    powerPerAP = Pt / L;
    apPower = (scaling .* D) * d2;
    eta = zeros(size(apPower));
    nonzeroAP = apPower > 0;
    eta(nonzeroAP) = sqrt(powerPerAP ./ apPower(nonzeroAP));
    rho = (eta.^2) .* (scaling .* D) .* (d2.');
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
