%% ================= Initialization =================
close all; clear;

% ========== 仿真配置参数 ==========
% 系统参数
L = 100; N = 1; K = 20;
tau_c = 200; tau_p = 10;

ASD_varphi = deg2rad(15);
ASD_theta  = deg2rad(15);

p = 100;

SNR_dB = 5:5:30;
num_snr = length(SNR_dB);

% ===== Robust参数 =====
sigma_e = 0.3;     % CSI误差
nIter = 5;

% ===== 多场景实验参数 =====
numScenarios = 3; % 场景实验次数
nbrOfRealizations = 200; % 每个场景下的信道数

% 接口：设置要展示的仿真场景数量
numScenariosToPlot = numScenarios; % 用户可在此修改
isSaveFig = true;       % 是否保存图像
savePath = 'Imgs/';  % 图像保存路径

isSaveData = true;      % 是否保存数据
dataPath = 'SimulationData/'; % 数据保存路径

% 自动创建输出文件夹（如果不存在）
if ~exist(savePath, 'dir'); mkdir(savePath); end
if ~exist(dataPath, 'dir'); mkdir(dataPath); end

% 清理旧的仿真结果文件夹（可选）
if isSaveFig && exist(savePath, 'dir')
    fprintf('>>> Cleaning old Figures folder...\n');
    rmdir(savePath, 's');
    mkdir(savePath);
end

if isSaveData && exist(dataPath, 'dir')
    fprintf('>>> Cleaning old SimulationData folder...\n');
    rmdir(dataPath, 's');
    mkdir(dataPath);
end

% ========== 状态显示配置 ==========
SHOW_DETAILED_STATUS = true;    % 是否显示详细步骤

% 计算总迭代次数
totalIterations = numScenarios * num_snr * 6; % 6种算法(All-UE MR/L/R, DCC-UE MR/L/R)
completedIterations = 0;

% ========== 打印仿真配置信息 ==========
printSimConfig(L, N, K, tau_c, tau_p, SNR_dB, numScenarios, nbrOfRealizations, sigma_e, nIter, isSaveFig, isSaveData);

% 初始化结果存储 (用于累加后取均值)
ESR_MR_all_total = zeros(num_snr,1);
ESR_L_MMSE_all_total = zeros(num_snr,1);
ESR_R_MMSE_all_total = zeros(num_snr,1);
ESR_MR_dcc_total = zeros(num_snr,1);
ESR_L_MMSE_dcc_total = zeros(num_snr,1);
ESR_R_MMSE_dcc_total = zeros(num_snr,1);

%% ================= Scenario Loop =================
fprintf('▶ Starting main simulation loop...\n');
fprintf('  Total scenarios: %d | SNR points: %d | Algorithms: 6 (MR, L-MMSE, R-MMSE)\n\n', numScenarios, num_snr);

for s = 1:numScenarios
    scenarioProgress = (s - 1) / numScenarios * 100;
    
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║  SCENARIO %2d / %2d  [ %5.1f%% ]                              ║\n', s, numScenarios, scenarioProgress);
    fprintf('╚══════════════════════════════════════════════════════════════╝\n');
    
    % ========== Step 1: 生成场景布局 ==========
    if SHOW_DETAILED_STATUS
        fprintf('  ┌─ [1/5] Generating scenario setup (AP/UE distribution)...\n');
    end
    [gainOverNoisedB,R,pilotIndex,D,~,APpositions,UEpositions] = generateSetup(L,K,N,tau_p,1,s,ASD_varphi,ASD_theta);
    if SHOW_DETAILED_STATUS
        fprintf('  │   └─ Done\n');
    end
    
    % 绘制场景布局（根据用户设置的数量）
    if s <= numScenariosToPlot
        plotScenarioSetup(APpositions, UEpositions, s, isSaveFig, savePath, isSaveData, dataPath);
    end
    
    % ========== Step 2: 信道估计 ==========
    if SHOW_DETAILED_STATUS
        fprintf('  ┌─ [2/5] Performing channel estimation...\n');
        fprintf('  │   └─ Realizations: %d, Pilots: %d\n', nbrOfRealizations, tau_p);
    end
    [Hhat,H_ideal,~,C] = functionChannelEstimates(R,nbrOfRealizations,L,K,N,tau_p,pilotIndex,p);
    if SHOW_DETAILED_STATUS
        fprintf('  │   └─ Done\n');
    end
    
    % ========== Step 3: 引入 CSI 误差 ==========
    if SHOW_DETAILED_STATUS
        fprintf('  ┌─ [3/5] Introducing CSI error (sigma_e = %.2f)...\n', sigma_e);
    end
    H = Hhat + sqrt(sigma_e^2/2)*(randn(size(Hhat))+1i*randn(size(Hhat)));
    if SHOW_DETAILED_STATUS
        fprintf('  │   └─ Done\n');
    end
    
    gainOverNoise = db2pow(gainOverNoisedB(:,:,1));
    D_dcc = D(:,:,1); % 获取 DCC 接入矩阵

    %% ================= SNR扫描 =================
    fprintf('  ┌─ [4/5] Running SNR sweep (%d points)...\n', num_snr);
    
    for snr_idx = 1:num_snr
        Pt = db2pow(SNR_dB(snr_idx)); % 总功率
        
        % --- All-UE Case ---
        D_all = ones(L,K);
        rho_dist_all = computeRhoDist(D_all, gainOverNoise, Pt, L, K);
        
        if SHOW_DETAILED_STATUS
            fprintf('  │   ┌─ SNR %2d/%2d (%.1f dB) - All-UE\n', snr_idx, num_snr, SNR_dB(snr_idx));
        end
        
        % MR (Maximum Ratio) precoding
        [V_MR_all, scaling_MR_all] = functionPrecoding_MR(Hhat, nbrOfRealizations, N, K, L);
        SE_MR_all = functionComputeSE_downlink_MR(H, V_MR_all, scaling_MR_all, D_all, tau_c, tau_p, nbrOfRealizations, N, K, L, rho_dist_all);
        
        SE_L_all = functionComputeSE_downlink_LMMSE(Hhat,H,D_all,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,rho_dist_all);
        SE_R_all = functionComputeSE_downlink_RobustMMSE(Hhat,H,D_all,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,rho_dist_all,sigma_e,Pt,nIter);
        
        if SHOW_DETAILED_STATUS
            fprintf('  │   │   ├─ MR:          SE=%.4f\n', sum(SE_MR_all));
            fprintf('  │   │   ├─ L-MMSE:      SE=%.4f\n', sum(SE_L_all));
            fprintf('  │   │   └─ Robust-MMSE: SE=%.4f\n', sum(SE_R_all));
        end
        
        ESR_MR_all_total(snr_idx) = ESR_MR_all_total(snr_idx) + sum(SE_MR_all);
        ESR_L_MMSE_all_total(snr_idx) = ESR_L_MMSE_all_total(snr_idx) + sum(SE_L_all);
        ESR_R_MMSE_all_total(snr_idx) = ESR_R_MMSE_all_total(snr_idx) + sum(SE_R_all);
        
        % --- DCC-UE Case ---
        rho_dist_dcc = computeRhoDist(D_dcc, gainOverNoise, Pt, L, K);
        
        if SHOW_DETAILED_STATUS
            fprintf('  │   └─ SNR %2d/%2d (%.1f dB) - DCC-UE\n', snr_idx, num_snr, SNR_dB(snr_idx));
        end
        
        % MR (Maximum Ratio) precoding for DCC
        [V_MR_dcc, scaling_MR_dcc] = functionPrecoding_MR(Hhat, nbrOfRealizations, N, K, L);
        SE_MR_dcc = functionComputeSE_downlink_MR(H, V_MR_dcc, scaling_MR_dcc, D_dcc, tau_c, tau_p, nbrOfRealizations, N, K, L, rho_dist_dcc);
        
        SE_L_dcc = functionComputeSE_downlink_LMMSE(Hhat,H,D_dcc,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,rho_dist_dcc);
        SE_R_dcc = functionComputeSE_downlink_RobustMMSE(Hhat,H,D_dcc,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,rho_dist_dcc,sigma_e,Pt,nIter);
        
        if SHOW_DETAILED_STATUS
            fprintf('  │       ├─ MR:          SE=%.4f\n', sum(SE_MR_dcc));
            fprintf('  │       ├─ L-MMSE:      SE=%.4f\n', sum(SE_L_dcc));
            fprintf('  │       └─ Robust-MMSE: SE=%.4f\n', sum(SE_R_dcc));
        end
        
        ESR_MR_dcc_total(snr_idx) = ESR_MR_dcc_total(snr_idx) + sum(SE_MR_dcc);
        ESR_L_MMSE_dcc_total(snr_idx) = ESR_L_MMSE_dcc_total(snr_idx) + sum(SE_L_dcc);
        ESR_R_MMSE_dcc_total(snr_idx) = ESR_R_MMSE_dcc_total(snr_idx) + sum(SE_R_dcc);
        
        % 更新进度
        completedIterations = completedIterations + 6;
        overallProgress = completedIterations / totalIterations * 100;
        
        % 显示进度条
        barLen = 40;
        filled = round(barLen * overallProgress / 100);
        barStr = repmat('=', 1, filled);
        spaces = repmat(' ', 1, barLen - filled);
        fprintf('  │  [%s%s] %5.1f%%\n', barStr, spaces, overallProgress);
    end
    
    fprintf('  └─ SNR sweep completed\n');
    
    % ========== Step 5: 场景汇总 ==========
    fprintf('\n');
    fprintf('  ┌─ [5/5] Scenario %d Summary:\n', s);
    fprintf('  │   └─ Progress: %d/%d scenarios completed\n', s, numScenarios);
    fprintf('  └─ Overall: %.1f%% done\n\n', s / numScenarios * 100);
end

% 计算均值
ESR_MR_all = ESR_MR_all_total / numScenarios;
ESR_L_MMSE_all = ESR_L_MMSE_all_total / numScenarios;
ESR_R_MMSE_all = ESR_R_MMSE_all_total / numScenarios;
ESR_MR_dcc = ESR_MR_dcc_total / numScenarios;
ESR_L_MMSE_dcc = ESR_L_MMSE_dcc_total / numScenarios;
ESR_R_MMSE_dcc = ESR_R_MMSE_dcc_total / numScenarios;

%% ================= Output Results =================
printFinalResults(ESR_MR_all, ESR_L_MMSE_all, ESR_R_MMSE_all, ESR_MR_dcc, ESR_L_MMSE_dcc, ESR_R_MMSE_dcc, ...
    numScenarios, nbrOfRealizations, num_snr, totalIterations, isSaveFig, isSaveData, savePath, dataPath);

plotESRResults(SNR_dB, ESR_MR_all, ESR_L_MMSE_all, ESR_R_MMSE_all, ESR_MR_dcc, ESR_L_MMSE_dcc, ESR_R_MMSE_dcc, ...
    numScenarios, isSaveFig, savePath, isSaveData, dataPath);
