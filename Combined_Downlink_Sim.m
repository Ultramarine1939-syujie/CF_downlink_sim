%% ================= Initialization =================
close all; clear;

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
numScenarios = 10; % 场景实验次数
nbrOfRealizations = 200; % 每个场景下的信道数

% 接口：设置要展示的仿真场景数量
numScenariosToPlot = 1; % 用户可在此修改
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

% 初始化结果存储 (用于累加后取均值)
ESR_L_MMSE_all_total = zeros(num_snr,1);
ESR_R_MMSE_all_total = zeros(num_snr,1);
ESR_L_MMSE_dcc_total = zeros(num_snr,1);
ESR_R_MMSE_dcc_total = zeros(num_snr,1);

%% ================= Scenario Loop =================
for s = 1:numScenarios
    fprintf('>>> Running Scenario %d/%d...\n', s, numScenarios);
    
    % 1. 生成当前场景的 AP/UE 分布及相关矩阵
    [gainOverNoisedB,R,pilotIndex,D,~,APpositions,UEpositions] = generateSetup(L,K,N,tau_p,1,s,ASD_varphi,ASD_theta);
    
    % 绘制场景布局（根据用户设置的数量）
    if s <= numScenariosToPlot
        plotScenarioSetup(APpositions, UEpositions, s, isSaveFig, savePath, isSaveData, dataPath);
    end
    
    % 2. 生成信道估计和真实信道
    [Hhat,H_ideal,~,C] = functionChannelEstimates(R,nbrOfRealizations,L,K,N,tau_p,pilotIndex,p);
    
    % 3. 引入 CSI 误差
    H = Hhat + sqrt(sigma_e^2/2)*(randn(size(Hhat))+1i*randn(size(Hhat)));
    
    gainOverNoise = db2pow(gainOverNoisedB(:,:,1));
    D_dcc = D(:,:,1); % 获取 DCC 接入矩阵

    %% ================= SNR扫描 =================
    for snr_idx = 1:num_snr
        Pt = db2pow(SNR_dB(snr_idx)); % 总功率
        fprintf('    SNR Index: %d/%d (SNR = %d dB)\n', snr_idx, num_snr, SNR_dB(snr_idx));
        
        % --- All-UE Case ---
        D_all = ones(L,K);
        rho_dist_all = zeros(L,K);
        for l=1:L
            servedUEs = find(D_all(l,:)==1);
            normFactor = sum(sqrt(gainOverNoise(l,servedUEs)));
            for k=servedUEs
                rho_dist_all(l,k) = Pt * sqrt(gainOverNoise(l,k))/normFactor;
            end
        end
        
        SE_L_all = functionComputeSE_downlink_LMMSE(Hhat,H,D_all,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,rho_dist_all);
        SE_R_all = functionComputeSE_downlink_RobustMMSE(Hhat,H,D_all,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,rho_dist_all,sigma_e,Pt,nIter);
        
        ESR_L_MMSE_all_total(snr_idx) = ESR_L_MMSE_all_total(snr_idx) + sum(SE_L_all);
        ESR_R_MMSE_all_total(snr_idx) = ESR_R_MMSE_all_total(snr_idx) + sum(SE_R_all);
        
        % --- DCC-UE Case ---
        rho_dist_dcc = zeros(L,K);
        for l=1:L
            servedUEs = find(D_dcc(l,:)==1);
            if isempty(servedUEs); continue; end
            normFactor = sum(sqrt(gainOverNoise(l,servedUEs)));
            for k=servedUEs
                rho_dist_dcc(l,k) = Pt * sqrt(gainOverNoise(l,k))/normFactor;
            end
        end
        
        SE_L_dcc = functionComputeSE_downlink_LMMSE(Hhat,H,D_dcc,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,rho_dist_dcc);
        SE_R_dcc = functionComputeSE_downlink_RobustMMSE(Hhat,H,D_dcc,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,rho_dist_dcc,sigma_e,Pt,nIter);
        
        ESR_L_MMSE_dcc_total(snr_idx) = ESR_L_MMSE_dcc_total(snr_idx) + sum(SE_L_dcc);
        ESR_R_MMSE_dcc_total(snr_idx) = ESR_R_MMSE_dcc_total(snr_idx) + sum(SE_R_dcc);
    end
end

% 计算均值
ESR_L_MMSE_all = ESR_L_MMSE_all_total / numScenarios;
ESR_R_MMSE_all = ESR_R_MMSE_all_total / numScenarios;
ESR_L_MMSE_dcc = ESR_L_MMSE_dcc_total / numScenarios;
ESR_R_MMSE_dcc = ESR_R_MMSE_dcc_total / numScenarios;

fprintf('>>> Simulation Completed Successfully.\n');

%% ================= Plot Results =================
figure;
plot(SNR_dB, ESR_L_MMSE_all,'-s','LineWidth',2); hold on;
plot(SNR_dB, ESR_R_MMSE_all,'-o','LineWidth',2);
plot(SNR_dB, ESR_L_MMSE_dcc,'--s','LineWidth',2);
plot(SNR_dB, ESR_R_MMSE_dcc,'--o','LineWidth',2);

legend('L-MMSE (All)','Robust MMSE (All)','L-MMSE (DCC)','Robust MMSE (DCC)');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title(sprintf('Averaged ESR over %d Scenarios: All-AP vs DCC', numScenarios));
grid on;

% 保存结果图
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Averaged_ESR_Results.fig'));
    saveas(gcf, fullfile(savePath, 'Averaged_ESR_Results.png'));
end

% 保存仿真数据
if isSaveData
    save(fullfile(dataPath, 'Simulation_Results_Data.mat'), ...
        'SNR_dB', 'ESR_L_MMSE_all', 'ESR_R_MMSE_all', 'ESR_L_MMSE_dcc', 'ESR_R_MMSE_dcc', 'numScenarios');
end
