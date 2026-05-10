% 绘制 ESR 结果图并保存
function plotESRResults(SNR_dB, ESR_MR_all, ESR_L_all, ESR_R_all, ESR_PSO_MR_all, ESR_PSO_L_all, ESR_PSO_R_all, ...
    ESR_MR_dcc, ESR_L_dcc, ESR_R_dcc, ESR_PSO_MR_dcc, ESR_PSO_L_dcc, ESR_PSO_R_dcc, ...
    ESR_Random_MR_all, ESR_Random_L_all, ESR_Random_R_all, ...
    ESR_Random_MR_dcc, ESR_Random_L_dcc, ESR_Random_R_dcc, ...
    ESR_EPA_MR_all, ESR_EPA_L_all, ESR_EPA_R_all, ...
    ESR_EPA_MR_dcc, ESR_EPA_L_dcc, ESR_EPA_R_dcc, ...
    ESR_WMMSE_MR_all, ESR_WMMSE_L_all, ESR_WMMSE_R_all, ...
    ESR_WMMSE_MR_dcc, ESR_WMMSE_L_dcc, ESR_WMMSE_R_dcc, ...
    numScenarios, isSaveFig, savePath, isSaveData, dataPath)

% 图1: PSO+MR vs PSO+L-MMSE vs PSO+R-MMSE (All场景)
figure('Visible', 'off');
plot(SNR_dB, ESR_PSO_MR_all, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_PSO_L_all, '-s', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_R_all, '-o', 'LineWidth', 2);
legend('PSO+MR (All)', 'PSO+L-MMSE (All)', 'PSO+R-MMSE (All)');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title(sprintf('All: PSO+MR vs PSO+L-MMSE vs PSO+R-MMSE (Avg over %d Scenarios)', numScenarios));
grid on;
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig1_All_PSO_Comparison.fig'));
    saveas(gcf, fullfile(savePath, 'Fig1_All_PSO_Comparison.png'));
end

% 图2: PSO+MR vs PSO+L-MMSE vs PSO+R-MMSE (DCC场景)
figure('Visible', 'off');
plot(SNR_dB, ESR_PSO_MR_dcc, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_PSO_L_dcc, '-s', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_R_dcc, '-o', 'LineWidth', 2);
legend('PSO+MR (DCC)', 'PSO+L-MMSE (DCC)', 'PSO+R-MMSE (DCC)');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title(sprintf('DCC: PSO+MR vs PSO+L-MMSE vs PSO+R-MMSE (Avg over %d Scenarios)', numScenarios));
grid on;
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig2_DCC_PSO_Comparison.fig'));
    saveas(gcf, fullfile(savePath, 'Fig2_DCC_PSO_Comparison.png'));
end

% 图3: L-MMSE vs 所有功率分配方法 (All + DCC)
figure('Visible', 'off');
plot(SNR_dB, ESR_L_all, '-s', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_Random_L_all, '-*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_L_all, '-^', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_L_all, '-d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_L_all, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_L_dcc, '--s', 'LineWidth', 2);
plot(SNR_dB, ESR_Random_L_dcc, '--*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_L_dcc, '--^', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_L_dcc, '--d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_L_dcc, '--v', 'LineWidth', 2);
legend('Baseline (All)', 'Random (All)', 'PSO (All)', 'EPA (All)', 'WMMSE (All)', ...
    'Baseline (DCC)', 'Random (DCC)', 'PSO (DCC)', 'EPA (DCC)', 'WMMSE (DCC)');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title(sprintf('L-MMSE: All Power Allocation Methods (Avg over %d Scenarios)', numScenarios));
grid on;
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig3_LMMSE_AllMethods.fig'));
    saveas(gcf, fullfile(savePath, 'Fig3_LMMSE_AllMethods.png'));
end

% 图4: R-MMSE vs 所有功率分配方法 (All + DCC)
figure('Visible', 'off');
plot(SNR_dB, ESR_R_all, '-o', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_Random_R_all, '-*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_R_all, '-^', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_R_all, '-d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_R_all, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_R_dcc, '--o', 'LineWidth', 2);
plot(SNR_dB, ESR_Random_R_dcc, '--*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_R_dcc, '--^', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_R_dcc, '--d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_R_dcc, '--v', 'LineWidth', 2);
legend('Baseline (All)', 'Random (All)', 'PSO (All)', 'EPA (All)', 'WMMSE (All)', ...
    'Baseline (DCC)', 'Random (DCC)', 'PSO (DCC)', 'EPA (DCC)', 'WMMSE (DCC)');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title(sprintf('R-MMSE: All Power Allocation Methods (Avg over %d Scenarios)', numScenarios));
grid on;
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig4_RMMSE_AllMethods.fig'));
    saveas(gcf, fullfile(savePath, 'Fig4_RMMSE_AllMethods.png'));
end

% 图5: MR vs 所有功率分配方法 (All + DCC)
figure('Visible', 'off');
plot(SNR_dB, ESR_MR_all, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_Random_MR_all, '-*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_MR_all, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_MR_all, '-d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_MR_all, '-p', 'LineWidth', 2);
plot(SNR_dB, ESR_MR_dcc, '--^', 'LineWidth', 2);
plot(SNR_dB, ESR_Random_MR_dcc, '--*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_MR_dcc, '--v', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_MR_dcc, '--d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_MR_dcc, '--p', 'LineWidth', 2);
legend('Baseline (All)', 'Random (All)', 'PSO (All)', 'EPA (All)', 'WMMSE (All)', ...
    'Baseline (DCC)', 'Random (DCC)', 'PSO (DCC)', 'EPA (DCC)', 'WMMSE (DCC)');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title(sprintf('MR: All Power Allocation Methods (Avg over %d Scenarios)', numScenarios));
grid on;
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig5_MR_AllMethods.fig'));
    saveas(gcf, fullfile(savePath, 'Fig5_MR_AllMethods.png'));
end

% 图6: 功率分配算法综合对比 - MR预编码 (All + DCC 在同一图中)
figure('Visible', 'off');
subplot(1,2,1);
plot(SNR_dB, ESR_MR_all, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_Random_MR_all, '-*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_MR_all, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_MR_all, '-d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_MR_all, '-p', 'LineWidth', 2);
legend('Baseline', 'Random', 'PSO', 'EPA', 'WMMSE', 'Location', 'Best');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title('MR Precoding - All-UE');
grid on;

subplot(1,2,2);
plot(SNR_dB, ESR_MR_dcc, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_Random_MR_dcc, '-*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_MR_dcc, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_MR_dcc, '-d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_MR_dcc, '-p', 'LineWidth', 2);
legend('Baseline', 'Random', 'PSO', 'EPA', 'WMMSE', 'Location', 'Best');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title('MR Precoding - DCC-UE');
grid on;
% 添加总标题
try
    sgtitle(sprintf('Power Allocation Methods Comparison (MR, Avg over %d Scenarios)', numScenarios));
catch
    % 兼容旧版 MATLAB，手动添加总标题
    axes('Position', [0, 0.92, 1, 0.08], 'Visible', 'off');
    text(0.5, 0.5, sprintf('Power Allocation Methods Comparison (MR, Avg over %d Scenarios)', numScenarios), ...
        'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
end
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig6_PowerAlloc_MR_Comparison.fig'));
    saveas(gcf, fullfile(savePath, 'Fig6_PowerAlloc_MR_Comparison.png'));
end

% 图7: 功率分配算法综合对比 - L-MMSE预编码 (All + DCC 在同一图中)
figure('Visible', 'off');
subplot(1,2,1);
plot(SNR_dB, ESR_L_all, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_Random_L_all, '-*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_L_all, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_L_all, '-d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_L_all, '-p', 'LineWidth', 2);
legend('Baseline', 'Random', 'PSO', 'EPA', 'WMMSE', 'Location', 'Best');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title('L-MMSE Precoding - All-UE');
grid on;

subplot(1,2,2);
plot(SNR_dB, ESR_L_dcc, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_Random_L_dcc, '-*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_L_dcc, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_L_dcc, '-d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_L_dcc, '-p', 'LineWidth', 2);
legend('Baseline', 'Random', 'PSO', 'EPA', 'WMMSE', 'Location', 'Best');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title('L-MMSE Precoding - DCC-UE');
grid on;
% 添加总标题
try
    sgtitle(sprintf('Power Allocation Methods Comparison (L-MMSE, Avg over %d Scenarios)', numScenarios));
catch
    axes('Position', [0, 0.92, 1, 0.08], 'Visible', 'off');
    text(0.5, 0.5, sprintf('Power Allocation Methods Comparison (L-MMSE, Avg over %d Scenarios)', numScenarios), ...
        'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
end
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig7_PowerAlloc_LMMSE_Comparison.fig'));
    saveas(gcf, fullfile(savePath, 'Fig7_PowerAlloc_LMMSE_Comparison.png'));
end

% 图8: 功率分配算法综合对比 - R-MMSE预编码 (All + DCC 在同一图中)
figure('Visible', 'off');
subplot(1,2,1);
plot(SNR_dB, ESR_R_all, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_Random_R_all, '-*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_R_all, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_R_all, '-d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_R_all, '-p', 'LineWidth', 2);
legend('Baseline', 'Random', 'PSO', 'EPA', 'WMMSE', 'Location', 'Best');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title('R-MMSE Precoding - All-UE');
grid on;

subplot(1,2,2);
plot(SNR_dB, ESR_R_dcc, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_Random_R_dcc, '-*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_R_dcc, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_R_dcc, '-d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_R_dcc, '-p', 'LineWidth', 2);
legend('Baseline', 'Random', 'PSO', 'EPA', 'WMMSE', 'Location', 'Best');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title('R-MMSE Precoding - DCC-UE');
grid on;
% 添加总标题
try
    sgtitle(sprintf('Power Allocation Methods Comparison (R-MMSE, Avg over %d Scenarios)', numScenarios));
catch
    axes('Position', [0, 0.92, 1, 0.08], 'Visible', 'off');
    text(0.5, 0.5, sprintf('Power Allocation Methods Comparison (R-MMSE, Avg over %d Scenarios)', numScenarios), ...
        'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
end
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig8_PowerAlloc_RMMSE_Comparison.fig'));
    saveas(gcf, fullfile(savePath, 'Fig8_PowerAlloc_RMMSE_Comparison.png'));
end

% 图9: 所有功率分配方法汇总对比 - 选最优的预编码方案 (All场景)
figure('Visible', 'off');
% 找出每种功率分配方法在All场景下的最优预编码
ESR_Baseline_All_best = max([ESR_MR_all, ESR_L_all, ESR_R_all], [], 2);
ESR_Random_All_best = max([ESR_Random_MR_all, ESR_Random_L_all, ESR_Random_R_all], [], 2);
ESR_PSO_All_best = max([ESR_PSO_MR_all, ESR_PSO_L_all, ESR_PSO_R_all], [], 2);
ESR_EPA_All_best = max([ESR_EPA_MR_all, ESR_EPA_L_all, ESR_EPA_R_all], [], 2);
ESR_WMMSE_All_best = max([ESR_WMMSE_MR_all, ESR_WMMSE_L_all, ESR_WMMSE_R_all], [], 2);

plot(SNR_dB, ESR_Baseline_All_best, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_Random_All_best, '-*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_All_best, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_All_best, '-d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_All_best, '-p', 'LineWidth', 2);
legend('Baseline (Best Precoding)', 'Random (Best Precoding)', 'PSO (Best Precoding)', ...
    'EPA (Best Precoding)', 'WMMSE (Best Precoding)', 'Location', 'Best');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title(sprintf('All Power Allocation Methods - Best Performance (All-UE, Avg over %d Scenarios)', numScenarios));
grid on;
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig9_AllMethods_All_Best.fig'));
    saveas(gcf, fullfile(savePath, 'Fig9_AllMethods_All_Best.png'));
end

% 图10: 所有功率分配方法汇总对比 - 选最优的预编码方案 (DCC场景)
figure('Visible', 'off');
ESR_Baseline_DCC_best = max([ESR_MR_dcc, ESR_L_dcc, ESR_R_dcc], [], 2);
ESR_Random_DCC_best = max([ESR_Random_MR_dcc, ESR_Random_L_dcc, ESR_Random_R_dcc], [], 2);
ESR_PSO_DCC_best = max([ESR_PSO_MR_dcc, ESR_PSO_L_dcc, ESR_PSO_R_dcc], [], 2);
ESR_EPA_DCC_best = max([ESR_EPA_MR_dcc, ESR_EPA_L_dcc, ESR_EPA_R_dcc], [], 2);
ESR_WMMSE_DCC_best = max([ESR_WMMSE_MR_dcc, ESR_WMMSE_L_dcc, ESR_WMMSE_R_dcc], [], 2);

plot(SNR_dB, ESR_Baseline_DCC_best, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_Random_DCC_best, '-*', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_DCC_best, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_EPA_DCC_best, '-d', 'LineWidth', 2);
plot(SNR_dB, ESR_WMMSE_DCC_best, '-p', 'LineWidth', 2);
legend('Baseline (Best Precoding)', 'Random (Best Precoding)', 'PSO (Best Precoding)', ...
    'EPA (Best Precoding)', 'WMMSE (Best Precoding)', 'Location', 'Best');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title(sprintf('All Power Allocation Methods - Best Performance (DCC-UE, Avg over %d Scenarios)', numScenarios));
grid on;
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig10_AllMethods_DCC_Best.fig'));
    saveas(gcf, fullfile(savePath, 'Fig10_AllMethods_DCC_Best.png'));
end

% 保存仿真数据
if isSaveData
    save(fullfile(dataPath, 'Simulation_Results_Data.mat'), ...
        'SNR_dB', ...
        'ESR_MR_all', 'ESR_L_all', 'ESR_R_all', 'ESR_PSO_MR_all', 'ESR_PSO_L_all', 'ESR_PSO_R_all', ...
        'ESR_MR_dcc', 'ESR_L_dcc', 'ESR_R_dcc', 'ESR_PSO_MR_dcc', 'ESR_PSO_L_dcc', 'ESR_PSO_R_dcc', ...
        'ESR_Random_MR_all', 'ESR_Random_L_all', 'ESR_Random_R_all', ...
        'ESR_Random_MR_dcc', 'ESR_Random_L_dcc', 'ESR_Random_R_dcc', ...
        'ESR_EPA_MR_all', 'ESR_EPA_L_all', 'ESR_EPA_R_all', ...
        'ESR_EPA_MR_dcc', 'ESR_EPA_L_dcc', 'ESR_EPA_R_dcc', ...
        'ESR_WMMSE_MR_all', 'ESR_WMMSE_L_all', 'ESR_WMMSE_R_all', ...
        'ESR_WMMSE_MR_dcc', 'ESR_WMMSE_L_dcc', 'ESR_WMMSE_R_dcc', ...
        'numScenarios');
end
end
