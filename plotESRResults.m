% 绘制 ESR 结果图并保存
function plotESRResults(SNR_dB, ESR_MR_all, ESR_L_all, ESR_R_all, ESR_PSO_MR_all, ESR_PSO_L_all, ESR_PSO_R_all, ...
    ESR_MR_dcc, ESR_L_dcc, ESR_R_dcc, ESR_PSO_MR_dcc, ESR_PSO_L_dcc, ESR_PSO_R_dcc, ...
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

% 图3: L-MMSE vs PSO+L-MMSE (All + DCC)
figure('Visible', 'off');
plot(SNR_dB, ESR_L_all, '-s', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_PSO_L_all, '-^', 'LineWidth', 2);
plot(SNR_dB, ESR_L_dcc, '--s', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_L_dcc, '--^', 'LineWidth', 2);
legend('L-MMSE (All)', 'PSO+L-MMSE (All)', 'L-MMSE (DCC)', 'PSO+L-MMSE (DCC)');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title(sprintf('L-MMSE vs PSO+L-MMSE (Avg over %d Scenarios)', numScenarios));
grid on;
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig3_LMMSE_vs_PSO_LMMSE.fig'));
    saveas(gcf, fullfile(savePath, 'Fig3_LMMSE_vs_PSO_LMMSE.png'));
end

% 图4: R-MMSE vs PSO+R-MMSE (All + DCC)
figure('Visible', 'off');
plot(SNR_dB, ESR_R_all, '-o', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_PSO_R_all, '-^', 'LineWidth', 2);
plot(SNR_dB, ESR_R_dcc, '--o', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_R_dcc, '--^', 'LineWidth', 2);
legend('R-MMSE (All)', 'PSO+R-MMSE (All)', 'R-MMSE (DCC)', 'PSO+R-MMSE (DCC)');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title(sprintf('R-MMSE vs PSO+R-MMSE (Avg over %d Scenarios)', numScenarios));
grid on;
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig4_RMMSE_vs_PSO_RMMSE.fig'));
    saveas(gcf, fullfile(savePath, 'Fig4_RMMSE_vs_PSO_RMMSE.png'));
end

% 图5: MR vs PSO+MR (All + DCC)
figure('Visible', 'off');
plot(SNR_dB, ESR_MR_all, '-^', 'LineWidth', 2); hold on;
plot(SNR_dB, ESR_PSO_MR_all, '-v', 'LineWidth', 2);
plot(SNR_dB, ESR_MR_dcc, '--^', 'LineWidth', 2);
plot(SNR_dB, ESR_PSO_MR_dcc, '--v', 'LineWidth', 2);
legend('MR (All)', 'PSO+MR (All)', 'MR (DCC)', 'PSO+MR (DCC)');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (ESR)');
title(sprintf('MR vs PSO+MR (Avg over %d Scenarios)', numScenarios));
grid on;
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig5_MR_vs_PSO_MR.fig'));
    saveas(gcf, fullfile(savePath, 'Fig5_MR_vs_PSO_MR.png'));
end

% 保存仿真数据
if isSaveData
    save(fullfile(dataPath, 'Simulation_Results_Data.mat'), ...
        'SNR_dB', ...
        'ESR_MR_all', 'ESR_L_all', 'ESR_R_all', 'ESR_PSO_MR_all', 'ESR_PSO_L_all', 'ESR_PSO_R_all', ...
        'ESR_MR_dcc', 'ESR_L_dcc', 'ESR_R_dcc', 'ESR_PSO_MR_dcc', 'ESR_PSO_L_dcc', 'ESR_PSO_R_dcc', ...
        'numScenarios');
end
end