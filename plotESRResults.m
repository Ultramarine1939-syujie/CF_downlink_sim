% 绘制 ESR 结果图并保存
function plotESRResults(SNR_dB, ESR_L_all, ESR_R_all, ESR_L_dcc, ESR_R_dcc, numScenarios, isSaveFig, savePath, isSaveData, dataPath)
    figure;
    plot(SNR_dB, ESR_L_all,  '-s', 'LineWidth', 2); hold on;
    plot(SNR_dB, ESR_R_all,  '-o', 'LineWidth', 2);
    plot(SNR_dB, ESR_L_dcc, '--s', 'LineWidth', 2);
    plot(SNR_dB, ESR_R_dcc, '--o', 'LineWidth', 2);

    legend('L-MMSE (All)', 'Robust MMSE (All)', 'L-MMSE (DCC)', 'Robust MMSE (DCC)');
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
            'SNR_dB', 'ESR_L_all', 'ESR_R_all', 'ESR_L_dcc', 'ESR_R_dcc', 'numScenarios');
    end
end
