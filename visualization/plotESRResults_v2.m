%% 绘图模块 v2 - 聚焦最优方法发现
function plotESRResults_v2(ESR_mean, ESR_best, ESR_best_algo, algoTable, SNR_dB, ...
    numScenarios, isSaveFig, savePath, isSaveData, dataPath, Perf)

[numAlgos, num_snr] = size(ESR_mean);
avgESR = mean(ESR_mean, 2);
[~, globalRank] = sort(avgESR, 'descend');

% 将 struct array 提取为 cell 数组供 cellfun 使用
algoNames = {algoTable.name};

%% ===== 图1: Top-5 算法 ESR 曲线 (核心发现图) =====
figure('Visible', 'off', 'Position', [100 100 900 500]);
colors = lines(5);
markers = {'o', 's', '^', 'd', 'v'};

for i = 1:min(5, numAlgos)
    idx = globalRank(i);
    style = '-';
    if i == 1; lw = 2.5; else lw = 1.8; end
    plot(SNR_dB, ESR_mean(idx, :), [style markers{i}], ...
        'Color', colors(i, :), 'LineWidth', lw, 'MarkerSize', 8); hold on;
end

legendStr = cell(1, min(5, numAlgos));
for i = 1:min(5, numAlgos)
    idx = globalRank(i);
    rankMark = '';
    if i == 1; rankMark = ' [BEST]'; end
    legendStr{i} = [algoTable(idx).name rankMark];
end
legend(legendStr, 'Location', 'SouthEast');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (bit/s/Hz)');
title(sprintf('Top-5 Algorithms ESR Performance (Cell-Free, %d Scenarios)', numScenarios));
grid on;
set(gca, 'FontSize', 11);
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig1_Top5_Algorithms.png'));
    saveas(gcf, fullfile(savePath, 'Fig1_Top5_Algorithms.fig'));
end

%% ===== 图2: 每种功率分配方法的最优预编码对比 =====
figure('Visible', 'off', 'Position', [100 100 900 500]);

paList = {'Baseline', 'Random', 'EPA', 'PSO', 'WMMSE', 'GNN', 'GNN+FL'};
paColor = struct();
paColor.Baseline = [0.2 0.2 0.2];
paColor.Random   = [0.5 0.5 0.5];
paColor.EPA      = [0.0 0.6 0.0];
paColor.PSO      = [0.0 0.0 0.8];
paColor.WMMSE    = [0.8 0.0 0.0];
paColor.GNN      = [0.45 0.0 0.75];
paColor.GNNFL    = [0.93 0.69 0.13];

for pi = 1:length(paList)
    paName = paList{pi};
    paMask = cellfun(@(x) contains(x, paName), algoNames);
    if any(paMask)
        paESR = ESR_mean(paMask, :);
        paAvg = mean(paESR, 2);
        [~, bestIdx] = max(paAvg);

        % 找该PA对应的PC
        maskIdx = find(paMask);
        bestAlgoIdx = maskIdx(bestIdx);

        colorKey = paName;
        if strcmp(paName, 'GNN+FL'); colorKey = 'GNNFL'; end
        plot(SNR_dB, ESR_mean(bestAlgoIdx, :), '-', ...
            'Color', paColor.(colorKey), 'LineWidth', 2.2, 'DisplayName', paName); hold on;
    end
end

legend('Location', 'SouthEast');
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (bit/s/Hz)');
title('Best Precoding per Power Allocation Method');
grid on;
set(gca, 'FontSize', 11);
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig2_PA_BestPC.png'));
    saveas(gcf, fullfile(savePath, 'Fig2_PA_BestPC.fig'));
end

%% ===== 图3: 预编码方法横向对比 (All vs DCC) =====
figure('Visible', 'off', 'Position', [100 100 1100 500]);

pcList = {'MR', 'LMMSE', 'RMMSE'};
pcColor = struct();
pcColor.MR     = [0.0 0.45 0.75];
pcColor.LMMSE  = [0.75 0.0 0.45];
pcColor.RMMSE  = [0.0 0.6 0.0];

subplot(1, 2, 1);  % All-UE
for ci = 1:length(pcList)
    pcName = pcList{ci};
    mask = cellfun(@(x) contains(x, 'Baseline') && contains(x, pcName) && contains(x, 'All'), algoNames);
    if any(mask)
        idx = find(mask);
        plot(SNR_dB, ESR_mean(idx, :), '-o', ...
            'Color', pcColor.(pcName), 'LineWidth', 2, 'MarkerSize', 7); hold on;
    end
end
legend('Location', 'SouthEast'); xlabel('SNR (dB)'); ylabel('ESR');
title('All-UE: Baseline Precoding Comparison');
grid on;

subplot(1, 2, 2);  % DCC
for ci = 1:length(pcList)
    pcName = pcList{ci};
    mask = cellfun(@(x) contains(x, 'Baseline') && contains(x, pcName) && contains(x, 'DCC'), algoNames);
    if any(mask)
        idx = find(mask);
        plot(SNR_dB, ESR_mean(idx, :), '-o', ...
            'Color', pcColor.(pcName), 'LineWidth', 2, 'MarkerSize', 7); hold on;
    end
end
legend('Location', 'SouthEast'); xlabel('SNR (dB)'); ylabel('ESR');
title('DCC-UE: Baseline Precoding Comparison');
grid on;

set(gcf, 'Position', [100 100 1100 500]);
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig3_PC_All_vs_DCC.png'));
    saveas(gcf, fullfile(savePath, 'Fig3_PC_All_vs_DCC.fig'));
end

%% ===== 图4: 相对Baseline的增益热力图 =====
figure('Visible', 'off', 'Position', [100 100 1200 600]);

% 计算相对MR+Baseline+All的增益(dB)
baselineIdx = find(cellfun(@(x) contains(x,'Baseline') && contains(x,'MR') && contains(x,'All'), algoNames));
if isempty(baselineIdx); baselineIdx = 1; end
baselineESR = ESR_mean(baselineIdx, :);
gain_dB = 10 * log10(ESR_mean ./ baselineESR);

% 选取Top15算法绘制
topN = min(15, numAlgos);
topIdx = globalRank(1:topN);

imagesc(SNR_dB, 1:topN, gain_dB(topIdx, :));
colorbar;
caxis([-2 5]);
colormap jet;
set(gca, 'YTick', 1:topN, 'YTickLabel', algoNames(topIdx));
xlabel('SNR (dB)'); ylabel('Algorithm');
title('Relative Gain (dB) vs Baseline (MR+Baseline+All)');
set(gca, 'FontSize', 9);

% 标注每行最大值
for r = 1:topN
    [mx, ci] = max(gain_dB(topIdx(r), :));
    text(SNR_dB(ci)+0.3, r, sprintf('%+.1f', mx), 'FontSize', 8, 'Color', 'white', 'FontWeight', 'bold');
end
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig4_Gain_Heatmap.png'));
    saveas(gcf, fullfile(savePath, 'Fig4_Gain_Heatmap.fig'));
end

%% ===== 图5: 功率分配方法箱线图 (All SNR平均增益) =====
figure('Visible', 'off', 'Position', [100 100 800 500]);

paShortNames = {'BL', 'Rnd', 'EPA', 'PSO', 'WMMSE', 'GNN', 'GNN+FL'};
paGains = cell(1, length(paList));
for pi = 1:length(paList)
    paName = paList{pi};
    paMask = cellfun(@(x) contains(x, paName), algoNames);
    if any(paMask)
        paESR = ESR_mean(paMask, :);
        paAvg = mean(paESR, 2);
        [~, bestRelIdx] = max(paAvg);
        maskIdx = find(paMask);
        gain_vs_bl = ESR_mean(maskIdx(bestRelIdx), :) - ESR_mean(baselineIdx, :);
        paGains{pi} = gain_vs_bl;
    else
        paGains{pi} = zeros(1, num_snr);
    end
end

% 每种PA的平均增益
paMeanGain = cellfun(@mean, paGains);
[~, bestPA] = max(paMeanGain);

bar(1:length(paList), paMeanGain, 0.6, 'FaceColor', [0.3 0.3 0.8]);
hold on;
for pi = 1:length(paList)
    if pi == bestPA
        bar(pi, paMeanGain(pi), 0.6, 'FaceColor', [0.8 0.2 0.2]);
        text(pi, paMeanGain(pi)+0.1, sprintf('BEST', paMeanGain(pi)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.8 0.2 0.2]);
    else
        text(pi, paMeanGain(pi)+0.05, sprintf('%.2f', paMeanGain(pi)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
    end
end
set(gca, 'XTick', 1:length(paList), 'XTickLabel', paShortNames);
xlabel('Power Allocation Method'); ylabel('Average ESR Gain vs Baseline');
title('Power Allocation Method Ranking (Avg Gain over All SNR)');
grid on; yline(0, '--k', 'LineWidth', 0.5);
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig5_PA_Ranking.png'));
    saveas(gcf, fullfile(savePath, 'Fig5_PA_Ranking.fig'));
end

%% ===== 图6: 最优组合追踪 (SNR vs Best Algo) =====
figure('Visible', 'off', 'Position', [100 100 900 400]);
colors2 = lines(num_snr);
for si = 1:num_snr
    topAlgo = ESR_best_algo{si};
    plot(si, ESR_best(si), 'o', 'MarkerSize', 14, ...
        'Color', colors2(si, :), 'LineWidth', 2); hold on;
    text(si+0.1, ESR_best(si), topAlgo, 'FontSize', 8, 'Rotation', 15);
end
plot(1:num_snr, ESR_best, '-k', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('SNR Index'); ylabel('Best ESR Found');
title('Optimal Algorithm Selection vs SNR Point');
xticks(1:num_snr); xticklabels(arrayfun(@(x) sprintf('%.0f dB', SNR_dB(x)), 1:num_snr, 'UniformOutput', false));
grid on;
if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig6_BestAlgo_Tracker.png'));
    saveas(gcf, fullfile(savePath, 'Fig6_BestAlgo_Tracker.fig'));
end

%% ===== 图7: Phase 4 指标 (ESR损失 / 同步时延与通信开销降低) =====
figure('Visible', 'off', 'Position', [100 100 1100 450]);

maskW = cellfun(@(x) contains(x, 'WMMSE'), algoNames);
maskGF = cellfun(@(x) contains(x, 'GNN+FL'), algoNames);
maskG = cellfun(@(x) contains(x, 'GNN') && ~contains(x, 'GNN+FL'), algoNames);

subplot(1, 2, 1);
if any(maskW) && any(maskGF)
    idxW = find(maskW);
    idxGF = find(maskGF);
    [~, wRel] = max(mean(ESR_mean(idxW, :), 2));
    [~, gfRel] = max(mean(ESR_mean(idxGF, :), 2));
    wBest = idxW(wRel);
    gfBest = idxGF(gfRel);
    esrLoss = 1 - (ESR_mean(gfBest, :) ./ ESR_mean(wBest, :));
    plot(SNR_dB, esrLoss, '-o', 'LineWidth', 2.2); grid on;
    xlabel('SNR (dB)'); ylabel('1 - ESR_{GNN+FL}/ESR_{WMMSE}');
    title('ESR Loss vs WMMSE (Best-PC per PA)');
else
    plot(SNR_dB, zeros(size(SNR_dB)), '-'); grid on;
    xlabel('SNR (dB)'); ylabel('ESR Loss');
    title('ESR Loss (missing WMMSE or GNN+FL curves)');
end

subplot(1, 2, 2);
hasPerf = (nargin >= 11) && exist('Perf', 'var') && isstruct(Perf) && isfield(Perf, 'time_pa_sec');
if hasPerf
    modeIdx = find(strcmp(Perf.modeNames, 'All'), 1);
    if isempty(modeIdx); modeIdx = 1; end
    mW = find(strcmp(Perf.methodNames, 'WMMSE'), 1);
    mGF = find(strcmp(Perf.methodNames, 'GNN+FL'), 1);
    if ~isempty(mW) && ~isempty(mGF)
        T_w = squeeze(Perf.time_pa_sec(mW, :, modeIdx));
        T_gf = squeeze(Perf.time_pa_sec(mGF, :, modeIdx));
        D_w = squeeze(Perf.comm_bytes(mW, :, modeIdx));
        D_gf = squeeze(Perf.comm_bytes(mGF, :, modeIdx));
        tRed = (T_w - T_gf) ./ max(T_w, eps);
        dRed = (D_w - D_gf) ./ max(D_w, eps);
        plot(SNR_dB, tRed, '-s', 'LineWidth', 2.2); hold on;
        plot(SNR_dB, dRed, '-d', 'LineWidth', 2.2); grid on;
        legend({'Sync Latency Reduction', 'Comm Reduction'}, 'Location', 'SouthEast');
        xlabel('SNR (dB)'); ylabel('Reduction Ratio');
        title('Phase 4 Sync Performance (estimated)');
    else
        plot(SNR_dB, zeros(size(SNR_dB)), '-'); grid on;
        xlabel('SNR (dB)'); ylabel('Reduction Ratio');
        title('Phase 4 Sync Performance (missing Perf indices)');
    end
else
    plot(SNR_dB, zeros(size(SNR_dB)), '-'); grid on;
    xlabel('SNR (dB)'); ylabel('Reduction Ratio');
    title('Phase 4 Sync Performance (Perf not provided)');
end

if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig7_Phase4_Metrics.png'));
    saveas(gcf, fullfile(savePath, 'Fig7_Phase4_Metrics.fig'));
end

%% ===== 保存数据 =====
if isSaveData
    save(fullfile(dataPath, 'Simulation_Results_v2.mat'), ...
        'ESR_mean', 'ESR_best', 'ESR_best_algo', 'algoTable', 'SNR_dB', ...
        'avgESR', 'globalRank', 'numAlgos', 'num_snr', 'Perf');
end

fprintf('[plotESRResults_v2] Generated 7 figures.\n');
end
