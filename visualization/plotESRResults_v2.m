function plotESRResults_v2(ESR_mean, ESR_best, ESR_best_algo, algoTable, SNR_dB, ...
    numScenarios, isSaveFig, savePath, isSaveData, dataPath, Perf)

[numAlgos, num_snr] = size(ESR_mean);
algoNames = {algoTable.name};
algoPAs   = {algoTable.pa};
algoPCs   = {algoTable.pc};
algoModes = {algoTable.mode};

C_GNNFL = [0.93 0.69 0.13];
C_GNN   = [0.45 0.0  0.75];
C_WMMSE = [0.85 0.15 0.15];
C_PSO   = [0.0  0.45 0.75];
C_EPA   = [0.0  0.6  0.0];
C_BL    = [0.5  0.5  0.5];
LW_PROP = 2.5;
LW_BASE = 1.8;
FS_AXIS = 12;
FS_TITLE = 13;
FS_LEG = 10;
MARKERS = {'o', 's', '^', 'd', 'v', 'p'};

%% ===== Fig 1: GNN vs Traditional Methods ESR Curves =====
figure('Visible', 'off', 'Position', [100 100 900 550]);

paTargets = {'GNN+FL', 'GNN', 'WMMSE', 'PSO', 'EPA', 'Baseline'};
paColors  = {C_GNNFL, C_GNN, C_WMMSE, C_PSO, C_EPA, C_BL};
paLines   = {'-', '-', '-', '-', '-', '--'};
paLW      = [LW_PROP, LW_PROP, LW_BASE, LW_BASE, LW_BASE, LW_BASE];
paMarkers = {MARKERS{1}, MARKERS{2}, MARKERS{3}, MARKERS{4}, MARKERS{5}, MARKERS{6}};
legendLabels = {};
hold on;

for pi = 1:length(paTargets)
    paName = paTargets{pi};
    bestIdx = getBestPC(ESR_mean, algoTable, paName, 'All');
    if isempty(bestIdx)
        bestIdx = getBestPC(ESR_mean, algoTable, paName, 'DCC');
    end
    if isempty(bestIdx); continue; end

    plot(SNR_dB, ESR_mean(bestIdx, :), [paLines{pi} paMarkers{pi}], ...
        'Color', paColors{pi}, 'LineWidth', paLW(pi), 'MarkerSize', 7);

    label = paName;
    if pi <= 2; label = [label, ' (Proposed)']; end
    legendLabels{end+1} = label; %#ok<AGROW>
end

legend(legendLabels, 'Location', 'SouthEast', 'FontSize', FS_LEG);
xlabel('SNR (dB)', 'FontSize', FS_AXIS);
ylabel('Ergodic Sum Rate (bit/s/Hz)', 'FontSize', FS_AXIS);
title('GNN vs Traditional Power Allocation Methods', 'FontSize', FS_TITLE);
grid on; box on;
set(gca, 'FontSize', FS_AXIS);

if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig1_GNN_vs_Traditional_ESR.png'));
    saveas(gcf, fullfile(savePath, 'Fig1_GNN_vs_Traditional_ESR.fig'));
end

%% ===== Fig 2: GNN Performance Gain Analysis =====
figure('Visible', 'off', 'Position', [100 100 900 600]);

wmmseIdx = getBestPC(ESR_mean, algoTable, 'WMMSE', 'All');
gnnflIdx = getBestPC(ESR_mean, algoTable, 'GNN+FL', 'All');
gnnIdx   = getBestPC(ESR_mean, algoTable, 'GNN', 'All');

if isempty(wmmseIdx)
    wmmseIdx = getBestPC(ESR_mean, algoTable, 'WMMSE', 'DCC');
end
if isempty(gnnflIdx)
    gnnflIdx = getBestPC(ESR_mean, algoTable, 'GNN+FL', 'DCC');
end
if isempty(gnnIdx)
    gnnIdx = getBestPC(ESR_mean, algoTable, 'GNN', 'DCC');
end

snrLabels = arrayfun(@(s) sprintf('%g dB', s), SNR_dB, 'UniformOutput', false);

subplot(2, 1, 1);
if ~isempty(gnnflIdx) && ~isempty(wmmseIdx)
    gain_gnnfl = (ESR_mean(gnnflIdx, :) - ESR_mean(wmmseIdx, :)) ./ ESR_mean(wmmseIdx, :) * 100;
    barColors = repmat(C_GNNFL, num_snr, 1);
    negMask = gain_gnnfl < 0;
    if any(negMask); barColors(negMask, :) = repmat([0.85 0.15 0.15], sum(negMask), 1); end

    b = bar(SNR_dB, gain_gnnfl, 0.6);
    b.FaceColor = 'flat';
    b.CData = barColors;
    hold on;
    yline(0, 'k--', 'LineWidth', 1);
    for si = 1:num_snr
        txt = sprintf('%+.1f%%', gain_gnnfl(si));
        yPos = gain_gnnfl(si);
        if yPos >= 0; va = 'bottom'; else; va = 'top'; end
        text(SNR_dB(si), yPos, txt, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', va, 'FontSize', 9, 'FontWeight', 'bold');
    end
    ylabel('ESR Gain (%)');
    title('GNN+FL vs WMMSE', 'FontSize', FS_TITLE);
    grid on; box on;
    set(gca, 'FontSize', FS_AXIS);
else
    text(0.5, 0.5, 'Insufficient data', 'Units', 'normalized', 'HorizontalAlignment', 'center');
end

subplot(2, 1, 2);
if ~isempty(gnnIdx) && ~isempty(wmmseIdx)
    gain_gnn = (ESR_mean(gnnIdx, :) - ESR_mean(wmmseIdx, :)) ./ ESR_mean(wmmseIdx, :) * 100;
    barColors = repmat(C_GNN, num_snr, 1);
    negMask = gain_gnn < 0;
    if any(negMask); barColors(negMask, :) = repmat([0.85 0.15 0.15], sum(negMask), 1); end

    b = bar(SNR_dB, gain_gnn, 0.6);
    b.FaceColor = 'flat';
    b.CData = barColors;
    hold on;
    yline(0, 'k--', 'LineWidth', 1);
    for si = 1:num_snr
        txt = sprintf('%+.1f%%', gain_gnn(si));
        yPos = gain_gnn(si);
        if yPos >= 0; va = 'bottom'; else; va = 'top'; end
        text(SNR_dB(si), yPos, txt, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', va, 'FontSize', 9, 'FontWeight', 'bold');
    end
    xlabel('SNR (dB)', 'FontSize', FS_AXIS);
    ylabel('ESR Gain (%)');
    title('GNN vs WMMSE', 'FontSize', FS_TITLE);
    grid on; box on;
    set(gca, 'FontSize', FS_AXIS);
else
    text(0.5, 0.5, 'Insufficient data', 'Units', 'normalized', 'HorizontalAlignment', 'center');
end

if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig2_GNN_Gain_Analysis.png'));
    saveas(gcf, fullfile(savePath, 'Fig2_GNN_Gain_Analysis.fig'));
end

%% ===== Fig 3: All vs DCC Access Mode Comparison =====
figure('Visible', 'off', 'Position', [100 100 1100 800]);

paCompare = {'GNN+FL', 'GNN', 'WMMSE', 'EPA'};
paColorsC = {C_GNNFL, C_GNN, C_WMMSE, C_EPA};

subplot(2, 2, 1);
hold on;
for pi = 1:length(paCompare)
    idx = getBestPC(ESR_mean, algoTable, paCompare{pi}, 'All');
    if isempty(idx); continue; end
    lw = LW_PROP; if pi > 2; lw = LW_BASE; end
    plot(SNR_dB, ESR_mean(idx, :), ['-' MARKERS{pi}], ...
        'Color', paColorsC{pi}, 'LineWidth', lw, 'MarkerSize', 7);
end
legend(paCompare, 'Location', 'SouthEast', 'FontSize', FS_LEG);
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (bit/s/Hz)');
title('All APs Mode', 'FontSize', FS_TITLE);
grid on; box on; set(gca, 'FontSize', FS_AXIS);

subplot(2, 2, 2);
hold on;
for pi = 1:length(paCompare)
    idx = getBestPC(ESR_mean, algoTable, paCompare{pi}, 'DCC');
    if isempty(idx); continue; end
    lw = LW_PROP; if pi > 2; lw = LW_BASE; end
    plot(SNR_dB, ESR_mean(idx, :), ['-' MARKERS{pi}], ...
        'Color', paColorsC{pi}, 'LineWidth', lw, 'MarkerSize', 7);
end
legend(paCompare, 'Location', 'SouthEast', 'FontSize', FS_LEG);
xlabel('SNR (dB)'); ylabel('Ergodic Sum Rate (bit/s/Hz)');
title('DCC Mode', 'FontSize', FS_TITLE);
grid on; box on; set(gca, 'FontSize', FS_AXIS);

fixedSNR = 25;
snrSel = find(SNR_dB == fixedSNR, 1);
if isempty(snrSel); snrSel = num_snr; fixedSNR = SNR_dB(end); end

subplot(2, 2, 3);
esrAll = zeros(length(paCompare), 1);
for pi = 1:length(paCompare)
    idx = getBestPC(ESR_mean, algoTable, paCompare{pi}, 'All');
    if ~isempty(idx); esrAll(pi) = ESR_mean(idx, snrSel); end
end
b = bar(esrAll, 0.6);
b.FaceColor = 'flat';
b.CData = cell2mat(paColorsC');
set(gca, 'XTickLabel', paCompare, 'FontSize', 10);
ylabel('ESR (bit/s/Hz)');
title(sprintf('All Mode @ SNR=%d dB', fixedSNR), 'FontSize', FS_TITLE);
grid on; box on; set(gca, 'FontSize', FS_AXIS);

subplot(2, 2, 4);
esrDCC = zeros(length(paCompare), 1);
for pi = 1:length(paCompare)
    idx = getBestPC(ESR_mean, algoTable, paCompare{pi}, 'DCC');
    if ~isempty(idx); esrDCC(pi) = ESR_mean(idx, snrSel); end
end
b = bar(esrDCC, 0.6);
b.FaceColor = 'flat';
b.CData = cell2mat(paColorsC');
set(gca, 'XTickLabel', paCompare, 'FontSize', 10);
ylabel('ESR (bit/s/Hz)');
title(sprintf('DCC Mode @ SNR=%d dB', fixedSNR), 'FontSize', FS_TITLE);
grid on; box on; set(gca, 'FontSize', FS_AXIS);

if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig3_All_vs_DCC_Comparison.png'));
    saveas(gcf, fullfile(savePath, 'Fig3_All_vs_DCC_Comparison.fig'));
end

%% ===== Fig 4: Comprehensive Performance Summary Table =====
fig4 = figure('Visible', 'off', 'Position', [100 100 950 420]);
ax = axes('Position', [0.02 0.05 0.96 0.88]);
axis off;
xlim([0 1]); ylim([0 1]);

baselineIdx = getBestPC(ESR_mean, algoTable, 'Baseline', 'All');
if isempty(baselineIdx); baselineIdx = getBestPC(ESR_mean, algoTable, 'Baseline', 'DCC'); end
wmmseIdx = getBestPC(ESR_mean, algoTable, 'WMMSE', 'All');
if isempty(wmmseIdx); wmmseIdx = getBestPC(ESR_mean, algoTable, 'WMMSE', 'DCC'); end

if ~isempty(baselineIdx); baselineESR = mean(ESR_mean(baselineIdx, :)); else; baselineESR = 0; end
if ~isempty(wmmseIdx); wmmseESR = mean(ESR_mean(wmmseIdx, :)); else; wmmseESR = 0; end

paSummary = {'GNN+FL', 'GNN', 'WMMSE', 'PSO', 'EPA', 'Baseline'};
complexityLabels = {'O(forward)', 'O(forward)', 'O(iter*LK^2)', 'O(nPart*iter)', 'O(LK)', 'O(LK)'};

colX = [0.02, 0.18, 0.34, 0.50, 0.66, 0.82];
headers = {'PA Method', 'Best PC', 'Avg ESR', 'vs Baseline', 'vs WMMSE', 'Complexity'};
headerY = 0.92;
rowH = 0.12;
rowStartY = 0.80;

for ri = 1:2
    rowY = rowStartY - (ri - 1) * rowH;
    patch([0, 1, 1, 0], [rowY - rowH*0.4, rowY - rowH*0.4, rowY + rowH*0.6, rowY + rowH*0.6], ...
        [1.0 0.97 0.85], 'EdgeColor', 'none');
end

for ci = 1:length(headers)
    text(colX(ci), headerY, headers{ci}, 'FontWeight', 'bold', 'FontSize', 11);
end
line([0 1], [headerY - 0.04 headerY - 0.04], 'LineWidth', 1.5, 'Color', [0.3 0.3 0.3]);

for ri = 1:length(paSummary)
    paName = paSummary{ri};
    bestIdx = getBestPC(ESR_mean, algoTable, paName, 'All');
    if isempty(bestIdx); bestIdx = getBestPC(ESR_mean, algoTable, paName, 'DCC'); end

    rowY = rowStartY - (ri - 1) * rowH;

    text(colX(1), rowY, paName, 'FontSize', 10, 'FontWeight', 'bold');

    if ~isempty(bestIdx)
        bestPC = algoPCs{bestIdx};
        avgE = mean(ESR_mean(bestIdx, :));
        vsBL = avgE - baselineESR;
        vsWM = avgE - wmmseESR;
        pctBL = vsBL / max(baselineESR, eps) * 100;

        text(colX(2), rowY, bestPC, 'FontSize', 10);
        text(colX(3), rowY, sprintf('%.2f', avgE), 'FontSize', 10);

        if ri == length(paSummary)
            text(colX(4), rowY, '--', 'FontSize', 10);
            text(colX(5), rowY, sprintf('%+.2f', vsWM), 'FontSize', 10);
        else
            text(colX(4), rowY, sprintf('%+.1f%%', pctBL), 'FontSize', 10);
            text(colX(5), rowY, sprintf('%+.2f', vsWM), 'FontSize', 10);
        end
    else
        text(colX(3), rowY, 'N/A', 'FontSize', 10);
    end

    text(colX(6), rowY, complexityLabels{ri}, 'FontSize', 10);
end

title('Comprehensive Performance Summary', 'FontSize', FS_TITLE);

if isSaveFig
    saveas(gcf, fullfile(savePath, 'Fig4_Performance_Summary.png'));
    saveas(gcf, fullfile(savePath, 'Fig4_Performance_Summary.fig'));
end

%% ===== Save Data =====
if isSaveData
    if ~exist(dataPath, 'dir'); mkdir(dataPath); end
    save(fullfile(dataPath, 'Simulation_Results_v2.mat'), ...
        'ESR_mean', 'ESR_best', 'ESR_best_algo', 'algoTable', 'SNR_dB', 'Perf');
    fprintf('[INFO] Simulation results saved to: %s\n', fullfile(dataPath, 'Simulation_Results_v2.mat'));
end

fprintf('[INFO] 4 figures generated successfully.\n');
end

%% ===== Helper Functions =====
function bestIdx = getBestPC(ESR_mean, algoTable, paKeyword, modeKeyword)
algoNames = {algoTable.name};
algoModes = {algoTable.mode};

mask = cellfun(@(x, m) contains(x, paKeyword) && strcmp(m, modeKeyword), algoNames, algoModes);
if ~any(mask)
    mask = cellfun(@(x) contains(x, paKeyword), algoNames);
end
if ~any(mask); bestIdx = []; return; end

idxList = find(mask);
avgE = mean(ESR_mean(idxList, :), 2);
[~, relBest] = max(avgE);
bestIdx = idxList(relBest);
end
