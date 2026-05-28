function plotLatencyAblationResults(Ablation, savePath, isSaveFig, isSaveData, dataPath)
%PLOTLATENCYABLATIONRESULTS Plot PA x PC synchronization-latency ablations.
%
% The input Ablation struct is created by main/Combined_Downlink_Sim.m and
% contains one row per PA+PC+mode algorithm combination.

algoTable = Ablation.algoTable;
algoNames = {algoTable.name}.';
algoPAs = {algoTable.pa}.';
algoPCs = {algoTable.pc}.';
algoModes = {algoTable.mode}.';
if isfield(algoTable, 'pcArch')
    pcArch = {algoTable.pcArch}.';
    paArch = {algoTable.paArch}.';
    isDistributed = [algoTable.isDistributed].';
else
    pcArch = repmat({'unknown'}, numel(algoTable), 1);
    paArch = repmat({'unknown'}, numel(algoTable), 1);
    isDistributed = false(numel(algoTable), 1);
end

avgESR = Ablation.avgESR(:);
avgSyncMs = mean(Ablation.sync_delay_ms, 2);
avgControlMs = mean(Ablation.control_delay_ms, 2);
avgComputeMs = mean(Ablation.compute_delay_ms, 2, 'omitnan');
if isfield(Ablation, 'pc_compute_delay_ms')
    avgPCComputeMs = mean(Ablation.pc_compute_delay_ms, 2, 'omitnan');
else
    avgPCComputeMs = zeros(size(avgComputeMs));
end
avgBytes = mean(Ablation.sync_bytes, 2);
avgRounds = mean(Ablation.sync_rounds, 2);

paOrder = {'LocalGNN', 'UGNN', 'DCGNN', 'GNN', 'DDPG', 'DQN', 'DWMMSE', 'WMMSE', 'FPCP', 'EPA', 'random', 'baseline'};
paLabels = {'Local-GNN', 'U-GNN', 'DCGNN', 'GNN', 'DDPG', 'DQN', 'D-WMMSE', 'WMMSE', 'FPCP', 'EPA', 'Random', 'Baseline'};
pcOrder = {'MR', 'LMMSE', 'RMMSE', 'LMMSE_G'};
pcLabels = {'MR', 'L-MMSE', 'R-MMSE', 'L-MMSE-G'};
paColors = [
    0.00 0.52 0.58
    0.00 0.38 0.38
    0.10 0.35 0.95
    0.45 0.00 0.75
    0.80 0.25 0.65
    0.35 0.45 0.90
    0.95 0.55 0.10
    0.85 0.15 0.15
    0.10 0.55 0.25
    0.00 0.60 0.00
    0.55 0.55 0.55
    0.20 0.20 0.20
];

if isSaveFig && ~exist(savePath, 'dir')
    mkdir(savePath);
end
if isSaveData && ~exist(dataPath, 'dir')
    mkdir(dataPath);
end

%% Fig A5_2: Best distributed-precode result per PA, sorted by ESR
modeMask = strcmp(algoModes, 'DCC');
if ~any(modeMask)
    modeMask = true(size(algoModes));
end
distPCMask = strcmp(pcArch, 'distributed');
paFocusMask = modeMask & distPCMask;
bestIdxByPA = nan(numel(paOrder), 1);
bestESRByPA = nan(numel(paOrder), 1);
bestDelayByPA = nan(numel(paOrder), 1);
bestComputeByPA = nan(numel(paOrder), 1);
bestControlByPA = nan(numel(paOrder), 1);
bestLabelByPA = strings(numel(paOrder), 1);
for pi = 1:numel(paOrder)
    idxCandidates = find(paFocusMask & strcmp(algoPAs, paOrder{pi}));
    if isempty(idxCandidates); continue; end
    [~, bestLocal] = max(avgESR(idxCandidates));
    bestIdx = idxCandidates(bestLocal);
    bestIdxByPA(pi) = bestIdx;
    bestESRByPA(pi) = avgESR(bestIdx);
    bestDelayByPA(pi) = avgControlMs(bestIdx);
    bestComputeByPA(pi) = avgComputeMs(bestIdx);
    bestControlByPA(pi) = avgControlMs(bestIdx);
    bestLabelByPA(pi) = string(sprintf('%s+%s', paLabels{pi}, displayPC(algoPCs{bestIdx})));
end
validBest = isfinite(bestESRByPA);
[~, rankOrder] = sort(bestESRByPA(validBest), 'descend');
validPAIdx = find(validBest);
rankPAIdx = validPAIdx(rankOrder);

figA52 = figure('Visible', 'off', 'Position', [100 100 1080 560]);
axA52 = axes(figA52);
hold(axA52, 'on');
rankColors = paColors(rankPAIdx, :);
b = bar(axA52, 1:numel(rankPAIdx), bestESRByPA(rankPAIdx), 0.62, 'FaceColor', 'flat');
b.CData = rankColors;
set(axA52, 'XTick', 1:numel(rankPAIdx), 'XTickLabel', cellstr(bestLabelByPA(rankPAIdx)), ...
    'XTickLabelRotation', 20, 'FontSize', 11);
ylabel('Average ESR (bit/s/Hz)');
xlabel('Best DCC + distributed-precode combination for each PA');
title({'A5\_2 PA Ranking by Best Achievable ESR', ...
    'Higher bar = stronger power allocation under the same distributed-precode constraint'});
grid on; box on;
saveFigure(figA52, savePath, isSaveFig, 'FigA5_2_PA_Best_ESR_Ranking');

%% Fig A6: DCC PA x PC heatmap
fig2 = figure('Visible', 'off', 'Position', [100 100 900 520]);
heat = nan(numel(paOrder), numel(pcOrder));
for pi = 1:numel(paOrder)
    for ci = 1:numel(pcOrder)
        idx = find(strcmp(algoPAs, paOrder{pi}) & strcmp(algoPCs, pcOrder{ci}) & strcmp(algoModes, 'DCC'), 1);
        if isempty(idx)
            idx = find(strcmp(algoPAs, paOrder{pi}) & strcmp(algoPCs, pcOrder{ci}), 1);
        end
        if ~isempty(idx)
            heat(pi, ci) = avgControlMs(idx);
        end
    end
end
imagesc(log10(heat + 1e-3));
colormap(parula);
cb = colorbar;
cb.Label.String = 'Color legend: log10(control delay in ms + 1e-3)';
set(gca, 'XTick', 1:numel(pcOrder), 'XTickLabel', pcLabels, ...
    'YTick', 1:numel(paOrder), 'YTickLabel', paLabels, 'FontSize', 12);
xlabel('Precoding method');
ylabel('Power allocation method');
title({'PA x PC Control Delay Ablation (DCC)', ...
    'Cell text = modeled control delay in ms; color = logarithmic delay scale'});
for pi = 1:numel(paOrder)
    for ci = 1:numel(pcOrder)
        if isfinite(heat(pi, ci))
            text(ci, pi, sprintf('%.3g', heat(pi, ci)), ...
                'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold');
        end
    end
end
saveFigure(fig2, savePath, isSaveFig, 'FigA6_PA_PC_SyncLatency_Heatmap');

%% Fig A7: Fixed R-MMSE, PA ablation
fig3 = figure('Visible', 'off', 'Position', [100 100 980 520]);
fixedPC = 'RMMSE';
barDelay = nan(1, numel(paOrder));
lineESR = nan(1, numel(paOrder));
for pi = 1:numel(paOrder)
    idx = find(strcmp(algoPAs, paOrder{pi}) & strcmp(algoPCs, fixedPC) & strcmp(algoModes, 'DCC'), 1);
    if isempty(idx)
        idx = find(strcmp(algoPAs, paOrder{pi}) & strcmp(algoPCs, fixedPC), 1);
    end
    if ~isempty(idx)
        barDelay(pi) = avgControlMs(idx);
        lineESR(pi) = avgESR(idx);
    end
end
leftColor = [0.00 0.45 0.74];
rightColor = [0.85 0.33 0.10];
set(gca, 'YColor', leftColor);
yyaxis left;
b = bar(1:numel(paOrder), barDelay, 0.58);
b.FaceColor = leftColor;
b.EdgeColor = [0.10 0.25 0.40];
ylabel('Control delay (ms)');
yyaxis right;
plotSmoothCurve(1:numel(paOrder), lineESR, rightColor);
ylabel('Average ESR (bit/s/Hz)');
set(gca, 'XTick', 1:numel(paOrder), 'XTickLabel', paLabels, 'FontSize', 12);
ax3 = gca;
ax3.YAxis(1).Color = leftColor;
ax3.YAxis(2).Color = rightColor;
xlabel('Power allocation method');
title('PA Ablation under R-MMSE Precoding: Control Delay vs ESR');
grid on; box on;
saveFigure(fig3, savePath, isSaveFig, 'FigA7_RMMSE_PA_Latency_Ablation');

%% Fig A8: Fixed L-MMSE, PA ablation
fig4 = figure('Visible', 'off', 'Position', [100 100 980 520]);
fixedPC = 'LMMSE';
barDelay = nan(1, numel(paOrder));
lineESR = nan(1, numel(paOrder));
for pi = 1:numel(paOrder)
    idx = find(strcmp(algoPAs, paOrder{pi}) & strcmp(algoPCs, fixedPC) & strcmp(algoModes, 'DCC'), 1);
    if isempty(idx)
        idx = find(strcmp(algoPAs, paOrder{pi}) & strcmp(algoPCs, fixedPC), 1);
    end
    if ~isempty(idx)
        barDelay(pi) = avgControlMs(idx);
        lineESR(pi) = avgESR(idx);
    end
end
leftColor = [0.00 0.45 0.74];
rightColor = [0.85 0.33 0.10];
set(gca, 'YColor', leftColor);
yyaxis left;
b = bar(1:numel(paOrder), barDelay, 0.58);
b.FaceColor = leftColor;
b.EdgeColor = [0.10 0.25 0.40];
ylabel('Control delay (ms)');
yyaxis right;
plotSmoothCurve(1:numel(paOrder), lineESR, rightColor);
ylabel('Average ESR (bit/s/Hz)');
set(gca, 'XTick', 1:numel(paOrder), 'XTickLabel', paLabels, 'FontSize', 12);
ax4 = gca;
ax4.YAxis(1).Color = leftColor;
ax4.YAxis(2).Color = rightColor;
xlabel('Power allocation method');
title('PA Ablation under L-MMSE Precoding: Control Delay vs ESR');
grid on; box on;
saveFigure(fig4, savePath, isSaveFig, 'FigA8_LMMSE_PA_Latency_Ablation');

%% Fig A9: Fixed MR, PA ablation
fig5 = figure('Visible', 'off', 'Position', [100 100 980 520]);
fixedPC = 'MR';
barDelay = nan(1, numel(paOrder));
lineESR = nan(1, numel(paOrder));
for pi = 1:numel(paOrder)
    idx = find(strcmp(algoPAs, paOrder{pi}) & strcmp(algoPCs, fixedPC) & strcmp(algoModes, 'DCC'), 1);
    if isempty(idx)
        idx = find(strcmp(algoPAs, paOrder{pi}) & strcmp(algoPCs, fixedPC), 1);
    end
    if ~isempty(idx)
        barDelay(pi) = avgControlMs(idx);
        lineESR(pi) = avgESR(idx);
    end
end
leftColor = [0.00 0.45 0.74];
rightColor = [0.85 0.33 0.10];
set(gca, 'YColor', leftColor);
yyaxis left;
b = bar(1:numel(paOrder), barDelay, 0.58);
b.FaceColor = leftColor;
b.EdgeColor = [0.10 0.25 0.40];
ylabel('Control delay (ms)');
yyaxis right;
plotSmoothCurve(1:numel(paOrder), lineESR, rightColor);
ylabel('Average ESR (bit/s/Hz)');
set(gca, 'XTick', 1:numel(paOrder), 'XTickLabel', paLabels, 'FontSize', 12);
ax5 = gca;
ax5.YAxis(1).Color = leftColor;
ax5.YAxis(2).Color = rightColor;
xlabel('Power allocation method');
title('PA Ablation under MR Precoding: Control Delay vs ESR');
grid on; box on;
saveFigure(fig5, savePath, isSaveFig, 'FigA9_MR_PA_Latency_Ablation');

%% Save data table
if isSaveData
    T = table((1:numel(algoTable)).', algoNames, algoPAs, algoPCs, algoModes, ...
        pcArch, paArch, isDistributed, avgESR, avgSyncMs, avgControlMs, avgComputeMs, avgPCComputeMs, avgBytes, avgRounds, ...
        'VariableNames', {'id', 'algorithm', 'pa', 'pc', 'mode', ...
        'pc_arch', 'pa_arch', 'is_distributed', ...
        'avg_esr', 'sync_delay_ms', 'control_delay_ms', 'pa_compute_ms', ...
        'pc_compute_ms', 'sync_bytes', 'sync_rounds'});
    writetable(T, fullfile(dataPath, 'Sync_Ablation_Table.csv'));
    save(fullfile(dataPath, 'Sync_Ablation_Results.mat'), 'Ablation', 'T');
    fprintf('[INFO] Sync ablation table saved to: %s\n', ...
        fullfile(dataPath, 'Sync_Ablation_Table.csv'));
end

fprintf('[INFO] 5 synchronization ablation figures generated successfully.\n');
end

function label = displayPC(pcKey)
switch pcKey
    case 'LMMSE'
        label = 'L-MMSE';
    case 'RMMSE'
        label = 'R-MMSE';
    case 'LMMSE_G'
        label = 'L-MMSE-G';
    otherwise
        label = pcKey;
end
end

function h = plotSmoothCurve(x, y, color)
valid = isfinite(x) & isfinite(y);
xv = x(valid);
yv = y(valid);
if numel(xv) >= 3
    xq = linspace(min(xv), max(xv), 240);
    yq = interp1(xv, yv, xq, 'pchip');
    h = plot(xq, yq, '-', 'Color', color, 'LineWidth', 2.8);
    hold on;
    plot(xv, yv, 'o', 'Color', color, 'MarkerFaceColor', color, ...
        'MarkerEdgeColor', 'w', 'LineWidth', 1.2, 'MarkerSize', 6);
else
    h = plot(xv, yv, '-o', 'Color', color, 'LineWidth', 2.8, ...
        'MarkerFaceColor', color);
end
end

function saveFigure(fig, savePath, isSaveFig, baseName)
if ~isSaveFig; return; end
saveas(fig, fullfile(savePath, [baseName '.png']));
saveas(fig, fullfile(savePath, [baseName '.fig']));
end
