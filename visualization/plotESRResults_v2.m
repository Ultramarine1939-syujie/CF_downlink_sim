function plotESRResults_v2(ESR_mean, ESR_best, ESR_best_algo, algoTable, SNR_dB, ...
    numScenarios, isSaveFig, savePath, isSaveData, dataPath, Perf)

%PLOTESRRESULTS_V2 Plot current simulation results.
% The plots intentionally use exact PA identifiers from algoTable.pa, so
% methods cannot be mixed by substring matching.

[~, num_snr] = size(ESR_mean);
algoPAs = {algoTable.pa};
algoPCs = {algoTable.pc};
algoModes = {algoTable.mode};

paOrder = {'LocalGNN', 'GNN', 'DWMMSE', 'WMMSE', 'PSO', 'EPA', 'random', 'baseline'};
paLabels = {'Local-GNN', 'GNN', 'D-WMMSE', 'WMMSE', 'PSO', 'EPA', 'Random', 'Baseline'};
paColors = [
    0.00 0.52 0.58
    0.45 0.00 0.75
    0.95 0.55 0.10
    0.85 0.15 0.15
    0.00 0.45 0.75
    0.00 0.60 0.00
    0.55 0.55 0.55
    0.20 0.20 0.20
];
paMarkers = {'x', 'o', 'h', 's', '^', 'd', 'v', 'p'};

FS_AXIS = 12;
FS_TITLE = 13;
FS_LEG = 10;
LW_GNN = 2.8;
LW_BASE = 1.8;

if isSaveFig && ~exist(savePath, 'dir')
    mkdir(savePath);
end

%% Fig 1: Best PC per PA, DCC mode preferred for current GNN deployment
fig1 = figure('Visible', 'off', 'Position', [100 100 950 560]);
hold on;
legendLabels = {};
for pi = 1:numel(paOrder)
    idx = getBestCurve(ESR_mean, algoTable, paOrder{pi}, 'DCC');
    if isempty(idx)
        idx = getBestCurve(ESR_mean, algoTable, paOrder{pi}, 'All');
    end
    if isempty(idx); continue; end
    lw = LW_BASE;
    if any(strcmp(paOrder{pi}, {'GNN', 'LocalGNN'})); lw = LW_GNN; end
    plot(SNR_dB, ESR_mean(idx, :), ['-' paMarkers{pi}], ...
        'Color', paColors(pi, :), 'LineWidth', lw, 'MarkerSize', 7);
    legendLabels{end+1} = sprintf('%s + %s (%s)', paLabels{pi}, ...
        displayPC(algoPCs{idx}), algoModes{idx}); %#ok<AGROW>
end
xlabel('SNR (dB)', 'FontSize', FS_AXIS);
ylabel('Ergodic Sum Rate (bit/s/Hz)', 'FontSize', FS_AXIS);
title('Best Precoder per Power Allocation Method', 'FontSize', FS_TITLE);
legend(legendLabels, 'Location', 'SouthEast', 'FontSize', FS_LEG);
grid on; box on; set(gca, 'FontSize', FS_AXIS);
saveFigure(fig1, savePath, isSaveFig, 'Fig1_Best_PA_ESR');

%% Fig 2: Same precoder comparison for R-MMSE, the strongest practical PC here
fig2 = figure('Visible', 'off', 'Position', [100 100 950 560]);
hold on;
legendLabels = {};
fixedPC = 'RMMSE';
for pi = 1:numel(paOrder)
    idx = findExact(algoTable, paOrder{pi}, fixedPC, 'DCC');
    if isempty(idx)
        idx = findExact(algoTable, paOrder{pi}, fixedPC, 'All');
    end
    if isempty(idx); continue; end
    lw = LW_BASE;
    if any(strcmp(paOrder{pi}, {'GNN', 'LocalGNN'})); lw = LW_GNN; end
    plot(SNR_dB, ESR_mean(idx, :), ['-' paMarkers{pi}], ...
        'Color', paColors(pi, :), 'LineWidth', lw, 'MarkerSize', 7);
    legendLabels{end+1} = sprintf('%s (%s)', paLabels{pi}, algoModes{idx}); %#ok<AGROW>
end
xlabel('SNR (dB)', 'FontSize', FS_AXIS);
ylabel('Ergodic Sum Rate (bit/s/Hz)', 'FontSize', FS_AXIS);
title('Power Allocation Comparison with R-MMSE Precoding', 'FontSize', FS_TITLE);
legend(legendLabels, 'Location', 'SouthEast', 'FontSize', FS_LEG);
grid on; box on; set(gca, 'FontSize', FS_AXIS);
saveFigure(fig2, savePath, isSaveFig, 'Fig2_RMMSE_PA_Comparison');

%% Fig 3: GNN gap to WMMSE under the same PC/mode
fig3 = figure('Visible', 'off', 'Position', [100 100 900 560]);
gnnIdx = findExact(algoTable, 'GNN', fixedPC, 'DCC');
wmmseIdx = findExact(algoTable, 'WMMSE', fixedPC, 'DCC');
if isempty(gnnIdx) || isempty(wmmseIdx)
    text(0.5, 0.5, 'GNN/WMMSE R-MMSE DCC results unavailable', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', FS_AXIS);
else
    gapPct = (ESR_mean(gnnIdx, :) - ESR_mean(wmmseIdx, :)) ./ max(ESR_mean(wmmseIdx, :), eps) * 100;
    b = bar(SNR_dB, gapPct, 0.58);
    b.FaceColor = 'flat';
    b.CData = repmat(paColors(2, :), num_snr, 1);
    neg = gapPct < 0;
    b.CData(neg, :) = repmat([0.85 0.15 0.15], sum(neg), 1);
    hold on; yline(0, 'k--', 'LineWidth', 1);
    for si = 1:num_snr
        if gapPct(si) >= 0; va = 'bottom'; else; va = 'top'; end
        text(SNR_dB(si), gapPct(si), sprintf('%+.1f%%', gapPct(si)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', va, ...
            'FontSize', 9, 'FontWeight', 'bold');
    end
    xlabel('SNR (dB)', 'FontSize', FS_AXIS);
    ylabel('ESR Gap to WMMSE (%)', 'FontSize', FS_AXIS);
    title('GNN Gap to WMMSE (R-MMSE, DCC)', 'FontSize', FS_TITLE);
    grid on; box on; set(gca, 'FontSize', FS_AXIS);
end
saveFigure(fig3, savePath, isSaveFig, 'Fig3_GNN_WMMSE_Gap');

%% Fig 4: Compact method summary
fig4 = figure('Visible', 'off', 'Position', [100 100 980 430]);
ax = axes('Position', [0.03 0.08 0.94 0.84]); %#ok<LAXES>
axis off; xlim([0 1]); ylim([0 1]);

headers = {'PA', 'Best PC', 'Mode', 'Avg ESR', 'vs WMMSE', 'Notes'};
colX = [0.03, 0.17, 0.31, 0.45, 0.60, 0.75];
rowY = 0.84;
rowH = 0.115;
for ci = 1:numel(headers)
    text(colX(ci), 0.94, headers{ci}, 'FontWeight', 'bold', 'FontSize', 11);
end
line([0.02 0.98], [0.90 0.90], 'Color', [0.25 0.25 0.25], 'LineWidth', 1.2);

wmmseBestIdx = getBestCurve(ESR_mean, algoTable, 'WMMSE', 'DCC');
if isempty(wmmseBestIdx); wmmseBestIdx = getBestCurve(ESR_mean, algoTable, 'WMMSE', 'All'); end
wmmseAvg = mean(ESR_mean(wmmseBestIdx, :));

notes = containers.Map(paOrder, {'AP-local learned split', 'full-graph learned split', ...
    'fixed-round distributed update', 'iterative reference', 'heuristic optimizer', 'equal power', ...
    'random baseline', 'large-scale baseline'});

for pi = 1:numel(paOrder)
    idx = getBestCurve(ESR_mean, algoTable, paOrder{pi}, 'DCC');
    if isempty(idx); idx = getBestCurve(ESR_mean, algoTable, paOrder{pi}, 'All'); end
    if isempty(idx); continue; end
    if pi == 1
        patch([0.01 0.99 0.99 0.01], [rowY - 0.035 rowY - 0.035 rowY + 0.055 rowY + 0.055], ...
            [0.96 0.92 1.00], 'EdgeColor', 'none');
    end
    avgE = mean(ESR_mean(idx, :));
    text(colX(1), rowY, paLabels{pi}, 'FontSize', 10, 'FontWeight', 'bold');
    text(colX(2), rowY, displayPC(algoPCs{idx}), 'FontSize', 10);
    text(colX(3), rowY, algoModes{idx}, 'FontSize', 10);
    text(colX(4), rowY, sprintf('%.2f', avgE), 'FontSize', 10);
    text(colX(5), rowY, sprintf('%+.2f', avgE - wmmseAvg), 'FontSize', 10);
    text(colX(6), rowY, notes(paOrder{pi}), 'FontSize', 10);
    rowY = rowY - rowH;
end
title('Current Method Summary', 'FontSize', FS_TITLE);
saveFigure(fig4, savePath, isSaveFig, 'Fig4_Method_Summary');

%% Fig 5: WMMSE vs GNN timing, split by end-to-end and core compute
fig5 = figure('Visible', 'off', 'Position', [100 100 1120 720]);
if nargin < 11 || isempty(Perf) || ~isfield(Perf, 'time_pa_sec') || ~isfield(Perf, 'methodNames')
    text(0.5, 0.5, 'Timing data unavailable', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', FS_AXIS);
else
    wIdx = find(strcmp(Perf.methodNames, 'WMMSE'), 1);
    gIdx = find(strcmp(Perf.methodNames, 'GNN'), 1);
    if isempty(wIdx) || isempty(gIdx)
        text(0.5, 0.5, 'WMMSE/GNN timing rows unavailable', ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', FS_AXIS);
    else
        wmmseE2E = max(squeeze(mean(Perf.time_pa_sec(wIdx, :, :), 3)), eps);
        gnnE2E = max(squeeze(mean(Perf.time_pa_sec(gIdx, :, :), 3)), eps);
        wmmseE2E = wmmseE2E(:).';
        gnnE2E = gnnE2E(:).';

        if isfield(Perf, 'time_core_sec')
            wmmseCore = max(squeeze(mean(Perf.time_core_sec(wIdx, :, :), 3)), eps);
            gnnCore = max(squeeze(mean(Perf.time_core_sec(gIdx, :, :), 3)), eps);
        else
            wmmseCore = wmmseE2E;
            gnnCore = gnnE2E;
        end
        wmmseCore = wmmseCore(:).';
        gnnCore = gnnCore(:).';

        tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        plotTimePair(SNR_dB, wmmseE2E, gnnE2E, paColors, FS_AXIS, FS_TITLE, FS_LEG, ...
            'End-to-End PA Time', 'Includes MATLAB-Python Bridge');
        plotSpeedup(SNR_dB, wmmseE2E ./ gnnE2E, paColors, num_snr, FS_AXIS, FS_TITLE, ...
            'End-to-End Speedup');
        plotTimePair(SNR_dB, wmmseCore, gnnCore, paColors, FS_AXIS, FS_TITLE, FS_LEG, ...
            'Core Compute Time', 'WMMSE loop vs GNN forward');
        plotSpeedup(SNR_dB, wmmseCore ./ gnnCore, paColors, num_snr, FS_AXIS, FS_TITLE, ...
            'Core Compute Speedup');

        sgtitle(sprintf('WMMSE vs GNN Timing: E2E %.3f/%.3f ms, Core %.3f/%.3f ms', ...
            mean(wmmseE2E) * 1000, mean(gnnE2E) * 1000, ...
            mean(wmmseCore) * 1000, mean(gnnCore) * 1000), 'FontSize', FS_TITLE);
    end
end
saveFigure(fig5, savePath, isSaveFig, 'Fig5_Time_Overhead_Comparison');

%% Save data
if isSaveData
    if ~exist(dataPath, 'dir'); mkdir(dataPath); end
    save(fullfile(dataPath, 'Simulation_Results_v2.mat'), ...
        'ESR_mean', 'ESR_best', 'ESR_best_algo', 'algoTable', 'SNR_dB', 'Perf');
    fprintf('[INFO] Simulation results saved to: %s\n', fullfile(dataPath, 'Simulation_Results_v2.mat'));
end

fprintf('[INFO] 5 figures generated successfully.\n');
end

function idx = findExact(algoTable, paKey, pcKey, modeKey)
algoPAs = {algoTable.pa};
algoPCs = {algoTable.pc};
algoModes = {algoTable.mode};
idx = find(strcmp(algoPAs, paKey) & strcmp(algoPCs, pcKey) & strcmp(algoModes, modeKey), 1);
end

function bestIdx = getBestCurve(ESR_mean, algoTable, paKey, modeKey)
algoPAs = {algoTable.pa};
algoModes = {algoTable.mode};
idxList = find(strcmp(algoPAs, paKey) & strcmp(algoModes, modeKey));
if isempty(idxList)
    idxList = find(strcmp(algoPAs, paKey));
end
if isempty(idxList); bestIdx = []; return; end
[~, relBest] = max(mean(ESR_mean(idxList, :), 2));
bestIdx = idxList(relBest);
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

function plotTimePair(SNR_dB, wmmseTime, gnnTime, paColors, FS_AXIS, FS_TITLE, FS_LEG, titleText, subText)
nexttile;
semilogy(SNR_dB, wmmseTime * 1000, '-s', ...
    'Color', paColors(4, :), 'LineWidth', 2.2, 'MarkerSize', 7);
hold on;
semilogy(SNR_dB, gnnTime * 1000, '-o', ...
    'Color', paColors(2, :), 'LineWidth', 2.8, 'MarkerSize', 7);
xlabel('SNR (dB)', 'FontSize', FS_AXIS);
ylabel('Time (ms)', 'FontSize', FS_AXIS);
title({titleText, subText}, 'FontSize', FS_TITLE);
legend({'WMMSE', 'GNN'}, 'Location', 'best', 'FontSize', FS_LEG);
grid on; box on; set(gca, 'FontSize', FS_AXIS);
end

function plotSpeedup(SNR_dB, speedup, paColors, num_snr, FS_AXIS, FS_TITLE, titleText)
nexttile;
b = bar(SNR_dB, speedup, 0.58);
b.FaceColor = 'flat';
b.CData = repmat(paColors(2, :), num_snr, 1);
slower = speedup < 1;
b.CData(slower, :) = repmat([0.85 0.15 0.15], sum(slower), 1);
hold on; yline(1, 'k--', 'LineWidth', 1);
for si = 1:num_snr
    if speedup(si) >= 1; va = 'bottom'; else; va = 'top'; end
    text(SNR_dB(si), speedup(si), sprintf('%.2fx', speedup(si)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', va, ...
        'FontSize', 9, 'FontWeight', 'bold');
end
xlabel('SNR (dB)', 'FontSize', FS_AXIS);
ylabel('WMMSE / GNN Time', 'FontSize', FS_AXIS);
title(titleText, 'FontSize', FS_TITLE);
grid on; box on; set(gca, 'FontSize', FS_AXIS);
end

function saveFigure(fig, savePath, isSaveFig, baseName)
if ~isSaveFig; return; end
saveas(fig, fullfile(savePath, [baseName '.png']));
saveas(fig, fullfile(savePath, [baseName '.fig']));
end
