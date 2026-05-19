function plotESRResults_v2(ESR_mean, ESR_best, ESR_best_algo, algoTable, SNR_dB, ...
    numScenarios, isSaveFig, savePath, isSaveData, dataPath, Perf)

%PLOTESRRESULTS_V2 Plot current simulation results.
% The plots intentionally use exact PA identifiers from algoTable.pa, so
% methods cannot be mixed by substring matching.

[~, num_snr] = size(ESR_mean);
algoPAs = {algoTable.pa};
algoPCs = {algoTable.pc};
algoModes = {algoTable.mode};

paOrder = {'LocalGNN', 'DCGNN', 'GNN', 'DDPG', 'DQN', 'DWMMSE', 'WMMSE', 'FPCP', 'EPA', 'random', 'baseline'};
paLabels = {'Local-GNN', 'DCGNN', 'GNN', 'DDPG', 'DQN', 'D-WMMSE', 'WMMSE', 'FPCP', 'EPA', 'Random', 'Baseline'};
paColors = [
    0.00 0.52 0.58
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
paMarkers = {'x', '^', 'o', '+', '*', 'h', 's', '>', 'd', 'v', 'p'};

FS_AXIS = 12;
FS_TITLE = 13;
FS_LEG = 10;
LW_GNN = 2.8;
LW_BASE = 1.8;

if isSaveFig && ~exist(savePath, 'dir')
    mkdir(savePath);
end

%% Fig 1: Best PC per PA in DCC mode
fig1 = figure('Visible', 'off', 'Position', [100 100 950 560]);
hold on;
legendLabels = {};
for pi = 1:numel(paOrder)
    idx = getBestCurve(ESR_mean, algoTable, paOrder{pi}, 'DCC');
    if isempty(idx); continue; end
    lw = LW_BASE;
    if any(strcmp(paOrder{pi}, {'GNN', 'LocalGNN', 'DCGNN', 'DQN', 'DDPG'})); lw = LW_GNN; end
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
    if isempty(idx); continue; end
    lw = LW_BASE;
    if any(strcmp(paOrder{pi}, {'GNN', 'LocalGNN', 'DCGNN', 'DQN', 'DDPG'})); lw = LW_GNN; end
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

%% Fig 3: Learning-family gap to WMMSE under the same PC/mode
fig3 = figure('Visible', 'off', 'Position', [100 100 900 560]);
wmmseIdx = findExact(algoTable, 'WMMSE', fixedPC, 'DCC');
gnnFamily = {'LocalGNN', 'DCGNN', 'GNN', 'DDPG', 'DQN'};
gnnFamilyLabels = {'Local-GNN', 'DCGNN', 'GNN', 'DDPG', 'DQN'};
gnnFamilyColors = [paColors(1, :); paColors(2, :); paColors(3, :); paColors(4, :); paColors(5, :)];
gapPctMat = [];
gapLabels = {};
gapColors = [];
if isempty(wmmseIdx)
    text(0.5, 0.5, 'WMMSE R-MMSE DCC reference unavailable', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', FS_AXIS);
else
    for gi = 1:numel(gnnFamily)
        curIdx = findExact(algoTable, gnnFamily{gi}, fixedPC, 'DCC');
        if isempty(curIdx); continue; end
        gapPctMat(:, end+1) = ((ESR_mean(curIdx, :) - ESR_mean(wmmseIdx, :)) ./ ...
            max(ESR_mean(wmmseIdx, :), eps) * 100).'; %#ok<AGROW>
        gapLabels{end+1} = gnnFamilyLabels{gi}; %#ok<AGROW>
        gapColors(end+1, :) = gnnFamilyColors(gi, :); %#ok<AGROW>
    end

    if isempty(gapPctMat)
        text(0.5, 0.5, 'Learning-family R-MMSE DCC results unavailable', ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', FS_AXIS);
    else
        b = bar(SNR_dB, gapPctMat, 0.68, 'grouped');
        for gi = 1:numel(b)
            b(gi).FaceColor = gapColors(gi, :);
            b(gi).EdgeColor = [0.15 0.15 0.15];
        end
    end
    hold on; yline(0, 'k--', 'LineWidth', 1);
    if ~isempty(gapPctMat)
        for gi = 1:size(gapPctMat, 2)
            xEnd = b(gi).XEndPoints;
            for si = 1:num_snr
                if gapPctMat(si, gi) >= 0; va = 'bottom'; else; va = 'top'; end
                text(xEnd(si), gapPctMat(si, gi), sprintf('%+.1f%%', gapPctMat(si, gi)), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', va, ...
                    'FontSize', 8, 'FontWeight', 'bold');
            end
        end
        legend(gapLabels, 'Location', 'best', 'FontSize', FS_LEG);
    end
    xlabel('SNR (dB)', 'FontSize', FS_AXIS);
    ylabel('ESR Gap to WMMSE (%)', 'FontSize', FS_AXIS);
    title('Learning-family Gap to WMMSE (R-MMSE, DCC)', 'FontSize', FS_TITLE);
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
wmmseAvg = mean(ESR_mean(wmmseBestIdx, :));

notes = containers.Map(paOrder, {'AP-local learned split', 'dynamic graph learned split', ...
    'full-graph learned split', 'DDPG reward-trained split', 'DQN reward-trained alpha', ...
    'fixed-round distributed update', 'iterative reference', 'fractional power control', ...
    'equal power', 'random baseline', 'large-scale baseline'});

for pi = 1:numel(paOrder)
    idx = getBestCurve(ESR_mean, algoTable, paOrder{pi}, 'DCC');
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

%% Fig 5: WMMSE vs learning-family timing, split by end-to-end and core compute
fig5 = figure('Visible', 'off', 'Position', [100 100 1120 720]);
if nargin < 11 || isempty(Perf) || ~isfield(Perf, 'time_pa_sec') || ~isfield(Perf, 'methodNames')
    text(0.5, 0.5, 'Timing data unavailable', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', FS_AXIS);
else
    wIdx = find(strcmp(Perf.methodNames, 'WMMSE'), 1);
    gIdx = find(strcmp(Perf.methodNames, 'GNN'), 1);
    lIdx = find(strcmp(Perf.methodNames, 'Local-GNN'), 1);
    dIdx = find(strcmp(Perf.methodNames, 'DCGNN'), 1);
    dqIdx = find(strcmp(Perf.methodNames, 'DQN'), 1);
    ddIdx = find(strcmp(Perf.methodNames, 'DDPG'), 1);
    if isempty(wIdx) || (isempty(gIdx) && isempty(lIdx) && isempty(dIdx) && isempty(dqIdx) && isempty(ddIdx))
        text(0.5, 0.5, 'WMMSE/learning-family timing rows unavailable', ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', FS_AXIS);
    else
        wmmseE2E = getPerfCurve(Perf.time_pa_sec, wIdx);
        methodIdx = [];
        methodLabels = {};
        methodColors = [];
        if ~isempty(lIdx)
            methodIdx(end+1) = lIdx; %#ok<AGROW>
            methodLabels{end+1} = 'Local-GNN'; %#ok<AGROW>
            methodColors(end+1, :) = paColors(1, :); %#ok<AGROW>
        end
        if ~isempty(gIdx)
            methodIdx(end+1) = gIdx; %#ok<AGROW>
            methodLabels{end+1} = 'GNN'; %#ok<AGROW>
            methodColors(end+1, :) = paColors(3, :); %#ok<AGROW>
        end
        if ~isempty(dIdx)
            methodIdx(end+1) = dIdx; %#ok<AGROW>
            methodLabels{end+1} = 'DCGNN'; %#ok<AGROW>
            methodColors(end+1, :) = paColors(2, :); %#ok<AGROW>
        end
        if ~isempty(ddIdx)
            methodIdx(end+1) = ddIdx; %#ok<AGROW>
            methodLabels{end+1} = 'DDPG'; %#ok<AGROW>
            methodColors(end+1, :) = paColors(4, :); %#ok<AGROW>
        end
        if ~isempty(dqIdx)
            methodIdx(end+1) = dqIdx; %#ok<AGROW>
            methodLabels{end+1} = 'DQN'; %#ok<AGROW>
            methodColors(end+1, :) = paColors(5, :); %#ok<AGROW>
        end

        e2eMat = zeros(numel(methodIdx), num_snr);
        for mi = 1:numel(methodIdx)
            e2eMat(mi, :) = getPerfCurve(Perf.time_pa_sec, methodIdx(mi));
        end

        if isfield(Perf, 'time_core_sec')
            wmmseCore = getPerfCurve(Perf.time_core_sec, wIdx);
            coreMat = zeros(numel(methodIdx), num_snr);
            for mi = 1:numel(methodIdx)
                coreMat(mi, :) = getPerfCurve(Perf.time_core_sec, methodIdx(mi));
            end
        else
            wmmseCore = wmmseE2E;
            coreMat = e2eMat;
        end

        tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        plotTimeComparison(SNR_dB, wmmseE2E, e2eMat, methodLabels, methodColors, paColors(7, :), ...
            FS_AXIS, FS_TITLE, FS_LEG, ...
            'End-to-End PA Time', 'Includes MATLAB-Python Bridge');
        plotSpeedupComparison(SNR_dB, wmmseE2E ./ e2eMat, methodLabels, methodColors, ...
            FS_AXIS, FS_TITLE, FS_LEG, ...
            'End-to-End Speedup');
        plotTimeComparison(SNR_dB, wmmseCore, coreMat, methodLabels, methodColors, paColors(7, :), ...
            FS_AXIS, FS_TITLE, FS_LEG, ...
            'Core Compute Time', 'WMMSE loop vs GNN forward');
        plotSpeedupComparison(SNR_dB, wmmseCore ./ coreMat, methodLabels, methodColors, ...
            FS_AXIS, FS_TITLE, FS_LEG, ...
            'Core Compute Speedup');

        sgtitle(sprintf('WMMSE vs Learning-family Timing: WMMSE E2E %.3f ms, Core %.3f ms', ...
            mean(wmmseE2E) * 1000, mean(wmmseCore) * 1000), 'FontSize', FS_TITLE);
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

function curve = getPerfCurve(perfArray, methodIdx)
curve = max(squeeze(mean(perfArray(methodIdx, :, :), 3)), eps);
curve = curve(:).';
end

function plotTimeComparison(SNR_dB, wmmseTime, methodTimes, methodLabels, methodColors, wmmseColor, ...
    FS_AXIS, FS_TITLE, FS_LEG, titleText, subText)
nexttile;
semilogy(SNR_dB, wmmseTime * 1000, '-s', ...
    'Color', wmmseColor, 'LineWidth', 2.2, 'MarkerSize', 7);
hold on;
for mi = 1:size(methodTimes, 1)
    semilogy(SNR_dB, methodTimes(mi, :) * 1000, '-o', ...
        'Color', methodColors(mi, :), 'LineWidth', 2.6, 'MarkerSize', 7);
end
xlabel('SNR (dB)', 'FontSize', FS_AXIS);
ylabel('Time (ms)', 'FontSize', FS_AXIS);
title({titleText, subText}, 'FontSize', FS_TITLE);
legend([{'WMMSE'}, methodLabels], 'Location', 'best', 'FontSize', FS_LEG);
grid on; box on; set(gca, 'FontSize', FS_AXIS);
end

function plotSpeedupComparison(SNR_dB, speedupMat, methodLabels, methodColors, ...
    FS_AXIS, FS_TITLE, FS_LEG, titleText)
nexttile;
b = bar(SNR_dB, speedupMat.', 0.68, 'grouped');
for mi = 1:numel(b)
    b(mi).FaceColor = methodColors(mi, :);
    b(mi).EdgeColor = [0.15 0.15 0.15];
end
hold on; yline(1, 'k--', 'LineWidth', 1);
for mi = 1:size(speedupMat, 1)
    xEnd = b(mi).XEndPoints;
    for si = 1:numel(SNR_dB)
        if speedupMat(mi, si) >= 1; va = 'bottom'; else; va = 'top'; end
        text(xEnd(si), speedupMat(mi, si), sprintf('%.2fx', speedupMat(mi, si)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', va, ...
            'FontSize', 8, 'FontWeight', 'bold');
    end
end
xlabel('SNR (dB)', 'FontSize', FS_AXIS);
ylabel('WMMSE / method time', 'FontSize', FS_AXIS);
title(titleText, 'FontSize', FS_TITLE);
legend(methodLabels, 'Location', 'best', 'FontSize', FS_LEG);
grid on; box on; set(gca, 'FontSize', FS_AXIS);
end

function saveFigure(fig, savePath, isSaveFig, baseName)
if ~isSaveFig; return; end
saveas(fig, fullfile(savePath, [baseName '.png']));
saveas(fig, fullfile(savePath, [baseName '.fig']));
end
