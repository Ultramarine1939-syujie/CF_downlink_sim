%% 打印仿真最终结果 (v2 - 聚焦排名与增益)
function printFinalResults_v2(ESR_mean, algoTable, SNR_dB, ESR_best, ESR_best_algo, ...
    PSO_info, numScenarios, nbrOfRealizations, isSaveFig, isSaveData, savePath, dataPath, Perf)

[numAlgos, num_snr] = size(ESR_mean);

% 将 struct array 提取为 cell 数组，避免重复调用 arrayfun
algoNames  = {algoTable.name};
algoPCs    = {algoTable.pc};

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║               SIMULATION RESULTS  -  ALGORITHM RANKING                    ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════════════════╣\n');

%% ---- 按 SNR 分组显示最优 ----
fprintf('║  Best Algorithm per SNR Point                                                  ║\n');
fprintf('║  ────────────────────────────────────────────────────────────────────────────║\n');
for si = 1:num_snr
    [~, sortIdx] = sort(ESR_mean(:, si), 'descend');
    top3 = sortIdx(1:min(3, numAlgos));
    fprintf('║  SNR=%5.1f dB  │  1st: %-22s  %6.2f  │\n', ...
        SNR_dB(si), algoTable(top3(1)).name, ESR_mean(top3(1), si));
    if length(top3) >= 2
        fprintf('║               │  2nd: %-22s  %6.2f  │\n', ...
            algoTable(top3(2)).name, ESR_mean(top3(2), si));
    end
    if length(top3) >= 3
        fprintf('║               │  3rd: %-22s  %6.2f  │\n', ...
            algoTable(top3(3)).name, ESR_mean(top3(3), si));
    end
    fprintf('║               │  Best: %-20s  ESR=%.3f    │\n', ...
        ESR_best_algo(si), ESR_best(si));
    if si < num_snr
        fprintf('║               │───────────────────────────────────-│\n');
    end
end

%% ---- 汇总排名表 ----
fprintf('╠══════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Full Ranking Table (sorted by average ESR over all SNR points)             ║\n');
fprintf('║  ────────────────────────────────────────────────────────────────────────────║\n');

avgESR = mean(ESR_mean, 2);
baselineIdx = find(cellfun(@(x) contains(x,'Baseline') && contains(x,'MR') && contains(x,'All'), algoNames), 1);
if isempty(baselineIdx)
    baselineIdx = find(cellfun(@(x) contains(x,'Baseline'), algoNames), 1);
end
if isempty(baselineIdx)
    baseline_avg = mean(avgESR);
else
    baseline_avg = avgESR(baselineIdx);
end
[~, globalRank] = sort(avgESR, 'descend');

fprintf('║  %-4s │ %-22s │ %8s │ %9s │ %9s ║\n', ...
    'Rank', 'Algorithm', 'Avg ESR', 'vs Baseline', 'Gain (%%)');
fprintf('║  ─────┼────────────────────────┼──────────┼───────────┼───────────║\n');

for r = 1:numAlgos
    idx = globalRank(r);
    gain_vs_base = avgESR(idx) - baseline_avg;
    pct = (gain_vs_base / baseline_avg) * 100;

    rankIcon = '   ';
    if r == 1; rankIcon = '[1]'; elseif r == 2; rankIcon = '[2]'; elseif r == 3; rankIcon = '[3]'; end

    pctStr = sprintf('%+.2f%%', pct);
    fprintf('║  %2d%s │ %-22s │ %8.2f │ %9.3f │ %9s ║\n', ...
        r, rankIcon, algoTable(idx).name, avgESR(idx), gain_vs_base, pctStr);
    if r < numAlgos && mod(r, 5) == 0
        fprintf('║  ─────┼────────────────────────┼──────────┼───────────┼───────────║\n');
    end
end

%% ---- 功率分配方法横向对比 ----
fprintf('╠══════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Power Allocation Method Comparison (Best PC per PA)                        ║\n');
fprintf('║  ────────────────────────────────────────────────────────────────────────────║\n');

paList = {'Baseline', 'Random', 'EPA', 'PSO', 'WMMSE', 'GNN', 'GNN+FL'};

fprintf('║  %-12s │ %10s │ %10s │ %10s │ %10s ║\n', ...
    'Power Alloc', 'Best PC', 'Avg ESR', 'Max ESR', 'Gain vs BL');
fprintf('║  ───────────┼────────────┼────────────┼────────────┼───────────║\n');

baseline_best_avg = -inf;
for pi = 1:length(paList)
    paName = paList{pi};
    paMask = cellfun(@(x) contains(x, paName), algoNames);
    if any(paMask)
        paESR = ESR_mean(paMask, :);
        paAvg = mean(paESR, 2);
        [bestESR, bestIdx] = max(paAvg);
        bestPC_cell = algoPCs(paMask);   % 用 () 得到子集 cell，再用 {} 取元素
        bestPC = bestPC_cell{bestIdx};

        blMask = cellfun(@(x) contains(x, 'Baseline') && contains(x, bestPC), algoNames);
        if any(blMask)
            blESR = mean(mean(ESR_mean(blMask, :)));
        else
            blESR = bestESR;
        end
        gain = bestESR - blESR;

        if strcmp(paName, 'Baseline')
            baseline_best_avg = bestESR;
        end

        fprintf('║  %-12s │ %10s │ %10.2f │ %10.2f │ %+10.2f ║\n', ...
            paName, bestPC, bestESR, max(paESR(:)), gain);
    end
end

%% ---- Phase 4: 同步性能评估摘要 ----
fprintf('╠══════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Phase 4: Sync Performance Summary                                          ║\n');
fprintf('║  ────────────────────────────────────────────────────────────────────────────║\n');

maskW = cellfun(@(x) contains(x, 'WMMSE'), algoNames);
maskGF = cellfun(@(x) contains(x, 'GNN+FL'), algoNames);
if any(maskW) && any(maskGF)
    idxW = find(maskW);
    idxGF = find(maskGF);
    [~, wRel] = max(mean(ESR_mean(idxW, :), 2));
    [~, gfRel] = max(mean(ESR_mean(idxGF, :), 2));
    wBest = idxW(wRel);
    gfBest = idxGF(gfRel);
    esrLoss = 1 - (ESR_mean(gfBest, :) ./ ESR_mean(wBest, :));
    fprintf('║  ESR loss (1-ESR_GNN+FL/ESR_WMMSE) @SNR:  min=%8.4f  mean=%8.4f  max=%8.4f ║\n', ...
        min(esrLoss), mean(esrLoss), max(esrLoss));
else
    fprintf('║  ESR loss: unavailable (missing WMMSE or GNN+FL rows)                        ║\n');
end

hasPerf = (nargin >= 14) && exist('Perf', 'var') && isstruct(Perf) && isfield(Perf, 'time_pa_sec');
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
        fprintf('║  Sync latency reduction ratio (All mode):  min=%8.4f  mean=%8.4f  max=%8.4f ║\n', ...
            min(tRed), mean(tRed), max(tRed));
        fprintf('║  Comm reduction ratio (All mode):          min=%8.4f  mean=%8.4f  max=%8.4f ║\n', ...
            min(dRed), mean(dRed), max(dRed));
    else
        fprintf('║  Sync/comm reduction: unavailable (missing Perf indices)                     ║\n');
    end
else
    fprintf('║  Sync/comm reduction: unavailable (Perf not provided)                        ║\n');
end

%% ---- SNR点详细表 (Top 10) ----
fprintf('╠══════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  ESR vs SNR (Top 10 algorithms)                                             ║\n');
fprintf('║  ────────────────────────────────────────────────────────────────────────────║\n');
fprintf('║  %-22s │', 'Algorithm');
for si = 1:num_snr
    fprintf(' SNR%2ddB ', SNR_dB(si));
end
fprintf(' ║\n');
fprintf('║  ────────────────────────┼');
for si = 1:num_snr; fprintf('────────'); end
fprintf('─┫\n');

topAlgos = globalRank(1:min(10, numAlgos));
for ai = 1:length(topAlgos)
    idx = topAlgos(ai);
    fprintf('║  %2d. %-19s │', ai, algoTable(idx).name);
    for si = 1:num_snr
        fprintf(' %6.1f ', ESR_mean(idx, si));
    end
    fprintf(' ║\n');
end

%% ---- 统计摘要 ----
fprintf('╠══════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Simulation Summary                                                           ║\n');
fprintf('║  ────────────────────────────────────────────────────────────────────────────║\n');
fprintf('║    Scenarios:            %3d                                                        ║\n', numScenarios);
fprintf('║    Realizations/scenario: %3d                                                       ║\n', nbrOfRealizations);
fprintf('║    SNR points:           %3d                                                        ║\n', num_snr);
fprintf('║    Total algorithms:     %3d                                                        ║\n', numAlgos);
fprintf('║    Best overall:         %-30s                          ║\n', ESR_best_algo(end));
fprintf('║    Best ESR @max SNR:   %8.2f                                                   ║\n', ESR_best(end));
if ~isempty(PSO_info.iterUsed)
    fprintf('║    PSO iterations (avg): %5.1f                                                   ║\n', ...
        mean(PSO_info.iterUsed));
end
fprintf('╚══════════════════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');
end
