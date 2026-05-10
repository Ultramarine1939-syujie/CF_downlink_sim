function printFinalResults_v2(ESR_mean, algoTable, SNR_dB, ESR_best, ESR_best_algo, ...
    PSO_info, numScenarios, nbrOfRealizations, isSaveFig, isSaveData, savePath, dataPath, Perf)

[numAlgos, num_snr] = size(ESR_mean);
algoNames = {algoTable.name};
algoPCs   = {algoTable.pc};

LMMSE_G_mask = cellfun(@(x) contains(x, 'L-MMSE-G'), algoNames);
isProposed = cellfun(@(x) contains(x, 'GNN'), algoNames);

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    SIMULATION RESULTS  -  GNN vs TRADITIONAL                   ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════════════════════╣\n');

avgESR = mean(ESR_mean, 2);
[~, globalRank] = sort(avgESR, 'descend');

baselineIdx = find(cellfun(@(x) contains(x,'Baseline') && contains(x,'MR') && contains(x,'All'), algoNames), 1);
if isempty(baselineIdx)
    baselineIdx = find(cellfun(@(x) contains(x,'Baseline'), algoNames), 1);
end
if isempty(baselineIdx); baseline_avg = mean(avgESR); else; baseline_avg = avgESR(baselineIdx); end

wmmseIdx = find(cellfun(@(x) contains(x,'WMMSE') && ~contains(x,'L-MMSE-G'), algoNames), 1);
if isempty(wmmseIdx); wmmse_avg = mean(avgESR); else; wmmse_avg = avgESR(wmmseIdx); end

fprintf('║  %-4s │ %-24s │ %8s │ %10s │ %10s ║\n', ...
    'Rank', 'Algorithm', 'Avg ESR', 'vs Baseline', 'vs WMMSE');
fprintf('║  ─────┼──────────────────────────┼──────────┼────────────┼────────────║\n');

printed = 0;
for r = 1:numAlgos
    idx = globalRank(r);
    if LMMSE_G_mask(idx); continue; end

    gain_vs_base = avgESR(idx) - baseline_avg;
    pct_base = gain_vs_base / max(baseline_eps(), 1) * 100;
    gain_vs_wmmse = avgESR(idx) - wmmse_avg;

    marker = '   ';
    if isProposed(idx); marker = ' * '; end
    if r == 1; marker = '[1]'; elseif r == 2; marker = '[2]'; elseif r == 3; marker = '[3]'; end

    fprintf('║  %2d%s │ %-24s │ %8.2f │ %+10.2f%% │ %+10.2f ║\n', ...
        r, marker, algoTable(idx).name, avgESR(idx), pct_base, gain_vs_wmmse);

    printed = printed + 1;
    if printed < numAlgos - sum(LMMSE_G_mask) && mod(printed, 5) == 0
        fprintf('║  ─────┼──────────────────────────┼──────────┼────────────┼────────────║\n');
    end
end

fprintf('╠══════════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Best Algorithm per SNR Point                                                    ║\n');
fprintf('║  ────────────────────────────────────────────────────────────────────────────────║\n');

for si = 1:num_snr
    [sortedESR, sortIdx] = sort(ESR_mean(:, si), 'descend');
    topN = 0;
    for ti = 1:length(sortIdx)
        if ~LMMSE_G_mask(sortIdx(ti))
            topN = topN + 1;
            if topN == 1
                fprintf('║  SNR=%5.1f dB  │  Best: %-24s  ESR=%8.2f         ║\n', ...
                    SNR_dB(si), algoTable(sortIdx(ti)).name, sortedESR(ti));
            elseif topN <= 3
                fprintf('║               │  #%d:    %-24s  ESR=%8.2f         ║\n', ...
                    topN, algoTable(sortIdx(ti)).name, sortedESR(ti));
            end
            if topN >= 3; break; end
        end
    end
end

if ~isempty(PSO_info.iterUsed)
    fprintf('╠══════════════════════════════════════════════════════════════════════════════════╣\n');
    fprintf('║  PSO: avg %5.1f iterations  |  best fitness: %.4f                              ║\n', ...
        mean(PSO_info.iterUsed), PSO_info.bestFitness);
end

fprintf('╠══════════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Summary: %d scenarios × %d realizations × %d SNR × %d algos (excl. L-MMSE-G)  ║\n', ...
    numScenarios, nbrOfRealizations, num_snr, numAlgos - sum(LMMSE_G_mask));
fprintf('║  Overall best: %-30s  ESR=%.2f @ %d dB             ║\n', ...
    ESR_best_algo(end), ESR_best(end), SNR_dB(end));
fprintf('╚══════════════════════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

    function v = baseline_eps()
        if isempty(baselineIdx) || baseline_avg == 0
            v = 1;
        else
            v = abs(baseline_avg);
        end
    end
end
