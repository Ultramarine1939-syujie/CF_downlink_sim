function printFinalResults_v2(ESR_mean, algoTable, SNR_dB, ESR_best, ESR_best_algo, ...
    numScenarios, nbrOfRealizations, isSaveFig, isSaveData, savePath, dataPath, Perf) %#ok<INUSD>
%printFinalResults_v2 Print ranked distributed candidates and references.

[numAlgos, num_snr] = size(ESR_mean);
algoNames = {algoTable.name};
avgESR = mean(ESR_mean, 2);
[~, globalRank] = sort(avgESR, 'descend');

if isfield(algoTable, 'isDistributed')
    distributedMask = [algoTable.isDistributed];
else
    distributedMask = true(1, numAlgos);
end

baselineIdx = find(contains(algoNames, 'Baseline') & contains(algoNames, 'MR') & contains(algoNames, 'All'), 1);
if isempty(baselineIdx)
    baselineIdx = find(contains(algoNames, 'Baseline'), 1);
end
if isempty(baselineIdx)
    baseline_avg = mean(avgESR, 'omitnan');
else
    baseline_avg = avgESR(baselineIdx);
end

wmmseIdx = find(contains(algoNames, 'WMMSE') & ~contains(algoNames, 'D-WMMSE') & ~contains(algoNames, 'L-MMSE-G'), 1);
if isempty(wmmseIdx)
    wmmse_avg = mean(avgESR, 'omitnan');
else
    wmmse_avg = avgESR(wmmseIdx);
end

fprintf('\n');
fprintf('=====================================================================\n');
fprintf('  DISTRIBUTED DOWNLINK CANDIDATES\n');
fprintf('=====================================================================\n');
fprintf('  %-4s  %-30s  %10s  %12s  %12s\n', ...
    'Rank', 'Algorithm', 'Avg ESR', 'vs Baseline', 'vs WMMSE');
fprintf('---------------------------------------------------------------------\n');

rankNo = 0;
for r = 1:numAlgos
    idx = globalRank(r);
    if ~distributedMask(idx)
        continue;
    end

    rankNo = rankNo + 1;
    pct_base = (avgESR(idx) - baseline_avg) / max(abs(baseline_avg), 1) * 100;
    gain_wmmse = avgESR(idx) - wmmse_avg;
    fprintf('  %-4d  %-30s  %10.2f  %+11.2f%%  %+12.2f\n', ...
        rankNo, algoTable(idx).name, avgESR(idx), pct_base, gain_wmmse);
end

fprintf('\n');
fprintf('Best distributed algorithms per SNR:\n');
for si = 1:num_snr
    [sortedESR, sortIdx] = sort(ESR_mean(:, si), 'descend');
    printed = 0;
    for ti = 1:numel(sortIdx)
        idx = sortIdx(ti);
        if distributedMask(idx)
            printed = printed + 1;
            fprintf('  SNR=%5.1f dB  #%d  %-30s  ESR=%8.2f\n', ...
                SNR_dB(si), printed, algoTable(idx).name, sortedESR(ti));
            if printed >= 3
                break;
            end
        end
    end
end

referenceMask = ~distributedMask;
if any(referenceMask)
    fprintf('\n');
    fprintf('Centralized / offline references:\n');
    refIdx = globalRank(referenceMask(globalRank));
    for ri = 1:min(5, numel(refIdx))
        idx = refIdx(ri);
        fprintf('  Ref #%d  %-30s  Avg ESR=%8.2f  PC=%s  PA=%s\n', ...
            ri, algoTable(idx).name, avgESR(idx), algoTable(idx).pcArch, algoTable(idx).paArch);
    end
end

fprintf('\n');
fprintf('Summary: %d scenarios x %d realizations x %d SNR x %d distributed algos\n', ...
    numScenarios, nbrOfRealizations, num_snr, sum(distributedMask));
fprintf('Overall best at highest SNR: %s  ESR=%.2f @ %.1f dB\n', ...
    ESR_best_algo(end), ESR_best(end), SNR_dB(end));
fprintf('=====================================================================\n\n');
end
