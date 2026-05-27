function Results = runUGNNGuardExperiments(varargin)
%runUGNNGuardExperiments Paper-oriented U-GNN guard ablations.
%
% This experiment reuses cached channels/precoders from Combined_Downlink_Sim
% and evaluates:
%   1) pure U-GNN without entropy guard
%   2) guarded U-GNN across entropy thresholds
%   3) FPCP fallback alpha ablation
%
% Outputs are saved under main/SimulationData and main/Imgs.

opts = parseOptions(varargin{:});

scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);
addpath(genpath(rootDir));
params = getDefaultParams();

L = params.system.L;
K = params.system.K;
N = params.system.N;
tau_c = params.system.tau_c;
tau_p = params.system.tau_p;
p = params.power.p;
SNR_dB = params.power.SNR_dB;
sigma_e = params.csi.sigma_e;
nIter = params.csi.nIter;
nbrOfRealizations = params.simulation.nbrOfRealizations;
numScenarios = min(params.simulation.numScenarios, opts.maxScenarios);

dataPath = fullfile(scriptDir, params.output.dataPath);
savePath = fullfile(scriptDir, params.output.savePath);
if ~exist(dataPath, 'dir'); mkdir(dataPath); end
if ~exist(savePath, 'dir'); mkdir(savePath); end

ugnnModelPath = fullfile(rootDir, params.gnn.ugnnModelFile);
if ~isfile(ugnnModelPath)
    error('U-GNN checkpoint not found: %s', ugnnModelPath);
end

modeName = params.simulation.accessModes{1};
pcList = {'MR', 'LMMSE', 'RMMSE', 'LMMSE_G'};
pcLabels = {'MR', 'L-MMSE', 'R-MMSE', 'L-MMSE-G'};
thresholds = opts.thresholds(:).';
alphas = opts.alphas(:).';
defaultAlpha = opts.defaultAlpha;

methodLabels = buildMethodLabels(thresholds, alphas);
numMethods = numel(methodLabels);
numPC = numel(pcList);
numSNR = numel(SNR_dB);
numThr = numel(thresholds);
numAlpha = numel(alphas);

ESR_sum = zeros(numMethods, numPC, numSNR);
time_sum = zeros(numMethods, numSNR);
trigger_count = zeros(numThr, numSNR);
sample_count = zeros(1, numSNR);
entropy_sum = zeros(1, numSNR);
invalid_count = zeros(1, numSNR);

fprintf('\n=====================================================================\n');
fprintf('  U-GNN GUARD ABLATION EXPERIMENTS\n');
fprintf('=====================================================================\n');
fprintf('  Scenarios: %d | SNR points: %d | PC: %d | Methods: %d\n', ...
    numScenarios, numSNR, numPC, numMethods);
fprintf('  Thresholds: %s\n', mat2str(thresholds));
fprintf('  FPCP alphas: %s\n', mat2str(alphas));
fprintf('  Model: %s\n', ugnnModelPath);

for s = 1:numScenarios
    scenarioCacheFile = fullfile(dataPath, sprintf('cache_scenario_s%d.mat', s));
    if ~isfile(scenarioCacheFile)
        error('Scenario cache missing: %s. Run main simulation first.', scenarioCacheFile);
    end
    S = load(scenarioCacheFile, 'H', 'Hhat', 'D_dcc', 'gainOverNoise');
    H = S.H;
    Hhat = S.Hhat;
    D = S.D_dcc;
    gainOverNoise = S.gainOverNoise;

    fprintf('\n  Scenario %d/%d\n', s, numScenarios);
    for si = 1:numSNR
        snrCacheFile = fullfile(dataPath, sprintf('cache_snr_s%d_sn%.0f.mat', s, SNR_dB(si)));
        if ~isfile(snrCacheFile)
            error('SNR cache missing: %s. Run main simulation first.', snrCacheFile);
        end
        C = load(snrCacheFile, 'Pt', 'V_cache_all');
        Pt = C.Pt;
        V_cache_mode = C.V_cache_all.(modeName);

        pureOpts = struct('guardEnabled', false, 'minEntropy', 0, 'guardAlpha', defaultAlpha);
        [rhoPure, timingPure] = computeRhoUGNN(Hhat, D, gainOverNoise, Pt, ...
            ugnnModelPath, sigma_e, pureOpts);
        entropy = timingPure.guard_entropy;
        invalid = timingPure.guard_invalid;

        fpcpDefaultTic = tic;
        rhoFpcpDefault = computeRhoFPCP(D, gainOverNoise, Pt, L, K, defaultAlpha);
        fpcpDefaultSec = toc(fpcpDefaultTic);

        methodTimes = zeros(numMethods, 1);
        methodTimes(1) = timingPure.total_sec;

        pureESR = zeros(1, numPC);
        for ci = 1:numPC
            pcName = pcList{ci};
            V = V_cache_mode.(pcName).V;
            scaling = V_cache_mode.(pcName).scaling;
            pureESR(ci) = computeESR(pcName, Hhat, H, V, scaling, D, tau_c, tau_p, ...
                nbrOfRealizations, N, K, L, p, rhoPure, sigma_e, Pt, nIter);
        end
        ESR_sum(1, :, si) = ESR_sum(1, :, si) + pureESR;

        alphaESR = zeros(numAlpha, numPC);
        alphaTimes = zeros(numAlpha, 1);
        alphaRhos = cell(numAlpha, 1);
        for ai = 1:numAlpha
            alphaTic = tic;
            alphaRhos{ai} = computeRhoFPCP(D, gainOverNoise, Pt, L, K, alphas(ai));
            alphaTimes(ai) = toc(alphaTic);
            for ci = 1:numPC
                pcName = pcList{ci};
                V = V_cache_mode.(pcName).V;
                scaling = V_cache_mode.(pcName).scaling;
                alphaESR(ai, ci) = computeESR(pcName, Hhat, H, V, scaling, D, tau_c, tau_p, ...
                    nbrOfRealizations, N, K, L, p, alphaRhos{ai}, sigma_e, Pt, nIter);
            end
            methodIdx = 1 + numThr + ai;
            ESR_sum(methodIdx, :, si) = ESR_sum(methodIdx, :, si) + alphaESR(ai, :);
            methodTimes(methodIdx) = alphaTimes(ai);
        end

        defaultAlphaIdx = find(abs(alphas - defaultAlpha) < 1e-12, 1);
        if isempty(defaultAlphaIdx)
            defaultESR = zeros(1, numPC);
            for ci = 1:numPC
                pcName = pcList{ci};
                V = V_cache_mode.(pcName).V;
                scaling = V_cache_mode.(pcName).scaling;
                defaultESR(ci) = computeESR(pcName, Hhat, H, V, scaling, D, tau_c, tau_p, ...
                    nbrOfRealizations, N, K, L, p, rhoFpcpDefault, sigma_e, Pt, nIter);
            end
        else
            defaultESR = alphaESR(defaultAlphaIdx, :);
        end

        for ti = 1:numThr
            methodIdx = 1 + ti;
            doFallback = invalid || entropy < thresholds(ti);
            if doFallback
                ESR_sum(methodIdx, :, si) = ESR_sum(methodIdx, :, si) + defaultESR;
                methodTimes(methodIdx) = timingPure.total_sec + fpcpDefaultSec;
                trigger_count(ti, si) = trigger_count(ti, si) + 1;
            else
                ESR_sum(methodIdx, :, si) = ESR_sum(methodIdx, :, si) + pureESR;
                methodTimes(methodIdx) = timingPure.total_sec;
            end
        end
        time_sum(:, si) = time_sum(:, si) + methodTimes;

        sample_count(si) = sample_count(si) + 1;
        entropy_sum(si) = entropy_sum(si) + entropy;
        invalid_count(si) = invalid_count(si) + double(invalid);
        fprintf('    SNR=%2.0f dB | entropy=%.4f | invalid=%d\n', ...
            SNR_dB(si), entropy, invalid);
    end
end

ESR_mean = ESR_sum / max(numScenarios, 1);
time_mean = time_sum / max(numScenarios, 1);
trigger_rate = trigger_count ./ max(sample_count, 1);
entropy_mean = entropy_sum ./ max(sample_count, 1);
invalid_rate = invalid_count ./ max(sample_count, 1);

[T, TriggerTable] = buildTables(methodLabels, pcLabels, SNR_dB, ESR_mean, ...
    time_mean, thresholds, trigger_rate, entropy_mean, invalid_rate);

Results = struct();
Results.methodLabels = methodLabels;
Results.pcLabels = pcLabels;
Results.SNR_dB = SNR_dB;
Results.thresholds = thresholds;
Results.alphas = alphas;
Results.defaultAlpha = defaultAlpha;
Results.ESR_mean = ESR_mean;
Results.time_mean = time_mean;
Results.trigger_rate = trigger_rate;
Results.entropy_mean = entropy_mean;
Results.invalid_rate = invalid_rate;
Results.table = T;
Results.triggerTable = TriggerTable;

save(fullfile(dataPath, 'UGNN_Guard_Ablation_Results.mat'), 'Results');
writetable(T, fullfile(dataPath, 'UGNN_Guard_Ablation_Table.csv'));
writetable(TriggerTable, fullfile(dataPath, 'UGNN_Guard_Trigger_Table.csv'));

plotUGNNGuardAblations(Results, savePath, params.output.isSaveFig);

fprintf('\n[OK] U-GNN guard ablation saved:\n');
fprintf('  %s\n', fullfile(dataPath, 'UGNN_Guard_Ablation_Results.mat'));
fprintf('  %s\n', fullfile(dataPath, 'UGNN_Guard_Ablation_Table.csv'));
fprintf('  %s\n', fullfile(dataPath, 'UGNN_Guard_Trigger_Table.csv'));
end

function opts = parseOptions(varargin)
    p = inputParser;
    addParameter(p, 'thresholds', [0.05 0.10 0.20 0.35 0.50 0.70]);
    addParameter(p, 'alphas', [-2 -1 -0.5 0 0.5 1 2]);
    addParameter(p, 'defaultAlpha', -1.0);
    addParameter(p, 'maxScenarios', inf);
    parse(p, varargin{:});
    opts = p.Results;
end

function labels = buildMethodLabels(thresholds, alphas)
    labels = strings(1, 1 + numel(thresholds) + numel(alphas));
    labels(1) = "Pure U-GNN";
    for i = 1:numel(thresholds)
        labels(1 + i) = sprintf("Guarded U-GNN Hmin=%.2f", thresholds(i));
    end
    for i = 1:numel(alphas)
        labels(1 + numel(thresholds) + i) = sprintf("FPCP alpha=%.1f", alphas(i));
    end
end

function ESR = computeESR(pcName, Hhat, H, V, scaling, D, tau_c, tau_p, ...
        nbrOfRealizations, N, K, L, p, rho, sigma_e, Pt, nIter)
    switch pcName
        case 'MR'
            SE_k = functionComputeSE_downlink_MR(H, V, scaling, D, ...
                tau_c, tau_p, nbrOfRealizations, N, K, L, rho);
        case {'LMMSE', 'LMMSE_G'}
            SE_k = functionComputeSE_downlink_LMMSE(Hhat, H, V, scaling, D, ...
                tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho);
        case 'RMMSE'
            SE_k = functionComputeSE_downlink_RobustMMSE(Hhat, H, V, scaling, D, ...
                tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho, sigma_e, Pt, nIter);
        otherwise
            error('Unknown PC method: %s', pcName);
    end
    ESR = sum(SE_k);
end

function [T, TriggerTable] = buildTables(methodLabels, pcLabels, SNR_dB, ESR_mean, ...
        time_mean, thresholds, trigger_rate, entropy_mean, invalid_rate)
    rows = {};
    for mi = 1:numel(methodLabels)
        for ci = 1:numel(pcLabels)
            for si = 1:numel(SNR_dB)
                rows(end+1, :) = {char(methodLabels(mi)), pcLabels{ci}, SNR_dB(si), ...
                    ESR_mean(mi, ci, si), time_mean(mi, si)}; %#ok<AGROW>
            end
        end
    end
    T = cell2table(rows, 'VariableNames', ...
        {'Method', 'Precoding', 'SNR_dB', 'ESR', 'PowerAllocationTime_s'});

    triggerRows = {};
    for ti = 1:numel(thresholds)
        for si = 1:numel(SNR_dB)
            triggerRows(end+1, :) = {thresholds(ti), SNR_dB(si), ...
                trigger_rate(ti, si), entropy_mean(si), invalid_rate(si)}; %#ok<AGROW>
        end
    end
    TriggerTable = cell2table(triggerRows, 'VariableNames', ...
        {'EntropyThreshold', 'SNR_dB', 'TriggerRate', 'MeanEntropy', 'InvalidRate'});
end

function plotUGNNGuardAblations(Results, savePath, isSaveFig)
    if ~isSaveFig
        return;
    end

    snr = Results.SNR_dB;
    rmmseIdx = find(strcmp(Results.pcLabels, 'R-MMSE'), 1);
    pureIdx = find(Results.methodLabels == "Pure U-GNN", 1);
    defaultGuardIdx = find(Results.methodLabels == "Guarded U-GNN Hmin=0.35", 1);
    if isempty(defaultGuardIdx)
        defaultGuardIdx = 2;
    end
    defaultFpcpIdx = find(Results.methodLabels == "FPCP alpha=-1.0", 1);

    fig1 = figure('Color', 'w', 'Position', [120, 120, 900, 520]);
    hold on; grid on; box on;
    plot(snr, squeeze(Results.ESR_mean(pureIdx, rmmseIdx, :)), '-o', 'LineWidth', 2);
    plot(snr, squeeze(Results.ESR_mean(defaultGuardIdx, rmmseIdx, :)), '-s', 'LineWidth', 2);
    if ~isempty(defaultFpcpIdx)
        plot(snr, squeeze(Results.ESR_mean(defaultFpcpIdx, rmmseIdx, :)), '--^', 'LineWidth', 2);
    end
    xlabel('SNR (dB)');
    ylabel('ESR (bit/s/Hz)');
    title('Pure vs Entropy-Guarded U-GNN under R-MMSE');
    legend('Pure U-GNN', 'Guarded U-GNN', 'FPCP alpha=-1', 'Location', 'best');
    saveFigure(fig1, savePath, 'FigU1_UGNN_Guard_ESR_vs_SNR');

    fig2 = figure('Color', 'w', 'Position', [140, 140, 900, 520]);
    hold on; grid on; box on;
    for ti = 1:numel(Results.thresholds)
        plot(snr, Results.trigger_rate(ti, :), '-o', 'LineWidth', 1.8);
    end
    xlabel('SNR (dB)');
    ylabel('Guard trigger rate');
    title('U-GNN Entropy Guard Trigger Rate');
    legend(arrayfun(@(x) sprintf('Hmin=%.2f', x), Results.thresholds, 'UniformOutput', false), ...
        'Location', 'best');
    ylim([0 1.05]);
    saveFigure(fig2, savePath, 'FigU2_UGNN_Guard_TriggerRate');

    alphaIdx = find(startsWith(cellstr(Results.methodLabels), 'FPCP alpha='));
    alphaESR = mean(Results.ESR_mean(alphaIdx, rmmseIdx, :), 3);
    fig3 = figure('Color', 'w', 'Position', [160, 160, 760, 500]);
    bar(Results.alphas, alphaESR, 0.65);
    grid on; box on;
    xlabel('FPCP alpha');
    ylabel('Average ESR under R-MMSE (bit/s/Hz)');
    title('FPCP Fallback Alpha Ablation');
    saveFigure(fig3, savePath, 'FigU3_UGNN_FPCP_Alpha_Ablation');
end

function saveFigure(fig, savePath, fileStem)
    if ~exist(savePath, 'dir'); mkdir(savePath); end
    saveas(fig, fullfile(savePath, [fileStem '.png']));
    saveas(fig, fullfile(savePath, [fileStem '.fig']));
end
