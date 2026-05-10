function exportTrainingData(outputDir, nSnapshotsPerSNR, useParallel)
%exportTrainingData 生成 GNN-FL 训练数据集 (Phase 1)
%   outputDir: 输出目录路径
%   nSnapshotsPerSNR: 每个 SNR 点的快照数（默认 500）
%   useParallel: 是否使用并行计算（默认 true，如有警告则自动降级）
%
% 生成内容:
%   features.sqrtGain  - sqrt(gainOverNoise) (L×K×N_snaps)
%   features.D         - AP-UE 关联矩阵 (L×K×N_snaps)
%   features.sigma_e   - 信道估计误差方差 (1×1×N_snaps)
%   labels.rho_WMMSE   - WMMSE 最优功率分配 (L×K×N_snaps)
%   labels.ESR_WMMSE   - WMMSE ESR 性能 (N_snaps×1)
%   meta.SNR_dB, meta.mode, meta.seed, meta.L, meta.K, meta.N
%
% 数据增强策略:
%   1. 拓扑扰动: 随机丢弃 10%~30% 的 AP-UE 连接
%   2. 噪声扰动: 对 sigma_e 添加 ±20% 随机扰动
%   3. SNR 覆盖: -10:5:30 dB，共 9 个 SNR 点

addpath(genpath(pwd));

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(pwd, 'data', 'gnn_training');
end
if nargin < 2 || isempty(nSnapshotsPerSNR)
    nSnapshotsPerSNR = 500;
end
if nargin < 3
    useParallel = true;
end

L = 100; N = 1; K = 20;
tau_c = 200; tau_p = 10;
ASD_varphi = deg2rad(15);
ASD_theta  = deg2rad(15);
p = 100;
sigma_e = 0.3;
nIter = 5;
nbrOfRealizations = 100;
nbrOfSetups = 1;
maxIterWMMSE = 30;
tolWMMSE = 1e-4;
SNR_dB_range = -10:5:30;
modes = {'All', 'DCC'};

N_snaps = length(SNR_dB_range) * nSnapshotsPerSNR * length(modes);

sqrtGain_cell  = cell(N_snaps, 1);
D_cell         = cell(N_snaps, 1);
sigma_e_cell   = cell(N_snaps, 1);
rho_WMMSE_cell = cell(N_snaps, 1);
rho_Dist_cell  = cell(N_snaps, 1);
rho_EPA_cell   = cell(N_snaps, 1);
ESR_WMMSE_cell = cell(N_snaps, 1);
ESR_Dist_cell  = cell(N_snaps, 1);
ESR_EPA_cell   = cell(N_snaps, 1);
meta_cell      = cell(N_snaps, 1);

fprintf('=== GNN-FL 训练数据集生成 (Phase 1) ===\n');
fprintf('输出目录: %s\n', outputDir);
fprintf('每个 SNR 快照数: %d\n', nSnapshotsPerSNR);
fprintf('SNR 范围: -10:5:30 dB\n');
fprintf('接入模式: All + DCC\n');
fprintf('预计生成: %d 快照\n', N_snaps);

if ~exist(outputDir, 'dir'); mkdir(outputDir); end

useParallel = useParallel && ~isempty(ver('parallel'));
if useParallel
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        try
            parpool;
        catch
            useParallel = false;
            fprintf('警告: 并行池启动失败，切换为串行模式\n');
        end
    end
end
fprintf('并行模式: %s\n\n', mat2str(useParallel));

tStart = tic;

totalWorkers = 0;
if useParallel
    try
        poolobj = gcp('nocreate');
        if ~isempty(poolobj)
            totalWorkers = poolobj.NumWorkers;
        end
    catch
        totalWorkers = 0;
    end
end

fprintf('并行工作进程数: %d\n', max(1, totalWorkers));

for si = 1:length(SNR_dB_range)
    SNR_dB = SNR_dB_range(si);
    Pt = db2pow(SNR_dB);

    fprintf('--- SNR = %2d dB (Pt = %.4f W) ---\n', SNR_dB, Pt);

    for modeIdx = 1:length(modes)
        mode = modes{modeIdx};
        modeStart = tic;

        if useParallel
            parfor snap = 1:nSnapshotsPerSNR
                [sqrtGain_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 D_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 sigma_e_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 rho_WMMSE_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 rho_Dist_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 rho_EPA_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 ESR_WMMSE_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 ESR_Dist_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 ESR_EPA_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 meta_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}] = ...
                    generateSingleSnapshot(L, K, N, tau_p, tau_c, ASD_varphi, ASD_theta, p, sigma_e, ...
                                          nbrOfRealizations, nbrOfSetups, maxIterWMMSE, tolWMMSE, ...
                                          SNR_dB, Pt, mode, si, modeIdx, snap);
            end
        else
            for snap = 1:nSnapshotsPerSNR
                [sqrtGain_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 D_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 sigma_e_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 rho_WMMSE_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 rho_Dist_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 rho_EPA_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 ESR_WMMSE_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 ESR_Dist_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 ESR_EPA_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}, ...
                 meta_cell{snap + (modeIdx-1)*nSnapshotsPerSNR + (si-1)*nSnapshotsPerSNR*length(modes)}] = ...
                    generateSingleSnapshot(L, K, N, tau_p, tau_c, ASD_varphi, ASD_theta, p, sigma_e, ...
                                          nbrOfRealizations, nbrOfSetups, maxIterWMMSE, tolWMMSE, ...
                                          SNR_dB, Pt, mode, si, modeIdx, snap);
            end
        end

        modeElapsed = toc(modeStart);
        done = (si-1) * nSnapshotsPerSNR * length(modes) + modeIdx * nSnapshotsPerSNR;
        fprintf('  [%s] 完成 %d/%d 快照 (%.1f%%) - 用时 %.1f 秒\n', ...
                mode, done, N_snaps, done/N_snaps*100, modeElapsed);
    end
end

elapsed = toc(tStart);
fprintf('\n数据生成总用时: %.1f 秒 (%.1f 分钟)\n', elapsed, elapsed/60);

fprintf('\n正在拼接数据...\n');
sqrtGain_all  = cat(3, sqrtGain_cell{:});
D_all         = cat(3, D_cell{:});
sigma_e_all   = reshape([sigma_e_cell{:}], 1, 1, N_snaps);
rho_WMMSE_all = cat(3, rho_WMMSE_cell{:});
rho_Dist_all  = cat(3, rho_Dist_cell{:});
rho_EPA_all   = cat(3, rho_EPA_cell{:});
ESR_WMMSE_all = cell2mat(ESR_WMMSE_cell);
ESR_Dist_all  = cell2mat(ESR_Dist_cell);
ESR_EPA_all   = cell2mat(ESR_EPA_cell);
meta = meta_cell;

fprintf('数据拼接完成!\n');

features.sqrtGain = sqrtGain_all;
features.D         = D_all;
features.sigma_e   = sigma_e_all;

labels.rho_WMMSE   = rho_WMMSE_all;
labels.rho_Dist    = rho_Dist_all;
labels.rho_EPA     = rho_EPA_all;
labels.ESR_WMMSE   = ESR_WMMSE_all;
labels.ESR_Dist    = ESR_Dist_all;
labels.ESPA        = ESR_EPA_all;

sysConfig = struct('L', L, 'K', K, 'N', N, 'tau_c', tau_c, 'tau_p', tau_p, ...
                   'ASD_varphi', ASD_varphi, 'ASD_theta', ASD_theta, ...
                   'p', p, 'sigma_e', sigma_e, ...
                   'nbrOfRealizations', nbrOfRealizations, ...
                   'SNR_dB_range', SNR_dB_range, ...
                   'nSnapshotsPerSNR', nSnapshotsPerSNR, ...
                   'useParallel', useParallel);

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
outputFile = fullfile(outputDir, sprintf('gnn_training_data_%s.mat', timestamp));
fprintf('\n保存至: %s\n', outputFile);

save(outputFile, 'features', 'labels', 'meta', 'sysConfig', '-v7.3');

fprintf('\n========================================\n');
fprintf('  数据集生成完成！\n');
fprintf('========================================\n');
fprintf('特征维度:\n');
fprintf('  sqrtGain: %s\n', mat2str(size(features.sqrtGain)));
fprintf('  D:         %s\n', mat2str(size(features.D)));
fprintf('标签维度:\n');
fprintf('  rho_WMMSE: %s\n', mat2str(size(labels.rho_WMMSE)));
fprintf('  ESR_WMMSE: %s\n', mat2str(size(labels.ESR_WMMSE)));

fprintf('\n快照统计:\n');
fprintf('  总数: %d\n', N_snaps);
fprintf('  All:  %d\n', sum(cellfun(@(x) strcmp(x.mode, 'All'), meta)));
fprintf('  DCC:  %d\n', sum(cellfun(@(x) strcmp(x.mode, 'DCC'), meta)));

fprintf('\nESR 性能范围:\n');
fprintf('  WMMSE: %.2f ~ %.2f  (mean=%.2f)\n', min(ESR_WMMSE_all), max(ESR_WMMSE_all), mean(ESR_WMMSE_all));
fprintf('  Dist:  %.2f ~ %.2f  (mean=%.2f)\n', min(ESR_Dist_all), max(ESR_Dist_all), mean(ESR_Dist_all));
fprintf('  EPA:   %.2f ~ %.2f  (mean=%.2f)\n', min(ESR_EPA_all), max(ESR_EPA_all), mean(ESR_EPA_all));

gain_vs_Dist = (ESR_WMMSE_all - ESR_Dist_all) ./ max(ESR_Dist_all, eps) * 100;
fprintf('\nWMMSE vs Dist 增益: %.1f%% ~ %.1f%%  (mean=%.1f%%)\n', ...
    min(gain_vs_Dist), max(gain_vs_Dist), mean(gain_vs_Dist));

fprintf('\n========================================\n');
fprintf('下一步: python/train_gnn.py\n');
fprintf('========================================\n');
end


function [sqrtGain, D_aug, sigma_e_aug, rho_WMMSE_aug, rho_Dist_aug, rho_EPA_aug, ...
          ESR_W, ESR_D, ESR_E, meta_out] = ...
    generateSingleSnapshot(L, K, N, tau_p, tau_c, ASD_varphi, ASD_theta, p, sigma_e, ...
                          nbrOfRealizations, nbrOfSetups, maxIterWMMSE, tolWMMSE, ...
                          SNR_dB, Pt, mode, si, modeIdx, snap)
seed = si * 100000 + modeIdx * 10000 + snap;
rng(seed);

[gainOverNoisedB_3D, R_5D, pilotIndex_2D, ~, D_small_3D] = ...
    generateSetup(L, K, N, tau_p, nbrOfSetups, seed, ASD_varphi, ASD_theta);

gainOverNoisedB = gainOverNoisedB_3D(:, :, 1);
R               = R_5D(:, :, :, :, 1);
pilotIndex      = pilotIndex_2D(:, 1);
D_small         = D_small_3D(:, :, 1);

gainOverNoise = db2pow(gainOverNoisedB);

if strcmp(mode, 'All')
    D = ones(L, K);
else
    D = D_small;
end

[Hhat, ~, ~, C] = functionChannelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p);
H = Hhat + sqrt(sigma_e^2/2) * (randn(size(Hhat)) + 1i*randn(size(Hhat)));

[V_MR, sc_MR] = functionPrecoding_MR(Hhat, nbrOfRealizations, N, K, L);

rho_WMMSE = computeRhoWMMSE(Hhat, D, Pt, N, K, L, nbrOfRealizations, maxIterWMMSE, tolWMMSE);
rho_Dist  = computeRhoDist(D, gainOverNoise, Pt, L, K);
rho_EPA   = computeRhoEPA(D, Pt, L, K);

SE_WMMSE = functionComputeSE_downlink_MR(H, V_MR, sc_MR, D, tau_c, tau_p, nbrOfRealizations, N, K, L, rho_WMMSE);
SE_Dist  = functionComputeSE_downlink_MR(H, V_MR, sc_MR, D, tau_c, tau_p, nbrOfRealizations, N, K, L, rho_Dist);
SE_EPA   = functionComputeSE_downlink_MR(H, V_MR, sc_MR, D, tau_c, tau_p, nbrOfRealizations, N, K, L, rho_EPA);

ESR_W = sum(SE_WMMSE);
ESR_D = sum(SE_Dist);
ESR_E = sum(SE_EPA);

dropRate = 0.1 + 0.2 * rand();
D_aug = D;
aug_mask = rand(L, K);
dropped = (aug_mask < dropRate) & (D_aug == 1);
D_aug(dropped) = 0;

rho_WMMSE_aug = rho_WMMSE;
rho_WMMSE_aug(dropped) = 0;
rho_Dist_aug  = rho_Dist;
rho_Dist_aug(dropped)  = 0;
rho_EPA_aug   = rho_EPA;
rho_EPA_aug(dropped)   = 0;

sigma_e_aug = sigma_e * (1 + 0.2 * (rand() - 0.5) * 2);

sqrtGain = sqrt(gainOverNoise);
meta_out = struct('SNR_dB', SNR_dB, 'mode', mode, 'seed', seed, 'snapIdx', snap);
end
