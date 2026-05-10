%% 消融实验结果可视化
function plotAblationResults(ablationDir, savePath, isSaveFig)
% PLOTABLATIONRESULTS 绘制消融实验对比图
%
% 输入:
%   ablationDir - models/ablation/ 目录路径
%   savePath    - 图表保存路径
%   isSaveFig   - 是否保存图像
%
% 读取 ablationDir/ablation_results.csv 生成对比图表

resultsFile = fullfile(ablationDir, 'ablation_results.csv');
if ~exist(resultsFile, 'file')
    fprintf('[plotAblationResults] ERROR: %s not found\n', resultsFile);
    return;
end

% 读取结果
T = readtable(resultsFile);
variantNames = T.variant;
valCorr = T.val_corr;
valMSE = T.val_mse;
modelTypes = T.model_type;
trainTypes = T.training_type;
params = T.model_params;

nVariants = height(T);

%% ===== 图A1: Val Correlation 对比柱状图 =====
figure('Visible', 'off', 'Position', [100 100 1000 500]);

colors = lines(nVariants);
barh(1:nVariants, valCorr, 0.6, 'FaceColor', 'flat', 'CData', colors);
set(gca, 'YTick', 1:nVariants, 'YTickLabel', variantNames);
xlabel('Validation Correlation (Val Corr)');
title('Ablation Study: Validation Correlation');
grid on;
set(gca, 'FontSize', 11);

% 标注数值
for i = 1:nVariants
    text(valCorr(i) + 0.005, i, sprintf('%.4f', valCorr(i)), ...
        'FontSize', 10, 'VerticalAlignment', 'middle');
end

if isSaveFig
    saveas(gcf, fullfile(savePath, 'FigA1_Ablation_Corr.png'));
    saveas(gcf, fullfile(savePath, 'FigA1_Ablation_Corr.fig'));
end

%% ===== 图A2: Val NZ-MSE 对比柱状图 =====
figure('Visible', 'off', 'Position', [100 100 1000 500]);

barh(1:nVariants, valMSE, 0.6, 'FaceColor', 'flat', 'CData', colors);
set(gca, 'YTick', 1:nVariants, 'YTickLabel', variantNames);
xlabel('Validation NZ-MSE');
title('Ablation Study: Nonzero MSE (lower is better)');
grid on;
set(gca, 'FontSize', 11);

for i = 1:nVariants
    text(valMSE(i) + max(valMSE)*0.01, i, sprintf('%.4f', valMSE(i)), ...
        'FontSize', 10, 'VerticalAlignment', 'middle');
end

if isSaveFig
    saveas(gcf, fullfile(savePath, 'FigA2_Ablation_MSE.png'));
    saveas(gcf, fullfile(savePath, 'FigA2_Ablation_MSE.fig'));
end

%% ===== 图A3: 多指标雷达图 =====
figure('Visible', 'off', 'Position', [100 100 700 700]);

% 选取核心对比变体
coreVariants = {'full', 'mlp_only', 'fedavg_full', 'global_norm', 'no_augment'};
coreMask = ismember(variantNames, coreVariants);
coreNames = variantNames(coreMask);
coreCorr = valCorr(coreMask);
coreMSE = valMSE(coreMask);
coreParams = params(coreMask);
coreTime = T.train_time_sec(coreMask);

nCore = sum(coreMask);
if nCore >= 3
    % 归一化各指标到 [0, 1]（MSE 和时间取反，越小越好）
    corrNorm = coreCorr / max(coreCorr);
    mseNorm = 1 - coreMSE / max(coreMSE);  % 反转：MSE越小分越高
    paramNorm = 1 - coreParams / max(coreParams);  % 参数越少分越高
    timeNorm = 1 - coreTime / max(coreTime);  % 时间越短分越高

    metrics = [corrNorm, mseNorm, paramNorm, timeNorm];
    metricLabels = {'Corr', '1-MSE', 'Efficiency', 'Speed'};

    % 雷达图
    angles = linspace(0, 2*pi, size(metrics, 2) + 1);
    metricsClosed = [metrics, metrics(:, 1)];

    hold on;
    coreColors = lines(nCore);
    for i = 1:nCore
        fill(angles, metricsClosed(i, :), coreColors(i, :), ...
            'FaceAlpha', 0.1, 'EdgeColor', coreColors(i, :), 'LineWidth', 2);
        plot(angles, metricsClosed(i, :), '-o', ...
            'Color', coreColors(i, :), 'LineWidth', 2, ...
            'MarkerSize', 6, 'DisplayName', coreNames{i});
    end
    hold off;

    set(gca, 'XTick', angles(1:end-1));
    set(gca, 'XTickLabel', metricLabels);
    axis([0 2*pi 0 1.05]);
    legend('Location', 'SouthEast');
    title('Ablation Study: Multi-metric Radar');
    grid on;
    set(gca, 'FontSize', 11);
end

if isSaveFig
    saveas(gcf, fullfile(savePath, 'FigA3_Ablation_Radar.png'));
    saveas(gcf, fullfile(savePath, 'FigA3_Ablation_Radar.fig'));
end

%% ===== 图A4: 消融维度成对对比 =====
figure('Visible', 'off', 'Position', [100 100 1000 400]);

% 定义对比组
comparisons = {
    'GNN vs No-GNN', {'full', 'mlp_only'};
    'FedAvg vs Central', {'full', 'fedavg_full'};
    'Per-sample vs Global', {'full', 'global_norm'};
    'With vs No Augment', {'full', 'no_augment'};
};

nComp = size(comparisons, 1);
xPositions = 1:nComp;
deltaCorr = zeros(1, nComp);
labels = cell(1, nComp);

for c = 1:nComp
    v1 = comparisons{c, 2}{1};
    v2 = comparisons{c, 2}{2};
    idx1 = find(strcmp(variantNames, v1), 1);
    idx2 = find(strcmp(variantNames, v2), 1);

    if ~isempty(idx1) && ~isempty(idx2)
        deltaCorr(c) = valCorr(idx2) - valCorr(idx1);
        labels{c} = comparisons{c, 1};
    else
        deltaCorr(c) = NaN;
        labels{c} = comparisons{c, 1};
    end
end

b = bar(xPositions, deltaCorr, 0.5);
b.FaceColor = 'flat';
for c = 1:nComp
    if deltaCorr(c) >= 0
        b.CData(c, :) = [0.2 0.7 0.2];
    else
        b.CData(c, :) = [0.8 0.2 0.2];
    end
end

set(gca, 'XTick', xPositions, 'XTickLabel', labels);
ylabel('Val Corr Difference (Ablation - Baseline)');
title('Ablation Impact on Correlation');
grid on;
yline(0, '--k', 'LineWidth', 1);
set(gca, 'FontSize', 10);

% 标注
for c = 1:nComp
    txt = sprintf('%+.4f', deltaCorr(c));
    yOffset = 0.003;
    if deltaCorr(c) < 0
        yOffset = -0.01;
    end
    text(xPositions(c), deltaCorr(c) + yOffset, txt, ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end

if isSaveFig
    saveas(gcf, fullfile(savePath, 'FigA4_Ablation_Pairwise.png'));
    saveas(gcf, fullfile(savePath, 'FigA4_Ablation_Pairwise.fig'));
end

fprintf('[plotAblationResults] Generated 4 ablation figures.\n');
end
