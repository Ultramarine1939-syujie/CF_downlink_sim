function [rho, timing, rho_nn] = computeRhoUGNN(Hhat, D, gainOverNoise, Pt, ugnnModelPath, sigma_e, options)
%computeRhoUGNN Teacher-free U-GNN power allocation.
%   U-GNN checkpoints are trained with a differentiable large-scale SE proxy
%   instead of WMMSE labels. Inference reuses the cached GNN runtime because
%   the checkpoint emits the same per-AP share logits.
%
%   The teacher-free proxy can over-reward nearly one-hot AP power splits.
%   U-GNN-v2 checkpoints encode the FPCP safety prior inside the network output
%   through a learned mixture gate. The entropy guard is now optional and should
%   be used only for ablations; by default, inference evaluates the U-GNN output
%   itself and falls back only when the Python output is numerically invalid.
%
%   Optional options fields:
%     guardEnabled  - true to enable entropy guard, false for pure U-GNN
%     minEntropy    - normalized entropy threshold for collapse detection
%     guardAlpha    - FPCP alpha used when the guard triggers

    guardTic = tic;
    if nargin < 7 || isempty(options)
        options = struct();
    end
    guardEnabled = getOption(options, 'guardEnabled', false);
    minEntropy = getOption(options, 'minEntropy', 0.35);
    guardAlpha = getOption(options, 'guardAlpha', -1.0);

    [rho_nn, timing] = computeRhoGNN(Hhat, D, gainOverNoise, Pt, ugnnModelPath, sigma_e);

    [L, K] = size(D);
    entropy = normalizedPowerEntropy(rho_nn, D);
    rowPower = sum(max(rho_nn, 0), 2);
    invalid = any(~isfinite(rho_nn(:))) || any(rowPower(D * ones(K, 1) > 0) <= 0);

    if invalid || (guardEnabled && entropy < minEntropy)
        rho = computeRhoFPCP(D, gainOverNoise, Pt, L, K, guardAlpha);
        timing.guard_triggered = true;
    else
        rho = rho_nn;
        timing.guard_triggered = false;
    end

    timing.guard_entropy = entropy;
    timing.guard_alpha = guardAlpha;
    timing.guard_min_entropy = minEntropy;
    timing.guard_enabled = guardEnabled;
    timing.guard_invalid = invalid;
    timing.guard_sec = toc(guardTic);
    timing.total_sec = timing.total_sec + timing.guard_sec;
end

function value = getOption(options, name, defaultValue)
    value = defaultValue;
    if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
        value = options.(name);
    end
end

function entropy = normalizedPowerEntropy(rho, D)
    D = double(D > 0.5);
    rho = max(double(rho), 0) .* D;
    rowSum = sum(rho, 2);
    served = sum(D, 2);
    active = rowSum > 0 & served > 1;
    if ~any(active)
        entropy = 1;
        return;
    end

    p = rho(active, :) ./ max(rowSum(active), eps);
    h = -sum(p .* log(max(p, eps)), 2);
    entropy = mean(h ./ log(max(served(active), 2)));
end
