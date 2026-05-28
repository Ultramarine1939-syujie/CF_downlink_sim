function [V, scaling] = functionPrecoding_LMMSE_global(Hhat, D, C, nbrOfRealizations, N, K, L, p)
% FUNCTIONPRECODING_LMMSE_GLOBAL 集中式 L-MMSE 预编码 (消融实验用)
%
% 与 functionPrecoding_LMMSE.m 的唯一区别:
%   使用所有 AP 的协方差矩阵之和 (cSum = sum(C,4))，而非仅本 AP 自身的协方差。
%   这是原始的集中式 L-MMSE 方案，用于消融实验维度 6:
%   验证分布式改造（仅用本地协方差）带来的信息损失。
%
% 输入/输出接口与 functionPrecoding_LMMSE.m 完全一致。

eyeN = eye(N);
scaling = zeros(L, K);
V = complex(zeros(L*N, K, nbrOfRealizations), 0);

if N == 1
    % [集中式] N==1 时，使用所有 AP 协方差之和
    cSum = squeeze(sum(C(1,1,:,:), 4));  % L x 1（所有 AP 协方差之和）
    for l = 1:L
        rowIdx = (l-1)*N + 1;
        hhat_ln = squeeze(Hhat(rowIdx, :, :)); % nbrOfRealizations x K

        denom = p * sum(abs(hhat_ln).^2, 2) + p * cSum(l) + 1; % nbrOfRealizations x 1
        v_ln = (p ./ denom) .* hhat_ln; % nbrOfRealizations x K
        v_ln(:, D(l, :) == 0) = 0;

        V(rowIdx, :, :) = permute(v_ln, [3 2 1]); % 1 x K x nbrOfRealizations
        scaling(l, :) = mean(abs(v_ln).^2, 1);
    end
else
    % [集中式] N>1 时，使用所有 AP 协方差矩阵之和
    cSum = sum(C, 4);  % N x N（所有 AP 协方差堆叠后求和）
    for n = 1:nbrOfRealizations
        for l = 1:L
            servedMask = (D(l, :) == 1);
            if ~any(servedMask); continue; end

            rows = (l-1)*N+1:l*N;
            Hhat_all = reshape(Hhat(rows, n, :), [N, K]);

            A = p * (Hhat_all * Hhat_all') + p * cSum + eyeN;
            V_full = p * (A \ Hhat_all);
            V_full(:, ~servedMask) = 0;

            V(rows, :, n) = V_full;
            scaling(l, :) = scaling(l, :) + sum(abs(V_full).^2, 1) / nbrOfRealizations;
        end
    end
end

end
