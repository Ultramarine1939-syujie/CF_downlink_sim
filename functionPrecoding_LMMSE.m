function [V, scaling] = functionPrecoding_LMMSE(Hhat, D, C, nbrOfRealizations, N, K, L, p)
% FUNCTIONPRECODING_LMMSE 实现 L-MMSE 预编码向量的计算
%
% 输入:
%   Hhat - 信道估计 (L*N x nbrOfRealizations x K)
%   D - DCC 关联矩阵 (L x K)
%   C - 协方差矩阵 (N x N x L x K)
%   nbrOfRealizations - 信道实现数量
%   N, K, L - 系统维度
%   p - 功率参数
%
% 输出:
%   V - 预编码向量 (L*N x K x nbrOfRealizations)，未服务用户列为 0
%   scaling - 归一化缩放因子 (L x K)

eyeN = eye(N);
scaling = zeros(L, K);
V = complex(zeros(L*N, K, nbrOfRealizations), 0);

if N == 1
    % [已向量化] N==1 时，L-MMSE 可写成逐 realization 的标量分母，避免 nbrOfRealizations 循环
    cSum = squeeze(sum(C(1, 1, :, :), 4)); % L x 1
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
    % N>1 的通用实现：保留 (l,n) 外层循环，但消除对 K/servedUEs 的内层循环与 cell 开销
    cSum = sum(C, 4); % N x N x L
    for n = 1:nbrOfRealizations
        for l = 1:L
            servedMask = (D(l, :) == 1);
            if ~any(servedMask); continue; end

            rows = (l-1)*N+1:l*N;
            Hhat_all = reshape(Hhat(rows, n, :), [N, K]);

            A = p * (Hhat_all * Hhat_all') + p * cSum(:, :, l) + eyeN;
            V_full = p * (A \ Hhat_all);
            V_full(:, ~servedMask) = 0;

            V(rows, :, n) = V_full;
            scaling(l, :) = scaling(l, :) + sum(abs(V_full).^2, 1) / nbrOfRealizations;
        end
    end
end

end