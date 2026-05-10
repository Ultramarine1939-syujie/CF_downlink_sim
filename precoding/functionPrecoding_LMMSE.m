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
    Hhat_sq = sum(abs(Hhat).^2, 1);
    c_local = reshape(C(1, 1, :, :), L, K);
    denom = p * Hhat_sq + reshape(p * c_local, 1, K, 1) + 1;
    V_all = p ./ denom .* Hhat;
    D_expanded = reshape(D, L, K, 1);
    V_all = V_all .* D_expanded;
    V = reshape(V_all, L, K, nbrOfRealizations);
    V = permute(V, [1 3 2]);
    scaling = mean(abs(V_all).^2, 2);
else
    % [分布式改造] N>1 时，仅使用本AP自身的协方差矩阵，而非所有AP之和
    % 原集中式: cSum = sum(C, 4);  % N x N x L（所有AP协方差堆叠）
    % 分布式:    每个AP只用自身对应的 C(:, :, l, :)
    for n = 1:nbrOfRealizations
        for l = 1:L
            servedMask = (D(l, :) == 1);
            if ~any(servedMask); continue; end

            rows = (l-1)*N+1:l*N;
            Hhat_all = reshape(Hhat(rows, n, :), [N, K]);

            % 分布式L-MMSE: 仅用第l个AP自身的协方差，而非所有AP之和
            cSum_l = sum(C(:, :, l, :), 4);  % N x N，第l个AP的协方差（沿用户维度求和）
            A = p * (Hhat_all * Hhat_all') + p * cSum_l + eyeN;
            V_full = p * (A \ Hhat_all);
            V_full(:, ~servedMask) = 0;

            V(rows, :, n) = V_full;
            scaling(l, :) = scaling(l, :) + sum(abs(V_full).^2, 1) / nbrOfRealizations;
        end
    end
end

end