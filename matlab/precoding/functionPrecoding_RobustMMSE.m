function [V, scaling] = functionPrecoding_RobustMMSE(Hhat, D, nbrOfRealizations, N, K, L, Pt, sigma_e, nIter)
% FUNCTIONPRECODING_ROBUSTMMSE 实现鲁棒 MMSE 预编码向量的计算
%
% 输入:
%   Hhat - 信道估计 (L*N x nbrOfRealizations x K)
%   D - DCC 关联矩阵 (L x K)
%   nbrOfRealizations - 信道实现数量
%   N, K, L - 系统维度
%   Pt - 总发射功率
%   sigma_e - CSI 误差
%   nIter - 迭代次数
%
% 输出:
%   V - 预编码向量 (L*N x K x nbrOfRealizations)，未服务用户列为 0
%   scaling - 归一化缩放因子 (L x K)

eyeN = eye(N);
scaling = zeros(L, K);
V = complex(zeros(L*N, K, nbrOfRealizations), 0);
tau = sqrt(1 + sigma_e^2);

for n = 1:nbrOfRealizations
    for l = 1:L
        servedMask = (D(l, :) == 1);
        if ~any(servedMask); continue; end
        
        rows = (l-1)*N+1:l*N;
        Hhat_all = reshape(Hhat(rows, n, :), [N, K]);
        Hhat_local = Hhat_all(:, servedMask);
        
        % 鲁棒 MMSE 预编码迭代逻辑
        P_bar = (Hhat_local * Hhat_local' + (K / Pt) * eyeN) \ Hhat_local;
        
        for iter = 1:nIter
            f = sqrt(Pt / real(trace(P_bar * P_bar')));
            Theta = sigma_e^2 * (Hhat_local * Hhat_local' / K);
            P = f * tau * P_bar;
            
            lambda = trace(eyeN) / (f^2 * Pt) - trace(P * P' * Theta) / Pt;
            lambda = max(lambda, 1e-6);
            
            P_bar = (Hhat_local * Hhat_local' + (1 + f^2) * Theta + lambda * f^2 * tau * eyeN) \ Hhat_local;
        end
        
        f = sqrt(Pt / real(trace(P_bar * P_bar')));
        V_tmp = f * tau * P_bar;

        V_full = zeros(N, K);
        V_full(:, servedMask) = V_tmp;
        V(rows, :, n) = V_full;
        scaling(l, :) = scaling(l, :) + sum(abs(V_full).^2, 1) / nbrOfRealizations;
    end
end
end