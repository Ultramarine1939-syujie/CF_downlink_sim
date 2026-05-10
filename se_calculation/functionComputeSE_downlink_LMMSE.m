function SE = functionComputeSE_downlink_LMMSE(Hhat, H, V, scaling, D, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho_dist)
% functionComputeSE_downlink_LMMSE 计算L-MMSE预编码下的下行链路频谱效率
%
% 输入参数:
%   Hhat                - 信道估计 (L*N x nbrOfRealizations x K)
%   H                   - 实际信道，含CSI误差 (L*N x nbrOfRealizations x K)
%   V                   - 预编码向量 (L*N x K x nbrOfRealizations) [已由预编码器计算]
%   scaling             - 功率缩放因子 (L x K) [已由预编码器计算]
%   D                   - 接入矩阵 (L x K)
%   tau_c, tau_p        - 相干块长度, 导频长度
%   nbrOfRealizations   - 信道实现数
%   N, K, L             - 天线数, 用户数, AP数
%   p                   - 导频功率
%   rho_dist            - 功率分配系数 (L x K)
%
% 输出参数:
%   SE                  - 用户频谱效率 (K x 1)

prelogFactor = (1 - tau_p / tau_c);

powerScale = zeros(L, K);
sc2d = reshape(scaling, L, K);
rho2d = reshape(rho_dist, L, K);
validMask = (D == 1) & (sc2d > 0) & (rho2d > 0);
powerScale(validMask) = sqrt(rho2d(validMask) ./ sc2d(validMask));

if N == 1
    powerScaleExp = reshape(powerScale, L, K, 1);
    W = V .* powerScaleExp;

    H_all = permute(H, [1 3 2]);
    H_conj = permute(conj(H_all), [2 1 3]);

    G = pagemtimes(H_conj, W);

    G_vec = reshape(G, K*K, nbrOfRealizations);
    diagIdx = 1:(K+1):(K*K);
    signal = mean(G_vec(diagIdx, :), 2);

    interf = mean(sum(abs(G).^2, 2), 3);
    interf = reshape(interf, [K 1]);
else
    powerScaleRows = reshape(powerScale, L, 1, K);
    powerScaleRows = repmat(powerScaleRows, [1, N, 1]);
    powerScaleRows = reshape(powerScaleRows, L*N, K);
    W = V .* reshape(powerScaleRows, L*N, K, 1);

    H_all = permute(H, [1 3 2]);
    H_conj = permute(conj(H_all), [2 1 3]);

    G = pagemtimes(H_conj, W);

    G_vec = reshape(G, K*K, nbrOfRealizations);
    diagIdx = 1:(K+1):(K*K);
    signal = mean(G_vec(diagIdx, :), 2);

    interf = mean(sum(abs(G).^2, 2), 3);
    interf = reshape(interf, [K 1]);
end

SINR = abs(signal).^2 ./ (interf - abs(signal).^2 + 1);
SE = prelogFactor * real(log2(1 + SINR));
end