function SE = functionComputeSE_downlink_MR(H, V, scaling, D, tau_c, tau_p, nbrOfRealizations, N, K, L, rho_dist)
% functionComputeSE_downlink_MR 计算MR预编码下的下行链路频谱效率
%
% 输入参数:
%   H                 - 信道估计值 (L*N x nbrOfRealizations x K)
%   V                 - MR预编码方向 (L*N x K x nbrOfRealizations)
%   scaling           - MR功率缩放系数 (L x K)
%   D                 - 接入矩阵 (L x K)
%   tau_c, tau_p      - 相干时间长度, 导频长度
%   nbrOfRealizations - 信道实现数
%   N, K, L           - 天线数, 用户数, AP数
%   rho_dist          - 功率分配系数 (L x K)
%
% 输出参数:
%   SE                - 用户频谱效率 (K x 1)

prelogFactor = (1 - tau_p / tau_c);

powerScale = zeros(L, K);
validMask = (D == 1) & (scaling > 0) & (rho_dist > 0);
powerScale(validMask) = sqrt(rho_dist(validMask) ./ scaling(validMask));

powerScaleRows = kron(powerScale, ones(N, 1)); % (L*N) x K
W = V .* reshape(powerScaleRows, L*N, K, 1);

H_all = permute(H, [1 3 2]); % (L*N) x K x nbrOfRealizations
G = pagemtimes(permute(conj(H_all), [2 1 3]), W); % K x K x nbrOfRealizations

G_vec = reshape(G, K*K, nbrOfRealizations);
diagIdx = 1:(K+1):(K*K);
signal = mean(G_vec(diagIdx, :), 2);

interf = mean(sum(abs(G).^2, 2), 3);
interf = reshape(interf, [K 1]);

SINR = abs(signal).^2 ./ (interf - abs(signal).^2 + 1);
SE = prelogFactor * real(log2(1 + SINR));
end