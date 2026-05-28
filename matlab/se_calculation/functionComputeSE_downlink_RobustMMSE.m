function SE = functionComputeSE_downlink_RobustMMSE(Hhat, H, V, scaling, D, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho_dist, sigma_e, Pt, nIter)
% functionComputeSE_downlink_RobustMMSE 计算鲁棒MMSE预编码下的下行链路频谱效率
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
%   sigma_e, Pt, nIter  - 鲁棒MMSE参数（本函数中未使用，保留接口兼容性）
%
% 输出参数:
%   SE                  - 用户频谱效率 (K x 1)

prelogFactor = (1 - tau_p / tau_c);

% 预处理功率缩放因子
powerScale = zeros(L, K);
validMask = (D == 1) & (scaling > 0) & (rho_dist > 0);
powerScale(validMask) = sqrt(rho_dist(validMask) ./ scaling(validMask));

powerScaleRows = kron(powerScale, ones(N, 1)); % (L*N) x K
W = V .* reshape(powerScaleRows, L*N, K, 1);   % (L*N) x K x nbrOfRealizations

% ---- Signal 计算 (向量化 realization 维度) ----
H_all = permute(H, [1 3 2]);          % (L*N) x K x nbrOfRealizations
H_conj = permute(conj(H_all), [2 1 3]); % K x (L*N) x nbrOfRealizations
G = pagemtimes(H_conj, W);            % K x K x nbrOfRealizations

G_vec = reshape(G, K*K, nbrOfRealizations);
diagIdx = 1:(K+1):(K*K);
signal = mean(G_vec(diagIdx, :), 2);  % K x 1

% ---- Interference 计算 (向量化 realization 维度) ----
interf = mean(sum(abs(G).^2, 2), 3);  % K x 1
interf = reshape(interf, [K 1]);

% ---- SINR ----
SINR = abs(signal).^2 ./ (interf - abs(signal).^2 + 1);
SE = prelogFactor * real(log2(1 + SINR));
end