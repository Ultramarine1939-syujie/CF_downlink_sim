function [V, scaling] = functionPrecoding_MR(Hhat, nbrOfRealizations, N, K, L)
% functionPrecoding_MR 计算MR(Maximum Ratio)预编码的方向V和功率缩放scaling
%
% 输入参数:
%   Hhat              - 信道估计 (L*N x nbrOfRealizations x K)
%   nbrOfRealizations - 信道实现数
%   N                 - 每AP天线数
%   K                 - UE用户数
%   L                 - AP数量
%
% 输出参数:
%   V       - 预编码方向 (L*N x K x nbrOfRealizations)
%   scaling - 功率缩放系数 (L x K)

V = permute(Hhat, [1 3 2]); % (L*N) x K x nbrOfRealizations

Hhat4 = reshape(Hhat, [N, L, nbrOfRealizations, K]); % N x L x nbrOfRealizations x K
scaling = squeeze(mean(sum(abs(Hhat4).^2, 1), 3)); % L x K
end