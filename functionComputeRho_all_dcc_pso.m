function [rho_all, rho_dcc, rho_pso] = functionComputeRho_all_dcc_pso(gainOverNoise, D_dcc, Pt, d_opt, scaling_pso, D_pso)
% functionComputeRho_all_dcc_pso 计算三种功率分配方案
%
% 输入参数:
%   gainOverNoise - 大尺度增益 (L x K)
%   D_dcc - DCC接入矩阵 (L x K)
%   Pt - 总发射功率
%   d_opt - PSO优化得到的功率分配向量 (K x 1)
%   scaling_pso - 预编码scaling系数 (L x K)
%   D_pso - PSO功率分配的接入矩阵 (L x K)
%
% 输出:
%   rho_all - 基于大尺度增益的功率分配 (L x K)
%   rho_dcc - DCC约束下的功率分配 (L x K)
%   rho_pso - PSO优化的功率分配 (L x K)

L = size(gainOverNoise, 1);
K = size(gainOverNoise, 2);

sqrtG = sqrt(gainOverNoise);
normFactor = sum(sqrtG, 2);
rho_all = zeros(L, K);
activeAP = normFactor > 0;
rho_all(activeAP, :) = Pt * (sqrtG(activeAP, :) ./ normFactor(activeAP));

if nargin >= 2 && ~isempty(D_dcc)
    sqrtG_dcc = sqrtG .* D_dcc;
    normFactor_dcc = sum(sqrtG_dcc, 2);
    rho_dcc = zeros(L, K);
    activeAP_dcc = normFactor_dcc > 0;
    rho_dcc(activeAP_dcc, :) = Pt * (sqrtG_dcc(activeAP_dcc, :) ./ normFactor_dcc(activeAP_dcc));
else
    rho_dcc = [];
end

if nargin >= 4 && ~isempty(d_opt)
    if nargin < 6 || isempty(D_pso)
        D_pso = ones(L, K);
    end
    rho_pso = functionPowerControl_from_d_equalPerAP(d_opt, scaling_pso, D_pso, Pt);
else
    rho_pso = [];
end
end

%% rho_pso 计算子函数
function rho_dist = functionPowerControl_from_d_equalPerAP(d_opt, scaling, D, Pt)
d2 = d_opt(:).^2; % K x 1
powerPerAP = Pt / size(D, 1);

apPower = (scaling .* D) * d2; % L x 1
eta = zeros(size(apPower));
nonzeroAP = apPower > 0;
eta(nonzeroAP) = sqrt(powerPerAP ./ apPower(nonzeroAP));

rho_dist = (eta.^2) .* (scaling .* D) .* (d2.');
end