% 计算下行功率分配系数 rho_dist
% rho_dist(l,k) = Pt * sqrt(gain(l,k)) / sum(sqrt(gain(l,servedUEs)))
%
% 输入:
%   D             - L x K 接入矩阵 (1=服务, 0=不服务)
%   gainOverNoise - L x K 大尺度增益 (线性值)
%   Pt            - 总发射功率 (线性值)
%   L, K          - AP数, UE数
%
% 输出:
%   rho_dist      - L x K 功率分配系数矩阵
function rho_dist = computeRhoDist(D, gainOverNoise, Pt, L, K)
    rho_dist = zeros(L, K);
    for l = 1:L
        servedUEs = find(D(l, :) == 1);
        if isempty(servedUEs); continue; end
        normFactor = sum(sqrt(gainOverNoise(l, servedUEs)));
        for k = servedUEs
            rho_dist(l, k) = Pt * sqrt(gainOverNoise(l, k)) / normFactor;
        end
    end
end
