% 计算下行链路等功率分配 (Equal Power Allocation, EPA)
% 每个 AP 将总功率平均分配给其服务的所有用户
%
% 输入:
%   D             - L x K 接入矩阵 (1=服务, 0=不服务)
%   Pt            - 总发射功率 (线性值)
%   L, K          - AP数, UE数
%
% 输出:
%   rho_dist      - L x K 功率分配系数矩阵
%
% 公式: rho(l,k) = Pt / N_served(l)  如果 D(l,k)=1
%              = 0                   如果 D(l,k)=0
%
% 其中 N_served(l) = sum(D(l,:)) 是 AP l 服务的用户数
%
% 作者: WorkBuddy
% 日期: 2026-04-22

function rho_dist = computeRhoEPA(D, Pt, L, K)
    rho_dist = zeros(L, K);
    
    for l = 1:L
        servedUEs = find(D(l, :) == 1);
        n_served = length(servedUEs);
        
        if n_served == 0
            continue;  % 该AP不服务任何用户
        end
        
        % 等功率分配：每个服务的用户获得 Pt / n_served
        rho_dist(l, servedUEs) = Pt / n_served;
    end
end
