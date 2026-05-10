% WMMSE (Weighted Minimum Mean Square Error) 功率分配算法
% 适用于 Cell-Free 大规模 MIMO 下行链路系统
%
% 原理：
%   WMMSE 通过迭代最小化加权和均方误差来优化发射功率
%   对于Cell-Free分布式系统，每个AP独立优化其功率分配
%   每次迭代：
%     1. 基于当前功率分配计算每个AP对各用户的等效SINR
%     2. 计算 MMSE 权重
%     3. 每个AP按局部权重更新功率
%
% 输入:
%   Hhat         - L*N x nbrOfRealizations x K 信道估计矩阵
%   D            - L x K 接入矩阵 (1=服务, 0=不服务)
%   Pt           - 总发射功率 (线性值)
%   N            - 每个 AP 的天线数
%   K            - 用户数
%   L            - AP数
%   nbrOfRealizations - 信道实现数
%   maxIter      - 最大迭代次数
%   tol          - 收敛容差
%
% 输出:
%   rho_dist     - L x K 功率分配系数矩阵
%   iter_used    - 实际使用的迭代次数
%
% 作者: WorkBuddy
% 日期: 2026-04-22

function [rho_dist, iter_used] = computeRhoWMMSE(Hhat, D, Pt, N, K, L, nbrOfRealizations, maxIter, tol)
    % 默认参数
    if nargin < 8
        maxIter = 30;
        tol = 1e-4;
    end
    
    % ========== 步骤1: 预计算信道统计（向量化，无循环）==========
    % Hhat: L*N x nbrOfRealizations x K
    % 变形: N x L x nbrOfRealizations x K，然后对N维度求模方得 L x K x nbrOfRealizations
    Hhat_4d = reshape(Hhat, N, L, nbrOfRealizations, K);     % N x L x R x K
    Hhat_4d = permute(Hhat_4d, [2, 1, 3, 4]);                % L x N x R x K
    g_lk_sq_raw = sum(abs(Hhat_4d).^2, 2);                    % L x 1 x R x K
    if any(size(g_lk_sq_raw) == 1) && ndims(g_lk_sq_raw) > 2
        g_lk_sq = squeeze(g_lk_sq_raw);                       % 去除singleton维度
    else
        g_lk_sq = g_lk_sq_raw;
    end                                                       % L x R x K (或 L x R x K)
    channel_power = mean(g_lk_sq, 2);                         % 可能是 L x 1 x K
    channel_power = reshape(channel_power, [L, K]);            % 强制 L x K 二维

    % ========== 步骤2: 初始化功率分配（向量化）==========
    % D_masked(l,k) = D(l,k) .* sqrt(channel_power(l,k))
    D_sqrt_cp = D .* sqrt(channel_power);                   % L x K
    total_sqrt_power = sum(D_sqrt_cp, 2) + eps;              % L x 1
    % rho_dist(l,k) = Pt * D(l,k) * sqrt(cp(l,k)) / total(l)
    rho_dist = Pt * D_sqrt_cp ./ total_sqrt_power;           % L x K
    % D(l,k)=0 处自动为0（0/eps=0）
    
    % ========== 步骤3: WMMSE 迭代优化 ==========
    
    for iter = 1:maxIter
        rho_old = rho_dist;
        
        % ----- 计算每个用户的等效SINR（完全向量化，无循环）-----
        % g_lk_sq:      L x R x K
        % rho_dist:     L x K
        % g_masked(l,r,k) = D(l,k) * g_lk_sq(l,r,k)
        g_masked = g_lk_sq .* reshape(D, [L, 1, K]);            % L x R x K

        % total_power_per_UE(r,k) = sum_l rho_dist(l,k) * g_masked(l,r,k)
        % 对每个用户 k，将 rho_dist(:,k) 扩展到 R 维，与 g_masked(:,:,k) 逐元素相乘
        weighted = g_masked .* reshape(rho_dist, [L, 1, K]);   % L x R x K，rho_dist(l,k) 广播到 R 维
        total_power_per_UE = sum(weighted, 1);                   % 1 x R x K -> squeeze
        total_power_per_UE = squeeze(total_power_per_UE);        % R x K

        % 信号功率 = sum_l D_rho(l,k) * g_lk_sq(l,r,k) for all (k,r)
        % D_rho: L x K，g_lk_sq: L x R x K
        % 对每个 k: signal(k,r) = D_rho(:,k)' * g_lk_sq(:,r,k)
        D_rho = rho_dist .* D;                                    % 应该是 L x K
        D_rho = reshape(D_rho, [L, K]);                           % 强制二维，防N维异常
        D_rho_t = D_rho.';                                        % K x L
        signal_power_per_UE = zeros(K, nbrOfRealizations);      % K x R
        for k = 1:K
            signal_power_per_UE(k, :) = D_rho_t(k, :) * g_lk_sq(:, :, k);  % 1xL * LxR = 1xR
        end

        % 干扰 = 总功率 - 信号功率
        interference_power_per_UE = total_power_per_UE.' - signal_power_per_UE; % K x R

        % SINR = signal / (interference + 1)，按realization平均
        SINR_k = mean(signal_power_per_UE ./ (interference_power_per_UE + 1), 2); % K x 1

        % ----- WMMSE 权重计算 -----
        % w_k = SINR_k / (1 + SINR_k)
        weights = SINR_k ./ (1 + SINR_k + eps);

        % ----- 更新功率分配（向量化）-----
        % local_weight(l,k) = weights(k) * channel_power(l,k)，D(l,k)=0自动屏蔽
        lw = channel_power .* (weights.');                              % L x K
        lw = lw .* D;                                                   % L x K
        total_lw = sum(lw, 2);                                         % L x 1
        safe_denom = total_lw + eps * (total_lw == 0);                % 避免NaN
        rho_dist_new = Pt * lw ./ safe_denom;                          % L x K
        
        rho_dist = rho_dist_new;
        
        % 检查收敛
        power_change = norm(rho_dist(:) - rho_old(:)) / (norm(rho_old(:)) + eps);
        
        % 每5次迭代输出一次状态
        if mod(iter, 5) == 0 || iter == 1
            fprintf('    [WMMSE] Iter %2d: ESR=%.4f, PowerChange=%.6f, SINR=[%.2f, %.2f, %.2f]\n', ...
                iter, sum(log2(1+SINR_k)), power_change, SINR_k(1), SINR_k(2), SINR_k(3));
        end
        
        if power_change < tol
            iter_used = iter;
            return;
        end
    end
    
    iter_used = maxIter;
end
