function [d_opt, iterUsed, bestFitness] = functionOptimize_d_PSO(Hhat, H, D, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, Pt)
% functionOptimize_d_PSO 使用粒子群优化 (PSO) 算法优化每个用户的预编码缩放系数 d_k
% 以最大化系统的 ESR (Ergodic Sum Rate)
%
% 输入参数:
%   Hhat - 信道估计 (L*N x K x nbrOfRealizations)
%   H - 真实信道 (L*N x K x nbrOfRealizations)
%   D - DCC 关联矩阵 (L x K)
%   C - 协方差矩阵 (用于 L-MMSE 接口一致性，当前忽略)
%   tau_c, tau_p - 系统参数
%   nbrOfRealizations - 信道实现数量
%   N, K, L - 系统维度
%   p - 功率参数 (用于 L-MMSE 接口一致性，当前忽略)
%   Pt - 总发射功率
%
% 输出:
%   d_opt - 最优功率分配向量 (K x 1)
%   iterUsed - 实际使用的迭代次数
%   bestFitness - 最优适应度值 (ESR)

% 1. 参数设置
numParticles = 30;    % 粒子数 (要求 20~30)
numIterations = 50;   % 迭代次数 (要求 30~50)

w_inertia = 0.7;      % 惯性权重 (要求 0.7)
c1 = 1.5;             % 个体学习因子 (要求 1.5)
c2 = 1.5;             % 社会学习因子 (要求 1.5)

% 2. PSO 初始化
d = zeros(K, numParticles);
v = zeros(K, numParticles);

% [已向量化] 预计算与 d 无关的常量，供 fitnessFunction 重复调用
pre = struct();
pre.nbrOfRealizations = nbrOfRealizations;
pre.N = N;
pre.K = K;
pre.L = L;
pre.prelogFactor = (1 - tau_p / tau_c);
pre.powerPerAP = Pt / L;
pre.Hhat_all = permute(Hhat, [1 3 2]); % (L*N) x K x nbrOfRealizations
pre.H_all = permute(H, [1 3 2]);       % (L*N) x K x nbrOfRealizations
pre.D_rows = kron(D, ones(N, 1));      % (L*N) x K

Hhat_abs2 = sum(abs(reshape(pre.Hhat_all, [N, L, K, nbrOfRealizations])).^2, 1);
pre.E_Hhat2 = squeeze(mean(Hhat_abs2, 4)); % L x K

% 随机初始化 (满足约束：d_k >= 0, sum(d_k^2) <= Pt)
for i = 1:numParticles
    tmp_d = rand(K, 1);
    current_Pt = Pt * rand();
    d(:, i) = tmp_d * sqrt(current_Pt / sum(tmp_d.^2));
    v(:, i) = 0.05 * randn(K, 1);
end

pBest = d;
pBestFitness = zeros(1, numParticles);
for i = 1:numParticles
    pBestFitness(i) = -fitnessFunction_PSO(d(:, i), pre);
end

[gBestFitness, gIdx] = min(pBestFitness);
gBest = pBest(:, gIdx);

% 3. PSO 迭代
convergenceThreshold = 1e-6;  % 收敛阈值
maxStableCount = 10;          % 连续10次迭代无显著提升则收敛
stableCount = 0;
prevBestFitness = gBestFitness;

fprintf('{      PSO Iter: ');

for iter = 1:numIterations
    v = w_inertia * v + ...
        c1 * rand(K, numParticles) .* (pBest - d) + ...
        c2 * rand(K, numParticles) .* (gBest - d);
    
    d = d + v;
    d(d < 0) = 0;
    
    sum_d2 = sum(d.^2, 1);
    violation = sum_d2 > Pt;
    if any(violation)
        d(:, violation) = d(:, violation) .* sqrt(Pt ./ sum_d2(violation));
    end
    
    for i = 1:numParticles
        currentFitness = -fitnessFunction_PSO(d(:, i), pre);
        
        if currentFitness < pBestFitness(i)
            pBest(:, i) = d(:, i);
            pBestFitness(i) = currentFitness;
        end
    end
    
    [iterBestFitness, iterIdx] = min(pBestFitness);
    if iterBestFitness < gBestFitness
        gBestFitness = iterBestFitness;
        gBest = pBest(:, iterIdx);
    end
    
    % 每10次迭代打印当前最佳适应度
    if mod(iter, 10) == 0 || iter == 1
        fprintf('%d : %.4f ', iter, -gBestFitness);
    end
    
    % 检查收敛
    if abs(prevBestFitness - gBestFitness) < convergenceThreshold
        stableCount = stableCount + 1;
    else
        stableCount = 0;
    end
    prevBestFitness = gBestFitness;
    
    % 提前终止条件
    if stableCount >= maxStableCount
        iterUsed = iter;
        break;
    end
end

fprintf('}\n');

% 如果达到最大迭代次数
if ~exist('iterUsed', 'var')
    iterUsed = numIterations;
end

bestFitness = -gBestFitness;
d_opt = gBest;

end

%% fitness function
function [fitness] = fitnessFunction_PSO(d_vec, varargin)
% FITNESSFUNCTION_PSO 计算给定功率分配向量 d_vec 时的 ESR (Ergodic Sum Rate)
% 作为 PSO 的适应度函数。

if (nargin == 2) && isstruct(varargin{1})
    pre = varargin{1};
    Hhat_all = pre.Hhat_all;
    H_all = pre.H_all;
    D_rows = pre.D_rows;
    E_Hhat2 = pre.E_Hhat2;
    prelogFactor = pre.prelogFactor;
    powerPerAP = pre.powerPerAP;
    nbrOfRealizations = pre.nbrOfRealizations;
    N = pre.N;
    K = pre.K;
    L = pre.L;
else
    Hhat = varargin{1};
    H = varargin{2};
    D = varargin{3};
    tau_c = varargin{4};
    tau_p = varargin{5};
    nbrOfRealizations = varargin{6};
    N = varargin{7};
    K = varargin{8};
    L = varargin{9};
    Pt = varargin{10};

    prelogFactor = (1 - tau_p / tau_c);
    powerPerAP = Pt / L;
    Hhat_all = permute(Hhat, [1 3 2]); % (L*N) x K x nbrOfRealizations
    H_all = permute(H, [1 3 2]);       % (L*N) x K x nbrOfRealizations
    D_rows = kron(D, ones(N, 1));      % (L*N) x K

    Hhat_abs2 = sum(abs(reshape(Hhat_all, [N, L, K, nbrOfRealizations])).^2, 1);
    E_Hhat2 = squeeze(mean(Hhat_abs2, 4)); % L x K
end

% 1. [已向量化] 归一化系数 scalingFactor
d2 = d_vec.^2;
D_ap = D_rows(1:N:end, :); % L x K
apPower = (E_Hhat2 .* D_ap) * d2; % L x 1
scalingFactor = zeros(L, 1);
nonzeroAP = apPower > 0;
scalingFactor(nonzeroAP) = sqrt(powerPerAP ./ apPower(nonzeroAP));
scalingRows = kron(scalingFactor, ones(N, 1)); % (L*N) x 1

% 2. [已向量化] KxK 有效信道矩阵 G(:,:,n) 的 Monte Carlo 平均
W = Hhat_all .* reshape(d_vec.', 1, K, 1);
W = W .* reshape(D_rows, L*N, K, 1);
W = W .* reshape(scalingRows, L*N, 1, 1);

G = pagemtimes(permute(conj(H_all), [2 1 3]), W); % K x K x nbrOfRealizations

G_vec = reshape(G, K*K, nbrOfRealizations);
diagIdx = 1:(K+1):(K*K);
signal = mean(G_vec(diagIdx, :), 2);
interf = mean(sum(abs(G).^2, 2), 3);
interf = reshape(interf, [K 1]);

signal_power = abs(signal).^2;
sinr = signal_power ./ (interf - signal_power + 1);
SE = prelogFactor * log2(1 + max(real(sinr), 0));
fitness = mean(SE);

end