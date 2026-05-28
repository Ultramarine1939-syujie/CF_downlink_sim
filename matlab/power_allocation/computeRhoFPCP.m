function rho_fpcp = computeRhoFPCP(D, gainOverNoise, Pt, L, K, alpha)
%computeRhoFPCP Fractional power control policy for downlink AP power split.
%   Each AP splits its power budget among its served UEs using a fractional
%   large-scale fading weight. alpha=0 gives EPA. With the convention below,
%   positive alpha favors weaker served links, while negative alpha favors
%   stronger served links. The default is tuned for downlink sum-rate
%   comparisons in this project.
%
%   rho(l,k) = Pt * w(l,k) / sum_i w(l,i), D(l,k)=1
%   w(l,k)   = gainOverNoise(l,k)^(-alpha)

    if nargin < 6 || isempty(alpha)
        alpha = -1.0;
    end

    rho_fpcp = zeros(L, K);
    safeGain = max(gainOverNoise, eps);

    for l = 1:L
        servedUEs = find(D(l, :) == 1);
        if isempty(servedUEs)
            continue;
        end

        weights = safeGain(l, servedUEs) .^ (-alpha);
        weightSum = sum(weights);
        if weightSum <= 0 || ~isfinite(weightSum)
            rho_fpcp(l, servedUEs) = Pt / numel(servedUEs);
        else
            rho_fpcp(l, servedUEs) = Pt * weights / weightSum;
        end
    end
end
