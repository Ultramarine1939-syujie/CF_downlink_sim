function [rho, info] = computeRhoDistributedWMMSE(D, gainOverNoise, Pt, L, K, maxRounds, damping)
%COMPUTERHODISTRIBUTEDWMMSE Fixed-round distributed WMMSE-style PA.
%
% This is a distributed baseline for downlink experiments. Each AP updates
% only its own power row using local large-scale gains and a fixed number of
% UE-side scalar aggregation rounds. It is not a centralized WMMSE solver.

    if nargin < 6 || isempty(maxRounds)
        maxRounds = 5;
    end
    if nargin < 7 || isempty(damping)
        damping = 0.6;
    end

    D = double(D > 0.5);
    gain = max(gainOverNoise, 0) .* D;
    rho = computeRhoEPA(D, Pt, L, K);

    for roundIdx = 1:maxRounds
        receivedPower = sum(gain .* rho, 1);  % UE-side scalar aggregation
        rhoNext = zeros(L, K);

        for l = 1:L
            served = D(l, :) > 0;
            if ~any(served); continue; end

            selfPower = gain(l, :) .* rho(l, :);
            interferenceProxy = max(receivedPower - selfPower, 0);
            localScore = gain(l, :) ./ (1 + interferenceProxy);
            localScore(~served) = 0;

            if sum(localScore) <= eps
                localScore = D(l, :);
            end
            rhoNext(l, :) = Pt * localScore / max(sum(localScore), eps);
        end

        rho = damping * rhoNext + (1 - damping) * rho;
        rho = enforcePerAPPower(rho, D, Pt);
    end

    info = struct();
    info.rounds = maxRounds;
    info.messageBytes = double(maxRounds * K * 8);
end

function rho = enforcePerAPPower(rho, D, Pt)
    rho = max(rho, 0) .* D;
    rowPower = sum(rho, 2);
    over = rowPower > Pt;
    if any(over)
        rho(over, :) = rho(over, :) .* (Pt ./ max(rowPower(over), eps));
    end
end
