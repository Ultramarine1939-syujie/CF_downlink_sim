function [rho_dist, iter_used] = computeRhoWMMSE(Hhat, D, Pt, N, K, L, nbrOfRealizations, maxIter, tol)
%computeRhoWMMSE Standard WMMSE power allocation with per-AP constraints.
%
% This implementation solves the weighted sum-rate power allocation problem
% on a deterministic equivalent downlink channel. The transmit variable is
% V(l,k)=sqrt(rho(l,k)), with the sparsity pattern imposed by D and one
% power constraint per AP: sum_k |V(l,k)|^2 <= Pt.
%
% The alternating WMMSE loop updates:
%   1) scalar MMSE receivers u_k
%   2) MSE weights w_k = 1/e_k
%   3) transmit amplitudes V through the WMMSE quadratic subproblem
%
% Hhat has dimensions (L*N) x nbrOfRealizations x K. Since this power
% allocator must stay independent of the selected precoder, it builds a
% deterministic equivalent channel H_eff(k,l) from the RMS local channel
% strength between AP l and UE k. The WMMSE updates themselves are the
% standard downlink WMMSE updates for that equivalent MISO BC.

    if nargin < 8 || isempty(maxIter)
        maxIter = 30;
    end
    if nargin < 9 || isempty(tol)
        tol = 1e-4;
    end

    D = double(D > 0.5);
    if isempty(Hhat) || all(D(:) == 0)
        rho_dist = computeRhoEPA(D, Pt, L, K);
        iter_used = 0;
        return;
    end

    H_eff = buildEffectiveChannel(Hhat, N, K, L, nbrOfRealizations);

    % Initialize with large-scale gain based allocation under per-AP power.
    rho0 = computeRhoDist(D, H_eff.', Pt, L, K);
    rho0(~isfinite(rho0)) = 0;
    V = sqrt(max(rho0, 0)) .* D;
    V = projectPerAP(V, D, Pt);

    prevRate = computeEquivalentRate(H_eff, V);
    bestRate = prevRate;
    bestV = V;
    reg = 1e-6 * max(real(trace(H_eff' * H_eff)) / max(L, 1), 1);
    maxDualIter = 25;

    for iter = 1:maxIter
        V_old = V;

        HV = H_eff * V;                         % K x K, user rows x streams
        totalRxPower = sum(abs(HV).^2, 2) + 1;  % unit noise variance
        desired = diag(HV);

        u = desired ./ max(totalRxPower, eps);
        mse = 1 - 2 * real(conj(u) .* desired) + abs(u).^2 .* totalRxPower;
        mse = max(real(mse), 1e-12);
        w = 1 ./ mse;

        A = H_eff' * (diag(w .* abs(u).^2) * H_eff);
        A = (A + A') / 2 + reg * eye(L);
        B = H_eff' * diag(w .* conj(u));        % L x K

        V_candidate = solveTransmitUpdate(A, B, D, Pt, maxDualIter, reg);
        [V, rate] = acceptMonotoneUpdate(H_eff, V_old, V_candidate, D, Pt, prevRate);

        sinr = computeEquivalentSINR(H_eff, V);

        powerChange = norm(V(:) - V_old(:)) / max(norm(V_old(:)), eps);
        rateChange = abs(rate - prevRate) / max(abs(prevRate), 1);

        if rate > bestRate
            bestRate = rate;
            bestV = V;
        end

        if mod(iter, 5) == 0 || iter == 1
            fprintf('    [WMMSE] Iter %2d: WSR=%.4f, PowerChange=%.6f, SINR=[%.2f, %.2f, %.2f]\n', ...
                iter, rate, powerChange, safeSINR(sinr, 1), safeSINR(sinr, 2), safeSINR(sinr, 3));
        end

        if iter > 1 && (powerChange < tol || rateChange < tol)
            iter_used = iter;
            rho_dist = finalizeRho(bestV, D, Pt);
            return;
        end
        prevRate = rate;
    end

    iter_used = maxIter;
    rho_dist = finalizeRho(bestV, D, Pt);
end

function H_eff = buildEffectiveChannel(Hhat, N, K, L, nbrOfRealizations)
    H_eff = zeros(K, L);
    for l = 1:L
        rows = (l-1)*N + (1:N);
        for k = 1:K
            H_lk = Hhat(rows, 1:nbrOfRealizations, k);
            perRealizationPower = sum(abs(H_lk).^2, 1);
            H_eff(k, l) = sqrt(max(mean(perRealizationPower), 0));
        end
    end
end

function V = solveTransmitUpdate(A, B, D, Pt, maxDualIter, reg)
    [L, K] = size(D);
    lambda = zeros(L, 1);
    V = zeros(L, K);

    diagScale = max(real(trace(A)) / max(L, 1), 1e-6);
    step0 = diagScale / max(Pt, 1);

    for dualIter = 1:maxDualIter
        V = solveForLambda(A, B, D, lambda, reg);
        rowPower = sum(abs(V).^2, 2);
        violation = rowPower - Pt;
        if max(violation) <= max(1e-6, 1e-5 * Pt)
            break;
        end
        step = step0 / sqrt(dualIter);
        lambda = max(0, lambda + step * violation);
    end

    V = projectPerAP(V, D, Pt);
end

function V = solveForLambda(A, B, D, lambda, reg)
    [L, K] = size(D);
    V = zeros(L, K);
    A_lam = A + diag(lambda) + reg * eye(L);

    if all(D(:) > 0.5)
        V = A_lam \ B;
        return;
    end

    for k = 1:K
        served = find(D(:, k) > 0.5);
        if isempty(served)
            continue;
        end
        V(served, k) = A_lam(served, served) \ B(served, k);
    end
end

function [V_next, rateNext] = acceptMonotoneUpdate(H_eff, V_old, V_candidate, D, Pt, prevRate)
    alpha = 1.0;
    V_next = V_old;
    rateNext = prevRate;
    while alpha >= 1e-3
        V_trial = projectPerAP((1 - alpha) * V_old + alpha * V_candidate, D, Pt);
        trialRate = computeEquivalentRate(H_eff, V_trial);
        if trialRate >= prevRate - 1e-9
            V_next = V_trial;
            rateNext = trialRate;
            return;
        end
        alpha = alpha / 2;
    end
end

function rate = computeEquivalentRate(H_eff, V)
    sinr = computeEquivalentSINR(H_eff, V);
    rate = sum(log2(1 + sinr));
end

function sinr = computeEquivalentSINR(H_eff, V)
    HV = H_eff * V;
    desiredPow = abs(diag(HV)).^2;
    interfPow = max(sum(abs(HV).^2, 2) - desiredPow, 0);
    sinr = desiredPow ./ max(interfPow + 1, eps);
end

function V = projectPerAP(V, D, Pt)
    V = V .* D;
    rowPower = sum(abs(V).^2, 2);
    over = rowPower > Pt;
    if any(over)
        scale = sqrt(Pt ./ max(rowPower(over), eps));
        V(over, :) = V(over, :) .* scale;
    end
end

function rho = finalizeRho(V, D, Pt)
    rho = abs(V).^2 .* D;
    rowPower = sum(rho, 2);
    over = rowPower > Pt * (1 + 1e-8);
    if any(over)
        rho(over, :) = rho(over, :) .* (Pt ./ max(rowPower(over), eps));
    end
    rho(~isfinite(rho)) = 0;
end

function value = safeSINR(sinr, idx)
    if numel(sinr) >= idx
        value = sinr(idx);
    else
        value = 0;
    end
end
