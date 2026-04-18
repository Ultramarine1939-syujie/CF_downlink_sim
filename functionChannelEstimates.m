% Function to generate channel realizations and estimates
% 用于生成信道估计的函数
function [Hhat,H,B,C] = functionChannelEstimates(R,nbrOfRealizations,L,K,N,tau_p,pilotIndex,p)
H = (randn(L*N,nbrOfRealizations,K)+1i*randn(L*N,nbrOfRealizations,K));
for l = 1:L
    for k = 1:K
        Rsqrt = sqrtm(R(:,:,l,k));
        H((l-1)*N+1:l*N,:,k) = sqrt(0.5)*Rsqrt*H((l-1)*N+1:l*N,:,k);
    end
end
eyeN = eye(N);
Np = sqrt(0.5)*(randn(N,nbrOfRealizations,L,tau_p) + 1i*randn(N,nbrOfRealizations,L,tau_p));
Hhat = zeros(L*N,nbrOfRealizations,K); 
if nargout>2
    B = zeros(size(R));
end
if nargout>3
    C = zeros(size(R));
end
for l = 1:L
    for t = 1:tau_p
        yp = sqrt(p)*tau_p*sum(H((l-1)*N+1:l*N,:,t==pilotIndex),3) + sqrt(tau_p)*Np(:,:,l,t);
        PsiInv = (p*tau_p*sum(R(:,:,l,t==pilotIndex),4) + eyeN);
        for k = find(t==pilotIndex)'
            RPsi = R(:,:,l,k) / PsiInv;
            Hhat((l-1)*N+1:l*N,:,k) = sqrt(p)*RPsi*yp;
            if nargout>2
                B(:,:,l,k) = p*tau_p*RPsi*R(:,:,l,k);
            end
            if nargout>3
                C(:,:,l,k) = R(:,:,l,k) - B(:,:,l,k);
            end
        end
    end
end
end
