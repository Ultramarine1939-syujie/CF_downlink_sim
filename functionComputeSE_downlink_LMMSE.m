% 用于计算L-MMSE的函数
function SE = functionComputeSE_downlink_LMMSE(Hhat,H,D,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,rho_dist)

eyeN = eye(N);
prelogFactor = (1-tau_p/tau_c);

signal = zeros(K,1);
interf = zeros(K,1);
scaling = zeros(L,K);

for n=1:nbrOfRealizations
    for l=1:L
        
        servedUEs = find(D(l,:)==1);
        if isempty(servedUEs); continue; end
        Hhat_all = reshape(Hhat((l-1)*N+1:l*N,n,:),[N K]);
        
        V = p*((p*(Hhat_all*Hhat_all')+p*sum(C(:,:,l,:),4)+eyeN)\Hhat_all(:,servedUEs));
        
        scaling(l,servedUEs) = scaling(l,servedUEs) + sum(abs(V).^2,1)/nbrOfRealizations;
    end
end

for n=1:nbrOfRealizations
    
    interf_n = zeros(K,K);
    
    for l=1:L
        
        servedUEs = find(D(l,:)==1);
        if isempty(servedUEs); continue; end
        Hallj = reshape(H((l-1)*N+1:l*N,n,:),[N K]);
        Hhat_all = reshape(Hhat((l-1)*N+1:l*N,n,:),[N K]);
        
        V = p*((p*(Hhat_all*Hhat_all')+p*sum(C(:,:,l,:),4)+eyeN)\Hhat_all(:,servedUEs));
        
        for i=1:length(servedUEs)
            k = servedUEs(i);
            w = V(:,i)*sqrt(rho_dist(l,k)/scaling(l,k));
            
            signal(k) = signal(k) + (Hallj(:,k)'*w)/nbrOfRealizations;
            interf_n(:,k) = interf_n(:,k) + Hallj'*w;
        end
    end
    
    interf = interf + sum(abs(interf_n).^2,2)/nbrOfRealizations;
end

SINR = abs(signal).^2 ./ (interf - abs(signal).^2 + 1);
SE = prelogFactor*real(log2(1+SINR));

end
