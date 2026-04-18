% 用于计算鲁棒MMSE的函数
function SE = functionComputeSE_downlink_RobustMMSE( ...
    Hhat,H,D,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,rho_dist,...
    sigma_e,Pt,nIter)

eyeN = eye(N);
prelogFactor = (1-tau_p/tau_c);

signal = zeros(K,1);
interf = zeros(K,1);
scaling = zeros(L,K);

tau = sqrt(1+sigma_e^2);

for n=1:nbrOfRealizations
    for l=1:L
        
        servedUEs = find(D(l,:)==1);
        if isempty(servedUEs); continue; end
        
        Hhat_all = reshape(Hhat((l-1)*N+1:l*N,n,:),[N K]);
        Hhat_local = Hhat_all(:,servedUEs);
        
        P_bar = (Hhat_local*Hhat_local' + (K/Pt)*eyeN)\Hhat_local;
        
        for iter=1:nIter
            
            f = sqrt(Pt/real(trace(P_bar*P_bar')));
            
            Theta = sigma_e^2*(Hhat_local*Hhat_local'/K);
            
            P = f*tau*P_bar;
            
            lambda = trace(eyeN)/(f^2*Pt) - trace(P*P'*Theta)/Pt;
            lambda = max(lambda,1e-6);
            
            P_bar = (Hhat_local*Hhat_local' + (1+f^2)*Theta + lambda*f^2*tau*eyeN)\Hhat_local;
        end
        
        f = sqrt(Pt/real(trace(P_bar*P_bar')));
        V = f*tau*P_bar;
        
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
        Hhat_local = Hhat_all(:,servedUEs);
        
        P_bar = (Hhat_local*Hhat_local' + (K/Pt)*eyeN)\Hhat_local;
        
        for iter=1:nIter
            
            f = sqrt(Pt/real(trace(P_bar*P_bar')));
            Theta = sigma_e^2*(Hhat_local*Hhat_local'/K);
            
            P = f*tau*P_bar;
            
            lambda = trace(eyeN)/(f^2*Pt) - trace(P*P'*Theta)/Pt;
            lambda = max(lambda,1e-6);
            
            P_bar = (Hhat_local*Hhat_local' + (1+f^2)*Theta + lambda*f^2*tau*eyeN)\Hhat_local;
        end
        
        f = sqrt(Pt/real(trace(P_bar*P_bar')));
        V = f*tau*P_bar;
        
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
