% Function to generate realizations of the simulation setup
% 用于生成模拟设置的随机实现的函数
function [gainOverNoisedB,R,pilotIndex,D,D_small,APpositions,UEpositions,distances] = generateSetup(L,K,N,tau_p,nbrOfSetups,seed,ASD_varphi,ASD_theta)
if (nargin>5)&&(seed>0)
    rng(seed); 
end
squareLength = 1000;    
B = 20e6;               
noiseFigure = 7;        
noiseVariancedBm = -174 + 10*log10(B) + noiseFigure;    
alpha = 36.7;           
constantTerm = -30.5;   
sigma_sf = 4;           
decorr = 9;             
distanceVertical = 10;  
antennaSpacing = 1/2;   

gainOverNoisedB = zeros(L,K,nbrOfSetups);               
R = zeros(N,N,L,K,nbrOfSetups);                         
distances = zeros(L,K,nbrOfSetups);                     
pilotIndex = zeros(K,nbrOfSetups);                      
D = zeros(L,K,nbrOfSetups);                             
D_small = zeros(L,K,nbrOfSetups);                       

for n = 1:nbrOfSetups
    APpositions = (rand(L,1) + 1i*rand(L,1)) * squareLength;
    UEpositions = zeros(K,1);
    wrapHorizontal = repmat([-squareLength 0 squareLength],[3 1]);
    wrapVertical = wrapHorizontal';
    wrapLocations = wrapHorizontal(:)' + 1i*wrapVertical(:)';
    APpositionsWrapped = repmat(APpositions,[1 length(wrapLocations)]) + repmat(wrapLocations,[L 1]);
    shadowCorrMatrix = sigma_sf^2*ones(K,K);
    shadowAPrealizations = zeros(K,L);
    for k = 1:K
        UEposition = (rand(1,1) + 1i*rand(1,1)) * squareLength;
        [distanceAPstoUE,whichpos] = min(abs(APpositionsWrapped - repmat(UEposition,size(APpositionsWrapped))),[],2);
        distances(:,k,n) = sqrt(distanceVertical^2+distanceAPstoUE.^2);
        if k-1>0
            shortestDistances = zeros(k-1,1);
            for i = 1:k-1
                shortestDistances(i) = min(abs(UEposition - UEpositions(i) + wrapLocations));
            end
            newcolumn = sigma_sf^2*2.^(-shortestDistances/decorr);
            term1 = newcolumn'/shadowCorrMatrix(1:k-1,1:k-1);
            meanvalues = term1*shadowAPrealizations(1:k-1,:);
            stdvalue = sqrt(sigma_sf^2 - term1*newcolumn);
        else
            meanvalues = 0;
            stdvalue = sigma_sf;
            newcolumn = [];
        end
        shadowing = meanvalues + stdvalue*randn(1,L);
        gainOverNoisedB(:,k,n) = constantTerm - alpha*log10(distances(:,k,n)) + shadowing' - noiseVariancedBm;
        shadowCorrMatrix(1:k-1,k) = newcolumn;
        shadowCorrMatrix(k,1:k-1) = newcolumn';
        shadowAPrealizations(k,:) = shadowing;
        UEpositions(k) = UEposition;
        [~,master] = max(gainOverNoisedB(:,k,n));
        D(master,k,n) = 1; 
        if k <= tau_p
            pilotIndex(k,n) = k;
        else
            pilotinterference = zeros(tau_p,1);
            for t = 1:tau_p
                pilotinterference(t) = sum(db2pow(gainOverNoisedB(master,pilotIndex(1:k-1,n)==t,n)));
            end
            [~,bestpilot] = min(pilotinterference);
            pilotIndex(k,n) = bestpilot;
        end
        for l = 1:L
            angletoUE_varphi = angle(UEpositions(k)-APpositionsWrapped(l,whichpos(l)));
            angletoUE_theta = asin(distanceVertical/distances(l,k,n));
            if nargin>6
                R(:,:,l,k,n) = db2pow(gainOverNoisedB(l,k,n))*functionRlocalscattering(N,angletoUE_varphi,angletoUE_theta,ASD_varphi,ASD_theta,antennaSpacing);
            else
                R(:,:,l,k,n) = db2pow(gainOverNoisedB(l,k,n))*eye(N);
            end
        end
    end
    for l = 1:L
        for t = 1:tau_p
            pilotUEs = find(t==pilotIndex(:,n));
            [~,UEindex] = max(gainOverNoisedB(l,pilotUEs,n));
            D(l,pilotUEs(UEindex),n) = 1;
        end
    end
    for k=1:K
        tempmat = -inf*ones(L,1);
        tempmat(D(:,k,n)==1,1) = gainOverNoisedB(D(:,k,n)==1,k,n);
        [~,servingAP] = max(tempmat);
        D_small(servingAP,k,n) = 1;
    end
end
end
