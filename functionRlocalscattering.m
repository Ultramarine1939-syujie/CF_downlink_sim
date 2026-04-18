% Function to generate the spatial correlation matrix for local scattering model
% 用于生成局部散射模型空间相关矩阵的函数
function R = functionRlocalscattering(N,varphi,theta,ASD_varphi,ASD_theta,antennaSpacing)
if  nargin < 6
    antennaSpacing = 1/2;
end
firstRow  = zeros(N,1);
firstRow(1) = 1; 
for column = 2:N
    distance = antennaSpacing*(column-1);
    if (ASD_theta>0) && (ASD_varphi>0)
        F = @(delta,epsilon)exp(1i*2*pi*distance*sin(varphi+delta).*cos(theta+epsilon)).*...
            exp(-delta.^2/(2*ASD_varphi^2))/(sqrt(2*pi)*ASD_varphi).*...
            exp(-epsilon.^2/(2*ASD_theta^2))/(sqrt(2*pi)*ASD_theta);
        firstRow(column) = integral2(F,-20*ASD_varphi,20*ASD_varphi,-20*ASD_theta,20*ASD_theta);
    elseif ASD_varphi>0
        F = @(delta)exp(1i*2*pi*distance*sin(varphi+delta).*cos(theta)).*...
            exp(-delta.^2/(2*ASD_varphi^2))/(sqrt(2*pi)*ASD_varphi);
        firstRow(column) = integral(F,-20*ASD_varphi,20*ASD_varphi);
    elseif ASD_theta>0
        F = @(epsilon)exp(1i*2*pi*distance*sin(varphi).*cos(theta+epsilon)).*...
            exp(-epsilon.^2/(2*ASD_theta^2))/(sqrt(2*pi)*ASD_theta);
        firstRow(column) = integral(F,-20*ASD_theta,20*ASD_theta);
    else
        firstRow(column) = exp(1i*2*pi*distance*sin(varphi).*cos(theta));
    end
end
R = toeplitz(firstRow);
R = R*(N/trace(R));
end
