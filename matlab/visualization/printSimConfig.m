% 打印仿真配置信息表
function printSimConfig(L, N, K, tau_c, tau_p, SNR_dB, numScenarios, nbrOfRealizations, sigma_e, nIter, isSaveFig, isSaveData)
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║            Downlink Simulation - Status Display              ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Simulation Configuration:                                   ║\n');
    fprintf('║    • Number of APs (L):         %d                           ║\n', L);
    fprintf('║    • Antennas per AP (N):       %d                           ║\n', N);
    fprintf('║    • Number of UEs (K):         %d                           ║\n', K);
    fprintf('║    • Coherence length (tau_c):  %d                           ║\n', tau_c);
    fprintf('║    • Pilot length (tau_p):       %d                          ║\n', tau_p);
    fprintf('║    • SNR range (dB):            %s                           ║\n', mat2str(SNR_dB));
    fprintf('║    • Scenarios:                 %d                           ║\n', numScenarios);
    fprintf('║    • Realizations per scenario:  %d                          ║\n', nbrOfRealizations);
    fprintf('║    • Robust CSI error (sigma_e): %.2f                        ║\n', sigma_e);
    fprintf('║    • Robust iterations (nIter):  %d                          ║\n', nIter);
    fprintf('║                                                              ║\n');
    fprintf('║  Output Settings:                                            ║\n');
    fprintf('║    • Save figures:            %s                             ║\n', mat2str(isSaveFig));
    fprintf('║    • Save data:               %s                             ║\n', mat2str(isSaveData));
    fprintf('╚══════════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
end
