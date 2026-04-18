% 打印仿真最终结果汇总表
function printFinalResults(ESR_MR_all, ESR_L_all, ESR_R_all, ESR_MR_dcc, ESR_L_dcc, ESR_R_dcc, numScenarios, nbrOfRealizations, num_snr, totalIterations, isSaveFig, isSaveData, savePath, dataPath)
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║              SIMULATION COMPLETED SUCCESSFULLY               ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Performance Summary:                                        ║\n');
    fprintf('║  ┌────────────────────────────────────────────────────────┐ ║\n');
    fprintf('║  │ Algorithm          │ Mode   │ Min ESR    │ Max ESR    │ ║\n');
    fprintf('║  ├────────────────────┼────────┼────────────┼────────────┤ ║\n');
    fprintf('║  │ MR                 │ All-UE │ %8.2f   │ %8.2f   │ ║\n', min(ESR_MR_all), max(ESR_MR_all));
    fprintf('║  │ L-MMSE             │ All-UE │ %8.2f   │ %8.2f   │ ║\n', min(ESR_L_all), max(ESR_L_all));
    fprintf('║  │ Robust-MMSE        │ All-UE │ %8.2f   │ %8.2f   │ ║\n', min(ESR_R_all), max(ESR_R_all));
    fprintf('║  ├────────────────────┼────────┼────────────┼────────────┤ ║\n');
    fprintf('║  │ MR                 │ DCC    │ %8.2f   │ %8.2f   │ ║\n', min(ESR_MR_dcc), max(ESR_MR_dcc));
    fprintf('║  │ L-MMSE             │ DCC    │ %8.2f   │ %8.2f   │ ║\n', min(ESR_L_dcc), max(ESR_L_dcc));
    fprintf('║  │ Robust-MMSE        │ DCC    │ %8.2f   │ %8.2f   │ ║\n', min(ESR_R_dcc), max(ESR_R_dcc));
    fprintf('║  └────────────────────────────────────────────────────────┘ ║\n');
    fprintf('║                                                              ║\n');
    fprintf('║  Statistics:                                                 ║\n');
    fprintf('║    • Total scenarios:           %d                           ║\n', numScenarios);
    fprintf('║    • Realizations per scenario:  %d                           ║\n', nbrOfRealizations);
    fprintf('║    • SNR points:                %d                           ║\n', num_snr);
    fprintf('║    • Total iterations:           %d                           ║\n', totalIterations);
    if isSaveFig
        fprintf('║  Output:                                                     ║\n');
        fprintf('║    • Figures saved to:       %s/                             ║\n', savePath(1:min(end,35)));
    end
    if isSaveData
        fprintf('║    • Data saved to:          %s/                             ║\n', dataPath(1:min(end,35)));
    end
    fprintf('╚══════════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
end