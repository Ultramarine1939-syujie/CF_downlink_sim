% 打印仿真最终结果汇总表
function printFinalResults(ESR_MR_all, ESR_L_all, ESR_R_all, ESR_PSO_MR_all, ESR_PSO_L_all, ESR_PSO_R_all, ...
    ESR_MR_dcc, ESR_L_dcc, ESR_R_dcc, ESR_PSO_MR_dcc, ESR_PSO_L_dcc, ESR_PSO_R_dcc, ...
    numScenarios, nbrOfRealizations, num_snr, totalIterations, ...
    d_opt_all, d_opt_dcc, iterUsed_all, iterUsed_dcc, bestFitness_all, bestFitness_dcc, ...
    isSaveFig, isSaveData, savePath, dataPath)

    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║              SIMULATION COMPLETED SUCCESSFULLY               ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Performance Summary:                                        ║\n');
    fprintf('║  ┌────────────────────────────────────────────────────────┐ ║\n');
    fprintf('║  │ Algorithm          │ Mode   │ Min ESR    │ Max ESR    │ ║\n');
    fprintf('║  ├────────────────────┼────────┼────────────┼────────────┤ ║\n');
    fprintf('║  │ MR                 │ All-UE │ %8.2f   │ %8.2f   │ ║\n', min(ESR_MR_all), max(ESR_MR_all));
    fprintf('║  │ PSO+MR             │ All-UE │ %8.2f   │ %8.2f   │ ║\n', min(ESR_PSO_MR_all), max(ESR_PSO_MR_all));
    fprintf('║  │ L-MMSE             │ All-UE │ %8.2f   │ %8.2f   │ ║\n', min(ESR_L_all), max(ESR_L_all));
    fprintf('║  │ PSO+L-MMSE         │ All-UE │ %8.2f   │ %8.2f   │ ║\n', min(ESR_PSO_L_all), max(ESR_PSO_L_all));
    fprintf('║  │ Robust-MMSE        │ All-UE │ %8.2f   │ %8.2f   │ ║\n', min(ESR_R_all), max(ESR_R_all));
    fprintf('║  │ PSO+R-MMSE         │ All-UE │ %8.2f   │ %8.2f   │ ║\n', min(ESR_PSO_R_all), max(ESR_PSO_R_all));
    fprintf('║  ├────────────────────┼────────┼────────────┼────────────┤ ║\n');
    fprintf('║  │ MR                 │ DCC    │ %8.2f   │ %8.2f   │ ║\n', min(ESR_MR_dcc), max(ESR_MR_dcc));
    fprintf('║  │ PSO+MR             │ DCC    │ %8.2f   │ %8.2f   │ ║\n', min(ESR_PSO_MR_dcc), max(ESR_PSO_MR_dcc));
    fprintf('║  │ L-MMSE             │ DCC    │ %8.2f   │ %8.2f   │ ║\n', min(ESR_L_dcc), max(ESR_L_dcc));
    fprintf('║  │ PSO+L-MMSE         │ DCC    │ %8.2f   │ %8.2f   │ ║\n', min(ESR_PSO_L_dcc), max(ESR_PSO_L_dcc));
    fprintf('║  │ Robust-MMSE        │ DCC    │ %8.2f   │ %8.2f   │ ║\n', min(ESR_R_dcc), max(ESR_R_dcc));
    fprintf('║  │ PSO+R-MMSE         │ DCC    │ %8.2f   │ %8.2f   │ ║\n', min(ESR_PSO_R_dcc), max(ESR_PSO_R_dcc));
    fprintf('║  └────────────────────────────────────────────────────────┘ ║\n');

    % PSO Optimization Details
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║              PSO OPTIMIZATION DETAILS                         ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');

    fprintf('║  All-UE Scenario PSO Results:                                ║\n');
    fprintf('║    • Iterations used:     %2d / 50                            ║\n', mean(iterUsed_all));
    fprintf('║    • Best Fitness (ESR):  %.4f                               ║\n', mean(bestFitness_all));
    fprintf('║    • Power Allocation (d_k):                                 ║\n');

    % 打印d_opt_all的值，每行4个
    K = length(d_opt_all);
    for row = 1:4:K
        rowEnd = min(row+3, K);
        if row == 1
            fprintf('║      d_k [%2d-%2d]: ', row-1, rowEnd-1);
        else
            fprintf('║               [%2d-%2d]: ', row-1, rowEnd-1);
        end
        for k = row:rowEnd
            fprintf('%.4f  ', d_opt_all(k));
        end
        fprintf('║\n');
    end

    fprintf('║                                                              ║\n');
    fprintf('║  DCC-UE Scenario PSO Results:                                ║\n');
    fprintf('║    • Iterations used:     %2d / 50                            ║\n', mean(iterUsed_dcc));
    fprintf('║    • Best Fitness (ESR):  %.4f                               ║\n', mean(bestFitness_dcc));
    fprintf('║    • Power Allocation (d_k):                                 ║\n');

    % 打印d_opt_dcc的值，每行4个
    for row = 1:4:K
        rowEnd = min(row+3, K);
        if row == 1
            fprintf('║      d_k [%2d-%2d]: ', row-1, rowEnd-1);
        else
            fprintf('║               [%2d-%2d]: ', row-1, rowEnd-1);
        end
        for k = row:rowEnd
            fprintf('%.4f  ', d_opt_dcc(k));
        end
        fprintf('║\n');
    end

    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
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