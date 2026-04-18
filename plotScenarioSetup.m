% 新增函数：绘制 AP 和 UE 的位置
function plotScenarioSetup(APpositions, UEpositions, scenarioIdx, isSaveFig, savePath, isSaveData, dataPath)
    % 创建一个隐藏的 figure 用于绘图和保存
    figHandle = figure('Visible', 'off');
    
    plot(real(APpositions), imag(APpositions), 'b^', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'APs');
    hold on;
    plot(real(UEpositions), imag(UEpositions), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r', 'DisplayName', 'UEs');
    
    xlabel('Horizontal Position (m)');
    ylabel('Vertical Position (m)');
    title(sprintf('Scenario %d Layout: %d APs, %d UEs', scenarioIdx, length(APpositions), length(UEpositions)));
    legend('Location', 'northeastoutside');
    grid on;
    axis square;
    
    % 保存图像
    if nargin > 3 && isSaveFig && isvalid(figHandle)
        try
            saveas(figHandle, fullfile(savePath, sprintf('Scenario_%d_Layout.fig', scenarioIdx)));
            saveas(figHandle, fullfile(savePath, sprintf('Scenario_%d_Layout.png', scenarioIdx)));
        catch ME
            fprintf('  ⚠ Warning: Failed to save scenario figure: %s\n', ME.message);
        end
    end
    
    % 保存数据
    if nargin > 5 && isSaveData
        save(fullfile(dataPath, sprintf('Scenario_%d_Positions.mat', scenarioIdx)), 'APpositions', 'UEpositions');
    end
    
    hold off;
    close(figHandle);
end
