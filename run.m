% CF Downlink Simulation - 入口脚本
% 运行此脚本启动仿真
%
% 项目结构:
%   main/            - 主程序
%   channel/         - 信道模型
%   precoding/       - 预编码方案
%   power_allocation/ - 功率分配算法
%   se_calculation/  - SE计算
%   visualization/    - 可视化

% 获取当前脚本所在目录
currentDir = fileparts(mfilename('fullpath'));
cd(currentDir);

% 添加所有子目录到 MATLAB 路径
addpath(genpath(pwd));

% 启动主仿真程序
cd main;
Combined_Downlink_Sim;
