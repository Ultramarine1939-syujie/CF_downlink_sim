cd C:\Users\Admin\Documents\组内资料\repos_git\Downlink_sim\CF_downlink_sim
addpath(genpath(pwd))
exportTrainingData(fullfile(pwd,'data','gnn_training'), 2)
diary off
exit
