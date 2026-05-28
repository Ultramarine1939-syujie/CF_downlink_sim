projectRoot = fileparts(mfilename('fullpath'));
matlabRoot = fullfile(projectRoot, 'matlab');
cd(projectRoot);
addpath(genpath(matlabRoot));

paths = getProjectPaths();
exportTrainingData(paths.trainingData, 2);
