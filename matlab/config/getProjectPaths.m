function paths = getProjectPaths()
%GETPROJECTPATHS Central path registry for the project.
%   Keep filesystem layout changes in one place. Callers should build paths
%   from this struct instead of hard-coding root-relative strings.

configDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(configDir);
rootDir = fileparts(matlabDir);

paths = struct();
paths.root = rootDir;
paths.matlab = matlabDir;
paths.config = configDir;
paths.main = fullfile(matlabDir, 'main');
paths.channel = fullfile(matlabDir, 'channel');
paths.precoding = fullfile(matlabDir, 'precoding');
paths.powerAllocation = fullfile(matlabDir, 'power_allocation');
paths.seCalculation = fullfile(matlabDir, 'se_calculation');
paths.visualization = fullfile(matlabDir, 'visualization');
paths.python = fullfile(rootDir, 'python');
paths.docs = fullfile(rootDir, 'docs');

paths.data = fullfile(rootDir, 'data');
paths.trainingData = fullfile(paths.data, 'gnn_training');
paths.models = fullfile(rootDir, 'models');

paths.outputRoot = fullfile(rootDir, 'main');
paths.figures = fullfile(paths.outputRoot, 'Imgs');
paths.simulationData = fullfile(paths.outputRoot, 'SimulationData');
end
