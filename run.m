% Root MATLAB entrypoint for CF_downlink_sim.
% Source code lives under matlab/. Generated figures and caches stay under
% main/ for compatibility with older runs.

projectRoot = fileparts(mfilename('fullpath'));
matlabRoot = fullfile(projectRoot, 'matlab');

if ~isfolder(matlabRoot)
    error('MATLAB source directory not found: %s', matlabRoot);
end

cd(projectRoot);
addpath(genpath(matlabRoot));

paths = getProjectPaths();
cd(paths.main);
Combined_Downlink_Sim;
