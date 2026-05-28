function [rho, timing] = computeRhoRL(~, D, gainOverNoise, Pt, rlModelPath, sigma_e)
%computeRhoRL DQN/DDPG power allocation through a cached Python runtime.
%   RL checkpoints are trained from a large-scale sum-rate reward proxy, not
%   from WMMSE labels. Missing checkpoints fall back to EPA.

    totalTic = tic;
    [L, K] = size(D);
    timing = emptyTiming();

    if nargin < 5 || isempty(rlModelPath) || ~isfile(rlModelPath)
        rho = computeRhoEPA(D, Pt, L, K);
        timing.total_sec = toc(totalTic);
        timing.forward_sec = timing.total_sec;
        return;
    end
    if nargin < 6 || isempty(sigma_e)
        sigma_e = 0.3;
    end

    sqrtGain = sqrt(max(gainOverNoise, 0));

    if exist('getProjectPaths', 'file') == 2
        paths = getProjectPaths();
        pyDir = paths.python;
    else
        thisDir = fileparts(mfilename('fullpath'));
        matlabDir = fileparts(thisDir);
        rootDir = fileparts(matlabDir);
        pyDir = fullfile(rootDir, 'python');
    end

    persistent cached_pyDir cached_runtime
    if isempty(cached_pyDir) || ~strcmp(string(cached_pyDir), string(pyDir))
        sysPathRaw = cell(py.sys.path);
        sysPath = cellfun(@char, sysPathRaw, 'UniformOutput', false);
        if ~any(strcmpi(sysPath, pyDir))
            insert(py.sys.path, int32(0), pyDir);
        end
        cached_pyDir = pyDir;
        cached_runtime = [];
    end

    if isempty(cached_runtime)
        cached_runtime = py.importlib.import_module('rl_runtime');
    end

    result = cached_runtime.infer(rlModelPath, sqrtGain, D, Pt, sigma_e);
    rho_np = result{'rho'};
    rho_np = py.numpy.ascontiguousarray(rho_np);
    rho_flat = double(py.array.array('d', py.numpy.ravel(rho_np, pyargs('order', 'C'))));
    rho = reshape(rho_flat, [K, L]).';

    timing.load_sec = pyFloat(result, 'load_sec');
    timing.feature_sec = pyFloat(result, 'feature_sec');
    timing.collate_sec = 0;
    timing.forward_sec = pyFloat(result, 'forward_sec');
    timing.post_sec = pyFloat(result, 'post_sec');
    timing.python_total_sec = pyFloat(result, 'python_total_sec');
    timing.total_sec = toc(totalTic);
    timing.bridge_sec = max(timing.total_sec - timing.python_total_sec, 0);
end

function timing = emptyTiming()
    timing = struct( ...
        'total_sec', 0, ...
        'bridge_sec', 0, ...
        'load_sec', 0, ...
        'feature_sec', 0, ...
        'collate_sec', 0, ...
        'forward_sec', 0, ...
        'post_sec', 0, ...
        'python_total_sec', 0);
end

function value = pyFloat(d, key)
    value = double(py.float(d{key}));
end
