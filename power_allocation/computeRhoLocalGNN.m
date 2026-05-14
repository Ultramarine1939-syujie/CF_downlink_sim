function [rho, timing] = computeRhoLocalGNN(~, D, gainOverNoise, Pt, localModelPath, sigma_e)
%computeRhoLocalGNN Strict AP-local neural power allocation.
%   Each AP row is inferred from only its local gain/DCC row and scalar
%   operating-point features. The Python call batches AP rows only to reduce
%   MATLAB-Python overhead; there is no cross-AP message passing.

    totalTic = tic;
    [L, K] = size(D);
    timing = emptyTiming();

    if nargin < 5 || isempty(localModelPath) || ~isfile(localModelPath)
        rho = computeRhoEPA(D, Pt, L, K);
        timing.total_sec = toc(totalTic);
        timing.forward_sec = timing.total_sec;
        return;
    end
    if nargin < 6 || isempty(sigma_e)
        sigma_e = 0.3;
    end

    sqrtGain = sqrt(max(gainOverNoise, 0));

    thisDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(thisDir);
    pyDir = fullfile(rootDir, 'python');

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
        cached_runtime = py.importlib.import_module('gnn_runtime_local');
    end

    result = cached_runtime.infer(localModelPath, sqrtGain, D, Pt, sigma_e);
    rho_np = result{'rho'};
    rho_flat = double(py.array.array('d', py.numpy.nditer(rho_np)));
    rho = reshape(rho_flat, [K, L]).';

    timing.load_sec = pyFloat(result, 'load_sec');
    timing.feature_sec = pyFloat(result, 'feature_sec');
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
        'forward_sec', 0, ...
        'post_sec', 0, ...
        'python_total_sec', 0);
end

function value = pyFloat(d, key)
    value = double(py.float(d{key}));
end
