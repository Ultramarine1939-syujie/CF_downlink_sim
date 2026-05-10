function rho = computeRhoGNN(Hhat, D, gainOverNoise, Pt, gnnModelPath, sigma_e)
    [L, K] = size(D);
    if nargin < 5 || isempty(gnnModelPath) || ~isfile(gnnModelPath)
        rho = computeRhoEPA(D, Pt, L, K);
        return;
    end
    if nargin < 6 || isempty(sigma_e)
        sigma_e = 0.3;
    end

    sqrtGain = sqrt(max(gainOverNoise, 0));

    thisDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(thisDir);
    pyDir = fullfile(rootDir, 'python');

    persistent cached_pyDir cached_torch cached_np cached_train_gnn cached_model cached_model_path cached_L cached_K

    needReload = isempty(cached_model) || isempty(cached_model_path) || ~strcmp(string(cached_model_path), string(gnnModelPath)) ...
        || isempty(cached_L) || isempty(cached_K) || cached_L ~= L || cached_K ~= K;

    if isempty(cached_pyDir) || ~strcmp(string(cached_pyDir), string(pyDir))
        sysPathRaw = cell(py.sys.path);
        sysPath = cellfun(@char, sysPathRaw, 'UniformOutput', false);
        if ~any(strcmpi(sysPath, pyDir))
            insert(py.sys.path, int32(0), pyDir);
        end
        cached_pyDir = pyDir;
        cached_torch = [];
        cached_np = [];
        cached_train_gnn = [];
        cached_model = [];
        cached_model_path = [];
        cached_L = [];
        cached_K = [];
        needReload = true;
    end

    if isempty(cached_torch); cached_torch = py.importlib.import_module('torch'); end
    if isempty(cached_np); cached_np = py.importlib.import_module('numpy'); end
    if isempty(cached_train_gnn); cached_train_gnn = py.importlib.import_module('train_gnn'); end

    if needReload
        ckpt = cached_torch.load(gnnModelPath, pyargs('map_location', 'cpu', 'weights_only', false));
    model_type = 'gat';  % 默认 GAT
    ckpt_keys = cellfun(@char, cell(ckpt.keys()), 'UniformOutput', false);
    if ismember('model_type', ckpt_keys)
        model_type = char(ckpt{'model_type'});
    end

    if strcmp(model_type, 'mlp')
        model = cached_train_gnn.PowerGNN_MLP(pyargs( ...
            'L', int32(L), ...
            'K', int32(K), ...
            'hidden_dim', int32(128), ...
            'num_layers', int32(3), ...
            'dropout', 0.1, ...
            'output_scale', 1.0 ...
        ));
    else
        model = cached_train_gnn.PowerGNN_GAT(pyargs( ...
            'L', int32(L), ...
            'K', int32(K), ...
            'hidden_dim', int32(128), ...
            'num_heads', int32(4), ...
            'num_layers', int32(3), ...
            'dropout', 0.1, ...
            'output_scale', 1.0 ...
        ));
    end

    if isa(ckpt, 'py.dict') && ismember('model_state_dict', ckpt_keys)
        state = ckpt{'model_state_dict'};
    else
        state = ckpt;
    end
    model.load_state_dict(state);
    model.eval();

        cached_model = model;
        cached_model_path = gnnModelPath;
        cached_L = L;
        cached_K = K;
    end

    torch = cached_torch;
    np = cached_np;
    train_gnn = cached_train_gnn;
    model = cached_model;

    x_ap = torch.tensor(np.array(sqrtGain), pyargs('dtype', torch.float32));
    x_ue_mat = [sqrtGain.', sigma_e * ones(K, 1)];
    x_ue = torch.tensor(np.array(x_ue_mat), pyargs('dtype', torch.float32));
    D_mask = torch.tensor(np.array(D), pyargs('dtype', torch.float32));
    y = torch.zeros(int32(L), int32(K), pyargs('dtype', torch.float32));
    rho_is_nonzero = torch.zeros(int32(L), int32(K), pyargs('dtype', torch.float32));
    z1 = torch.zeros(int32(1), pyargs('dtype', torch.float32));

    sample = py.dict(pyargs( ...
        'x_ap', x_ap, ...
        'x_ue', x_ue, ...
        'D_mask', D_mask, ...
        'rho_is_nonzero', rho_is_nonzero, ...
        'y', y, ...
        'esr', z1, ...
        'snr', z1, ...
        'mode', 'All', ...
        'idx', int32(0) ...
    ));

    batch = train_gnn.custom_collate(py.list({sample}));
    if strcmp(model_type, 'mlp')
        batch = train_gnn.custom_collate_mlp(py.list({sample}));
    end
    rho_t = model(batch);
    rho_np = np.squeeze(rho_t.detach().cpu().numpy());

    rho_flat = double(py.array.array('d', py.numpy.nditer(rho_np)));
    rho_pred = reshape(rho_flat, [K, L]).';

    weights = max((rho_pred + 1.0) / 2.0, 0.0);
    weights = weights .* D;

    rho = zeros(L, K);
    for l = 1:L
        served = find(D(l, :) > 0.5);
        if isempty(served)
            continue;
        end
        w = weights(l, served);
        s = sum(w);
        if s <= 0
            rho(l, served) = Pt / numel(served);
        else
            rho(l, served) = Pt * w / s;
        end
    end
end

