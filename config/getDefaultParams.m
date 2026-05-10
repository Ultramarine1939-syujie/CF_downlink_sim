function params = getDefaultParams()
%GETDEFAULTPARAMS 返回默认仿真参数配置
%   所有参数集中管理，确保 MATLAB 与 Python 模块参数一致性

params = struct();

params.system = struct();
params.system.L = 100;
params.system.K = 20;
params.system.N = 1;
params.system.tau_c = 200;
params.system.tau_p = 10;

params.channel = struct();
params.channel.ASD_varphi = deg2rad(15);
params.channel.ASD_theta = deg2rad(15);
params.channel.squareLength = 1000;
params.channel.B = 20e6;
params.channel.noiseFigure = 7;
params.channel.alpha = 36.7;
params.channel.constantTerm = -30.5;
params.channel.sigma_sf = 4;
params.channel.decorr = 9;
params.channel.distanceVertical = 10;
params.channel.antennaSpacing = 0.5;

params.power = struct();
params.power.p = 100;
params.power.SNR_dB_range = -10:5:30;
params.power.defaultSNR_dB = 10:5:30;

params.csi = struct();
params.csi.sigma_e = 0.3;
params.csi.nIter = 5;

params.simulation = struct();
params.simulation.nbrOfRealizations = 50;
params.simulation.numScenarios = 2;
params.simulation.seed = 42;

params.wmmse = struct();
params.wmmse.maxIter = 30;
params.wmmse.tol = 1e-4;

params.output = struct();
params.output.isSaveFig = true;
params.output.isSaveData = true;
params.output.savePath = 'Imgs';
params.output.dataPath = 'SimulationData';
params.output.isCleanOld = true;

params.gnn = struct();
params.gnn.hidden_dim = 128;
params.gnn.num_heads = 4;
params.gnn.num_layers = 3;
params.gnn.dropout = 0.1;
params.gnn.batch_size = 32;
params.gnn.lr_max = 1e-3;
params.gnn.val_split = 0.15;
params.gnn.patience = 50;
params.gnn.output_scale = 1.0;

params.fedavg = struct();
params.fedavg.rounds = 50;
params.fedavg.frac = 0.2;
params.fedavg.local_epochs = 1;
params.fedavg.lr = 5e-4;
params.fedavg.weight_decay = 5e-4;
params.fedavg.num_clients = 0;
params.fedavg.split_by = 'random';

params.training = struct();
params.training.nSnapshotsPerSNR = 500;
params.training.dataAug_dropRate_min = 0.1;
params.training.dataAug_dropRate_max = 0.3;
params.training.dataAug_sigma_e_var = 0.2;
end


function params = loadParams(varargin)
%LOADPARAMS 加载或创建参数配置
%   params = loadParams() - 使用默认参数
%   params = loadParams('paramName', value, ...) - 自定义参数

params = getDefaultParams();

for i = 1:2:length(varargin)
    name = varargin{i};
    value = varargin{i+1};

    nameParts = strsplit(name, '.');

    if length(nameParts) == 1
        if isfield(params, name)
            params.(name) = value;
        else
            warning('Unknown parameter: %s', name);
        end
    elseif length(nameParts) == 2
        if isfield(params, nameParts{1}) && isfield(params.(nameParts{1}), nameParts{2})
            params.(nameParts{1}).(nameParts{2}) = value;
        else
            warning('Unknown parameter: %s', name);
        end
    else
        warning('Parameter name too deep: %s', name);
    end
end
end


function config = getSimulationConfig()
%GETSIMULATIONCONFIG 返回标准仿真配置结构体
%   用于 Combined_Downlink_Sim.m 初始化

params = getDefaultParams();

config = struct();
config.L = params.system.L;
config.K = params.system.K;
config.N = params.system.N;
config.tau_c = params.system.tau_c;
config.tau_p = params.system.tau_p;
config.ASD_varphi = params.channel.ASD_varphi;
config.ASD_theta = params.channel.ASD_theta;
config.p = params.power.p;
config.SNR_dB = params.power.defaultSNR_dB;
config.sigma_e = params.csi.sigma_e;
config.nIter = params.csi.nIter;
config.numScenarios = params.simulation.numScenarios;
config.nbrOfRealizations = params.simulation.nbrOfRealizations;
config.isSaveFig = params.output.isSaveFig;
config.isSaveData = params.output.isSaveData;
config.VERBOSE = true;

end


function str = paramsToString(params)
%PARAMSTOSTRING 将参数结构体转换为可读字符串
    parts = {};
    fnames = fieldnames(params);
    for i = 1:length(fnames)
        fname = fnames{i};
        if isstruct(params.(fname))
            subnames = fieldnames(params.(fname));
            for j = 1:length(subnames)
                subname = subnames{j};
                value = params.(fname).(subname);
                if isnumeric(value)
                    if isscalar(value)
                        strval = sprintf('%.4g', value);
                    else
                        strval = mat2str(value);
                    end
                elseif ischar(value)
                    strval = value;
                else
                    strval = class(value);
                end
                parts{end+1} = sprintf('%s.%s = %s', fname, subname, strval);
            end
        end
    end
    str = strjoin(parts, '\n');
end


function validateParams(params)
%VALIDATEPARAMS 验证参数合法性
%   抛出错误如果参数无效

assert(params.system.L > 0, 'L must be positive');
assert(params.system.K > 0, 'K must be positive');
assert(params.system.N > 0, 'N must be positive');
assert(params.system.tau_c >= params.system.tau_p, 'tau_c must be >= tau_p');
assert(params.system.tau_p > 0, 'tau_p must be positive');
assert(params.csi.sigma_e >= 0 && params.csi.sigma_e <= 1, 'sigma_e must be in [0, 1]');
assert(params.simulation.numScenarios >= 1, 'numScenarios must be >= 1');
assert(params.simulation.nbrOfRealizations >= 1, 'nbrOfRealizations must be >= 1');

fprintf('参数验证通过\n');
end
