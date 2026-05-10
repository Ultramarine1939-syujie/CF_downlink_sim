classdef SimulationLogger
%SIMULATIONLOGGER 仿真日志记录器
%   提供统一的日志输出接口，支持不同日志级别和文件输出

properties
    LogLevel (1, 1) double {mustBeInteger, mustBePositive} = 1
    LogFile (1, :) char = ''
    EnableTimestamp (1, 1) logical = true
    EnableConsole (1, 1) logical = true
    EnableFile (1, 1) logical = false
end

properties (Access = private)
    FileHandle
end

methods
    function obj = SimulationLogger(varargin)
        p = inputParser;
        p.addParameter('LogLevel', 1);
        p.addParameter('LogFile', '');
        p.addParameter('EnableTimestamp', true);
        p.addParameter('EnableConsole', true);
        p.addParameter('EnableFile', false);
        p.parse(varargin{:});

        obj.LogLevel = p.Results.LogLevel;
        obj.LogFile = p.Results.LogFile;
        obj.EnableTimestamp = p.Results.EnableTimestamp;
        obj.EnableConsole = p.Results.EnableConsole;
        obj.EnableFile = p.Results.EnableFile;

        if obj.EnableFile && ~isempty(obj.LogFile)
            obj.FileHandle = fopen(obj.LogFile, 'a');
            if obj.FileHandle == -1
                warning('Cannot open log file: %s', obj.LogFile);
                obj.EnableFile = false;
            end
        end
    end

    function delete(obj)
        if obj.EnableFile && obj.FileHandle ~= -1
            fclose(obj.FileHandle);
        end
    end

    function obj = set.LogLevel(obj, value)
        if value < 0 || value > 3
            error('LogLevel must be between 0 (off) and 3 (debug)');
        end
        obj.LogLevel = value;
    end

    function info(obj, msg, varargin)
        if obj.LogLevel >= 1
            obj.log('INFO', msg, varargin{:});
        end
    end

    function warn(obj, msg, varargin)
        if obj.LogLevel >= 2
            obj.log('WARN', msg, varargin{:});
        end
    end

    function debug(obj, msg, varargin)
        if obj.LogLevel >= 3
            obj.log('DEBUG', msg, varargin{:});
        end
    end

    function error_log(obj, msg, varargin)
        if obj.LogLevel >= 1
            obj.log('ERROR', msg, varargin{:});
        end
    end

    function progress(obj, current, total, varargin)
        if obj.LogLevel >= 1
            pct = current / total * 100;
            msg = sprintf('Progress: %d/%d (%.1f%%)', current, total, pct);
            if nargin > 3
                msg = [msg, ' - ', sprintf(varargin{:})];
            end
            obj.info(msg);
        end
    end

    function section(obj, title)
        if obj.LogLevel >= 1
            separator = repmat('=', [1, 60]);
            obj.info(separator);
            obj.info(title);
            obj.info(separator);
        end
    end

    function subsection(obj, title)
        if obj.LogLevel >= 1
            separator = repmat('-', [1, 60]);
            obj.info(separator);
            obj.info(title);
        end
    end

    function table(obj, data, headers)
        if obj.LogLevel >= 1 && nargin >= 2
            if nargin < 3
                headers = {};
            end
            obj.printTable(data, headers);
        end
    end
end

methods (Access = private)
    function log(obj, level, msg, varargin)
        if ~isempty(varargin)
            fullMsg = sprintf(msg, varargin{:});
        else
            fullMsg = msg;
        end

        if obj.EnableTimestamp
            timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
            logLine = sprintf('[%s] [%s] %s', timestamp, level, fullMsg);
        else
            logLine = sprintf('[%s] %s', level, fullMsg);
        end

        if obj.EnableConsole
            if strcmp(level, 'ERROR')
                fprintf(2, '%s\n', logLine);
            elseif strcmp(level, 'WARN')
                fprintf(1, '%s\n', logLine);
            else
                fprintf('%s\n', logLine);
            end
        end

        if obj.EnableFile && obj.FileHandle ~= -1
            fprintf(obj.FileHandle, '%s\n', logLine);
        end
    end

    function printTable(obj, data, headers)
        if iscell(data)
            nRows = size(data, 1);
            nCols = size(data, 2);

            if ~isempty(headers) && length(headers) == nCols
                headerLine = sprintf('  %-20s', headers{:});
                obj.info(headerLine);
                obj.info(repmat('-', [1, 80]));
            end

            for i = 1:min(nRows, 100)
                rowData = data(i, :);
                rowStr = cellfun(@(x) sprintf('%-20s', obj.formatValue(x)), rowData, 'UniformOutput', false);
                obj.info(sprintf('  %s', rowStr{:}));
            end

            if nRows > 100
                obj.info(sprintf('  ... (showing 100 of %d rows)', nRows));
            end
        elseif isstruct(data)
            fields = fieldnames(data);
            for i = 1:min(length(fields), 100)
                fname = fields{i};
                fval = data.(fname);
                obj.info(sprintf('  %-30s: %s', fname, obj.formatValue(fval)));
            end
        end
    end

    function str = formatValue(obj, value)
        if isnumeric(value)
            if isscalar(value)
                if isreal(value) && abs(value) < 1e6 && abs(value) >= 1e-3
                    str = sprintf('%.4g', value);
                else
                    str = sprintf('%e', value);
                end
            else
                str = mat2str(value);
            end
        elseif ischar(value)
            str = value;
        elseif islogical(value)
            str = mat2str(value);
        else
            str = class(value);
        end
    end
end
end
