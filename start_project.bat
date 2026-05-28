@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0"
set "MATLAB_SRC=%PROJECT_ROOT%matlab"
set "PYTHON_DIR=%PROJECT_ROOT%python"
set "DATA_DIR=%PROJECT_ROOT%data\gnn_training"
set "MODELS_DIR=%PROJECT_ROOT%models"
set "FIGURE_DIR=%PROJECT_ROOT%main\Imgs"
set "SIM_DATA_DIR=%PROJECT_ROOT%main\SimulationData"

set "MODE=%~1"
set "AUTO_Y=0"
set "FORCE_RETRAIN=0"
if /i "%~1"=="/check" set "MODE=C"
for %%A in (%*) do (
    if /i "%%~A"=="/y" set "AUTO_Y=1"
    if /i "%%~A"=="/retrain" set "FORCE_RETRAIN=1"
)

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "%FIGURE_DIR%" mkdir "%FIGURE_DIR%"
if not exist "%SIM_DATA_DIR%" mkdir "%SIM_DATA_DIR%"

if "%MODE%"=="" goto :menu
goto :dispatch

:menu
echo.
echo ============================================================
echo             CF_downlink_sim Windows Launcher
echo ============================================================
echo Project root:
echo   %PROJECT_ROOT%
echo.
echo   [1] Quick full workflow
echo       small data export + training + smoke tests + simulation
echo.
echo   [2] Full experiment workflow
echo       use matlab\config\getDefaultParams.m settings
echo.
echo   [3] Simulation only
echo       reuse existing data, models, and caches
echo.
echo   [4] Redraw figures only
echo       redraw from main\SimulationData\Simulation_Results_v2.mat
echo.
echo   [5] Smoke tests only
echo       check Python inference and MATLAB bridge
echo.
echo   [6] Validate latest dataset
echo.
echo   [7] Quick data export
echo       export 2 snapshots per SNR for pipeline checks
echo.
echo   [8] Open MATLAB GUI
echo       add matlab\ source tree to MATLAB path
echo.
echo   [C] Environment check
echo   [Q] Quit
echo ============================================================
echo.
set /p MODE="Select [1/2/3/4/5/6/7/8/C/Q]: "

:dispatch
if /i "%MODE%"=="Q" exit /b 0
if /i "%MODE%"=="q" exit /b 0

cd /d "%PROJECT_ROOT%" || (
    echo [ERROR] Failed to enter project root: %PROJECT_ROOT%
    goto :fail
)

if /i "%MODE%"=="1" (
    set "SNAPSHOTS_PER_SNR=10"
    set "TRAIN_EPOCHS=20"
    goto :fullworkflow
)
if /i "%MODE%"=="2" (
    set "SNAPSHOTS_PER_SNR="
    set "TRAIN_EPOCHS=100"
    goto :fullworkflow
)
if /i "%MODE%"=="3" goto :simulation
if /i "%MODE%"=="4" goto :redraw
if /i "%MODE%"=="5" goto :smoketests
if /i "%MODE%"=="6" goto :validate
if /i "%MODE%"=="7" goto :exportquick
if /i "%MODE%"=="8" goto :matlabgui
if /i "%MODE%"=="C" goto :envcheck
if /i "%MODE%"=="c" goto :envcheck

echo [ERROR] Invalid option: %MODE%
goto :fail

:fullworkflow
call :require_matlab || goto :fail
call :require_python || goto :fail

echo.
echo ============================================================
echo   [1/5] Export training data
echo ============================================================
if defined SNAPSHOTS_PER_SNR (
    matlab -batch "cd('%PROJECT_ROOT%'); addpath(genpath(fullfile(pwd,'matlab'))); exportTrainingData([], %SNAPSHOTS_PER_SNR%, false);"
) else (
    matlab -batch "cd('%PROJECT_ROOT%'); addpath(genpath(fullfile(pwd,'matlab'))); exportTrainingData();"
)
if errorlevel 1 goto :finish

echo.
echo ============================================================
echo   [2/5] Validate latest dataset
echo ============================================================
python validate_dataset.py
if errorlevel 1 goto :finish
call :find_latest_data || goto :fail

echo.
echo ============================================================
echo   [3/5] Train or reuse models
echo ============================================================
call :train_models || goto :finish

echo.
echo ============================================================
echo   [4/5] Smoke tests
echo ============================================================
call :smoke_impl || goto :finish

echo.
echo ============================================================
echo   [5/5] Main simulation
echo ============================================================
matlab -batch "cd('%PROJECT_ROOT%'); addpath(genpath(fullfile(pwd,'matlab'))); paths=getProjectPaths(); cd(paths.main); Combined_Downlink_Sim;"
goto :finish

:simulation
call :require_matlab || goto :fail
echo [INFO] Starting main simulation...
matlab -batch "cd('%PROJECT_ROOT%'); addpath(genpath(fullfile(pwd,'matlab'))); paths=getProjectPaths(); cd(paths.main); Combined_Downlink_Sim;"
goto :finish

:redraw
call :require_matlab || goto :fail
if not exist "%SIM_DATA_DIR%\Simulation_Results_v2.mat" (
    echo [ERROR] Simulation_Results_v2.mat not found. Run simulation first.
    goto :fail
)
echo [INFO] Redrawing figures from saved results...
matlab -batch "cd('%PROJECT_ROOT%'); addpath(genpath(fullfile(pwd,'matlab'))); savePath=fullfile(pwd,'main','Imgs'); dataPath=fullfile(pwd,'main','SimulationData'); simFile=fullfile(dataPath,'Simulation_Results_v2.mat'); syncFile=fullfile(dataPath,'Sync_Ablation_Results.mat'); S=load(simFile); if isfield(S,'Perf'), Perf=S.Perf; else, Perf=[]; end; plotESRResults_v2(S.ESR_mean,S.ESR_best,S.ESR_best_algo,S.algoTable,S.SNR_dB,0,true,savePath,false,dataPath,Perf); if isfile(syncFile), A=load(syncFile); plotLatencyAblationResults(A.Ablation,savePath,true,false,dataPath); else, warning('Sync_Ablation_Results.mat not found; skipped latency-ablation redraw.'); end"
goto :finish

:smoketests
call :require_matlab || goto :fail
call :require_python || goto :fail
call :smoke_impl
goto :finish

:validate
call :require_python || goto :fail
python validate_dataset.py
goto :finish

:exportquick
call :require_matlab || goto :fail
matlab -batch "cd('%PROJECT_ROOT%'); addpath(genpath(fullfile(pwd,'matlab'))); exportTrainingData([], 2, false);"
goto :finish

:matlabgui
call :require_matlab || goto :fail
start "" matlab -r "cd('%PROJECT_ROOT%'); addpath(genpath(fullfile(pwd,'matlab'))); disp('CF_downlink_sim ready. Run run.m or start_project.bat.');"
goto :finish

:envcheck
echo.
echo [INFO] Checking tools and folders...
where matlab >nul 2>&1
if errorlevel 1 (echo [MISS] MATLAB not found in PATH.) else for /f "delims=" %%p in ('where matlab') do echo [OK] MATLAB: %%p
where python >nul 2>&1
if errorlevel 1 (echo [MISS] Python not found in PATH.) else for /f "delims=" %%p in ('where python') do echo [OK] Python: %%p
if exist "%MATLAB_SRC%" (echo [OK] MATLAB source: %MATLAB_SRC%) else echo [MISS] MATLAB source: %MATLAB_SRC%
if exist "%PYTHON_DIR%" (echo [OK] Python source: %PYTHON_DIR%) else echo [MISS] Python source: %PYTHON_DIR%
if exist "%DATA_DIR%" (echo [OK] Training data dir: %DATA_DIR%) else echo [MISS] Training data dir: %DATA_DIR%
if exist "%MODELS_DIR%" (echo [OK] Models dir: %MODELS_DIR%) else echo [MISS] Models dir: %MODELS_DIR%
if /i "%~1"=="/check" exit /b 0
goto :finish

:train_models
if "%FORCE_RETRAIN%"=="1" echo [INFO] /retrain enabled. Existing models will be replaced.
call :train_one "%MODELS_DIR%\best_gat_gnn_power.pt" "Full-graph GNN" "python train_gnn.py --data ""%LATEST_DATA%"" --epochs %TRAIN_EPOCHS% --output_dir ""%MODELS_DIR%""" || exit /b 1
call :train_one "%MODELS_DIR%\best_dcgnn_power.pt" "DCGNN" "python train_gnn.py --data ""%LATEST_DATA%"" --epochs %TRAIN_EPOCHS% --output_dir ""%MODELS_DIR%"" --model_type dcgnn --dcgnn_top_z 15" || exit /b 1
call :train_one "%MODELS_DIR%\best_ugnn_power.pt" "U-GNN" "python train_gnn_unsup.py --data ""%LATEST_DATA%"" --epochs %TRAIN_EPOCHS% --output_dir ""%MODELS_DIR%""" || exit /b 1
call :train_one "%MODELS_DIR%\best_local_gnn_power.pt" "Local-GNN" "python train_gnn_local.py --data ""%LATEST_DATA%"" --epochs %TRAIN_EPOCHS% --output_dir ""%MODELS_DIR%""" || exit /b 1
set "SKIP_RL=0"
if "%FORCE_RETRAIN%"=="0" if exist "%MODELS_DIR%\best_dqn_power.pt" if exist "%MODELS_DIR%\best_ddpg_power.pt" set "SKIP_RL=1"
if "%SKIP_RL%"=="1" (
    echo [SKIP] DQN/DDPG already exist. Use /retrain to replace them.
) else (
    cd /d "%PYTHON_DIR%"
    python train_rl_power.py --data "%LATEST_DATA%" --method all --epochs %TRAIN_EPOCHS% --output_dir "%MODELS_DIR%"
    if errorlevel 1 exit /b 1
)
cd /d "%PROJECT_ROOT%"
exit /b 0

:train_one
set "MODEL_FILE=%~1"
set "MODEL_NAME=%~2"
set "TRAIN_CMD=%~3"
if exist "%MODEL_FILE%" if "%FORCE_RETRAIN%"=="0" (
    echo [SKIP] %MODEL_NAME% already exists: %MODEL_FILE%
    exit /b 0
)
echo [TRAIN] %MODEL_NAME%
cd /d "%PYTHON_DIR%"
%TRAIN_CMD%
set "TRAIN_EXIT=%ERRORLEVEL%"
cd /d "%PROJECT_ROOT%"
exit /b %TRAIN_EXIT%

:smoke_impl
echo [INFO] Python inference smoke test...
if exist "%MODELS_DIR%\best_gat_gnn_power.pt" (
    python -c "import sys, numpy as np; sys.path.insert(0, r'%PYTHON_DIR%'); import gnn_runtime; sg=np.abs(np.random.randn(100,20)).astype('float32'); D=np.ones((100,20), dtype='float32'); out=gnn_runtime.infer(r'%MODELS_DIR%\best_gat_gnn_power.pt', sg, D, 10.0, 0.3); rho=out['rho']; assert rho.shape==(100,20); assert np.isfinite(rho).all(); print('Python GNN smoke OK')"
    if errorlevel 1 exit /b 1
) else (
    echo [WARN] best_gat_gnn_power.pt not found. GNN will fall back to EPA in simulation.
)

echo [INFO] MATLAB bridge smoke test...
matlab -batch "cd('%PROJECT_ROOT%'); addpath(genpath(fullfile(pwd,'matlab'))); D=ones(100,20); gain=abs(rand(100,20)); Hhat=zeros(100,1,20); modelPath=fullfile(pwd,'models','best_gat_gnn_power.pt'); rhoF=computeRhoFPCP(D,gain,10,100,20,0.5); [rhoG,tG]=computeRhoGNN(Hhat,D,gain,10,modelPath,0.3); assert(all(isfinite(rhoF(:)))); assert(all(isfinite(rhoG(:)))); assert(abs(sum(rhoF(:))-1000)<1e-3); assert(abs(sum(rhoG(:))-1000)<1e-3); assert(isfield(tG,'forward_sec')); fprintf('MATLAB bridge smoke OK: FPCP=%%.6f, GNN=%%.6f\n', sum(rhoF(:)), sum(rhoG(:)));"
exit /b %ERRORLEVEL%

:find_latest_data
set "LATEST_DATA="
for /f "delims=" %%f in ('dir /b /a-d /o-d "%DATA_DIR%\gnn_training_data_*.mat" 2^>nul') do (
    if not defined LATEST_DATA set "LATEST_DATA=%DATA_DIR%\%%f"
)
if not defined LATEST_DATA (
    echo [ERROR] No training data found: %DATA_DIR%\gnn_training_data_*.mat
    exit /b 1
)
echo [INFO] Using training data: %LATEST_DATA%
exit /b 0

:require_matlab
where matlab >nul 2>&1
if errorlevel 1 (
    echo [ERROR] MATLAB not found in PATH.
    echo Add MATLAB bin to PATH, for example:
    echo   C:\Program Files\MATLAB\R2025a\bin
    exit /b 1
)
exit /b 0

:require_python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    exit /b 1
)
exit /b 0

:finish
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
    echo [OK] Workflow finished.
) else (
    echo [ERROR] Workflow failed. Exit code: %EXIT_CODE%
)
echo.
echo Common output folders:
echo   Figures: %FIGURE_DIR%
echo   Data:    %SIM_DATA_DIR%
echo   Models:  %MODELS_DIR%
echo.
if "%AUTO_Y%"=="0" pause
exit /b %EXIT_CODE%

:fail
if "%AUTO_Y%"=="0" pause
exit /b 1
