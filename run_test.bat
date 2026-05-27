@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0"
set "PYTHON_DIR=%PROJECT_ROOT%python"
set "DATA_DIR=%PROJECT_ROOT%data\gnn_training"
set "MODELS_DIR=%PROJECT_ROOT%models"
set "GNN_MODEL=%MODELS_DIR%\best_gat_gnn_power.pt"
set "DCGNN_MODEL=%MODELS_DIR%\best_dcgnn_power.pt"
set "UGNN_MODEL=%MODELS_DIR%\best_ugnn_power.pt"
set "LOCAL_GNN_MODEL=%MODELS_DIR%\best_local_gnn_power.pt"
set "DQN_MODEL=%MODELS_DIR%\best_dqn_power.pt"
set "DDPG_MODEL=%MODELS_DIR%\best_ddpg_power.pt"

echo.
echo ============================================================
echo           CF_downlink_sim Automated Test Script
echo ============================================================
echo.
echo   [1] Quick Test Mode (Recommended for first run)
echo       - 10 snapshots/SNR, 20 epochs
echo       - Estimated time: 10-30 minutes
echo       - Purpose: Quick validation of GNN, DCGNN, U-GNN, Local-GNN, DQN, and DDPG effectiveness
echo.
echo   [2] Full Run Mode
echo       - snapshots/SNR from config/getDefaultParams.m, 100 epochs
echo       - Estimated time: depends on config/getDefaultParams.m
echo       - Purpose: Generate paper-quality results
echo.
echo   [3] Custom Mode
echo       - Manually set parameters
echo.
echo   [4] Simulation Only (Use existing models)
echo       - Skip data generation and model training
echo       - Estimated time: 5-10 minutes
echo.
echo   [5] Smoke Tests Only (Use existing data/model)
echo       - Validate dataset and test Python/MATLAB GNN + DCGNN + U-GNN + Local-GNN + RL inference
echo       - Skip data generation, training, and full simulation
echo.
echo   [6] Redraw Figures Only (Use existing simulation results)
echo       - Replot figures from main\SimulationData\
echo       - Skip data generation, training, smoke tests, and full simulation
echo.
echo   [Q] Exit
echo.
echo ============================================================

if not "%~1"=="" (
    set "MODE=%~1"
    echo Select option [1/2/3/4/5/6/Q]: !MODE!
) else (
    set /p MODE="Select option [1/2/3/4/5/6/Q]: "
)

if /i "!MODE!"=="Q" exit /b 0
if /i "!MODE!"=="q" exit /b 0

set FORCE_RETRAIN=0
if /i "%~1"=="/retrain" set FORCE_RETRAIN=1
if /i "%~2"=="/retrain" set FORCE_RETRAIN=1
if /i "%~3"=="/retrain" set FORCE_RETRAIN=1
if /i "%~4"=="/retrain" set FORCE_RETRAIN=1

set SNAPSHOTS_PER_SNR=
set GNN_EPOCHS=100
set SKIP_DATA_GEN=0
set SKIP_DATA_VALIDATE=0
set SKIP_GNN_TRAIN=0
set SKIP_DCGNN_TRAIN=0
set SKIP_LOCAL_GNN_TRAIN=0
set SKIP_RL_TRAIN=0
set RUN_SIMULATION=1
set REDRAW_FIGURES_ONLY=0

if "!MODE!"=="1" (
    echo.
    echo [Selected] Quick Test Mode
    set SNAPSHOTS_PER_SNR=10
    set GNN_EPOCHS=20
    goto :start
)

if "!MODE!"=="2" (
    echo.
    echo [Selected] Full Run Mode
    goto :start
)

if "!MODE!"=="3" (
    echo.
    echo [Selected] Custom Mode
    echo.
    set /p SNAPSHOTS_PER_SNR="Snapshots per SNR [blank = config default]: "
    set /p GNN_EPOCHS="GNN training epochs [default 100]: "
    if "!GNN_EPOCHS!"=="" set GNN_EPOCHS=100
    goto :start
)

if "!MODE!"=="4" (
    echo.
    echo [Selected] Simulation Only Mode
    set SKIP_DATA_GEN=1
    set SKIP_DATA_VALIDATE=1
    set SKIP_GNN_TRAIN=1
    goto :start
)

if "!MODE!"=="5" (
    echo.
    echo [Selected] Smoke Tests Only Mode
    set SKIP_DATA_GEN=1
    set SKIP_GNN_TRAIN=1
    set RUN_SIMULATION=0
    goto :start
)

if "!MODE!"=="6" (
    echo.
    echo [Selected] Redraw Figures Only Mode
    set SKIP_DATA_GEN=1
    set SKIP_DATA_VALIDATE=1
    set SKIP_GNN_TRAIN=1
    set RUN_SIMULATION=0
    set REDRAW_FIGURES_ONLY=1
    goto :start
)

echo [ERROR] Invalid option!
pause
exit /b 1

:start
echo.
echo ============================================================
echo   Run Parameters
echo ============================================================
if defined SNAPSHOTS_PER_SNR (
echo   Snapshots/SNR:  %SNAPSHOTS_PER_SNR%
) else (
echo   Snapshots/SNR:  config default
)
echo   GNN epochs:     %GNN_EPOCHS%
echo   DCGNN epochs:   %GNN_EPOCHS%
echo   U-GNN epochs:   %GNN_EPOCHS%
echo   Local-GNN epochs: %GNN_EPOCHS%
echo   DQN/DDPG epochs: %GNN_EPOCHS%
echo   Skip data gen:  %SKIP_DATA_GEN%
echo   Skip data val:  %SKIP_DATA_VALIDATE%
echo   Skip GNN train: %SKIP_GNN_TRAIN%
echo   Force retrain:  %FORCE_RETRAIN%
echo   GNN model:      auto-skip existing best_gat_gnn_power.pt
echo   DCGNN model:    auto-skip existing best_dcgnn_power.pt
echo   U-GNN model:    auto-skip existing best_ugnn_power.pt
echo   Local model:    auto-skip existing best_local_gnn_power.pt
echo   RL models:      auto-skip existing best_dqn_power.pt and best_ddpg_power.pt
echo   Run simulation: %RUN_SIMULATION%
echo   Redraw only:    %REDRAW_FIGURES_ONLY%
echo ============================================================
echo.

if /i "%~2"=="/y" (
    set "CONFIRM=Y"
    echo Confirm to start? [Y/n]: Y
) else (
    set /p CONFIRM="Confirm to start? [Y/n]: "
)
if /i "%CONFIRM%"=="n" (
    echo Cancelled.
    exit /b 0
)

set START_TIME=%TIME%

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

where matlab >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] MATLAB was not found in PATH.
    pause
    exit /b 1
)

if %REDRAW_FIGURES_ONLY%==1 goto :redrawfigures

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in PATH.
    pause
    exit /b 1
)

set LATEST_DATA=
for /f "delims=" %%f in ('dir /b /a-d /o-d "%DATA_DIR%\gnn_training_data_*.mat" 2^>nul') do (
    if not defined LATEST_DATA set "LATEST_DATA=%DATA_DIR%\%%f"
)

if %SKIP_DATA_GEN%==0 (
    if defined LATEST_DATA (
        echo.
        echo Existing training dataset found:
        echo   !LATEST_DATA!
        if /i "%~2"=="/y" (
            echo Auto-confirm enabled: generating a new dataset.
        ) else (
            set "GENERATE_NEW_DATA="
            set /p GENERATE_NEW_DATA="Generate a new training dataset? [y/N]: "
            if /i not "!GENERATE_NEW_DATA!"=="y" (
                set SKIP_DATA_GEN=1
                echo Reusing existing dataset for training.
            )
        )
    )
)

if %SKIP_DATA_GEN%==1 (
    echo.
    echo [SKIP] Step 1: Data generation skipped
    goto :step2
)

echo.
echo ============================================================
echo   [Step 1/9] Generating training dataset
echo ============================================================
echo.

if defined SNAPSHOTS_PER_SNR (
    set "MATLAB_CMD=addpath(genpath(pwd)); exportTrainingData([], %SNAPSHOTS_PER_SNR%);"
) else (
    set "MATLAB_CMD=addpath(genpath(pwd)); exportTrainingData();"
)
matlab -batch "cd('%PROJECT_ROOT%'); %MATLAB_CMD%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Data generation failed!
    pause
    exit /b 1
)

echo.
echo [OK] Data generation completed
echo.

:step2
if %SKIP_DATA_VALIDATE%==1 (
    echo.
    echo [SKIP] Step 2: Dataset validation skipped
    goto :step3
)

echo.
echo ============================================================
echo   [Step 2/9] Validating latest training dataset
echo ============================================================
echo.

set LATEST_DATA=
for /f "delims=" %%f in ('dir /b /a-d /o-d "%DATA_DIR%\gnn_training_data_*.mat" 2^>nul') do (
    if not defined LATEST_DATA set "LATEST_DATA=%DATA_DIR%\%%f"
)
if not defined LATEST_DATA (
    echo [ERROR] Training data file not found!
    pause
    exit /b 1
)
echo Using data file: %LATEST_DATA%
echo.

cd /d "%PROJECT_ROOT%"
python validate_dataset.py "%LATEST_DATA%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Dataset validation failed!
    pause
    exit /b 1
)

echo.
echo [OK] Dataset validation completed
echo.

:step3
if %SKIP_GNN_TRAIN%==1 (
    echo.
    echo [SKIP] Step 3-6: GNN/DCGNN/U-GNN/RL/Local-GNN training skipped
    goto :step7
)

echo.
echo ============================================================
echo   [Step 3/9] Checking full-graph GNN model
echo ============================================================
echo.

if exist "%GNN_MODEL%" if %FORCE_RETRAIN%==0 (
    echo Existing full-graph GNN model found:
    for %%F in ("%GNN_MODEL%") do (
        echo   Path: %%~fF
        echo   Size: %%~zF bytes
        echo   Last modified: %%~tF
    )
    echo.
    echo Reusing existing full-graph GNN model. Use /retrain to replace it.
    echo.
    echo [SKIP] Step 3: Full-graph GNN training skipped
    goto :step4
)

echo.
echo ============================================================
echo   [Step 3/9] Training full-graph GNN model
echo ============================================================
echo.

cd /d "%PYTHON_DIR%"
python train_gnn.py --data "%LATEST_DATA%" --epochs %GNN_EPOCHS% --output_dir "%MODELS_DIR%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] GNN training failed!
    pause
    exit /b 1
)

echo.
echo [OK] GNN training completed
echo.

:step4
echo.
echo ============================================================
echo   [Step 4/9] Checking DCGNN model
echo ============================================================
echo.

set SKIP_DCGNN_TRAIN=0
if exist "%DCGNN_MODEL%" (
    echo Existing DCGNN model found:
    for %%F in ("%DCGNN_MODEL%") do (
        echo   Path: %%~fF
        echo   Size: %%~zF bytes
        echo   Last modified: %%~tF
    )
    echo.
    if %FORCE_RETRAIN%==1 (
        echo Force retrain enabled: retraining DCGNN and replacing the existing model.
    ) else (
        set SKIP_DCGNN_TRAIN=1
        echo Reusing existing DCGNN model. Use /retrain to replace it.
    )
)

if !SKIP_DCGNN_TRAIN!==1 (
    echo.
    echo [SKIP] Step 4: DCGNN training skipped
    goto :step5
)

echo.
echo ============================================================
echo   [Step 4/9] Training dynamically-connected DCGNN model
echo ============================================================
echo.

cd /d "%PYTHON_DIR%"
python train_gnn.py --data "%LATEST_DATA%" --epochs %GNN_EPOCHS% --output_dir "%MODELS_DIR%" --model_type dcgnn --dcgnn_top_z 15

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] DCGNN training failed!
    pause
    exit /b 1
)

echo.
echo [OK] DCGNN training completed
echo.

:step5
echo.
echo ============================================================
echo   [Step 5/9] Checking U-GNN and DQN/DDPG models
echo ============================================================
echo.

set SKIP_UGNN_TRAIN=0
if exist "%UGNN_MODEL%" (
    echo Existing U-GNN model found:
    for %%F in ("%UGNN_MODEL%") do (
        echo   Path: %%~fF
        echo   Size: %%~zF bytes
        echo   Last modified: %%~tF
    )
    echo.
    if %FORCE_RETRAIN%==1 (
        echo Force retrain enabled: retraining U-GNN and replacing the existing model.
    ) else (
        set SKIP_UGNN_TRAIN=1
        echo Reusing existing U-GNN model. Use /retrain to replace it.
    )
)

if !SKIP_UGNN_TRAIN!==1 (
    echo.
    echo [SKIP] Step 5: U-GNN training skipped
) else (
    echo.
    echo ============================================================
    echo   [Step 5/9] Training teacher-free U-GNN model
    echo ============================================================
    echo.

    cd /d "%PYTHON_DIR%"
    python train_gnn_unsup.py --data "%LATEST_DATA%" --epochs %GNN_EPOCHS% --output_dir "%MODELS_DIR%"

    if !ERRORLEVEL! neq 0 (
        echo.
        echo [ERROR] U-GNN training failed!
        pause
        exit /b 1
    )

    echo.
    echo [OK] U-GNN training completed
    echo.
)

set SKIP_RL_TRAIN=0
if exist "%DQN_MODEL%" if exist "%DDPG_MODEL%" (
    echo Existing DQN/DDPG models found:
    for %%F in ("%DQN_MODEL%" "%DDPG_MODEL%") do (
        echo   Path: %%~fF
        echo   Size: %%~zF bytes
        echo   Last modified: %%~tF
    )
    echo.
    if %FORCE_RETRAIN%==1 (
        echo Force retrain enabled: retraining DQN/DDPG and replacing the existing models.
    ) else (
        set SKIP_RL_TRAIN=1
        echo Reusing existing DQN/DDPG models. Use /retrain to replace them.
    )
)

if !SKIP_RL_TRAIN!==1 (
    echo.
    echo [SKIP] Step 5: DQN/DDPG training skipped
    goto :step6
)

echo.
echo ============================================================
echo   [Step 5/9] Training DQN/DDPG reward-based RL models
echo ============================================================
echo.

cd /d "%PYTHON_DIR%"
python train_rl_power.py --data "%LATEST_DATA%" --method all --epochs %GNN_EPOCHS% --output_dir "%MODELS_DIR%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] DQN/DDPG training failed!
    pause
    exit /b 1
)

echo.
echo [OK] DQN/DDPG training completed
echo.

:step6
echo.
echo ============================================================
echo   [Step 6/9] Checking AP-local GNN model
echo ============================================================
echo.

set SKIP_LOCAL_GNN_TRAIN=0
if exist "%LOCAL_GNN_MODEL%" (
    echo Existing Local-GNN model found:
    for %%F in ("%LOCAL_GNN_MODEL%") do (
        echo   Path: %%~fF
        echo   Size: %%~zF bytes
        echo   Last modified: %%~tF
    )
    echo.
    if %FORCE_RETRAIN%==1 (
        echo Force retrain enabled: retraining Local-GNN and replacing the existing model.
    ) else (
        set SKIP_LOCAL_GNN_TRAIN=1
        echo Reusing existing Local-GNN model. Use /retrain to replace it.
    )
)

if !SKIP_LOCAL_GNN_TRAIN!==1 (
    echo.
    echo [SKIP] Step 6: Local-GNN training skipped
    goto :step7
)

echo.
echo ============================================================
echo   [Step 6/9] Training AP-local GNN model
echo ============================================================
echo.

cd /d "%PYTHON_DIR%"
python train_gnn_local.py --data "%LATEST_DATA%" --epochs %GNN_EPOCHS% --output_dir "%MODELS_DIR%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Local-GNN training failed!
    pause
    exit /b 1
)

echo.
echo [OK] Local-GNN training completed
echo.

:step7
echo.
echo ============================================================
echo   [Step 7/9] Python GNN/RL inference smoke tests
echo ============================================================
echo.

if exist "%MODELS_DIR%\best_gat_gnn_power.pt" (
    cd /d "%PROJECT_ROOT%"
    python -c "import sys, numpy as np; sys.path.insert(0, r'%PYTHON_DIR%'); import gnn_runtime; sg=np.abs(np.random.randn(100,20)).astype('float32'); D=np.ones((100,20), dtype='float32'); out=gnn_runtime.infer(r'%MODELS_DIR%\best_gat_gnn_power.pt', sg, D, 10.0, 0.3); rho=out['rho']; assert rho.shape==(100,20); assert np.isfinite(rho).all(); assert np.allclose(rho.sum(axis=1), 10.0, atol=1e-5); print('Python full-graph GNN smoke OK: rho sum=', float(rho.sum()))"

    if !ERRORLEVEL! neq 0 (
        echo.
        echo [ERROR] Python full-graph GNN inference smoke test failed!
        pause
        exit /b 1
    )
) else (
    echo [WARN] best_gat_gnn_power.pt not found; full-graph GNN will fall back to EPA in MATLAB simulation.
)

if exist "%MODELS_DIR%\best_dcgnn_power.pt" (
    cd /d "%PROJECT_ROOT%"
    python -c "import sys, numpy as np; sys.path.insert(0, r'%PYTHON_DIR%'); import gnn_runtime; sg=np.abs(np.random.randn(100,20)).astype('float32'); D=np.ones((100,20), dtype='float32'); out=gnn_runtime.infer(r'%MODELS_DIR%\best_dcgnn_power.pt', sg, D, 10.0, 0.3); rho=out['rho']; assert rho.shape==(100,20); assert np.isfinite(rho).all(); assert np.allclose(rho.sum(axis=1), 10.0, atol=1e-5); print('Python DCGNN smoke OK: rho sum=', float(rho.sum()), 'top_z model tested')"

    if !ERRORLEVEL! neq 0 (
        echo.
        echo [ERROR] Python DCGNN inference smoke test failed!
        pause
        exit /b 1
    )
) else (
    echo [WARN] best_dcgnn_power.pt not found; DCGNN will fall back to EPA in MATLAB simulation.
)

if exist "%MODELS_DIR%\best_ugnn_power.pt" (
    cd /d "%PROJECT_ROOT%"
    python -c "import sys, numpy as np; sys.path.insert(0, r'%PYTHON_DIR%'); import gnn_runtime; sg=np.abs(np.random.randn(100,20)).astype('float32'); D=np.ones((100,20), dtype='float32'); out=gnn_runtime.infer(r'%MODELS_DIR%\best_ugnn_power.pt', sg, D, 10.0, 0.3); rho=out['rho']; assert rho.shape==(100,20); assert np.isfinite(rho).all(); assert np.allclose(rho.sum(axis=1), 10.0, atol=1e-5); print('Python U-GNN smoke OK: rho sum=', float(rho.sum()))"

    if !ERRORLEVEL! neq 0 (
        echo.
        echo [ERROR] Python U-GNN inference smoke test failed!
        pause
        exit /b 1
    )
) else (
    echo [WARN] best_ugnn_power.pt not found; U-GNN will fall back to EPA in MATLAB simulation.
)

if exist "%MODELS_DIR%\best_dqn_power.pt" (
    cd /d "%PROJECT_ROOT%"
    python -c "import sys, numpy as np; sys.path.insert(0, r'%PYTHON_DIR%'); import rl_runtime; sg=np.abs(np.random.randn(100,20)).astype('float32'); D=np.ones((100,20), dtype='float32'); out=rl_runtime.infer(r'%MODELS_DIR%\best_dqn_power.pt', sg, D, 10.0, 0.3); rho=out['rho']; assert rho.shape==(100,20); assert np.isfinite(rho).all(); assert np.allclose(rho.sum(axis=1), 10.0, atol=1e-5); print('Python DQN smoke OK: rho sum=', float(rho.sum()), 'action=', out.get('action_idx'), 'alpha=', out.get('alpha'))"

    if !ERRORLEVEL! neq 0 (
        echo.
        echo [ERROR] Python DQN inference smoke test failed!
        pause
        exit /b 1
    )
) else (
    echo [WARN] best_dqn_power.pt not found; DQN will fall back to EPA in MATLAB simulation.
)

if exist "%MODELS_DIR%\best_ddpg_power.pt" (
    cd /d "%PROJECT_ROOT%"
    python -c "import sys, numpy as np; sys.path.insert(0, r'%PYTHON_DIR%'); import rl_runtime; sg=np.abs(np.random.randn(100,20)).astype('float32'); D=np.ones((100,20), dtype='float32'); out=rl_runtime.infer(r'%MODELS_DIR%\best_ddpg_power.pt', sg, D, 10.0, 0.3); rho=out['rho']; assert rho.shape==(100,20); assert np.isfinite(rho).all(); assert np.allclose(rho.sum(axis=1), 10.0, atol=1e-5); print('Python DDPG smoke OK: rho sum=', float(rho.sum()))"

    if !ERRORLEVEL! neq 0 (
        echo.
        echo [ERROR] Python DDPG inference smoke test failed!
        pause
        exit /b 1
    )
) else (
    echo [WARN] best_ddpg_power.pt not found; DDPG will fall back to EPA in MATLAB simulation.
)

if exist "%MODELS_DIR%\best_local_gnn_power.pt" (
    cd /d "%PROJECT_ROOT%"
    python -c "import sys, numpy as np; sys.path.insert(0, r'%PYTHON_DIR%'); import gnn_runtime_local; sg=np.abs(np.random.randn(100,20)).astype('float32'); D=np.ones((100,20), dtype='float32'); out=gnn_runtime_local.infer(r'%MODELS_DIR%\best_local_gnn_power.pt', sg, D, 10.0, 0.3); rho=out['rho']; assert rho.shape==(100,20); assert np.isfinite(rho).all(); assert np.allclose(rho.sum(axis=1), 10.0, atol=1e-5); print('Python Local-GNN smoke OK: rho sum=', float(rho.sum()))"

    if !ERRORLEVEL! neq 0 (
        echo.
        echo [ERROR] Python Local-GNN inference smoke test failed!
        pause
        exit /b 1
    )
) else (
    echo [WARN] best_local_gnn_power.pt not found; Local-GNN will fall back to EPA in MATLAB simulation.
)

echo.
echo [OK] Python inference smoke tests completed
echo.

echo.
echo ============================================================
echo   [Step 8/9] MATLAB GNN/RL bridge smoke tests
echo ============================================================
echo.

matlab -batch "cd('%PROJECT_ROOT%'); addpath(genpath(pwd)); D=ones(100,20); gain=abs(rand(100,20)); Hhat=zeros(100,1,20); modelPath=fullfile(pwd,'models','best_gat_gnn_power.pt'); dcgnnPath=fullfile(pwd,'models','best_dcgnn_power.pt'); ugnnPath=fullfile(pwd,'models','best_ugnn_power.pt'); localPath=fullfile(pwd,'models','best_local_gnn_power.pt'); dqnPath=fullfile(pwd,'models','best_dqn_power.pt'); ddpgPath=fullfile(pwd,'models','best_ddpg_power.pt'); rhoF=computeRhoFPCP(D,gain,10,100,20,0.5); [rho1,t1]=computeRhoGNN(Hhat,D,gain,10,modelPath,0.3); [rho2,t2]=computeRhoGNN(Hhat,D,gain,10,modelPath,0.3); [rhoD,tD]=computeRhoGNN(Hhat,D,gain,10,dcgnnPath,0.3); [rhoU,tU]=computeRhoUGNN(Hhat,D,gain,10,ugnnPath,0.3); [rhoL,tL]=computeRhoLocalGNN(Hhat,D,gain,10,localPath,0.3); [rhoQ,tQ]=computeRhoRL(Hhat,D,gain,10,dqnPath,0.3); [rhoP,tP]=computeRhoRL(Hhat,D,gain,10,ddpgPath,0.3); assert(all(isfinite(rhoF(:)))); assert(all(isfinite(rho1(:)))); assert(all(isfinite(rho2(:)))); assert(all(isfinite(rhoD(:)))); assert(all(isfinite(rhoU(:)))); assert(all(isfinite(rhoL(:)))); assert(all(isfinite(rhoQ(:)))); assert(all(isfinite(rhoP(:)))); assert(abs(sum(rhoF(:))-1000)<1e-3); assert(abs(sum(rho1(:))-1000)<1e-3); assert(abs(sum(rho2(:))-1000)<1e-3); assert(abs(sum(rhoD(:))-1000)<1e-3); assert(abs(sum(rhoU(:))-1000)<1e-3); assert(abs(sum(rhoL(:))-1000)<1e-3); assert(abs(sum(rhoQ(:))-1000)<1e-3); assert(abs(sum(rhoP(:))-1000)<1e-3); assert(isfield(t2,'forward_sec') && t2.forward_sec > 0); assert(isfield(tD,'forward_sec') && tD.forward_sec > 0); assert(isfield(tU,'forward_sec') && tU.forward_sec >= 0); assert(isfield(tL,'forward_sec') && tL.forward_sec > 0); assert(isfield(tQ,'forward_sec') && tQ.forward_sec > 0); assert(isfield(tP,'forward_sec') && tP.forward_sec > 0); fprintf('MATLAB bridge smoke OK: FPCP=%%.6f, GNN=%%.6f, DCGNN=%%.6f, U-GNN=%%.6f, Local-GNN=%%.6f, DQN=%%.6f, DDPG=%%.6f\n', sum(rhoF(:)), sum(rho2(:)), sum(rhoD(:)), sum(rhoU(:)), sum(rhoL(:)), sum(rhoQ(:)), sum(rhoP(:)));"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] MATLAB GNN bridge smoke tests failed!
    pause
    exit /b 1
)

echo.
echo [OK] MATLAB GNN bridge smoke tests completed
echo.

:step9
if %RUN_SIMULATION%==0 (
    echo.
    echo [SKIP] Step 9: Full simulation skipped
    goto :done
)

echo.
echo ============================================================
echo   [Step 9/9] Running Cell-Free simulation
echo ============================================================
echo.

cd /d "%PROJECT_ROOT%"
matlab -batch "run"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Simulation failed!
    pause
    exit /b 1
)

echo.
echo [OK] Simulation completed
echo.

goto :done

:redrawfigures
echo.
echo ============================================================
echo   [Redraw] Regenerating figures from saved simulation results
echo ============================================================
echo.

if not exist "%PROJECT_ROOT%main\SimulationData\Simulation_Results_v2.mat" (
    echo [ERROR] Simulation_Results_v2.mat not found.
    echo Please run a full simulation first.
    pause
    exit /b 1
)

cd /d "%PROJECT_ROOT%"
matlab -batch "cd('%PROJECT_ROOT%'); addpath(genpath(pwd)); savePath=fullfile(pwd,'main','Imgs'); dataPath=fullfile(pwd,'main','SimulationData'); simFile=fullfile(dataPath,'Simulation_Results_v2.mat'); syncFile=fullfile(dataPath,'Sync_Ablation_Results.mat'); S=load(simFile); if isfield(S,'Perf'), Perf=S.Perf; else, Perf=[]; end; plotESRResults_v2(S.ESR_mean,S.ESR_best,S.ESR_best_algo,S.algoTable,S.SNR_dB,0,true,savePath,false,dataPath,Perf); if isfile(syncFile), A=load(syncFile); plotLatencyAblationResults(A.Ablation,savePath,true,false,dataPath); else, warning('Sync_Ablation_Results.mat not found; skipped A5/A6/A7/A8/A9 redraw.'); end"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Figure redraw failed!
    pause
    exit /b 1
)

echo.
echo [OK] Figure redraw completed
echo.

:done
echo.
echo ============================================================
echo   Test Completed!
echo ============================================================
echo   Start time: %START_TIME%
echo   End time:   %TIME%
echo.
echo   Output files:
echo     - Figures: main\Imgs\
echo     - Data:    main\SimulationData\
echo     - Models:  models\
echo.
if %REDRAW_FIGURES_ONLY%==1 (
    echo   Redraw completed:
    echo     - ESR figures from Simulation_Results_v2.mat
    echo     - Sync ablation figures from Sync_Ablation_Results.mat when available
) else (
    echo   Validation criteria:
    echo     Figures include Baseline/Random/EPA/FPCP/D-WMMSE/WMMSE/GNN/Local-GNN/DCGNN/U-GNN/DQN/DDPG
    echo     Local-GNN enters distributed main ranking = distributed learning candidate active
    echo     DCGNN appears in learning-family plots    = dynamic graph model active
    echo     U-GNN appears in learning-family plots     = unsupervised teacher-free model active
    echo     DQN/DDPG appear in learning-family plots  = RL baselines active
    echo     Missing .pt model                         = corresponding learned method falls back to EPA
    echo.
    echo   Smoke tests completed:
    echo     - Dataset validation
    echo     - Python full-graph GNN inference
    echo     - Python DCGNN inference when model exists
    echo     - Python U-GNN inference when model exists
    echo     - Python DQN/DDPG inference when models exist
    echo     - Python Local-GNN inference when model exists
    echo     - MATLAB computeRhoFPCP, computeRhoGNN, computeRhoUGNN, computeRhoLocalGNN, and computeRhoRL bridges
)
echo ============================================================
echo.

cd /d "%PROJECT_ROOT%"
if /i not "%~2"=="/y" pause
endlocal
