@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0"
set "PYTHON_DIR=%PROJECT_ROOT%python"
set "DATA_DIR=%PROJECT_ROOT%data\gnn_training"
set "MODELS_DIR=%PROJECT_ROOT%models"

echo.
echo ============================================================
echo           CF_downlink_sim Automated Test Script
echo ============================================================
echo.
echo   [1] Quick Test Mode (Recommended for first run)
echo       - 10 snapshots/SNR, 20 epochs
echo       - Estimated time: 10-30 minutes
echo       - Purpose: Quick validation of GNN effectiveness
echo.
echo   [2] Full Run Mode
echo       - 100 snapshots/SNR, 100 epochs
echo       - Estimated time: 1-2 hours
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
echo       - Validate dataset and test Python/MATLAB GNN inference
echo       - Skip data generation, training, and full simulation
echo.
echo   [Q] Exit
echo.
echo ============================================================

if not "%~1"=="" (
    set "MODE=%~1"
    echo Select option [1/2/3/4/5/Q]: %MODE%
) else (
    set /p MODE="Select option [1/2/3/4/5/Q]: "
)

if /i "!MODE!"=="Q" exit /b 0
if /i "!MODE!"=="q" exit /b 0

set SNAPSHOTS_PER_SNR=100
set GNN_EPOCHS=100
set SKIP_DATA_GEN=0
set SKIP_DATA_VALIDATE=0
set SKIP_GNN_TRAIN=0
set RUN_SIMULATION=1

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
    set /p SNAPSHOTS_PER_SNR="Snapshots per SNR [default 100]: "
    if "!SNAPSHOTS_PER_SNR!"=="" set SNAPSHOTS_PER_SNR=100
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

echo [ERROR] Invalid option!
pause
exit /b 1

:start
echo.
echo ============================================================
echo   Run Parameters
echo ============================================================
echo   Snapshots/SNR:  %SNAPSHOTS_PER_SNR%
echo   GNN epochs:     %GNN_EPOCHS%
echo   Skip data gen:  %SKIP_DATA_GEN%
echo   Skip data val:  %SKIP_DATA_VALIDATE%
echo   Skip GNN train: %SKIP_GNN_TRAIN%
echo   Run simulation: %RUN_SIMULATION%
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

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in PATH.
    pause
    exit /b 1
)

where matlab >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] MATLAB was not found in PATH.
    pause
    exit /b 1
)

set START_TIME=%TIME%

if %SKIP_DATA_GEN%==1 (
    echo.
    echo [SKIP] Step 1: Data generation skipped
    goto :step2
)

echo.
echo ============================================================
echo   [Step 1/3] Generating training dataset
echo ============================================================
echo.

set MATLAB_CMD=addpath(genpath(pwd)); exportTrainingData([], %SNAPSHOTS_PER_SNR%);
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
echo   [Step 2/6] Validating latest training dataset
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
    echo [SKIP] Step 3: GNN training skipped
    goto :step4
)

echo.
echo ============================================================
echo   [Step 3/6] Training GNN model
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
echo   [Step 4/6] Python GNN inference smoke test
echo ============================================================
echo.

if not exist "%MODELS_DIR%\best_gat_gnn_power.pt" (
    echo [ERROR] best_gat_gnn_power.pt not found in models directory.
    pause
    exit /b 1
)

cd /d "%PROJECT_ROOT%"
python -c "import sys, numpy as np; sys.path.insert(0, r'%PYTHON_DIR%'); from inference import GNNInferrer; inf=GNNInferrer(r'%MODELS_DIR%\best_gat_gnn_power.pt'); sg=np.abs(np.random.randn(100,20)); D=np.ones((100,20)); rho=inf.infer(sg,D,0.3,10.0); assert rho.shape==(100,20); assert np.isfinite(rho).all(); assert abs(float(rho.sum())-1000.0)<1e-6; print('Python inference smoke OK: rho sum=', float(rho.sum()))"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Python GNN inference smoke test failed!
    pause
    exit /b 1
)

echo.
echo [OK] Python GNN inference smoke test completed
echo.

echo.
echo ============================================================
echo   [Step 5/6] MATLAB GNN bridge smoke test
echo ============================================================
echo.

matlab -batch "cd('%PROJECT_ROOT%'); addpath(genpath(pwd)); D=ones(100,20); gain=abs(rand(100,20)); Hhat=zeros(100,1,20); modelPath=fullfile(pwd,'models','best_gat_gnn_power.pt'); [rho1,t1]=computeRhoGNN(Hhat,D,gain,10,modelPath,0.3); [rho2,t2]=computeRhoGNN(Hhat,D,gain,10,modelPath,0.3); assert(all(isfinite(rho1(:)))); assert(all(isfinite(rho2(:)))); assert(abs(sum(rho1(:))-1000)<1e-3); assert(abs(sum(rho2(:))-1000)<1e-3); assert(isfield(t2,'forward_sec') && t2.forward_sec > 0); fprintf('MATLAB bridge smoke OK: rho1 sum=%%.6f, rho2 sum=%%.6f, second forward=%%.6fs, second total=%%.6fs\n', sum(rho1(:)), sum(rho2(:)), t2.forward_sec, t2.total_sec);"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] MATLAB GNN bridge smoke test failed!
    pause
    exit /b 1
)

echo.
echo [OK] MATLAB GNN bridge smoke test completed
echo.

:step6
if %RUN_SIMULATION%==0 (
    echo.
    echo [SKIP] Step 6: Full simulation skipped
    goto :done
)

echo.
echo ============================================================
echo   [Step 6/6] Running Cell-Free simulation
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
echo   Validation criteria:
echo     GNN ESR ^>= Dist ^> EPA  = Success
echo     GNN ESR ^<  EPA          = Still has issues
echo.
echo   Smoke tests completed:
echo     - Dataset validation
echo     - Python GNN inference
echo     - MATLAB computeRhoGNN bridge with cached second call
echo ============================================================
echo.

cd /d "%PROJECT_ROOT%"
if /i not "%~2"=="/y" pause
endlocal
