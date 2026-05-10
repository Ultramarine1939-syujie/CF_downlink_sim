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
echo       - 10 snapshots/SNR, 20 epochs, 5 FL rounds
echo       - Estimated time: 10-20 minutes
echo       - Purpose: Quick validation of GNN effectiveness
echo.
echo   [2] Full Run Mode
echo       - 100 snapshots/SNR, 100 epochs, 30 FL rounds
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
echo   [Q] Exit
echo.
echo ============================================================

set /p MODE="Select option [1/2/3/4/Q]: "

if /i "%MODE%"=="Q" exit /b 0
if /i "%MODE%"=="q" exit /b 0

set SNAPSHOTS_PER_SNR=100
set GNN_EPOCHS=100
set FEDAVG_ROUNDS=30
set SKIP_DATA_GEN=0
set SKIP_GNN_TRAIN=0
set SKIP_FEDAVG=0

if "%MODE%"=="1" (
    echo.
    echo [Selected] Quick Test Mode
    set SNAPSHOTS_PER_SNR=10
    set GNN_EPOCHS=20
    set FEDAVG_ROUNDS=5
    goto :start
)

if "%MODE%"=="2" (
    echo.
    echo [Selected] Full Run Mode
    goto :start
)

if "%MODE%"=="3" (
    echo.
    echo [Selected] Custom Mode
    echo.
    set /p SNAPSHOTS_PER_SNR="Snapshots per SNR [default 100]: "
    if "!SNAPSHOTS_PER_SNR!"=="" set SNAPSHOTS_PER_SNR=100
    set /p GNN_EPOCHS="GNN training epochs [default 100]: "
    if "!GNN_EPOCHS!"=="" set GNN_EPOCHS=100
    set /p FEDAVG_ROUNDS="FedAvg rounds [default 30]: "
    if "!FEDAVG_ROUNDS!"=="" set FEDAVG_ROUNDS=30
    goto :start
)

if "%MODE%"=="4" (
    echo.
    echo [Selected] Simulation Only Mode
    set SKIP_DATA_GEN=1
    set SKIP_GNN_TRAIN=1
    set SKIP_FEDAVG=1
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
echo   FedAvg rounds:  %FEDAVG_ROUNDS%
echo   Skip data gen:  %SKIP_DATA_GEN%
echo   Skip GNN train: %SKIP_GNN_TRAIN%
echo   Skip FedAvg:    %SKIP_FEDAVG%
echo ============================================================
echo.

set /p CONFIRM="Confirm to start? [Y/n]: "
if /i "%CONFIRM%"=="n" (
    echo Cancelled.
    exit /b 0
)

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

set START_TIME=%TIME%

if %SKIP_DATA_GEN%==1 (
    echo.
    echo [SKIP] Step 1: Data generation skipped
    goto :step2
)

echo.
echo ============================================================
echo   [Step 1/4] Generating training dataset
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
if %SKIP_GNN_TRAIN%==1 (
    echo.
    echo [SKIP] Step 2: GNN training skipped
    goto :step3
)

echo.
echo ============================================================
echo   [Step 2/4] Training GNN model
echo ============================================================
echo.

cd /d "%PYTHON_DIR%"

set LATEST_DATA=
for /f "delims=" %%f in ('dir /b /o-d "%DATA_DIR%\gnn_training_data_*.mat" 2^>nul') do (
    set LATEST_DATA=%DATA_DIR%\%%f
)
if not defined LATEST_DATA (
    echo [ERROR] Training data file not found!
    pause
    exit /b 1
)
echo Using data file: %LATEST_DATA%
echo.

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

:step3
if %SKIP_FEDAVG%==1 (
    echo.
    echo [SKIP] Step 3: FedAvg skipped
    goto :step4
)

echo.
echo ============================================================
echo   [Step 3/4] Federated learning fine-tuning
echo ============================================================
echo.

cd /d "%PYTHON_DIR%"

set GNN_MODEL=%MODELS_DIR%\best_gat_gnn_power.pt
if not exist "%GNN_MODEL%" (
    echo [WARN] GNN model not found, skipping FedAvg
    goto :step4
)

echo Using GNN model: %GNN_MODEL%
echo.

python fedavg.py --data "%LATEST_DATA%" --rounds %FEDAVG_ROUNDS% --init_ckpt "%GNN_MODEL%" --output_dir "%MODELS_DIR%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARN] FedAvg failed, continuing with simulation...
)

echo.
echo [OK] FedAvg completed
echo.

:step4
echo.
echo ============================================================
echo   [Step 4/4] Running Cell-Free simulation
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
echo     GNN ESR >= Dist > EPA  = Success
echo     GNN ESR <  EPA         = Still has issues
echo ============================================================
echo.

cd /d "%PROJECT_ROOT%"
pause
endlocal
