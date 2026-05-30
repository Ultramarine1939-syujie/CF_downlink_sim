@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Pure-Python full pipeline launcher for CF_downlink_sim.
rem Usage:
rem   run_python_full_pipeline.bat [small|full] [/retrain] [/skip-rl] [/y]
rem
rem Defaults to "small" so a first run finishes quickly. Use "full" for the
rem default simulation scale from python/cf_sim_core.py.

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "PYTHON_DIR=%ROOT%\python"
set "MODELS_DIR=%ROOT%\models"
set "DATA_DIR=%ROOT%\data\gnn_training"
set "MODE=small"
set "FORCE_RETRAIN=0"
set "SKIP_RL=0"
set "NO_PAUSE=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="/help" goto usage
if /I "%~1"=="--help" goto usage
if /I "%~1"=="-h" goto usage
if /I "%~1"=="small" set "MODE=small"
if /I "%~1"=="full" set "MODE=full"
if /I "%~1"=="/retrain" set "FORCE_RETRAIN=1"
if /I "%~1"=="--retrain" set "FORCE_RETRAIN=1"
if /I "%~1"=="/skip-rl" set "SKIP_RL=1"
if /I "%~1"=="--skip-rl" set "SKIP_RL=1"
if /I "%~1"=="/y" set "NO_PAUSE=1"
shift
goto parse_args
:args_done

if /I "%MODE%"=="small" (
    set "EXPORT_ARGS=--snapshots-per-snr 2 --snr-db 5 10 15 20 25 30 --realizations 10 --setups 1"
    set "TRAIN_EPOCHS=3"
    set "SIM_ARGS=--num-scenarios 1 --realizations 10 --snr-db 5 10 15 20 25 30 --pa baseline,EPA,FPCP,DWMMSE,LocalGNN,PaperDCGNN,DQN,DDPG --pc MR,LMMSE,RMMSE,LMMSE_G --no-sync-ablation"
) else (
    set "EXPORT_ARGS="
    set "TRAIN_EPOCHS=300"
    set "SIM_ARGS="
)

echo.
echo =====================================================================
echo        CF_downlink_sim Pure-Python Full Pipeline
echo =====================================================================
echo   Root:          %ROOT%
echo   Mode:          %MODE%
echo   Retrain:       %FORCE_RETRAIN%
echo   Skip RL:       %SKIP_RL%
echo =====================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python was not found on PATH.
    goto fail
)

if not exist "%PYTHON_DIR%\requirements.txt" (
    echo [ERROR] Missing requirements file: %PYTHON_DIR%\requirements.txt
    goto fail
)

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

echo [1/7] Checking Python imports...
python -c "import sys; sys.path.insert(0, r'%PYTHON_DIR%'); import numpy, h5py, torch, torch_geometric, matplotlib; import simulator; print('Python dependencies OK')" || goto fail

echo.
echo [2/7] Exporting training data...
python "%PYTHON_DIR%\export_training_data.py" %EXPORT_ARGS% || goto fail

call :find_latest_data
if not defined LATEST_DATA (
    echo [ERROR] No training dataset was found in %DATA_DIR%.
    goto fail
)
echo   Latest dataset: %LATEST_DATA%

echo.
echo [3/7] Validating training data...
python "%PYTHON_DIR%\validation.py" "%LATEST_DATA%" || goto fail

echo.
echo [4/7] Training or reusing learned power-allocation models...
call :train_if_needed "%MODELS_DIR%\best_paper_dcgnn.pt" "PaperDCGNN" "python ""%PYTHON_DIR%\train_dcgnn_paper.py"" --L 100 --K 20 --z 15 --epochs %TRAIN_EPOCHS% --output_dir ""%MODELS_DIR%""" || goto fail
call :train_if_needed "%MODELS_DIR%\best_local_gnn_power.pt" "Local-GNN" "python ""%PYTHON_DIR%\train_gnn_local.py"" --data ""%LATEST_DATA%"" --epochs %TRAIN_EPOCHS% --output_dir ""%MODELS_DIR%""" || goto fail

echo.
echo [5/7] Training or reusing RL baselines...
if "%SKIP_RL%"=="1" (
    echo   Skipped by flag.
) else (
    set "HAVE_RL=0"
    if "%FORCE_RETRAIN%"=="0" if exist "%MODELS_DIR%\best_dqn_power.pt" if exist "%MODELS_DIR%\best_ddpg_power.pt" set "HAVE_RL=1"
    if "!HAVE_RL!"=="1" (
        echo   Reusing existing DQN/DDPG checkpoints.
    ) else (
        python "%PYTHON_DIR%\train_rl.py" --data "%LATEST_DATA%" --method all --epochs %TRAIN_EPOCHS% --output_dir "%MODELS_DIR%" || goto fail
    )
)

echo.
echo [6/7] Running model smoke checks...
python -c "import sys, numpy as np; sys.path.insert(0, r'%PYTHON_DIR%'); import gnn_local_inference; sg=np.abs(np.random.randn(100,20)).astype('float32'); D=np.ones((100,20), dtype='float32'); out=gnn_local_inference.infer(r'%MODELS_DIR%\best_local_gnn_power.pt', sg, D, 10.0, 0.3); rho=out['rho']; assert rho.shape==(100,20); assert np.isfinite(rho).all(); print('LocalGNN smoke OK')" || goto fail

echo.
echo [7/7] Running pure-Python main simulation...
python "%PYTHON_DIR%\run_simulation.py" %SIM_ARGS% || goto fail

echo.
echo =====================================================================
echo   Pure-Python pipeline completed successfully.
echo   Data:    %ROOT%\main\SimulationData
echo   Figures: %ROOT%\main\Imgs
echo   Models:  %MODELS_DIR%
echo =====================================================================
goto done

:find_latest_data
set "LATEST_DATA="
for /f "delims=" %%F in ('dir /b /a:-d "%DATA_DIR%\*.mat" 2^>nul') do (
    set "LATEST_DATA=%DATA_DIR%\%%F"
)
goto :eof

:train_if_needed
set "MODEL_PATH=%~1"
set "MODEL_NAME=%~2"
set "HAVE_MODEL=0"
if "%FORCE_RETRAIN%"=="0" if exist "%MODEL_PATH%" set "HAVE_MODEL=1"
if "%HAVE_MODEL%"=="1" (
    echo   Reusing %MODEL_NAME%: %MODEL_PATH%
    goto :eof
)
echo   Training %MODEL_NAME%...
%~3 || exit /b 1
goto :eof

:fail
echo.
echo [FAILED] Pure-Python pipeline stopped with an error.
if not "%NO_PAUSE%"=="1" pause
exit /b 1

:usage
echo.
echo Usage:
echo   run_python_full_pipeline.bat [small^|full] [/retrain] [/skip-rl] [/y]
echo.
echo Modes:
echo   small    Quick end-to-end run: tiny dataset, 3 training epochs, full SNR grid.
echo   full     Default Python simulation/export scale from cf_sim_core.py.
echo.
echo Flags:
echo   /retrain          Retrain models even if checkpoints already exist.
echo   /skip-rl          Skip DQN/DDPG training.
echo   /y                Do not pause before exit.
exit /b 0

:done
if not "%NO_PAUSE%"=="1" pause
exit /b 0
