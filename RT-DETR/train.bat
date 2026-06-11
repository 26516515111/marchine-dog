@echo off
REM ============================================================
REM RT-DETR-R18 一键训练脚本
REM 使用前请确保 conda 环境已激活
REM ============================================================

echo [1/2] Syncing config to PaddleDetection...
copy /Y "%~dp0configs\rtdetr_r18vd_fire.yml" "%~dp0..\PaddleDetection\configs\custom\rtdetr_r18vd_fire.yml"

echo [2/2] Starting training...
cd /d "%~dp0..\PaddleDetection"
python tools/train.py -c configs/custom/rtdetr_r18vd_fire.yml --eval

echo.
echo Training completed. Check PaddleDetection/output/rtdetr_r18_fire/
pause
