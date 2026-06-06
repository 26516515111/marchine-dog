@echo off
REM 模型导出脚本 - Windows 批处理版本
REM 训练完成后运行此脚本导出模型

echo ============================================================
echo   模型导出 - Paddle Inference 格式
echo ============================================================

REM 激活 conda 环境
call conda activate dog

REM 检查 PaddleDetection 是否存在
if not exist "PaddleDetection" (
    echo 错误: PaddleDetection 目录不存在
    echo 请先运行训练脚本
    pause
    exit /b 1
)

REM 检查训练好的模型是否存在
if not exist "PaddleDetection\output\ppyoloe_fire\best_model.pdparams" (
    echo 错误: 训练好的模型不存在
    echo 请先完成训练
    pause
    exit /b 1
)

echo.
echo 导出模型...

REM 进入 PaddleDetection 目录
cd PaddleDetection

REM 复制配置文件
copy /Y ..\mycode\configs\ppyoloe_fire.yml configs\custom\

REM 导出模型
python tools/export_model.py ^
    -c configs/custom/ppyoloe_fire.yml ^
    --output_dir=./output_inference ^
    -o weights=output/ppyoloe_fire/best_model.pdparams

REM 复制模型到提交目录
echo.
echo 复制模型文件到 model/ 目录...
cd ..
if not exist "model" mkdir model
copy /Y PaddleDetection\output_inference\ppyoloe_fire\model.pdmodel model\
copy /Y PaddleDetection\output_inference\ppyoloe_fire\model.pdiparams model\
copy /Y PaddleDetection\output_inference\ppyoloe_fire\infer_cfg.yml model\

echo.
echo ============================================================
echo   导出完成！
echo ============================================================
echo.
echo 模型文件:
dir /b model\
echo.
echo 模型大小:
for %%f in (model\*) do echo   %%f: %%~zf bytes
echo.

pause
