@echo off
REM 火焰检测训练脚本 - Windows 批处理版本
REM 在实验室电脑上运行此脚本

echo ============================================================
echo   火焰检测模型训练 - PaddleDetection
echo ============================================================

REM 激活 conda 环境
call conda activate dog

REM 检查 PaddleDetection 是否存在
if not exist "PaddleDetection" (
    echo.
    echo PaddleDetection 目录不存在，开始克隆...
    git clone https://github.com/PaddlePaddle/PaddleDetection.git
    
    echo.
    echo 安装依赖...
    cd PaddleDetection
    pip install -r requirements.txt
    python setup.py install
    cd ..
)

echo.
echo ============================================================
echo   开始训练
echo ============================================================

REM 进入 PaddleDetection 目录
cd PaddleDetection

REM 复制配置文件
copy /Y ..\mycode\configs\ppyoloe_fire.yml configs\custom\

REM 训练
python -m paddle.distributed.launch --gpus 0 tools/train.py ^
    -c configs/custom/ppyoloe_fire.yml ^
    --eval ^
    --use_vdl=True ^
    --vdl_log_dir=vdl_log/fire_detection ^
    --output_dir=output/ppyoloe_fire

echo.
echo ============================================================
echo   评估模型
echo ============================================================

python tools/eval.py ^
    -c configs/custom/ppyoloe_fire.yml ^
    -o weights=output/ppyoloe_fire/best_model.pdparams

echo.
echo ============================================================
echo   导出模型
echo ============================================================

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
echo   训练完成！
echo ============================================================
echo.
echo 模型文件已保存到 model/ 目录
echo 可以运行打包脚本生成 submission.zip
echo.

pause
