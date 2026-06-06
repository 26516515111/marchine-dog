# -*- coding: utf-8 -*-
"""
使用 PaddleDetection 官方脚本训练火焰检测模型

使用方法:
    conda activate dog
    python mycode/scripts/train_paddledet.py
"""
import os
import sys
import subprocess


def run_command(cmd, desc=""):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    if desc:
        print(f"  {desc}")
    print(f"{'='*60}")
    print(f"执行命令: {cmd}\n")
    
    # 运行命令
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding='utf-8'
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\n错误: 命令执行失败，返回码 {process.returncode}")
        return False
    
    print(f"\n命令执行成功!")
    return True


def check_paddledetection():
    """检查 PaddleDetection 是否已安装"""
    try:
        import paddledet
        print(f"PaddleDetection 已安装，版本: {paddledet.__version__}")
        return True
    except ImportError:
        print("PaddleDetection 未安装")
        return False


def install_paddledetection():
    """安装 PaddleDetection"""
    print("\n开始安装 PaddleDetection...")
    
    # 克隆仓库
    if not os.path.exists("PaddleDetection"):
        cmd = "git clone https://github.com/PaddlePaddle/PaddleDetection.git"
        if not run_command(cmd, "克隆 PaddleDetection 仓库"):
            return False
    
    # 安装依赖
    os.chdir("PaddleDetection")
    
    cmd = "pip install -r requirements.txt"
    if not run_command(cmd, "安装依赖"):
        return False
    
    # 编译安装
    cmd = "python setup.py install"
    if not run_command(cmd, "编译安装 PaddleDetection"):
        return False
    
    os.chdir("..")
    
    return True


def train_model():
    """使用 PaddleDetection 训练模型"""
    
    # 配置文件路径
    config_path = "mycode/configs/ppyoloe_fire.yml"
    
    # 检查配置文件
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在 {config_path}")
        return False
    
    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    
    # 训练命令
    # 使用 paddle.distributed.launch 启动训练
    cmd = (
        f"python -m paddle.distributed.launch "
        f"--gpus 0 "
        f"tools/train.py "
        f"-c {config_path} "
        f"--eval "
        f"--use_vdl=True "
        f"--vdl_log_dir=vdl_log/fire_detection "
        f"--output_dir=output/ppyoloe_fire"
    )
    
    return run_command(cmd, "开始训练 PP-YOLOE-s 模型")


def evaluate_model():
    """评估模型"""
    config_path = "mycode/configs/ppyoloe_fire.yml"
    weights_path = "output/ppyoloe_fire/best_model.pdparams"
    
    if not os.path.exists(weights_path):
        print(f"错误: 模型权重文件不存在 {weights_path}")
        return False
    
    cmd = (
        f"python tools/eval.py "
        f"-c {config_path} "
        f"-o weights={weights_path}"
    )
    
    return run_command(cmd, "评估模型")


def export_model():
    """导出模型为 Paddle Inference 格式"""
    config_path = "mycode/configs/ppyoloe_fire.yml"
    weights_path = "output/ppyoloe_fire/best_model.pdparams"
    
    if not os.path.exists(weights_path):
        print(f"错误: 模型权重文件不存在 {weights_path}")
        return False
    
    cmd = (
        f"python tools/export_model.py "
        f"-c {config_path} "
        f"--output_dir=./output_inference "
        f"-o weights={weights_path}"
    )
    
    if not run_command(cmd, "导出模型"):
        return False
    
    # 复制到提交目录
    import shutil
    
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    # 查找导出的模型文件
    inference_dir = "output_inference/ppyoloe_fire"
    if os.path.exists(inference_dir):
        for file_name in ["model.pdmodel", "model.pdiparams", "infer_cfg.yml"]:
            src = os.path.join(inference_dir, file_name)
            dst = os.path.join(model_dir, file_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"复制: {src} -> {dst}")
    
    print(f"\n模型已导出到 {model_dir} 目录")
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("火焰检测模型训练 - PaddleDetection 官方脚本")
    print("=" * 60)
    
    # 检查并安装 PaddleDetection
    if not check_paddledetection():
        print("\n是否安装 PaddleDetection? (y/n): ")
        choice = input().strip().lower()
        if choice == 'y':
            if not install_paddledetection():
                print("安装失败，请手动安装")
                sys.exit(1)
        else:
            print("请先安装 PaddleDetection:")
            print("  git clone https://github.com/PaddlePaddle/PaddleDetection.git")
            print("  cd PaddleDetection")
            print("  pip install -r requirements.txt")
            print("  python setup.py install")
            sys.exit(1)
    
    # 选择操作
    print("\n请选择操作:")
    print("1. 训练模型")
    print("2. 评估模型")
    print("3. 导出模型")
    print("4. 完整流程 (训练 -> 评估 -> 导出)")
    print("\n输入选项 (1-4): ")
    
    choice = input().strip()
    
    if choice == '1':
        train_model()
    elif choice == '2':
        evaluate_model()
    elif choice == '3':
        export_model()
    elif choice == '4':
        # 完整流程
        if train_model():
            if evaluate_model():
                export_model()
    else:
        print("无效选项")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
