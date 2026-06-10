# -*- coding: utf-8 -*-
"""
训练完成后自动上传到 Git 仓库的工作流脚本
用法：python train_and_push.py
"""
import os
import subprocess
import sys
import time
from datetime import datetime

# 配置
PADDLEDETECTION_DIR = os.path.join(os.path.dirname(__file__), '..', 'PaddleDetection')
CONFIG_FILE = 'configs/custom/ppyoloe_fire.yml'
GIT_REPO_DIR = os.path.join(os.path.dirname(__file__), '..')

# GitHub 配置 - 在这里添加你的 Personal Access Token
GITHUB_TOKEN = ''  # 在这里填写你的 GitHub Token，例如：''
GITHUB_REPO = 'https://github.com/26516515111/marchine-dog.git'  # 在这里填写你的仓库地址

def run_command(cmd, cwd=None):
    """运行命令并打印输出"""
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Running: {cmd}')
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def train():
    """运行训练"""
    print('=' * 60)
    print('Step 1: Starting training...')
    print('=' * 60)
    
    cmd = f'python tools/train.py -c {CONFIG_FILE} --eval'
    return run_command(cmd, cwd=PADDLEDETECTION_DIR)

def export_model():
    """导出模型"""
    print('=' * 60)
    print('Step 2: Exporting model...')
    print('=' * 60)
    
    cmd = 'python tools/export_model.py -c configs/custom/ppyoloe_fire.yml --output_dir=./output_inference -o weights=output/ppyoloe_fire/best_model.pdparams'
    return run_command(cmd, cwd=PADDLEDETECTION_DIR)

def copy_model():
    """复制模型到项目根目录"""
    print('=' * 60)
    print('Step 3: Copying model to project root...')
    print('=' * 60)
    
    src_dir = os.path.join(PADDLEDETECTION_DIR, 'output_inference', 'ppyoloe_fire')
    dst_dir = os.path.join(GIT_REPO_DIR, 'model')
    
    # 清空目标目录
    if os.path.exists(dst_dir):
        import shutil
        shutil.rmtree(dst_dir)
    
    # 复制文件
    import shutil
    shutil.copytree(src_dir, dst_dir)
    print(f'Model copied to {dst_dir}')

def git_push():
    """上传到 Git 仓库"""
    print('=' * 60)
    print('Step 4: Pushing to Git...')
    print('=' * 60)
    
    # 检查 token 是否配置
    if not GITHUB_TOKEN:
        print('Error: GITHUB_TOKEN not configured!')
        print('Please add your GitHub Personal Access Token to the script.')
        return 1
    
    # 构建带 token 的远程仓库 URL
    # 格式：https://TOKEN@github.com/username/repo.git
    repo_url = GITHUB_REPO.replace('https://', f'https://{GITHUB_TOKEN}@')
    
    # 获取当前时间作为提交信息
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    commit_message = f'Training completed at {now}'
    
    # Git 操作
    commands = [
        'git add .',
        f'git commit -m "{commit_message}"',
        f'git push {repo_url} main'
    ]
    
    for cmd in commands:
        ret = run_command(cmd, cwd=GIT_REPO_DIR)
        if ret != 0:
            print(f'Error: {cmd} failed with return code {ret}')
            return ret
    
    return 0

def main():
    """主函数"""
    start_time = time.time()
    
    print('=' * 60)
    print('Training and Git Push Workflow')
    print('=' * 60)
    print(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    # Step 1: 训练
    ret = train()
    if ret != 0:
        print(f'Training failed with return code {ret}')
        sys.exit(1)
    
    # Step 2: 导出模型
    ret = export_model()
    if ret != 0:
        print(f'Export failed with return code {ret}')
        sys.exit(1)
    
    # Step 3: 复制模型
    copy_model()
    
    # Step 4: 上传 Git
    ret = git_push()
    if ret != 0:
        print(f'Git push failed with return code {ret}')
        sys.exit(1)
    
    # 完成
    end_time = time.time()
    duration = end_time - start_time
    
    print()
    print('=' * 60)
    print('Workflow completed successfully!')
    print(f'End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)')
    print('=' * 60)

if __name__ == '__main__':
    main()
