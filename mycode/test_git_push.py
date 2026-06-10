# -*- coding: utf-8 -*-
"""
测试 Git 提交功能
用法：python test_git_push.py
"""
import os
import subprocess
import sys
from datetime import datetime

# 配置
GIT_REPO_DIR = os.path.join(os.path.dirname(__file__), '..')

# GitHub 配置 - 从 train_and_push.py 中读取
from train_and_push import GITHUB_TOKEN, GITHUB_REPO

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

def test_git_push():
    """测试 Git 提交功能"""
    print('=' * 60)
    print('Testing Git Push...')
    print('=' * 60)
    
    # 检查 token 是否配置
    if not GITHUB_TOKEN:
        print('Error: GITHUB_TOKEN not configured!')
        print('Please add your GitHub Personal Access Token to train_and_push.py.')
        return 1
    
    # 检查仓库地址是否配置
    if not GITHUB_REPO:
        print('Error: GITHUB_REPO not configured!')
        print('Please add your GitHub repository URL to train_and_push.py.')
        return 1
    
    print(f'GitHub Token: {GITHUB_TOKEN[:10]}...')
    print(f'GitHub Repo: {GITHUB_REPO}')
    print()
    
    # 构建带 token 的远程仓库 URL
    repo_url = GITHUB_REPO.replace('https://', f'https://{GITHUB_TOKEN}@')
    
    # 创建测试文件
    test_file = os.path.join(GIT_REPO_DIR, 'test_git_push.txt')
    with open(test_file, 'w') as f:
        f.write(f'Test git push at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    
    # Git 操作
    commands = [
        'git add test_git_push.txt',
        f'git commit -m "Test git push at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"',
        f'git push {repo_url} main'
    ]
    
    for cmd in commands:
        ret = run_command(cmd, cwd=GIT_REPO_DIR)
        if ret != 0:
            print(f'Error: {cmd} failed with return code {ret}')
            # 清理测试文件
            if os.path.exists(test_file):
                os.remove(test_file)
            return ret
    
    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print()
    print('=' * 60)
    print('Git push test completed successfully!')
    print('=' * 60)
    return 0

if __name__ == '__main__':
    sys.exit(test_git_push())
