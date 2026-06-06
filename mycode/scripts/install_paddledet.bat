# 安装 PaddleDetection 脚本
# 在实验室电脑上运行此脚本

# 1. 激活环境
conda activate dog

# 2. 克隆 PaddleDetection
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# 3. 进入目录
cd PaddleDetection

# 4. 安装依赖
pip install -r requirements.txt

# 5. 编译安装
python setup.py install

# 6. 验证安装
python -c "import paddledet; print('PaddleDetection 安装成功:', paddledet.__version__)"
