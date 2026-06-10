# PaddleDetection 修改说明

本文档记录了对 PaddleDetection 框架的修改，需要手动应用到 PaddleDetection 目录中。

## 1. 新增数据增强算子

**文件**: `ppdet/data/transform/operators.py`

在文件末尾添加以下 4 个数据增强算子类：

### MotionBlur - 运动模糊
```python
@register_op
class MotionBlur(BaseOperator):
    """
    运动模糊增强：模拟相机或目标运动造成的模糊效果
    
    Args:
        k (int or tuple): 模糊核大小，越大模糊越强。默认 (3, 7)
        angle (float or tuple): 运动方向角度范围（度）。默认 (-45, 45)
        prob (float): 执行概率。默认 0.5
    """
    
    def __init__(self, k=(3, 7), angle=(-45, 45), prob=0.5):
        super(MotionBlur, self).__init__()
        if isinstance(k, (list, tuple)):
            self.k_min, self.k_max = k
        else:
            self.k_min = self.k_max = k
        if isinstance(angle, (list, tuple)):
            self.angle_min, self.angle_max = angle
        else:
            self.angle_min = self.angle_max = angle
        self.prob = prob
    
    def apply(self, sample, context=None):
        if random.random() > self.prob:
            return sample
        
        img = sample['image']
        k = random.randint(self.k_min, self.k_max)
        if k % 2 == 0:
            k += 1
        angle = random.uniform(self.angle_min, self.angle_max)
        
        kernel = np.zeros((k, k), dtype=np.float32)
        center = k // 2
        rad = np.deg2rad(angle)
        cos_val = np.cos(rad)
        sin_val = np.sin(rad)
        
        for i in range(k):
            offset = i - center
            x = int(center + offset * cos_val)
            y = int(center + offset * sin_val)
            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] = 1.0
        
        kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel
        img_blurred = cv2.filter2D(img, -1, kernel)
        sample['image'] = img_blurred
        return sample
```

### GaussianBlur - 高斯模糊
```python
@register_op
class GaussianBlur(BaseOperator):
    """
    高斯模糊增强
    
    Args:
        k (int or tuple): 模糊核大小。默认 (3, 7)
        sigma (float or tuple): 高斯标准差范围。默认 (0.1, 2.0)
        prob (float): 执行概率。默认 0.5
    """
    
    def __init__(self, k=(3, 7), sigma=(0.1, 2.0), prob=0.5):
        super(GaussianBlur, self).__init__()
        if isinstance(k, (list, tuple)):
            self.k_min, self.k_max = k
        else:
            self.k_min = self.k_max = k
        if isinstance(sigma, (list, tuple)):
            self.sigma_min, self.sigma_max = sigma
        else:
            self.sigma_min = self.sigma_max = sigma
        self.prob = prob
    
    def apply(self, sample, context=None):
        if random.random() > self.prob:
            return sample
        
        img = sample['image']
        k = random.randint(self.k_min, self.k_max)
        if k % 2 == 0:
            k += 1
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        img_blurred = cv2.GaussianBlur(img, (k, k), sigma)
        sample['image'] = img_blurred
        return sample
```

### MedianBlur - 中值模糊
```python
@register_op
class MedianBlur(BaseOperator):
    """
    中值模糊增强：对椒盐噪声特别有效
    
    Args:
        k (int or tuple): 模糊核大小。默认 (3, 7)
        prob (float): 执行概率。默认 0.5
    """
    
    def __init__(self, k=(3, 7), prob=0.5):
        super(MedianBlur, self).__init__()
        if isinstance(k, (list, tuple)):
            self.k_min, self.k_max = k
        else:
            self.k_min = self.k_max = k
        self.prob = prob
    
    def apply(self, sample, context=None):
        if random.random() > self.prob:
            return sample
        
        img = sample['image']
        k = random.randint(self.k_min, self.k_max)
        if k % 2 == 0:
            k += 1
        img_blurred = cv2.medianBlur(img, k)
        sample['image'] = img_blurred
        return sample
```

### GaussianNoise - 高斯噪声
```python
@register_op
class GaussianNoise(BaseOperator):
    """
    高斯噪声增强
    
    Args:
        mean (float): 噪声均值。默认 0
        std (float or tuple): 噪声标准差范围。默认 (5, 25)
        prob (float): 执行概率。默认 0.5
    """
    
    def __init__(self, mean=0, std=(5, 25), prob=0.5):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        if isinstance(std, (list, tuple)):
            self.std_min, self.std_max = std
        else:
            self.std_min = self.std_max = std
        self.prob = prob
    
    def apply(self, sample, context=None):
        if random.random() > self.prob:
            return sample
        
        img = sample['image']
        std = random.uniform(self.std_min, self.std_max)
        noise = np.random.normal(self.mean, std, img.shape).astype(np.float32)
        img_noisy = img.astype(np.float32) + noise
        img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
        sample['image'] = img_noisy
        return sample
```

## 2. 修复配置文件编码问题

**文件**: `ppdet/core/workspace.py`

找到 `_load_config_with_base` 函数，修改 `open` 调用：

```python
# 修改前
with open(file_path) as f:

# 修改后
with open(file_path, encoding='utf-8') as f:
```

## 3. 调整 NMS 置信度阈值

**文件**: `configs/ppyoloe/_base_/ppyoloe_crn.yml`

将 `score_threshold` 从 `0.01` 改为 `0.1`：

```yaml
PPYOLOEHead:
  nms:
    name: MultiClassNMS
    nms_top_k: 1000
    keep_top_k: 300
    score_threshold: 0.1  # 原为 0.01
    nms_threshold: 0.7
```

## 应用步骤

1. 进入 PaddleDetection 目录
2. 按照上述说明修改对应文件
3. 重新安装 PaddleDetection（如需要）：
   ```bash
   cd PaddleDetection
   python setup.py install
   ```

## 配置文件

新增的配置文件已保存在 `mycode/configs/` 目录：
- `ppyoloe_fire.yml` - 基础火焰检测配置
- `ppyoloe_fire_hn.yml` - 含 Hard Negative 样本的配置
- `ppyoloe_fire_test.yml` - 测试配置
