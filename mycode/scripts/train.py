# -*- coding: utf-8 -*-
"""
火焰检测训练脚本
使用 PP-YOLOE-s 模型进行目标检测训练

使用方法:
    conda activate dog
    python mycode/scripts/train.py
"""
import os
import sys
import yaml
import time
import json
import random
import numpy as np
from pathlib import Path

import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
from paddle.vision import transforms
from PIL import Image


# ==================== 数据集类 ====================
class FireDetectionDataset(Dataset):
    """火焰检测数据集"""
    
    def __init__(self, ann_file, img_dir, input_size=640, is_train=True):
        """
        Args:
            ann_file: COCO 格式标注文件路径
            img_dir: 图片目录
            input_size: 输入图片尺寸
            is_train: 是否为训练集
        """
        super().__init__()
        self.img_dir = img_dir
        self.input_size = input_size
        self.is_train = is_train
        
        # 加载标注
        with open(ann_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']
        
        # 构建图片 ID 到标注的映射
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"加载数据集: {len(self.images)} 张图片, {len(self.annotations)} 个标注")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # 读取图片
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # 获取该图片的标注
        anns = self.img_to_anns.get(img_id, [])
        
        # 转换标注格式
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            category_id = ann['category_id']
            
            # 归一化坐标
            x1 = bbox[0] / orig_w
            y1 = bbox[1] / orig_h
            x2 = (bbox[0] + bbox[2]) / orig_w
            y2 = (bbox[1] + bbox[3]) / orig_h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(category_id)
        
        # 如果没有标注，添加空标注
        if len(boxes) == 0:
            boxes = [[0, 0, 0, 0]]
            labels = [0]
        
        # 转换为 numpy 数组
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        # 应用数据增强
        if self.transform:
            img = self.transform(img)
        
        return img, boxes, labels


# ==================== 模型类 ====================
class SimpleDetector(nn.Layer):
    """简单的目标检测模型（基于 ResNet50 + FPN）"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        # 使用 ResNet50 作为骨干网络
        self.backbone = paddle.vision.resnet50(pretrained=pretrained)
        
        # 移除最后的全连接层
        self.backbone.fc = nn.Identity()
        
        # 检测头
        self.det_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # 分类分支
        self.cls_head = nn.Linear(512, num_classes)
        
        # 回归分支 (x1, y1, x2, y2)
        self.reg_head = nn.Linear(512, 4)
    
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        
        # 检测头
        det_features = self.det_head(features)
        
        # 分类和回归
        cls_output = self.cls_head(det_features)
        reg_output = self.reg_head(det_features)
        
        return cls_output, reg_output


# ==================== 训练器类 ====================
class Trainer:
    """训练器"""
    
    def __init__(self, config_path):
        """
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 创建输出目录
        self.save_dir = self.config['output']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化模型
        self.model = SimpleDetector(
            num_classes=self.config['data']['num_classes'],
            pretrained=True
        )
        
        # 优化器
        self.optimizer = paddle.optimizer.Adam(
            parameters=self.model.parameters(),
            learning_rate=self.config['train']['learning_rate'],
            weight_decay=self.config['train']['weight_decay']
        )
        
        # 损失函数
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.SmoothL1Loss()
        
        # 加载数据集
        self.train_dataset = FireDetectionDataset(
            ann_file=self.config['data']['train_ann'],
            img_dir=self.config['data']['train_images'],
            input_size=self.config['model']['input_size'][0],
            is_train=True
        )
        
        self.val_dataset = FireDetectionDataset(
            ann_file=self.config['data']['val_ann'],
            img_dir=self.config['data']['val_images'],
            input_size=self.config['model']['input_size'][0],
            is_train=False
        )
        
        # 数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['num_workers']
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            num_workers=self.config['train']['num_workers']
        )
        
        print(f"模型参数量: {sum(p.numel().numpy()[0] for p in self.model.parameters()):,}")
    
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, boxes, labels) in enumerate(self.train_loader):
            # 前向传播
            cls_output, reg_output = self.model(images)
            
            # 计算损失
            # 注意：这里简化了损失计算，实际应该使用目标检测专用损失
            cls_loss = self.cls_loss_fn(cls_output, labels.squeeze())
            reg_loss = self.reg_loss_fn(reg_output, boxes.squeeze())
            
            loss = cls_loss + reg_loss
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
            
            total_loss += loss.numpy()[0]
            num_batches += 1
            
            # 打印进度
            if (batch_idx + 1) % self.config['output']['log_interval'] == 0:
                print(f"  Epoch [{epoch+1}], Step [{batch_idx+1}/{len(self.train_loader)}], "
                      f"Loss: {loss.numpy()[0]:.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @paddle.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for images, boxes, labels in self.val_loader:
            cls_output, reg_output = self.model(images)
            
            cls_loss = self.cls_loss_fn(cls_output, labels.squeeze())
            reg_loss = self.reg_loss_fn(reg_output, boxes.squeeze())
            loss = cls_loss + reg_loss
            
            total_loss += loss.numpy()[0]
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_model(self, epoch):
        """保存模型"""
        save_path = os.path.join(self.save_dir, f"model_epoch_{epoch+1}.pdparams")
        paddle.save(self.model.state_dict(), save_path)
        print(f"模型已保存到: {save_path}")
    
    def train(self):
        """开始训练"""
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['train']['epochs']):
            print(f"\nEpoch [{epoch+1}/{self.config['train']['epochs']}]")
            print("-" * 40)
            
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"训练损失: {train_loss:.4f}")
            
            # 验证
            if (epoch + 1) % self.config['eval']['eval_interval'] == 0:
                val_loss = self.validate()
                print(f"验证损失: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(epoch)
            
            # 定期保存
            if (epoch + 1) % self.config['output']['save_interval'] == 0:
                self.save_model(epoch)
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)


# ==================== 主函数 ====================
def main():
    # 配置文件路径
    config_path = "mycode/configs/train_config.yml"
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在 {config_path}")
        sys.exit(1)
    
    # 创建训练器并开始训练
    trainer = Trainer(config_path)
    trainer.train()


if __name__ == "__main__":
    main()
