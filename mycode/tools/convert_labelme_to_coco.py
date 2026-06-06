# -*- coding: utf-8 -*-
"""
将 LabelMe 格式标注转换为 COCO 格式（用于 PaddleDetection 训练）
"""
import os
import json
import glob
from PIL import Image


# 类别映射
CATEGORIES = [
    {"id": 0, "name": "battery"},
    {"id": 1, "name": "board"},
    {"id": 2, "name": "fire"}
]

LABEL_TO_ID = {"battery": 0, "board": 1, "fire": 2}


def convert_labelme_to_coco(image_dir, label_dir, output_path, train_ratio=0.8):
    """
    将 LabelMe 标注转换为 COCO 格式
    
    Args:
        image_dir: 图片目录
        label_dir: 标注目录
        output_path: 输出 JSON 路径
        train_ratio: 训练集比例
    """
    # 获取所有标注文件
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.json")))
    
    print(f"找到 {len(label_files)} 个标注文件")
    
    # 准备 COCO 格式数据
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    
    ann_id = 1
    valid_count = 0
    skip_count = 0
    
    for label_file in label_files:
        with open(label_file, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        # 获取图片信息
        image_name = label_data.get("imagePath", "")
        if not image_name:
            # 从文件名推断
            base_name = os.path.splitext(os.path.basename(label_file))[0]
            image_name = base_name + ".jpg"
        
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"跳过: 图片不存在 {image_path}")
            skip_count += 1
            continue
        
        # 获取图片尺寸
        img_width = label_data.get("imageWidth", 1920)
        img_height = label_data.get("imageHeight", 1080)
        
        # 图片 ID
        image_id = len(coco_data["images"]) + 1
        
        # 添加图片信息
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": img_width,
            "height": img_height
        })
        
        # 处理标注
        shapes = label_data.get("shapes", [])
        for shape in shapes:
            label = shape.get("label", "")
            if label not in LABEL_TO_ID:
                print(f"跳过未知类别: {label}")
                continue
            
            category_id = LABEL_TO_ID[label]
            points = shape.get("points", [])
            
            if len(points) < 2:
                continue
            
            # 计算边界框 (x, y, width, height)
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            # 添加标注
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })
            ann_id += 1
        
        valid_count += 1
    
    # 保存 COCO 格式 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n转换完成:")
    print(f"  有效图片: {valid_count}")
    print(f"  跳过图片: {skip_count}")
    print(f"  总标注数: {len(coco_data['annotations'])}")
    print(f"  输出文件: {output_path}")
    
    # 统计各类别数量
    cat_counts = {0: 0, 1: 0, 2: 0}
    for ann in coco_data["annotations"]:
        cat_counts[ann["category_id"]] += 1
    
    print(f"\n类别统计:")
    for cat in CATEGORIES:
        print(f"  {cat['name']}: {cat_counts[cat['id']]}")
    
    return coco_data


def split_dataset(coco_data, train_ratio=0.8):
    """
    将数据集分割为训练集和验证集
    """
    import random
    
    images = coco_data["images"]
    random.shuffle(images)
    
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    train_ids = set(img["id"] for img in train_images)
    val_ids = set(img["id"] for img in val_images)
    
    train_anns = [ann for ann in coco_data["annotations"] if ann["image_id"] in train_ids]
    val_anns = [ann for ann in coco_data["annotations"] if ann["image_id"] in val_ids]
    
    train_data = {
        "images": train_images,
        "annotations": train_anns,
        "categories": coco_data["categories"]
    }
    
    val_data = {
        "images": val_images,
        "annotations": val_anns,
        "categories": coco_data["categories"]
    }
    
    return train_data, val_data


if __name__ == "__main__":
    # 路径配置
    IMAGE_DIR = r"D:\work\Marchine Dog\submission_template_firedetect\A_train\Image"
    LABEL_DIR = r"D:\work\Marchine Dog\submission_template_firedetect\A_train\label"
    OUTPUT_DIR = r"D:\work\Marchine Dog\submission_template_firedetect\mycode\data"
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 转换为 COCO 格式
    print("=" * 50)
    print("开始转换 LabelMe -> COCO 格式")
    print("=" * 50)
    
    coco_data = convert_labelme_to_coco(
        IMAGE_DIR, 
        LABEL_DIR, 
        os.path.join(OUTPUT_DIR, "annotations_all.json")
    )
    
    # 分割数据集
    print("\n" + "=" * 50)
    print("分割数据集")
    print("=" * 50)
    
    train_data, val_data = split_dataset(coco_data, train_ratio=0.8)
    
    # 保存训练集
    train_path = os.path.join(OUTPUT_DIR, "annotations_train.json")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"\n训练集: {len(train_data['images'])} 张图片, {len(train_data['annotations'])} 个标注")
    print(f"保存到: {train_path}")
    
    # 保存验证集
    val_path = os.path.join(OUTPUT_DIR, "annotations_val.json")
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"验证集: {len(val_data['images'])} 张图片, {len(val_data['annotations'])} 个标注")
    print(f"保存到: {val_path}")
    
    print("\n" + "=" * 50)
    print("完成！")
    print("=" * 50)
