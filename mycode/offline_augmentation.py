# -*- coding: utf-8 -*-
"""
离线数据增强脚本
对 board/battery 图片进行旋转、翻转、裁剪增强
保持原始图片不变，生成新的增强图片和标注
"""
import json
import os
import random
import cv2
import numpy as np
from pathlib import Path
import shutil

# 配置参数
CONFIG = {
    # 输入路径
    'image_dir': 'dog/A_train/coco/train',
    'anno_path': 'dog/A_train/coco/annotations/instance_train.json',
    
    # 输出路径
    'output_image_dir': 'dog/A_train/coco/train_augmented',
    'output_anno_path': 'dog/A_train/coco/annotations/instance_train_augmented.json',
    
    # 增强参数
    'target_classes': [1, 2],  # battery=1, board=2（需要增强的类别）
    'only_target_annotations': False,  # 保留所有类别标注（避免 fire 漏标）
    'augmentations_per_image': 1,  # 每张图片生成几个增强版本
    
    # 旋转参数
    'rotation_angles': [90],  # 只使用 90 度旋转
    
    # 翻转参数
    'flip_horizontal': True,  # 水平翻转
    'flip_vertical': False,  # 垂直翻转（通常不用于目标检测）
    
    # 裁剪参数
    'crop_scales': [],  # 不使用裁剪
    'crop_attempts': 3,  # 每个裁剪比例尝试几次
    
    # 随机种子
    'random_seed': 42
}

def load_coco_annotations(anno_path):
    """加载 COCO 标注文件"""
    with open(anno_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_images_with_classes(data, target_classes):
    """获取包含目标类别的图片"""
    # 创建 image_id 到标注的映射
    ann_by_img = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_by_img:
            ann_by_img[img_id] = []
        ann_by_img[img_id].append(ann)
    
    # 找出包含目标类别的图片
    target_images = []
    for img_info in data['images']:
        img_id = img_info['id']
        if img_id in ann_by_img:
            has_target = any(ann['category_id'] in target_classes for ann in ann_by_img[img_id])
            if has_target:
                target_images.append(img_info)
    
    return target_images, ann_by_img

def rotate_bbox(bbox, angle, img_w, img_h):
    """
    旋转边界框
    
    Args:
        bbox: [x, y, w, h]
        angle: 旋转角度（90, 180, 270）
        img_w, img_h: 图片宽高
    
    Returns:
        new_bbox: [x, y, w, h]
    """
    x, y, w, h = bbox
    
    if angle == 90:
        # 顺时针90度: (x,y) -> (y, img_w-x-w)
        new_x = y
        new_y = img_w - x - w
        new_w = h
        new_h = w
    elif angle == 180:
        # 180度: (x,y) -> (img_w-x-w, img_h-y-h)
        new_x = img_w - x - w
        new_y = img_h - y - h
        new_w = w
        new_h = h
    elif angle == 270:
        # 顺时针270度: (x,y) -> (img_h-y-h, x)
        new_x = img_h - y - h
        new_y = x
        new_w = h
        new_h = w
    else:
        return bbox
    
    return [new_x, new_y, new_w, new_h]

def flip_bbox(bbox, img_w, img_h, horizontal=True):
    """
    翻转边界框
    
    Args:
        bbox: [x, y, w, h]
        img_w, img_h: 图片宽高
        horizontal: 是否水平翻转
    
    Returns:
        new_bbox: [x, y, w, h]
    """
    x, y, w, h = bbox
    
    if horizontal:
        # 水平翻转: x -> img_w - x - w
        new_x = img_w - x - w
        new_y = y
    else:
        # 垂直翻转: y -> img_h - y - h
        new_x = x
        new_y = img_h - y - h
    
    return [new_x, new_y, w, h]

def crop_image_and_bboxes(image, bboxes, scale, attempts=3):
    """
    随机裁剪图片和边界框
    
    Args:
        image: 原始图片
        bboxes: 边界框列表 [[x, y, w, h], ...]
        scale: 裁剪比例（相对于原图）
        attempts: 尝试次数
    
    Returns:
        cropped_image: 裁剪后的图片
        new_bboxes: 新的边界框列表
        crop_region: 裁剪区域 (x, y, w, h)
    """
    img_h, img_w = image.shape[:2]
    crop_w = int(img_w * scale)
    crop_h = int(img_h * scale)
    
    for _ in range(attempts):
        # 随机裁剪位置
        x = random.randint(0, img_w - crop_w)
        y = random.randint(0, img_h - crop_h)
        
        # 裁剪图片
        cropped = image[y:y+crop_h, x:x+crop_w].copy()
        
        # 调整边界框
        new_bboxes = []
        for bbox in bboxes:
            bx, by, bw, bh = bbox
            
            # 计算与裁剪区域的交集
            ix1 = max(x, bx)
            iy1 = max(y, by)
            ix2 = min(x + crop_w, bx + bw)
            iy2 = min(y + crop_h, by + bh)
            
            # 检查是否在裁剪区域内
            if ix2 > ix1 and iy2 > iy1:
                # 调整边界框坐标
                new_x = ix1 - x
                new_y = iy1 - y
                new_w = ix2 - ix1
                new_h = iy2 - iy1
                new_bboxes.append([new_x, new_y, new_w, new_h])
        
        # 如果保留了足够的边界框，返回结果
        if len(new_bboxes) >= len(bboxes) * 0.5:  # 至少保留50%的边界框
            return cropped, new_bboxes, (x, y, crop_w, crop_h)
    
    # 如果所有尝试都失败，返回原图
    return image, bboxes, (0, 0, img_w, img_h)

def augment_image(image, annotations, img_info, config):
    """
    对单张图片进行增强（只增强包含目标类别的图片）
    
    Args:
        image: 原始图片
        annotations: 该图片的标注列表
        img_info: 图片信息
        config: 配置参数
    
    Returns:
        augmented_list: [(augmented_image, new_annotations, suffix), ...]
    """
    augmented_list = []
    img_h, img_w = image.shape[:2]
    
    # 检查是否包含目标类别
    has_target = any(ann['category_id'] in config['target_classes'] for ann in annotations)
    if not has_target:
        return []  # 不包含目标类别，不增强
    
    # 过滤标注：只保留目标类别
    if config.get('only_target_annotations', False):
        annotations = [ann for ann in annotations if ann['category_id'] in config['target_classes']]
    
    # 1. 旋转增强（随机选择一个角度）
    angle = random.choice(config['rotation_angles'])
    if angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        rotated = image
    
    # 旋转边界框
    new_anns = []
    for ann in annotations:
        new_bbox = rotate_bbox(ann['bbox'], angle, img_w, img_h)
        new_ann = ann.copy()
        new_ann['bbox'] = new_bbox
        new_ann['area'] = new_bbox[2] * new_bbox[3]
        new_anns.append(new_ann)
    
    augmented_list.append((rotated, new_anns, f'_rot{angle}'))
    
    # 2. 翻转增强
    if config['flip_horizontal']:
        flipped = cv2.flip(image, 1)
        
        new_anns = []
        for ann in annotations:
            new_bbox = flip_bbox(ann['bbox'], img_w, img_h, horizontal=True)
            new_ann = ann.copy()
            new_ann['bbox'] = new_bbox
            new_ann['area'] = new_bbox[2] * new_bbox[3]
            new_anns.append(new_ann)
        
        augmented_list.append((flipped, new_anns, '_flip'))
    
    # 3. 裁剪增强（随机选择一个比例）
    if config['crop_scales']:
        scale = random.choice(config['crop_scales'])
        cropped, new_bboxes, crop_region = crop_image_and_bboxes(
            image, [ann['bbox'] for ann in annotations], scale, config['crop_attempts']
        )
        
        if len(new_bboxes) > 0:
            new_anns = []
            for i, bbox in enumerate(new_bboxes):
                if i < len(annotations):
                    new_ann = annotations[i].copy()
                    new_ann['bbox'] = bbox
                    new_ann['area'] = bbox[2] * bbox[3]
                    new_anns.append(new_ann)
            
            augmented_list.append((cropped, new_anns, f'_crop{int(scale*100)}'))
    
    return augmented_list

def offline_augmentation(config):
    """执行离线数据增强"""
    print("=" * 60)
    print("离线数据增强")
    print("=" * 60)
    
    # 设置随机种子
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # 加载标注
    print(f"\n1. 加载标注文件: {config['anno_path']}")
    data = load_coco_annotations(config['anno_path'])
    print(f"   原始图片数: {len(data['images'])}")
    print(f"   原始标注数: {len(data['annotations'])}")
    
    # 获取包含目标类别的图片
    print(f"\n2. 获取包含目标类别的图片: {config['target_classes']}")
    target_images, ann_by_img = get_images_with_classes(data, config['target_classes'])
    print(f"   包含 battery/board 的图片数: {len(target_images)}")
    
    # 创建输出目录
    os.makedirs(config['output_image_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['output_anno_path']), exist_ok=True)
    
    # 复制所有原始图片
    print(f"\n3. 复制原始图片到: {config['output_image_dir']}")
    for img_info in data['images']:
        src_path = os.path.join(config['image_dir'], img_info['file_name'])
        dst_path = os.path.join(config['output_image_dir'], img_info['file_name'])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # 准备新的标注数据
    new_data = {
        'images': data['images'].copy(),
        'annotations': data['annotations'].copy(),
        'categories': data['categories'].copy()
    }
    
    max_img_id = max(img['id'] for img in data['images'])
    max_ann_id = max(ann['id'] for ann in data['annotations'])
    
    # 执行增强
    print(f"\n4. 执行离线增强...")
    total_augmented = 0
    
    for img_info in target_images:
        img_id = img_info['id']
        img_path = os.path.join(config['image_dir'], img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
        
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # 获取该图片的标注
        annotations = ann_by_img.get(img_id, [])
        if len(annotations) == 0:
            continue
        
        # 执行增强
        augmented_list = augment_image(image, annotations, img_info, config)
        
        # 保存增强后的图片和标注
        for aug_image, aug_anns, suffix in augmented_list:
            # 生成新的文件名
            base_name = os.path.splitext(img_info['file_name'])[0]
            new_filename = f"{base_name}{suffix}.jpg"
            
            # 保存图片
            dst_path = os.path.join(config['output_image_dir'], new_filename)
            cv2.imwrite(dst_path, aug_image)
            
            # 添加图片信息
            max_img_id += 1
            new_img_info = {
                'id': max_img_id,
                'file_name': new_filename,
                'width': aug_image.shape[1],
                'height': aug_image.shape[0]
            }
            new_data['images'].append(new_img_info)
            
            # 添加标注信息
            for ann in aug_anns:
                max_ann_id += 1
                new_ann = ann.copy()
                new_ann['id'] = max_ann_id
                new_ann['image_id'] = max_img_id
                new_data['annotations'].append(new_ann)
            
            total_augmented += 1
    
    # 保存新标注
    print(f"\n5. 保存新标注文件: {config['output_anno_path']}")
    with open(config['output_anno_path'], 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False)
    
    # 统计结果
    print("\n" + "=" * 60)
    print("离线增强完成！")
    print("=" * 60)
    print(f"\n统计信息:")
    print(f"  原始图片数: {len(data['images'])}")
    print(f"  增强图片数: {total_augmented}")
    print(f"  最终图片数: {len(new_data['images'])}")
    print(f"  原始标注数: {len(data['annotations'])}")
    print(f"  最终标注数: {len(new_data['annotations'])}")
    
    # 统计各类别数量
    cls_count = {}
    for ann in new_data['annotations']:
        cls_id = ann['category_id']
        cls_count[cls_id] = cls_count.get(cls_id, 0) + 1
    
    print(f"\n各类别标注数:")
    for cls_id, count in sorted(cls_count.items()):
        cls_name = 'battery' if cls_id == 1 else ('board' if cls_id == 2 else 'fire')
        original_count = sum(1 for a in data['annotations'] if a['category_id'] == cls_id)
        print(f"  {cls_name}: {original_count} -> {count} (+{count - original_count})")
    
    return new_data

if __name__ == '__main__':
    offline_augmentation(CONFIG)
