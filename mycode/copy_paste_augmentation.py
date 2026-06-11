# -*- coding: utf-8 -*-
"""
Copy-Paste 数据增强脚本（改进版）
使用 GrabCut 分割 + 泊松融合，实现真正的 Copy-Paste 增强
"""
import json
import os
import random
import cv2
import numpy as np
from pathlib import Path

# 配置参数
CONFIG = {
    # 输入路径
    'image_dir': 'dog/A_train/coco/train',
    'anno_path': 'dog/A_train/coco/annotations/instance_train.json',
    
    # 输出路径
    'output_image_dir': 'dog/A_train/coco/train_copypaste',
    'output_anno_path': 'dog/A_train/coco/annotations/instance_train_copypaste.json',
    
    # Copy-Paste 参数
    'target_classes': [1, 2],  # battery=1, board=2（需要增强的类别）
    'copies_per_instance': 1,  # 每个实例复制几次（减少到1次）
    'paste_prob': 0.8,  # 粘贴到目标图片的概率
    'scale_range': (0.8, 1.2),  # 粘贴时的缩放范围
    'rotation_range': (-15, 15),  # 粘贴时的旋转范围
    'blend_method': 'poisson',  # 融合方法: 'poisson' 或 'alpha'
    'mask_quality_threshold': 0.2,  # 掩码质量阈值（降低以保留更多 board）
    
    # 随机种子
    'random_seed': 42
}

def load_coco_annotations(anno_path):
    """加载 COCO 标注文件"""
    with open(anno_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_instances_by_class(data, target_classes):
    """按类别获取实例信息"""
    instances = {cls_id: [] for cls_id in target_classes}
    
    # 创建 image_id 到 image_info 的映射
    img_map = {img['id']: img for img in data['images']}
    
    for ann in data['annotations']:
        if ann['category_id'] in target_classes:
            img_info = img_map[ann['image_id']]
            instances[ann['category_id']].append({
                'ann': ann,
                'image_info': img_info
            })
    
    return instances

def extract_instance_with_mask(image, bbox, method='grabcut'):
    """
    从图像中提取实例及其掩码
    
    Args:
        image: 源图像
        bbox: 边界框 [x, y, w, h]
        method: 分割方法 ('grabcut' 或 'color')
    
    Returns:
        roi: 提取的实例图像
        mask: 实例掩码（0-255）
        offset: 在原图中的偏移量 (x1, y1)
    """
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]
    
    # 扩展边界框（增加上下文）
    margin = int(max(w, h) * 0.1)
    x1 = max(0, int(x) - margin)
    y1 = max(0, int(y) - margin)
    x2 = min(img_w, int(x + w) + margin)
    y2 = min(img_h, int(y + h) + margin)
    
    # 提取区域
    roi = image[y1:y2, x1:x2].copy()
    
    # 在 ROI 中的相对边界框
    rel_x = x - x1
    rel_y = y - y1
    
    # 创建掩码
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    
    if method == 'grabcut':
        # 使用 GrabCut 进行前景分割
        rect = (int(rel_x), int(rel_y), int(w), int(h))
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # 初始化掩码
        init_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        init_mask[int(rel_y):int(rel_y+h), int(rel_x):int(rel_x+w)] = cv2.GC_PR_FGD
        
        try:
            cv2.grabCut(roi, init_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            # 提取前景掩码
            mask = np.where((init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        except Exception as e:
            print(f"GrabCut 失败: {e}, 使用矩形掩码")
            mask[int(rel_y):int(rel_y+h), int(rel_x):int(rel_x+w)] = 255
    
    elif method == 'color':
        # 基于颜色的分割（适用于颜色对比明显的场景）
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 提取边界框区域的颜色范围
        roi_region = roi[int(rel_y):int(rel_y+h), int(rel_x):int(rel_x+w)]
        hsv_region = hsv[int(rel_y):int(rel_y+h), int(rel_x):int(rel_x+w)]
        
        # 计算颜色均值和标准差
        mean_color = np.mean(hsv_region, axis=(0, 1))
        std_color = np.std(hsv_region, axis=(0, 1))
        
        # 创建颜色掩码
        lower = np.maximum(0, mean_color - 2 * std_color).astype(np.uint8)
        upper = np.minimum(255, mean_color + 2 * std_color).astype(np.uint8)
        
        color_mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学操作清理掩码
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 只保留边界框区域内的掩码
        bbox_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        bbox_mask[int(rel_y):int(rel_y+h), int(rel_x):int(rel_x+w)] = 255
        mask = cv2.bitwise_and(color_mask, bbox_mask)
    
    else:
        # 默认使用矩形掩码
        mask[int(rel_y):int(rel_y+h), int(rel_x):int(rel_x+w)] = 255
    
    # 平滑掩码边缘
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return roi, mask, (x1, y1)

def rotate_instance(roi, mask, angle):
    """旋转实例和掩码"""
    h, w = roi.shape[:2]
    center = (w // 2, h // 2)
    
    # 计算旋转后的画布大小
    rad = np.radians(abs(angle))
    new_w = int(w * np.cos(rad) + h * np.sin(rad))
    new_h = int(w * np.sin(rad) + h * np.cos(rad))
    
    # 调整旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2
    
    # 旋转
    rotated_roi = cv2.warpAffine(roi, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return rotated_roi, rotated_mask

def poisson_blend(background, roi, mask, position):
    """
    使用泊松融合将实例粘贴到背景上
    
    Args:
        background: 背景图像
        roi: 实例图像
        mask: 实例掩码
        position: 粘贴位置 (x, y)
    
    Returns:
        blended: 融合后的图像
        bbox: 新的边界框
    """
    x, y = position
    roi_h, roi_w = roi.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    # 确保粘贴位置有效
    x = max(0, min(x, bg_w - roi_w))
    y = max(0, min(y, bg_h - roi_h))
    
    # 计算融合中心
    center = (x + roi_w // 2, y + roi_h // 2)
    
    try:
        # 使用泊松融合
        blended = cv2.seamlessClone(
            roi, background, mask, center, cv2.MIXED_CLONE
        )
        
        # 计算新的边界框
        bbox = [x, y, roi_w, roi_h]
        
        return blended, bbox
    
    except Exception as e:
        print(f"泊松融合失败: {e}, 使用 alpha 融合")
        return alpha_blend(background, roi, mask, position)

def alpha_blend(background, roi, mask, position):
    """
    使用 alpha 融合将实例粘贴到背景上
    
    Args:
        background: 背景图像
        roi: 实例图像
        mask: 实例掩码
        position: 粘贴位置 (x, y)
    
    Returns:
        blended: 融合后的图像
        bbox: 新的边界框
    """
    x, y = position
    roi_h, roi_w = roi.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    # 确保粘贴位置有效
    x = max(0, min(x, bg_w - roi_w))
    y = max(0, min(y, bg_h - roi_h))
    
    # 计算粘贴区域
    x1, y1 = x, y
    x2, y2 = min(x + roi_w, bg_w), min(y + roi_h, bg_h)
    
    # 计算 ROI 中对应的区域
    roi_x1, roi_y1 = 0, 0
    roi_x2, roi_y2 = x2 - x1, y2 - y1
    
    # 获取掩码
    mask_region = mask[roi_y1:roi_y2, roi_x1:roi_x2]
    mask_3ch = cv2.merge([mask_region, mask_region, mask_region])
    
    # 混合图像
    bg_region = background[y1:y2, x1:x2]
    roi_region = roi[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # 使用掩码混合
    mask_float = mask_3ch.astype(float) / 255
    blended_region = bg_region * (1 - mask_float) + roi_region * mask_float
    background[y1:y2, x1:x2] = blended_region.astype(np.uint8)
    
    # 计算新的边界框
    bbox = [x1, y1, x2 - x1, y2 - y1]
    
    return background, bbox

def find_paste_position(background, roi_shape, existing_bboxes, min_distance=50):
    """找到合适的粘贴位置（避免重叠）"""
    bg_h, bg_w = background.shape[:2]
    roi_h, roi_w = roi_shape[:2]
    
    max_attempts = 50
    for _ in range(max_attempts):
        # 随机位置
        x = random.randint(0, max(0, bg_w - roi_w))
        y = random.randint(0, max(0, bg_h - roi_h))
        
        # 检查与现有 bbox 的重叠
        new_bbox = [x, y, roi_w, roi_h]
        overlap = False
        
        for exist_bbox in existing_bboxes:
            iou = calculate_iou(new_bbox, exist_bbox)
            if iou > 0.1:  # 允许少量重叠
                overlap = True
                break
        
        if not overlap:
            return (x, y)
    
    # 如果找不到合适位置，返回随机位置
    return (random.randint(0, max(0, bg_w - roi_w)), 
            random.randint(0, max(0, bg_h - roi_h)))

def calculate_iou(bbox1, bbox2):
    """计算 IoU"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0

def copy_paste_augmentation(config):
    """执行 Copy-Paste 增强"""
    print("=" * 60)
    print("Copy-Paste 数据增强（改进版）")
    print("=" * 60)
    
    # 设置随机种子
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # 加载标注
    print(f"\n1. 加载标注文件: {config['anno_path']}")
    data = load_coco_annotations(config['anno_path'])
    print(f"   原始图片数: {len(data['images'])}")
    print(f"   原始标注数: {len(data['annotations'])}")
    
    # 获取目标类别的实例
    print(f"\n2. 获取目标类别实例: {config['target_classes']}")
    instances = get_instances_by_class(data, config['target_classes'])
    for cls_id, inst_list in instances.items():
        cls_name = 'battery' if cls_id == 1 else 'board'
        print(f"   {cls_name} (cls_id={cls_id}): {len(inst_list)} 个实例")
    
    # 创建输出目录
    os.makedirs(config['output_image_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['output_anno_path']), exist_ok=True)
    
    # 复制原始图片
    print(f"\n3. 复制原始图片到: {config['output_image_dir']}")
    for img_info in data['images']:
        src_path = os.path.join(config['image_dir'], img_info['file_name'])
        dst_path = os.path.join(config['output_image_dir'], img_info['file_name'])
        if os.path.exists(src_path):
            import shutil
            shutil.copy2(src_path, dst_path)
    
    # 准备新的标注数据
    new_data = {
        'images': data['images'].copy(),
        'annotations': data['annotations'].copy(),
        'categories': data['categories'].copy()
    }
    
    max_img_id = max(img['id'] for img in data['images'])
    max_ann_id = max(ann['id'] for ann in data['annotations'])
    
    # 执行 Copy-Paste
    print(f"\n4. 执行 Copy-Paste 增强...")
    print(f"   融合方法: {config['blend_method']}")
    total_pasted = 0
    
    for cls_id, inst_list in instances.items():
        cls_name = 'battery' if cls_id == 1 else 'board'
        print(f"\n   处理 {cls_name} 类别...")
        
        # 选择目标图片（不含当前类别的图片）
        target_images = []
        for img_info in data['images']:
            img_anns = [a for a in data['annotations'] if a['image_id'] == img_info['id']]
            has_cls = any(a['category_id'] == cls_id for a in img_anns)
            if not has_cls:  # 不含当前类别的图片
                target_images.append(img_info)
        
        print(f"   可用目标图片数: {len(target_images)}")
        
        if len(target_images) == 0:
            print(f"   警告: 没有可用的目标图片，跳过 {cls_name}")
            continue
        
        # 对每个实例进行 Copy-Paste
        pasted_count = 0
        for inst_info in inst_list:
            ann = inst_info['ann']
            img_info = inst_info['image_info']
            
            # 读取源图像
            src_path = os.path.join(config['image_dir'], img_info['file_name'])
            if not os.path.exists(src_path):
                continue
            src_image = cv2.imread(src_path)
            
            # 提取实例和掩码（使用 GrabCut 分割）
            roi, mask, offset = extract_instance_with_mask(src_image, ann['bbox'], method='grabcut')
            
            # 检查掩码质量
            mask_area = np.sum(mask > 128)
            bbox_area = ann['bbox'][2] * ann['bbox'][3]
            if mask_area < bbox_area * config['mask_quality_threshold']:  # 掩码面积太小，跳过
                print(f"   警告: 掩码质量差，跳过实例 {ann['id']}")
                continue
            
            # 随机缩放
            scale = random.uniform(*config['scale_range'])
            new_w = int(roi.shape[1] * scale)
            new_h = int(roi.shape[0] * scale)
            if new_w > 0 and new_h > 0:
                roi = cv2.resize(roi, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h))
            
            # 随机旋转
            angle = random.uniform(*config['rotation_range'])
            if abs(angle) > 1:
                roi, mask = rotate_instance(roi, mask, angle)
            
            # 粘贴到目标图片
            for _ in range(config['copies_per_instance']):
                if random.random() > config['paste_prob']:
                    continue
                
                # 随机选择目标图片
                target_img_info = random.choice(target_images)
                target_path = os.path.join(config['output_image_dir'], target_img_info['file_name'])
                
                if not os.path.exists(target_path):
                    continue
                
                target_image = cv2.imread(target_path)
                if target_image is None:
                    continue
                
                # 获取目标图片的现有 bbox
                existing_bboxes = [a['bbox'] for a in new_data['annotations'] 
                                  if a['image_id'] == target_img_info['id']]
                
                # 找到粘贴位置
                position = find_paste_position(target_image, roi.shape, existing_bboxes)
                
                # 使用指定的融合方法
                if config['blend_method'] == 'poisson':
                    pasted_image, new_bbox = poisson_blend(target_image, roi, mask, position)
                else:
                    pasted_image, new_bbox = alpha_blend(target_image, roi, mask, position)
                
                if new_bbox is not None:
                    # 保存粘贴后的图像（覆盖原图）
                    cv2.imwrite(target_path, pasted_image)
                    
                    # 添加新标注
                    max_ann_id += 1
                    new_ann = {
                        'id': max_ann_id,
                        'image_id': target_img_info['id'],
                        'category_id': cls_id,
                        'bbox': new_bbox,
                        'area': new_bbox[2] * new_bbox[3],
                        'iscrowd': 0,
                        'segmentation': []
                    }
                    new_data['annotations'].append(new_ann)
                    pasted_count += 1
                    total_pasted += 1
        
        print(f"   {cls_name} 粘贴完成: {pasted_count} 个实例")
    
    # 保存新标注
    print(f"\n5. 保存新标注文件: {config['output_anno_path']}")
    with open(config['output_anno_path'], 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False)
    
    # 统计结果
    print("\n" + "=" * 60)
    print("Copy-Paste 增强完成！")
    print("=" * 60)
    print(f"\n统计信息:")
    print(f"  原始图片数: {len(data['images'])}")
    print(f"  原始标注数: {len(data['annotations'])}")
    print(f"  新增标注数: {total_pasted}")
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
    copy_paste_augmentation(CONFIG)
