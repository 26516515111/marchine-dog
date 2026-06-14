# -*- coding: utf-8 -*-
"""
Battery 实例级 Copy-Paste 增强（保守版）
只增强 battery 类别，控制数量，避免过度增强
"""
import json
import os
import random
import cv2
import numpy as np
import shutil

CONFIG = {
    'image_dir': 'dog/A_train/coco/train',
    'anno_path': 'dog/A_train/coco/annotations/instance_train.json',
    'output_image_dir': 'dog/A_train/coco/train_battery_aug',
    'output_anno_path': 'dog/A_train/coco/annotations/instance_train_battery_aug.json',
    'target_class': 1,  # battery only
    'copies_per_image': 1,  # 每张目标图片最多粘贴 1 个 battery
    'max_total_copies': 50,  # 最多新增 50 个 battery 实例
    'scale_range': (0.9, 1.1),  # 缩放范围（保守）
    'rotation_range': (-10, 10),  # 旋转范围（保守）
    'random_seed': 42
}

def load_coco(anno_path):
    with open(anno_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_instance(image, bbox):
    x, y, w, h = [int(v) for v in bbox]
    img_h, img_w = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    roi = image[y1:y2, x1:x2].copy()
    mask = np.ones(roi.shape[:2], dtype=np.uint8) * 255
    return roi, mask, (x1, y1)

def paste_instance(background, roi, mask, position):
    x, y = position
    roi_h, roi_w = roi.shape[:2]
    bg_h, bg_w = background.shape[:2]
    x = max(0, min(x, bg_w - roi_w))
    y = max(0, min(y, bg_h - roi_h))
    x2, y2 = min(x + roi_w, bg_w), min(y + roi_h, bg_h)
    roi_x2, roi_y2 = x2 - x, y2 - y
    mask_3ch = cv2.merge([mask[:roi_y2, :roi_x2]] * 3)
    bg_region = background[y:y2, x:x2]
    roi_region = roi[:roi_y2, :roi_x2]
    mask_float = mask_3ch.astype(float) / 255
    blended = bg_region * (1 - mask_float) + roi_region * mask_float
    background[y:y2, x:x2] = blended.astype(np.uint8)
    return background, [x, y, roi_x2, roi_y2]

def find_paste_position(bg_shape, roi_shape, existing_bboxes, min_iou=0.1):
    bg_h, bg_w = bg_shape[:2]
    roi_h, roi_w = roi_shape[:2]
    for _ in range(30):
        x = random.randint(0, max(0, bg_w - roi_w))
        y = random.randint(0, max(0, bg_h - roi_h))
        new_bbox = [x, y, roi_w, roi_h]
        overlap = False
        for eb in existing_bboxes:
            x1 = max(new_bbox[0], eb[0])
            y1 = max(new_bbox[1], eb[1])
            x2 = min(new_bbox[0]+new_bbox[2], eb[0]+eb[2])
            y2 = min(new_bbox[1]+new_bbox[3], eb[1]+eb[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            area1 = new_bbox[2] * new_bbox[3]
            area2 = eb[2] * eb[3]
            iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
            if iou > min_iou:
                overlap = True
                break
        if not overlap:
            return (x, y)
    return (random.randint(0, max(0, bg_w - roi_w)), random.randint(0, max(0, bg_h - roi_h)))

def main():
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    print("加载标注...")
    data = load_coco(CONFIG['anno_path'])
    img_map = {img['id']: img for img in data['images']}
    
    # 获取 battery 实例
    battery_instances = []
    for ann in data['annotations']:
        if ann['category_id'] == CONFIG['target_class']:
            battery_instances.append({
                'ann': ann,
                'img': img_map[ann['image_id']]
            })
    print(f"battery 实例数: {len(battery_instances)}")
    
    # 选择目标图片（不含 battery 的图片）
    battery_img_ids = set(a['image_id'] for a in data['annotations'] if a['category_id'] == 1)
    target_images = [img for img in data['images'] if img['id'] not in battery_img_ids]
    print(f"不含 battery 的图片数: {len(target_images)}")
    
    if len(target_images) == 0:
        print("没有可用的目标图片")
        return
    
    # 创建输出目录
    os.makedirs(CONFIG['output_image_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG['output_anno_path']), exist_ok=True)
    
    # 复制原始图片
    print("复制原始图片...")
    for img_info in data['images']:
        src = os.path.join(CONFIG['image_dir'], img_info['file_name'])
        dst = os.path.join(CONFIG['output_image_dir'], img_info['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    # 准备新标注
    new_data = {
        'images': data['images'].copy(),
        'annotations': [a.copy() for a in data['annotations']],
        'categories': data['categories'].copy()
    }
    max_ann_id = max(a['id'] for a in data['annotations'])
    
    # 执行 Copy-Paste
    print("执行 battery Copy-Paste...")
    pasted_count = 0
    random.shuffle(battery_instances)
    
    for inst in battery_instances:
        if pasted_count >= CONFIG['max_total_copies']:
            break
        
        ann = inst['ann']
        img_info = inst['img']
        
        src_path = os.path.join(CONFIG['image_dir'], img_info['file_name'])
        if not os.path.exists(src_path):
            continue
        src_image = cv2.imread(src_path)
        if src_image is None:
            continue
        
        roi, mask, offset = extract_instance(src_image, ann['bbox'])
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            continue
        
        # 随机缩放
        scale = random.uniform(*CONFIG['scale_range'])
        new_w, new_h = int(roi.shape[1] * scale), int(roi.shape[0] * scale)
        if new_w > 0 and new_h > 0:
            roi = cv2.resize(roi, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h))
        
        # 随机旋转
        angle = random.uniform(*CONFIG['rotation_range'])
        if abs(angle) > 1:
            h, w = roi.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            cos, sin = abs(M[0,0]), abs(M[0,1])
            new_w2 = int(h*sin + w*cos)
            new_h2 = int(h*cos + w*sin)
            M[0,2] += (new_w2 - w) / 2
            M[1,2] += (new_h2 - h) / 2
            roi = cv2.warpAffine(roi, M, (new_w2, new_h2), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            mask = cv2.warpAffine(mask, M, (new_w2, new_h2), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # 选择目标图片
        target_img = random.choice(target_images)
        target_path = os.path.join(CONFIG['output_image_dir'], target_img['file_name'])
        if not os.path.exists(target_path):
            continue
        target_image = cv2.imread(target_path)
        if target_image is None:
            continue
        
        # 获取现有 bbox
        existing_bboxes = [a['bbox'] for a in new_data['annotations'] if a['image_id'] == target_img['id']]
        
        # 粘贴
        position = find_paste_position(target_image.shape, roi.shape, existing_bboxes)
        pasted_image, new_bbox = paste_instance(target_image, roi, mask, position)
        
        # 保存
        cv2.imwrite(target_path, pasted_image)
        
        # 添加标注
        max_ann_id += 1
        new_data['annotations'].append({
            'id': max_ann_id,
            'image_id': target_img['id'],
            'category_id': 1,
            'bbox': new_bbox,
            'area': new_bbox[2] * new_bbox[3],
            'iscrowd': 0,
            'segmentation': []
        })
        pasted_count += 1
    
    # 保存标注
    print(f"保存新标注到: {CONFIG['output_anno_path']}")
    with open(CONFIG['output_anno_path'], 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False)
    
    # 统计
    cats = {}
    for a in new_data['annotations']:
        c = a['category_id']
        cats[c] = cats.get(c, 0) + 1
    
    print(f"\n增强完成:")
    print(f"  新增 battery: {pasted_count}")
    print(f"  battery: {cats.get(1,0)}")
    print(f"  board: {cats.get(2,0)}")
    print(f"  fire: {cats.get(3,0)}")

if __name__ == '__main__':
    main()
