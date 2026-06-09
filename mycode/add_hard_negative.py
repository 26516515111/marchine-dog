# -*- coding: utf-8 -*-
"""
将 Hard Negative 样本加入训练集
- 创建新的 COCO 标注文件（包含 hard negative 空标注）
- 创建新的 train.txt
"""
import json
import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HN_DIR = os.path.join(BASE_DIR, 'A_train', 'hard_negative')
TRAIN_ANNOTATION = os.path.join(BASE_DIR, 'A_train', 'coco', 'annotations', 'instance_train.json')
OUTPUT_ANNOTATION = os.path.join(BASE_DIR, 'A_train', 'coco', 'annotations', 'instance_train_with_hn.json')
OUTPUT_TRAINTXT = os.path.join(BASE_DIR, '..', 'train_with_hn.txt')

def main():
    # 读取原始标注
    with open(TRAIN_ANNOTATION, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取 hard negative 文件
    hn_files = [f for f in os.listdir(HN_DIR) if f.endswith('.jpg')]
    print(f'Hard negative images: {len(hn_files)}')

    # 找到最大 image_id
    max_img_id = max(img['id'] for img in data['images'])
    print(f'Max original image_id: {max_img_id}')

    # 添加 hard negative 图片（空标注）
    for i, fname in enumerate(hn_files):
        new_id = max_img_id + i + 1
        
        # 读取实际图片尺寸
        img_path = os.path.join(HN_DIR, fname)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
        else:
            h, w = 0, 0
        
        data['images'].append({
            'id': new_id,
            'file_name': fname,
            'width': w,
            'height': h
        })
        # 不添加任何 annotation → 空标注 = 负样本

    # 保存新标注
    with open(OUTPUT_ANNOTATION, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f'New annotation: {len(data["images"])} images, {len(data["annotations"])} annotations')
    print(f'Saved to: {OUTPUT_ANNOTATION}')

    # 创建新的 train.txt
    # 读取原始 train.txt
    with open(os.path.join(BASE_DIR, '..', 'train.txt'), 'r', encoding='utf-8') as f:
        original_lines = [line.strip() for line in f if line.strip()]

    # 添加 hard negative 路径
    hn_lines = [os.path.join(HN_DIR, fname).replace('\\', '/') for fname in hn_files]
    all_lines = original_lines + hn_lines

    with open(OUTPUT_TRAINTXT, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')

    print(f'New train.txt: {len(all_lines)} images')
    print(f'Saved to: {OUTPUT_TRAINTXT}')

    # 统计
    print(f'\nSummary:')
    print(f'  Original: {len(original_lines)} images')
    print(f'  Hard negative: {len(hn_lines)} images')
    print(f'  Total: {len(all_lines)} images')

if __name__ == '__main__':
    main()
