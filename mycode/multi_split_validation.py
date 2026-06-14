# -*- coding: utf-8 -*-
"""
多划分验证脚本
将数据集随机划分为 train/val 多次，检查模型在不同划分下的稳定性
"""
import json
import os
import random
import numpy as np
from collections import defaultdict

CONFIG = {
    'anno_path': 'dog/A_train/coco/annotations/instance_train.json',
    'output_dir': 'dog/A_train/coco/splits',
    'n_splits': 5,
    'val_ratio': 0.2,
    'random_seed': 42
}

def load_coco(anno_path):
    with open(anno_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_split(data, val_ratio, seed):
    random.seed(seed)
    images = data['images'].copy()
    random.shuffle(images)
    
    n_val = int(len(images) * val_ratio)
    val_images = images[:n_val]
    train_images = images[n_val:]
    
    val_ids = set(img['id'] for img in val_images)
    train_ids = set(img['id'] for img in train_images)
    
    train_anns = [a for a in data['annotations'] if a['image_id'] in train_ids]
    val_anns = [a for a in data['annotations'] if a['image_id'] in val_ids]
    
    return {
        'train': {'images': train_images, 'annotations': train_anns},
        'val': {'images': val_images, 'annotations': val_anns}
    }

def count_categories(annotations):
    counts = defaultdict(int)
    for ann in annotations:
        counts[ann['category_id']] += 1
    return dict(counts)

def main():
    print("=" * 60)
    print("多划分验证")
    print("=" * 60)
    
    data = load_coco(CONFIG['anno_path'])
    print(f"总图片数: {len(data['images'])}")
    print(f"总标注数: {len(data['annotations'])}")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    all_stats = []
    
    for i in range(CONFIG['n_splits']):
        seed = CONFIG['random_seed'] + i
        split = create_split(data, CONFIG['val_ratio'], seed)
        
        train_cats = count_categories(split['train']['annotations'])
        val_cats = count_categories(split['val']['annotations'])
        
        stats = {
            'split_id': i,
            'seed': seed,
            'train_images': len(split['train']['images']),
            'val_images': len(split['val']['images']),
            'train_anns': len(split['train']['annotations']),
            'val_anns': len(split['val']['annotations']),
            'train_cats': train_cats,
            'val_cats': val_cats
        }
        all_stats.append(stats)
        
        print(f"\n--- Split {i} (seed={seed}) ---")
        print(f"  Train: {stats['train_images']} images, {stats['train_anns']} annotations")
        print(f"  Val: {stats['val_images']} images, {stats['val_anns']} annotations")
        print(f"  Train class distribution: battery={train_cats.get(1,0)}, board={train_cats.get(2,0)}, fire={train_cats.get(3,0)}")
        print(f"  Val class distribution: battery={val_cats.get(1,0)}, board={val_cats.get(2,0)}, fire={val_cats.get(3,0)}")
        
        # 保存划分
        split_file = os.path.join(CONFIG['output_dir'], f'split_{i}.json')
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump({
                'train': split['train'],
                'val': split['val'],
                'metadata': stats
            }, f, ensure_ascii=False)
    
    # 汇总统计
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)
    
    train_counts = [s['train_anns'] for s in all_stats]
    val_counts = [s['val_anns'] for s in all_stats]
    
    print(f"Train annotations: {np.mean(train_counts):.0f} ± {np.std(train_counts):.0f}")
    print(f"Val annotations: {np.mean(val_counts):.0f} ± {np.std(val_counts):.0f}")
    
    # 各类别在 val 集中的分布
    for cls_id, cls_name in [(1, 'battery'), (2, 'board'), (3, 'fire')]:
        cls_counts = [s['val_cats'].get(cls_id, 0) for s in all_stats]
        print(f"Val {cls_name}: {np.mean(cls_counts):.0f} ± {np.std(cls_counts):.0f}")
    
    print(f"\n划分文件保存在: {CONFIG['output_dir']}")

if __name__ == '__main__':
    main()
