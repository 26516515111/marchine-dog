# -*- coding: utf-8 -*-
"""
分析标注中 bbox 的大小分布
"""
import json
import numpy as np

def analyze_bbox_sizes(anno_path):
    """分析 bbox 大小分布"""
    with open(anno_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    
    widths = []
    heights = []
    areas = []
    
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        w, h = bbox[2], bbox[3]
        widths.append(w)
        heights.append(h)
        areas.append(w * h)
    
    widths = np.array(widths)
    heights = np.array(heights)
    areas = np.array(areas)
    
    print("=" * 60)
    print(f"标注文件: {anno_path}")
    print("=" * 60)
    print(f"\n总标注数: {len(annotations)}")
    
    print(f"\n宽度统计 (像素):")
    print(f"  最小值: {widths.min():.1f}")
    print(f"  最大值: {widths.max():.1f}")
    print(f"  平均值: {widths.mean():.1f}")
    print(f"  中位数: {np.median(widths):.1f}")
    print(f"  标准差: {widths.std():.1f}")
    
    print(f"\n高度统计 (像素):")
    print(f"  最小值: {heights.min():.1f}")
    print(f"  最大值: {heights.max():.1f}")
    print(f"  平均值: {heights.mean():.1f}")
    print(f"  中位数: {np.median(heights):.1f}")
    print(f"  标准差: {heights.std():.1f}")
    
    print(f"\n面积统计 (像素²):")
    print(f"  最小值: {areas.min():.1f}")
    print(f"  最大值: {areas.max():.1f}")
    print(f"  平均值: {areas.mean():.1f}")
    print(f"  中位数: {np.median(areas):.1f}")
    
    # 按类别统计
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    cat_sizes = {cat_id: {'widths': [], 'heights': []} for cat_id in categories}
    
    for ann in annotations:
        cat_id = ann['category_id']
        bbox = ann['bbox']
        cat_sizes[cat_id]['widths'].append(bbox[2])
        cat_sizes[cat_id]['heights'].append(bbox[3])
    
    print(f"\n各类别大小统计:")
    for cat_id, cat_name in categories.items():
        w = np.array(cat_sizes[cat_id]['widths'])
        h = np.array(cat_sizes[cat_id]['heights'])
        print(f"\n  {cat_name} (ID={cat_id}):")
        print(f"    数量: {len(w)}")
        print(f"    宽度: {w.mean():.1f} ± {w.std():.1f} (范围: {w.min():.1f} - {w.max():.1f})")
        print(f"    高度: {h.mean():.1f} ± {h.std():.1f} (范围: {h.min():.1f} - {h.max():.1f})")
    
    # 推荐输入尺寸
    max_dim = max(widths.max(), heights.max())
    avg_dim = (widths.mean() + heights.mean()) / 2
    
    print(f"\n" + "=" * 60)
    print("推荐输入尺寸:")
    print("=" * 60)
    print(f"  最大目标尺寸: {max_dim:.0f} 像素")
    print(f"  平均目标尺寸: {avg_dim:.0f} 像素")
    print(f"  图片原始尺寸: 1920 x 1080")
    print(f"\n  推荐 target_size: [512, 640, 768] 或 [640, 768, 896]")
    print(f"  (输入尺寸应大于最大目标尺寸的 2-3 倍)")
    print("=" * 60)


if __name__ == "__main__":
    # 分析训练集
    analyze_bbox_sizes(r"D:\work\Marchine Dog\dog\A_train\coco\annotations\instance_train.json")
    
    # 分析验证集
    analyze_bbox_sizes(r"D:\work\Marchine Dog\dog\A_train\coco\annotations\instance_val.json")
