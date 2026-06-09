# -*- coding: utf-8 -*-
"""
验证 COCO 标注文件是否有效
"""
import json
import os

def verify_coco_annotation(anno_path, image_dir):
    """验证 COCO 标注文件"""
    print("=" * 60)
    print(f"验证标注文件: {anno_path}")
    print(f"图片目录: {image_dir}")
    print("=" * 60)
    
    # 加载标注文件
    with open(anno_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 基本信息
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])
    
    print(f"\n基本信息:")
    print(f"  图片数量: {len(images)}")
    print(f"  标注数量: {len(annotations)}")
    print(f"  类别数量: {len(categories)}")
    
    # 类别信息
    print(f"\n类别信息:")
    for cat in categories:
        print(f"  ID={cat['id']}, Name={cat['name']}")
    
    # 检查图片文件
    print(f"\n检查图片文件:")
    missing_images = []
    valid_images = []
    invalid_images = []
    
    for img in images:
        img_path = os.path.join(image_dir, img['file_name'])
        if os.path.exists(img_path):
            # 检查图片尺寸
            if img.get('width', 0) > 0 and img.get('height', 0) > 0:
                valid_images.append(img)
            else:
                invalid_images.append(img)
        else:
            missing_images.append(img)
    
    print(f"  有效图片: {len(valid_images)}")
    print(f"  缺失图片: {len(missing_images)}")
    print(f"  无效图片(尺寸错误): {len(invalid_images)}")
    
    if missing_images:
        print(f"\n缺失图片列表(前10个):")
        for img in missing_images[:10]:
            print(f"  - {img['file_name']}")
    
    if invalid_images:
        print(f"\n无效图片列表(前10个):")
        for img in invalid_images[:10]:
            print(f"  - {img['file_name']}: width={img.get('width')}, height={img.get('height')}")
    
    # 检查标注
    print(f"\n检查标注:")
    valid_anns = []
    invalid_anns = []
    
    valid_image_ids = set(img['id'] for img in valid_images)
    valid_category_ids = set(cat['id'] for cat in categories)
    
    for ann in annotations:
        # 检查图片 ID
        if ann['image_id'] not in valid_image_ids:
            invalid_anns.append(('image_id 无效', ann))
            continue
        
        # 检查类别 ID
        if ann['category_id'] not in valid_category_ids:
            invalid_anns.append(('category_id 无效', ann))
            continue
        
        # 检查 bbox
        bbox = ann.get('bbox', [])
        if len(bbox) != 4:
            invalid_anns.append(('bbox 格式错误', ann))
            continue
        
        # 检查 bbox 值
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            invalid_anns.append(('bbox 尺寸无效', ann))
            continue
        
        valid_anns.append(ann)
    
    print(f"  有效标注: {len(valid_anns)}")
    print(f"  无效标注: {len(invalid_anns)}")
    
    if invalid_anns:
        print(f"\n无效标注示例(前5个):")
        for reason, ann in invalid_anns[:5]:
            print(f"  - 原因: {reason}")
            print(f"    标注: {ann}")
    
    # 统计各类别数量
    print(f"\n类别统计:")
    cat_counts = {}
    for ann in valid_anns:
        cat_id = ann['category_id']
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
    
    for cat in categories:
        count = cat_counts.get(cat['id'], 0)
        print(f"  {cat['name']}: {count}")
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结:")
    print(f"  有效图片: {len(valid_images)}/{len(images)}")
    print(f"  有效标注: {len(valid_anns)}/{len(annotations)}")
    
    if len(valid_images) == len(images) and len(valid_anns) == len(annotations):
        print("\n✅ 标注文件验证通过!")
    else:
        print("\n⚠️ 标注文件存在问题，请检查上述错误信息")
    print("=" * 60)
    
    return valid_images, valid_anns


if __name__ == "__main__":
    # 验证训练集
    print("\n" + "=" * 60)
    print("验证训练集")
    print("=" * 60)
    verify_coco_annotation(
        r"D:\work\Marchine Dog\dog\A_train\coco\annotations\instance_train.json",
        r"D:\work\Marchine Dog\dog\A_train\coco\train"
    )
    
    # 验证验证集
    print("\n" + "=" * 60)
    print("验证验证集")
    print("=" * 60)
    verify_coco_annotation(
        r"D:\work\Marchine Dog\dog\A_train\coco\annotations\instance_val.json",
        r"D:\work\Marchine Dog\dog\A_train\coco\val"
    )
