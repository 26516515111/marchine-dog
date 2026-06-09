# -*- coding: utf-8 -*-
"""
分析验证集无效样本原因
"""
import json

# 加载验证集标注
with open(r"D:\work\Marchine Dog\dog\A_train\coco\annotations\instance_val.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']

# 统计每个图片的标注数量
image_ann_count = {}
for ann in annotations:
    img_id = ann['image_id']
    image_ann_count[img_id] = image_ann_count.get(img_id, 0) + 1

# 找出没有标注的图片
no_ann_images = []
has_ann_images = []

for img in images:
    img_id = img['id']
    count = image_ann_count.get(img_id, 0)
    if count == 0:
        no_ann_images.append(img)
    else:
        has_ann_images.append((img, count))

print("=" * 60)
print("验证集标注分析")
print("=" * 60)

print(f"\n总图片数: {len(images)}")
print(f"有标注的图片数: {len(has_ann_images)}")
print(f"没有标注的图片数: {len(no_ann_images)}")
print(f"总标注数: {len(annotations)}")

if no_ann_images:
    print(f"\n没有标注的图片列表:")
    for img in no_ann_images:
        print(f"  - ID={img['id']}, 文件名={img['file_name']}")

print(f"\n有标注的图片统计:")
for img, count in has_ann_images[:10]:
    print(f"  - ID={img['id']}, 文件名={img['file_name']}, 标注数={count}")
if len(has_ann_images) > 10:
    print(f"  ... 还有 {len(has_ann_images) - 10} 个图片")

print("\n" + "=" * 60)
print("结论:")
print("=" * 60)
if no_ann_images:
    print(f"验证集有 {len(no_ann_images)} 个图片没有标注（空图片）")
    print("这些图片会被 PaddleDetection 标记为 invalid")
    print(f"有效图片数: {len(has_ann_images)}")
else:
    print("所有图片都有标注")
