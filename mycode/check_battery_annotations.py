# -*- coding: utf-8 -*-
import json
import numpy as np

data = json.load(open('dog/A_train/coco/annotations/instance_train.json'))
img_map = {img['id']: img for img in data['images']}
battery_anns = [a for a in data['annotations'] if a['category_id'] == 1]

print('=== Extreme aspect ratio battery annotations ===')
for ann in battery_anns:
    w, h = ann['bbox'][2], ann['bbox'][3]
    ratio = w / h if h > 0 else 0
    if ratio > 5 or (h / w if w > 0 else 0) > 5:
        img_info = img_map[ann['image_id']]
        fname = img_info['file_name']
        print(f'  Image: {fname}, bbox: {ann["bbox"]}, ratio: {ratio:.2f}')

print()
print('=== Image resolution stats ===')
resolutions = set()
for img in data['images']:
    resolutions.add((img['width'], img['height']))
print(f'Unique resolutions: {len(resolutions)}')
for res in sorted(resolutions):
    print(f'  {res[0]}x{res[1]}')

print()
print('=== Battery size distribution ===')
areas = [a['bbox'][2] * a['bbox'][3] for a in battery_anns]
print(f'  <1000 px^2: {sum(1 for a in areas if a < 1000)}')
print(f'  1000-5000 px^2: {sum(1 for a in areas if 1000 <= a < 5000)}')
print(f'  5000-10000 px^2: {sum(1 for a in areas if 5000 <= a < 10000)}')
print(f'  10000-50000 px^2: {sum(1 for a in areas if 10000 <= a < 50000)}')
print(f'  >50000 px^2: {sum(1 for a in areas if a >= 50000)}')
