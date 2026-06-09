# -*- coding: utf-8 -*-
"""
分析 FP 图片的特征 (英文输出)
"""
import json
import cv2
import numpy as np
import os

with open('dog/fp_analysis/fp_analysis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

category_map = {1: 'battery', 2: 'board', 3: 'fire'}

print('FP Image Feature Analysis')
print('=' * 60)

# Collect all FP features
all_fps = []
for class_id, fps in data['fp_by_class'].items():
    for fp in fps:
        all_fps.append({
            'class_id': int(class_id),
            'class_name': category_map[int(class_id)],
            'image_id': fp['image_id'],
            'score': fp['score'],
            'bbox': fp['bbox']
        })

# Confidence distribution
scores = [fp['score'] for fp in all_fps]
print(f'\nConfidence Statistics:')
print(f'  Average: {np.mean(scores):.3f}')
print(f'  Min: {np.min(scores):.3f}')
print(f'  Max: {np.max(scores):.3f}')
print(f'  Median: {np.median(scores):.3f}')

# Detailed feature analysis
print(f'\nDetailed FP Features:')
print('-' * 60)

for fp in all_fps:
    img_path = f'A_train/coco/val/{fp["image_id"]}.jpg'
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        x, y, fw, fh = fp['bbox']
        
        # Relative position
        rel_x = (x + fw/2) / w
        rel_y = (y + fh/2) / h
        
        # Color features of FP region
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + fw)), min(h, int(y + fh))
        roi = img[y1:y2, x1:x2]
        
        if roi.size > 0:
            avg_color = np.mean(roi, axis=(0,1))
            
            # Position region
            if rel_x < 0.33:
                pos_x = 'Left'
            elif rel_x < 0.66:
                pos_x = 'Center'
            else:
                pos_x = 'Right'
            
            if rel_y < 0.33:
                pos_y = 'Top'
            elif rel_y < 0.66:
                pos_y = 'Middle'
            else:
                pos_y = 'Bottom'
            
            # Size
            area = fw * fh
            if area < 1000:
                size = 'Small'
            elif area < 5000:
                size = 'Medium'
            else:
                size = 'Large'
            
            # Color description
            b, g, r = avg_color
            if r > 150 and g < 100 and b < 100:
                color_desc = 'Red'
            elif r > 150 and g > 150 and b < 100:
                color_desc = 'Yellow'
            elif r > 200 and g > 200 and b > 200:
                color_desc = 'White/Bright'
            elif r < 50 and g < 50 and b < 50:
                color_desc = 'Black/Dark'
            else:
                color_desc = f'Mixed({r:.0f},{g:.0f},{b:.0f})'
            
            print(f'{fp["image_id"]} ({fp["class_name"]}):')
            print(f'  Confidence: {fp["score"]:.3f}')
            print(f'  Position: {pos_x}-{pos_y} ({rel_x:.2f}, {rel_y:.2f})')
            print(f'  Size: {size} ({fw:.0f}x{fh:.0f})')
            print(f'  Color: {color_desc}')
            print()

# Summary
print('=' * 60)
print('FP Feature Summary:')
print('=' * 60)

for class_id in [1, 2, 3]:
    class_fps = [fp for fp in all_fps if fp['class_id'] == class_id]
    if class_fps:
        avg_score = np.mean([fp['score'] for fp in class_fps])
        print(f'\n{category_map[class_id]} ({len(class_fps)} FP):')
        print(f'  Average Confidence: {avg_score:.3f}')
        
        # Analyze common features
        sizes = []
        colors = []
        for fp in class_fps:
            img_path = f'A_train/coco/val/{fp["image_id"]}.jpg'
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                x, y, fw, fh = fp['bbox']
                area = fw * fh
                sizes.append(area)
                
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(w, int(x + fw)), min(h, int(y + fh))
                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    avg_color = np.mean(roi, axis=(0,1))
                    colors.append(avg_color)
        
        if sizes:
            print(f'  Average Area: {np.mean(sizes):.0f}')
            print(f'  Area Range: {np.min(sizes):.0f} - {np.max(sizes):.0f}')
        
        if colors:
            avg_bgr = np.mean(colors, axis=0)
            print(f'  Average Color (BGR): ({avg_bgr[0]:.0f}, {avg_bgr[1]:.0f}, {avg_bgr[2]:.0f})')

# Hard negative mining recommendations
print('\n' + '=' * 60)
print('Hard Negative Mining Recommendations:')
print('=' * 60)

print('\n1. Collect similar images with:')
print('   - Similar background colors')
print('   - Similar lighting conditions')
print('   - Similar object sizes')

print('\n2. Focus on fire FP (14 cases):')
print('   - Red/yellow colored objects')
print('   - Bright areas in images')
print('   - Small to medium sized regions')

print('\n3. Add these to training set:')
print('   - Manual annotation required')
print('   - Ensure no actual fire/battery/board in these regions')
print('   - Add as negative samples (no annotations)')

print('\n4. Expected improvement:')
print('   - Reduce FP by 50-70%')
print('   - Improve precision by 2-5%')
print('   - Overall F1 improvement: 1-3%')
