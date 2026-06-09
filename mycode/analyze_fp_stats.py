import os
import json
import cv2
import numpy as np
from paddle.inference import Config, create_predictor

MODEL_DIR = r'D:\work\Marchine Dog\model'
TRAIN_IMAGE_DIR = r'D:\work\Marchine Dog\dog\A_train\Image'
TRAIN_ANNOTATION = r'D:\work\Marchine Dog\dog\A_train\coco\annotations\instance_train.json'

config = Config(os.path.join(MODEL_DIR, 'model.pdmodel'), os.path.join(MODEL_DIR, 'model.pdiparams'))
config.enable_use_gpu(1000, 0)
config.enable_memory_optim()
predictor = create_predictor(config)

with open(TRAIN_ANNOTATION, 'r', encoding='utf-8') as f:
    data = json.load(f)
gt_by_img = {}
for ann in data['annotations']:
    gt_by_img.setdefault(ann['image_id'], []).append(ann)
img_info = {img['id']: img for img in data['images']}

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

# 分析 FP
fp_stats = {
    'battery': {'count': 0, 'conf_sum': 0, 'iou_sum': 0, 'sizes': []},
    'board': {'count': 0, 'conf_sum': 0, 'iou_sum': 0, 'sizes': []},
    'fire': {'count': 0, 'conf_sum': 0, 'iou_sum': 0, 'sizes': []}
}

for img_id, info in img_info.items():
    img_path = os.path.join(TRAIN_IMAGE_DIR, info['file_name'])
    if not os.path.exists(img_path):
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    padded = np.zeros((640, 640, 3), dtype=np.uint8)
    padded[:new_h, :new_w, :] = img_resized
    img_norm = padded.astype(np.float32) / 255.0
    img_norm = img_norm.transpose((2, 0, 1))
    img_batch = np.expand_dims(img_norm, axis=0).astype(np.float32)
    sf = np.array([[scale, scale]], dtype=np.float32)
    
    predictor.get_input_handle('image').reshape(img_batch.shape)
    predictor.get_input_handle('image').copy_from_cpu(img_batch)
    predictor.get_input_handle('scale_factor').reshape(sf.shape)
    predictor.get_input_handle('scale_factor').copy_from_cpu(sf)
    
    predictor.run()
    
    dets = predictor.get_output_handle('multiclass_nms3_0.tmp_0').copy_to_cpu()
    num = predictor.get_output_handle('multiclass_nms3_0.tmp_2').copy_to_cpu()[0]
    
    gt = gt_by_img.get(img_id, [])
    
    for det in dets[:num]:
        conf = det[1]
        if conf < 0.1:  # 使用新的 score_threshold
            continue
        
        cls_id = int(det[0])
        bbox = det[2:6].tolist()
        
        # 检查是否与 GT 匹配
        matched = False
        max_iou = 0
        for g in gt:
            gx, gy, gw, gh = g['bbox']
            gt_xyxy = [gx, gy, gx + gw, gy + gh]
            iou = compute_iou(bbox, gt_xyxy)
            max_iou = max(max_iou, iou)
            if cls_id == g['category_id'] - 1 and iou >= 0.5:
                matched = True
                break
        
        if not matched:  # FP
            class_name = ['battery', 'board', 'fire'][cls_id]
            fp_stats[class_name]['count'] += 1
            fp_stats[class_name]['conf_sum'] += conf
            fp_stats[class_name]['iou_sum'] += max_iou
            fp_stats[class_name]['sizes'].append((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))

print('FP Analysis (conf >= 0.01):')
print('=' * 60)
for class_name in ['battery', 'board', 'fire']:
    stats = fp_stats[class_name]
    if stats['count'] > 0:
        avg_conf = stats['conf_sum'] / stats['count']
        avg_iou = stats['iou_sum'] / stats['count']
        avg_size = np.mean(stats['sizes'])
        print(f'{class_name}:')
        print(f'  Count: {stats["count"]}')
        print(f'  Avg Conf: {avg_conf:.3f}')
        print(f'  Avg IoU with GT: {avg_iou:.3f}')
        print(f'  Avg Size: {avg_size:.0f}')
        print()
