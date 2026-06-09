# -*- coding: utf-8 -*-
"""
Pseudo Hard Negative Mining
用当前模型对训练集推理，找到 FP 区域 crop 作为 hard negative 样本
"""
import os
import json
import cv2
import numpy as np
from paddle.inference import Config, create_predictor
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')
TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, 'A_train', 'Image')
TRAIN_ANNOTATION = os.path.join(BASE_DIR, 'A_train', 'coco', 'annotations', 'instance_train.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'A_train', 'hard_negative')

CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5
CROP_PADDING = 10
MAX_CROPS_PER_IMAGE = 5

CATEGORY_MAP = {0: 'battery', 1: 'board', 2: 'fire'}


def load_model():
    config = Config(
        os.path.join(MODEL_DIR, 'model.pdmodel'),
        os.path.join(MODEL_DIR, 'model.pdiparams')
    )
    config.enable_use_gpu(1000, 0)
    config.enable_memory_optim()
    return create_predictor(config)


def preprocess(img_path, target_size=640):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w, :] = img_resized
    img_norm = padded.astype(np.float32) / 255.0
    img_norm = img_norm.transpose((2, 0, 1))
    return img_norm, (h, w), scale


def run_inference(predictor, img_norm, scale):
    img_batch = np.expand_dims(img_norm, axis=0).astype(np.float32)
    sf = np.array([[scale, scale]], dtype=np.float32)

    predictor.get_input_handle('image').reshape(img_batch.shape)
    predictor.get_input_handle('image').copy_from_cpu(img_batch)
    predictor.get_input_handle('scale_factor').reshape(sf.shape)
    predictor.get_input_handle('scale_factor').copy_from_cpu(sf)

    predictor.run()

    dets = predictor.get_output_handle('multiclass_nms3_0.tmp_0').copy_to_cpu()
    num = predictor.get_output_handle('multiclass_nms3_0.tmp_2').copy_to_cpu()[0]
    return dets[:num]


def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def find_fp(detections, gt_annots):
    fps = []
    for det in detections:
        cls_id_raw = int(det[0])       # model raw: 0,1,2
        conf = det[1]
        bbox = det[2:6].tolist()
        
        # 过滤低置信度
        if conf < CONF_THRESHOLD:
            continue
        
        cls_id_gt = cls_id_raw + 1     # GT uses: 1,2,3
        matched = False
        for gt in gt_annots:
            gt_cls = gt['category_id']
            gx, gy, gw, gh = gt['bbox']
            gt_xyxy = [gx, gy, gx + gw, gy + gh]
            if cls_id_gt == gt_cls and compute_iou(bbox, gt_xyxy) >= IOU_THRESHOLD:
                matched = True
                break
        if not matched:
            fps.append({'class_id': cls_id_raw, 'confidence': float(conf), 'bbox': bbox})
    return fps


def crop_save(img, bbox, out_path, pad=10):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    cv2.imwrite(out_path, crop)
    return True


def check_iou_with_gt(bbox, gt_annots, iou_threshold=0.0):
    """检查 bbox 与所有 GT 的 IoU，如果 IoU > threshold 则返回 False"""
    for gt in gt_annots:
        gx, gy, gw, gh = gt['bbox']
        gt_xyxy = [gx, gy, gx + gw, gy + gh]
        iou = compute_iou(bbox, gt_xyxy)
        if iou > iou_threshold:
            return False
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading model...')
    predictor = load_model()

    print('Loading ground truth...')
    with open(TRAIN_ANNOTATION, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_by_img = {}
    for ann in data['annotations']:
        gt_by_img.setdefault(ann['image_id'], []).append(ann)
    img_info = {img['id']: img for img in data['images']}

    total_fps = 0
    total_crops = 0
    cls_count = {0: 0, 1: 0, 2: 0}

    print(f'Processing {len(img_info)} training images...')
    for idx, (img_id, info) in enumerate(img_info.items()):
        img_path = os.path.join(TRAIN_IMAGE_DIR, info['file_name'])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_norm, orig_shape, scale = preprocess(img_path)
        if img_norm is None:
            continue

        dets = run_inference(predictor, img_norm, scale)
        gt = gt_by_img.get(img_id, [])
        fps = find_fp(dets, gt)

        if len(fps) > MAX_CROPS_PER_IMAGE:
            fps = random.sample(fps, MAX_CROPS_PER_IMAGE)

        for i, fp in enumerate(fps):
            # 检查与 GT 的 IoU，确保为 0
            if not check_iou_with_gt(fp['bbox'], gt, iou_threshold=0.0):
                continue
            
            cname = CATEGORY_MAP[fp['class_id']]
            fname = f'{info["file_name"].split(".")[0]}_{cname}_{i}_{fp["confidence"]:.2f}.jpg'
            out_path = os.path.join(OUTPUT_DIR, fname)
            if crop_save(img, fp['bbox'], out_path, CROP_PADDING):
                total_crops += 1
                cls_count[fp['class_id']] += 1

        total_fps += len(fps)
        if (idx + 1) % 50 == 0:
            print(f'  {idx+1}/{len(img_info)} processed, {total_fps} FP, {total_crops} crops')

    print(f'\nDone:')
    print(f'  Images: {len(img_info)}')
    print(f'  FP: {total_fps}')
    print(f'  Crops: {total_crops}')
    for cid in [0, 1, 2]:
        print(f'  {CATEGORY_MAP[cid]}: {cls_count[cid]}')

    stats = {
        'total_images': len(img_info),
        'total_fp': total_fps,
        'total_crops': total_crops,
        'fp_by_class': cls_count,
        'conf_threshold': CONF_THRESHOLD,
        'iou_threshold': IOU_THRESHOLD
    }
    with open(os.path.join(OUTPUT_DIR, 'mining_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()
