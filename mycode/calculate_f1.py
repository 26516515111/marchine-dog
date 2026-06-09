# -*- coding: utf-8 -*-
"""
计算 F1 分数的脚本
"""
import json
import time
import os
import sys

# 类别ID映射
# COCO 标注中的 category_id: 1=battery, 2=board, 3=fire
CATEGORY_MAP = {1: 'battery', 2: 'board', 3: 'fire'}

# 0-based 映射: 0=battery, 1=board, 2=fire
CATEGORY_MAP_0BASED = {0: 'battery', 1: 'board', 2: 'fire'}

def load_ground_truth(gt_file):
    """加载真实标注"""
    with open(gt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 构建图片ID到图片信息的映射
    images = {img['id']: img for img in data['images']}
    
    # 创建文件名到图片ID的映射
    filename_to_id = {}
    for img in data['images']:
        # 去掉扩展名
        filename = os.path.splitext(img['file_name'])[0]
        filename_to_id[filename] = img['id']
    
    # 按图片ID组织标注
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    return images, annotations_by_image, filename_to_id

def load_predictions(pred_file):
    """加载预测结果"""
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理不同的格式
    if isinstance(data, dict) and 'result' in data:
        predictions = data['result']
    elif isinstance(data, list):
        predictions = data
    else:
        predictions = [data]
    
    return predictions

def calculate_iou(box1, box2):
    """计算 IoU (Intersection over Union)
    box 格式: [x, y, w, h]
    """
    # COCO 格式 [x, y, w, h] -> [x1, y1, x2, y2]
    b1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    b2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    
    # 计算交集
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    
    # 计算并集
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_f1(ground_truth, predictions, filename_to_id, iou_threshold=0.5):
    """
    计算每个类别的 Precision, Recall, F1
    
    返回:
    - metrics: 每个类别的指标
    - overall_score: 综合得分
    """
    # 初始化统计
    metrics = {}
    for cat_id, cat_name in CATEGORY_MAP.items():
        metrics[cat_name] = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # 按图片ID组织预测结果 (使用数字ID)
    preds_by_image = {}
    for pred in predictions:
        # 处理不同的字段名
        img_filename = pred.get('image_id')
        if img_filename is None:
            continue
        
        # 将文件名转换为数字ID
        # 去掉扩展名
        filename = os.path.splitext(img_filename)[0]
        img_id = filename_to_id.get(filename)
        if img_id is None:
            continue
        
        # 获取类别ID (可能是 'type' 或 'category_id')
        pred_cat_raw = pred.get('type') or pred.get('category_id')
        if pred_cat_raw is None:
            continue
        
        # 尝试两种映射方式
        # 1. 直接使用 (1-based)
        pred_cat_1based = pred_cat_raw
        # 2. 减 1 (0-based)
        pred_cat_0based = pred_cat_raw - 1
        
        # 获取边界框 (可能是 x,y,w,h 或 bbox)
        if 'bbox' in pred:
            bbox = pred['bbox']
        elif 'x' in pred and 'y' in pred and 'width' in pred and 'height' in pred:
            bbox = [pred['x'], pred['y'], pred['width'], pred['height']]
        else:
            continue
        
        if img_id not in preds_by_image:
            preds_by_image[img_id] = []
        preds_by_image[img_id].append({
            'bbox': bbox,
            'category_id_1based': pred_cat_1based,
            'category_id_0based': pred_cat_0based,
            'score': pred.get('score', 1.0)
        })
    
    # 匹配预测和真实标注
    matched_gt = {}  # 记录已匹配的真实标注
    
    for img_id, gts in ground_truth.items():
        preds = preds_by_image.get(img_id, [])
        
        # 按置信度排序预测结果
        preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        for pred in preds_sorted:
            pred_box = pred['bbox']
            
            best_iou = 0
            best_gt_idx = -1
            best_mapping = None
            
            # 查找最佳匹配的真实标注
            for gt_idx, gt in enumerate(gts):
                gt_key = f"{img_id}_{gt_idx}"
                if gt_key in matched_gt:
                    continue
                
                gt_box = gt['bbox']
                gt_cat = gt['category_id']
                
                # 尝试两种映射方式
                # 1. 使用 1-based 映射
                if pred['category_id_1based'] == gt_cat:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        best_mapping = '1based'
                
                # 2. 使用 0-based 映射
                if pred['category_id_0based'] == gt_cat:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        best_mapping = '0based'
            
            # 判断是否匹配成功
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_key = f"{img_id}_{best_gt_idx}"
                if gt_key not in matched_gt:
                    matched_gt[gt_key] = True
                    # 使用对应的映射方式获取类别名称
                    if best_mapping == '1based':
                        cat_name = CATEGORY_MAP.get(pred['category_id_1based'])
                    else:
                        cat_name = CATEGORY_MAP_0BASED.get(pred['category_id_0based'])
                    if cat_name:
                        metrics[cat_name]['true_positives'] += 1
                else:
                    cat_name = CATEGORY_MAP.get(pred['category_id_1based'])
                    if cat_name:
                        metrics[cat_name]['false_positives'] += 1
            else:
                cat_name = CATEGORY_MAP.get(pred['category_id_1based'])
                if cat_name:
                    metrics[cat_name]['false_positives'] += 1
        
        # 计算未匹配的真实标注数量
        for gt_idx, gt in enumerate(gts):
            gt_key = f"{img_id}_{gt_idx}"
            if gt_key not in matched_gt:
                cat_name = CATEGORY_MAP.get(gt['category_id'])
                if cat_name:
                    metrics[cat_name]['false_negatives'] += 1
    
    # 计算 Precision, Recall, F1
    total_f1 = 0
    valid_classes = 0
    
    for cat_name, metric in metrics.items():
        tp = metric['true_positives']
        fp = metric['false_positives']
        fn = metric['false_negatives']
        
        if tp + fp > 0:
            metric['precision'] = tp / (tp + fp)
        else:
            metric['precision'] = 0.0
        
        if tp + fn > 0:
            metric['recall'] = tp / (tp + fn)
        else:
            metric['recall'] = 0.0
        
        if metric['precision'] + metric['recall'] > 0:
            metric['f1'] = (2 * metric['precision'] * metric['recall'] / 
                          (metric['precision'] + metric['recall']))
        else:
            metric['f1'] = 0.0
        
        total_f1 += metric['f1']
        valid_classes += 1
    
    # 计算综合得分
    overall_score = total_f1 / valid_classes if valid_classes > 0 else 0.0
    
    return metrics, overall_score

def main():
    if len(sys.argv) != 3:
        print("用法: python calculate_f1.py <预测文件> <真实标注文件>")
        print("示例: python calculate_f1.py val_result.json A_train/coco/annotations/instance_val.json")
        sys.exit(1)
    
    pred_file = sys.argv[1]
    gt_file = sys.argv[2]
    
    print("正在加载数据...")
    images, ground_truth, filename_to_id = load_ground_truth(gt_file)
    predictions = load_predictions(pred_file)
    
    print(f"真实标注数量: {sum(len(v) for v in ground_truth.values())}")
    print(f"预测结果数量: {len(predictions)}")
    
    print("\n正在计算 F1 分数...")
    metrics, overall_score = calculate_f1(ground_truth, predictions, filename_to_id)
    
    print("\n" + "=" * 60)
    print("F1 分数统计")
    print("=" * 60)
    
    for cat_name, metric in metrics.items():
        print(f"\n{cat_name}:")
        print(f"  真阳性 (TP): {metric['true_positives']}")
        print(f"  假阳性 (FP): {metric['false_positives']}")
        print(f"  假阴性 (FN): {metric['false_negatives']}")
        print(f"  精确率 (P): {metric['precision']:.4f}")
        print(f"  召回率 (R): {metric['recall']:.4f}")
        print(f"  F1 分数: {metric['f1']:.4f}")
    
    print("\n" + "=" * 60)
    print(f"综合得分 (F1 均值): {overall_score:.4f}")
    print("=" * 60)
    
    # FPS 检查
    print("\nFPS 检查:")
    print("FPS 需要在评测时单独测量，确保 >= 20")
    print("如果 FPS < 20，综合得分强制为 0")

if __name__ == '__main__':
    main()
