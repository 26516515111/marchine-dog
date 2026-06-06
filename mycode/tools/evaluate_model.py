# -*- coding: utf-8 -*-
"""
模型评估脚本
评估训练好的模型在验证集上的性能

使用方法:
    conda activate dog
    python mycode/tools/evaluate_model.py
"""
import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

import paddle
from paddle.io import DataLoader
from PIL import Image
import cv2


# 类别名称
CLASS_NAMES = ["battery", "board", "fire"]


def load_model(model_dir):
    """
    加载 Paddle Inference 模型
    
    Args:
        model_dir: 模型目录
    
    Returns:
        predictor: 推理器
    """
    from paddle.inference import Config, create_predictor
    
    model_file = os.path.join(model_dir, "model.pdmodel")
    params_file = os.path.join(model_dir, "model.pdiparams")
    
    if not os.path.exists(model_file) or not os.path.exists(params_file):
        print(f"错误: 模型文件不存在 {model_dir}")
        return None
    
    # 配置推理选项
    config = Config(model_file, params_file)
    config.enable_use_gpu(2000, 0)
    config.switch_ir_optim(False)
    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    
    # 创建推理器
    predictor = create_predictor(config)
    
    return predictor


def preprocess_image(image_path, input_size=640):
    """
    预处理图片
    
    Args:
        image_path: 图片路径
        input_size: 输入尺寸
    
    Returns:
        image_data: 预处理后的图片数据
        orig_size: 原始图片尺寸
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return None, None
    
    orig_h, orig_w = img.shape[:2]
    
    # 调整大小
    img_resized = cv2.resize(img, (input_size, input_size))
    
    # 归一化
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # 转换为 CHW 格式
    img_chw = img_normalized.transpose(2, 0, 1)
    
    # 添加 batch 维度
    image_data = np.expand_dims(img_chw, axis=0).astype(np.float32)
    
    return image_data, (orig_w, orig_h)


def postprocess_output(cls_output, reg_output, conf_threshold=0.3, nms_threshold=0.5):
    """
    后处理模型输出
    
    Args:
        cls_output: 分类输出
        reg_output: 回归输出
        conf_threshold: 置信度阈值
        nms_threshold: NMS 阈值
    
    Returns:
        detections: 检测结果列表
    """
    detections = []
    
    # 应用 softmax 到分类输出
    cls_probs = paddle.nn.functional.softmax(paddle.to_tensor(cls_output), axis=1).numpy()[0]
    
    # 获取回归输出
    reg_boxes = reg_output[0]
    
    # 对每个类别进行处理
    for class_id in range(len(CLASS_NAMES)):
        # 获取该类别的置信度
        scores = cls_probs[:, class_id]
        
        # 过滤低置信度检测
        mask = scores > conf_threshold
        if not np.any(mask):
            continue
        
        class_scores = scores[mask]
        class_boxes = reg_boxes[mask]
        
        # 应用 NMS
        keep_indices = apply_nms(class_boxes, class_scores, nms_threshold)
        
        for idx in keep_indices:
            detections.append({
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'confidence': float(class_scores[idx]),
                'bbox': class_boxes[idx].tolist()  # [x1, y1, x2, y2] 归一化坐标
            })
    
    return detections


def apply_nms(boxes, scores, nms_threshold):
    """
    应用非极大值抑制
    
    Args:
        boxes: 边界框数组
        scores: 置信度数组
        nms_threshold: NMS 阈值
    
    Returns:
        keep_indices: 保留的索引
    """
    if len(boxes) == 0:
        return []
    
    # 计算面积
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # 按置信度排序
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 计算 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # 保留 IoU 小于阈值的
        inds = np.where(iou <= nms_threshold)[0]
        order = order[inds + 1]
    
    return keep


def calculate_iou(box1, box2):
    """
    计算两个边界框的 IoU
    
    Args:
        box1: 边界框 1 [x1, y1, x2, y2]
        box2: 边界框 2 [x1, y1, x2, y2]
    
    Returns:
        iou: IoU 值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0
    
    return iou


def evaluate_model(model_dir, val_ann_file, img_dir, conf_threshold=0.3):
    """
    评估模型性能
    
    Args:
        model_dir: 模型目录
        val_ann_file: 验证集标注文件
        img_dir: 图片目录
        conf_threshold: 置信度阈值
    
    Returns:
        metrics: 评估指标
    """
    # 加载模型
    predictor = load_model(model_dir)
    if predictor is None:
        return None
    
    # 加载验证集标注
    with open(val_ann_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    images = val_data['images']
    annotations = val_data['annotations']
    
    # 构建图片 ID 到标注的映射
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)
    
    # 统计各类别的 TP, FP, FN
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    
    print(f"评估 {len(images)} 张图片...")
    
    for i, img_info in enumerate(images):
        img_id = img_info['id']
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        # 预处理图片
        image_data, orig_size = preprocess_image(img_path)
        if image_data is None:
            continue
        
        # 获取模型输入名称
        input_names = predictor.get_input_names()
        
        # 设置输入
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(image_data if name == 'image' else np.array([[640, 640]], dtype=np.float32))
        
        # 运行推理
        predictor.run()
        
        # 获取输出
        output_names = predictor.get_output_names()
        cls_output = predictor.get_output_handle(output_names[0]).copy_to_cpu()
        reg_output = predictor.get_output_handle(output_names[1]).copy_to_cpu()
        
        # 后处理
        detections = postprocess_output(cls_output, reg_output, conf_threshold)
        
        # 获取真实标注
        gt_anns = img_to_anns.get(img_id, [])
        
        # 转换真实标注格式
        gt_boxes = []
        gt_classes = []
        for ann in gt_anns:
            bbox = ann['bbox']  # [x, y, w, h]
            # 转换为 [x1, y1, x2, y2] 归一化坐标
            x1 = bbox[0] / img_info['width']
            y1 = bbox[1] / img_info['height']
            x2 = (bbox[0] + bbox[2]) / img_info['width']
            y2 = (bbox[1] + bbox[3]) / img_info['height']
            gt_boxes.append([x1, y1, x2, y2])
            gt_classes.append(ann['category_id'])
        
        # 匹配检测结果和真实标注
        matched_gt = set()
        for det in detections:
            det_class = det['class_id']
            det_box = det['bbox']
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_idx in matched_gt:
                    continue
                
                if det_class != gt_class:
                    continue
                
                iou = calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= 0.5 and best_gt_idx not in matched_gt:
                tp[det_class] += 1
                matched_gt.add(best_gt_idx)
            else:
                fp[det_class] += 1
        
        # 统计未匹配的真实标注
        for gt_idx, gt_class in enumerate(gt_classes):
            if gt_idx not in matched_gt:
                fn[gt_class] += 1
        
        # 打印进度
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(images)} 张图片")
    
    # 计算各类别的 Precision, Recall, F1
    metrics = {}
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    
    for class_id in range(len(CLASS_NAMES)):
        class_name = CLASS_NAMES[class_id]
        
        precision = tp[class_id] / (tp[class_id] + fp[class_id]) if (tp[class_id] + fp[class_id]) > 0 else 0
        recall = tp[class_id] / (tp[class_id] + fn[class_id]) if (tp[class_id] + fn[class_id]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp[class_id],
            'fp': fp[class_id],
            'fn': fn[class_id]
        }
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        
        print(f"\n{class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  TP: {tp[class_id]}, FP: {fp[class_id]}, FN: {fn[class_id]}")
    
    # 计算平均指标
    avg_precision = total_precision / len(CLASS_NAMES)
    avg_recall = total_recall / len(CLASS_NAMES)
    avg_f1 = total_f1 / len(CLASS_NAMES)
    
    print("\n" + "-" * 60)
    print("平均指标:")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Average Recall: {avg_recall:.4f}")
    print(f"  Average F1 Score: {avg_f1:.4f}")
    print("=" * 60)
    
    metrics['average'] = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }
    
    return metrics


def main():
    # 路径配置
    model_dir = "model"
    val_ann_file = "mycode/data/annotations_val.json"
    img_dir = "D:/work/Marchine Dog/A_train/Image"
    
    # 检查文件是否存在
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在 {model_dir}")
        print("请先运行模型导出脚本: python mycode/tools/export_model.py")
        sys.exit(1)
    
    if not os.path.exists(val_ann_file):
        print(f"错误: 验证集标注文件不存在 {val_ann_file}")
        print("请先运行数据转换脚本: python mycode/tools/convert_labelme_to_coco.py")
        sys.exit(1)
    
    # 评估模型
    metrics = evaluate_model(model_dir, val_ann_file, img_dir, conf_threshold=0.3)
    
    if metrics:
        # 保存评估结果
        output_path = "mycode/output/evaluation_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n评估结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
