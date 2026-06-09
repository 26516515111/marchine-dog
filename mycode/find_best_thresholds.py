# -*- coding: utf-8 -*-
"""
网格搜索最优置信度阈值
"""
import os
import sys
import json
import cv2
import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
import yaml


# 直接定义需要的类
class Resize:
    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            self.target_size = [target_size, target_size]
        else:
            self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp
    
    def __call__(self, img, im_info):
        img_shape = img.shape
        target_w, target_h = self.target_size[0], self.target_size[1]
        im_scale_y = target_h / float(img_shape[0])
        im_scale_x = target_w / float(img_shape[1])
        img = cv2.resize(img, (target_w, target_h), interpolation=self.interp)
        im_info['scale_factor'] = np.array([im_scale_y, im_scale_x], dtype=np.float32)
        im_info['im_shape'] = np.array([img.shape[0], img.shape[1]], dtype=np.float32)
        return img, im_info


class NormalizeImage:
    def __init__(self, mean=None, std=None, is_scale=True):
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.is_scale = is_scale
    
    def __call__(self, img, im_info):
        img = img.astype(np.float32)
        if self.is_scale:
            img = img / 255.0
        img -= np.array(self.mean, dtype=np.float32)
        img /= np.array(self.std, dtype=np.float32)
        return img, im_info


class Permute:
    def __init__(self, to_bgr=False):
        self.to_bgr = to_bgr
    
    def __call__(self, img, im_info):
        img = img.transpose((2, 0, 1))
        if self.to_bgr:
            img = img[[2, 1, 0], :, :]
        return img, im_info


class PadStride:
    def __init__(self, stride=32):
        self.stride = stride
    
    def __call__(self, img, im_info):
        img = np.pad(img, ((0, 0), (0, (self.stride - img.shape[1] % self.stride) % self.stride), (0, (self.stride - img.shape[2] % self.stride) % self.stride)), mode='constant', constant_values=0)
        return img, im_info


class PredictConfig():
    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf.get('min_subgraph_size', 3)
        self.labels = yml_conf['label_list']
        self.mask = yml_conf.get('mask', False)
        self.use_dynamic_shape = yml_conf.get('use_dynamic_shape', False)


def load_predictor(model_dir):
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams')
    )
    config.enable_use_gpu(2000, 0)
    config.switch_ir_optim(False)
    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor


def preprocess(image_path, preprocess_ops):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_info = {
        'im_shape': np.array([img.shape[0], img.shape[1]], dtype=np.float32),
        'scale_factor': np.array([1.0, 1.0], dtype=np.float32)
    }
    for op in preprocess_ops:
        img, im_info = op(img, im_info)
    return img, im_info


def create_inputs(imgs, im_info):
    inputs = {}
    im_shape = []
    scale_factor = []
    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))
    origin_scale_factor = np.concatenate(scale_factor, axis=0)
    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    padding_imgs_shape = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros((im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = np.array(img, dtype=np.float32)
        padding_imgs.append(padding_im)
        padding_imgs_shape.append(np.array([max_shape_h, max_shape_w]).astype('float32'))
    inputs['image'] = np.stack(padding_imgs, axis=0)
    inputs['im_shape'] = np.stack(padding_imgs_shape, axis=0)
    inputs['scale_factor'] = origin_scale_factor
    return inputs


def predict_with_thresholds(image_list, predictor, preprocess_ops, class_thresholds):
    """使用不同类别不同阈值进行预测"""
    c_results = {"result": []}
    
    for im_path in image_list:
        im, im_info = preprocess(im_path, preprocess_ops)
        if im is None:
            continue
        
        inputs = create_inputs([im], [im_info])
        image_id = os.path.basename(im_path).split('.')[0]
        
        input_names = predictor.get_input_names()
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])
        
        predictor.run()
        
        output_names = predictor.get_output_names()
        num_outs = int(len(output_names) / 2)
        np_boxes = predictor.get_output_handle(output_names[0]).copy_to_cpu()
        np_boxes_num = predictor.get_output_handle(output_names[num_outs]).copy_to_cpu()
        
        im_bboxes_num = np_boxes_num[0]
        
        if im_bboxes_num > 0:
            bbox_results = np_boxes[0:im_bboxes_num, 2:]
            id_results = np_boxes[0:im_bboxes_num, 0]
            score_results = np_boxes[0:im_bboxes_num, 1]
            
            for idx in range(im_bboxes_num):
                class_id = int(id_results[idx]) + 1
                score = float(score_results[idx])
                threshold = class_thresholds.get(class_id, 0.3)
                
                if score >= threshold:
                    c_results["result"].append({
                        "image_id": image_id,
                        "type": class_id,
                        "x": float(bbox_results[idx][0]),
                        "y": float(bbox_results[idx][1]),
                        "width": float(bbox_results[idx][2]) - float(bbox_results[idx][0]),
                        "height": float(bbox_results[idx][3]) - float(bbox_results[idx][1]),
                        "segmentation": []
                    })
    
    return c_results


def calculate_iou(box1, box2):
    """计算 IoU, box 格式: [x, y, w, h]"""
    b1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    b2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def calculate_f1(pred_data, gt_file):
    """计算 F1 分数 (复用官方 calculate_f1.py 逻辑)"""
    CATEGORY_MAP = {1: 'battery', 2: 'board', 3: 'fire'}
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_raw = json.load(f)
    
    # 构建 filename -> image_id 映射
    filename_to_id = {}
    for img in gt_raw['images']:
        filename = os.path.splitext(img['file_name'])[0]
        filename_to_id[filename] = img['id']
    
    # 按 image_id 组织 GT
    ground_truth = {}
    for ann in gt_raw['annotations']:
        img_id = ann['image_id']
        if img_id not in ground_truth:
            ground_truth[img_id] = []
        ground_truth[img_id].append(ann)
    
    # 按 image_id 组织预测 (filename -> 数字ID)
    preds_by_image = {}
    for pred in pred_data['result']:
        img_filename = pred.get('image_id')
        if img_filename is None:
            continue
        filename = os.path.splitext(img_filename)[0]
        img_id = filename_to_id.get(filename)
        if img_id is None:
            continue
        
        bbox = [pred['x'], pred['y'], pred['width'], pred['height']]
        pred_cat = pred['type']
        
        if img_id not in preds_by_image:
            preds_by_image[img_id] = []
        preds_by_image[img_id].append({
            'bbox': bbox,
            'category_id': pred_cat,
            'score': pred.get('score', 1.0)
        })
    
    # 匹配
    metrics = {}
    for cat_id, cat_name in CATEGORY_MAP.items():
        metrics[cat_name] = {'tp': 0, 'fp': 0, 'fn': 0}
    
    matched_gt = {}
    
    for img_id, gts in ground_truth.items():
        preds = preds_by_image.get(img_id, [])
        preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        for pred in preds_sorted:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gts):
                gt_key = f"{img_id}_{gt_idx}"
                if gt_key in matched_gt:
                    continue
                if pred['category_id'] != gt['category_id']:
                    continue
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            cat_name = CATEGORY_MAP.get(pred['category_id'])
            if cat_name is None:
                continue
            if best_iou >= 0.5 and best_gt_idx >= 0:
                gt_key = f"{img_id}_{best_gt_idx}"
                if gt_key not in matched_gt:
                    matched_gt[gt_key] = True
                    metrics[cat_name]['tp'] += 1
                else:
                    metrics[cat_name]['fp'] += 1
            else:
                metrics[cat_name]['fp'] += 1
        
        for gt_idx, gt in enumerate(gts):
            gt_key = f"{img_id}_{gt_idx}"
            if gt_key not in matched_gt:
                cat_name = CATEGORY_MAP.get(gt['category_id'])
                if cat_name:
                    metrics[cat_name]['fn'] += 1
    
    # 计算 P/R/F1
    results = {}
    for cat_name, m in metrics.items():
        tp, fp, fn = m['tp'], m['fp'], m['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[cat_name] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f1': f1}
    
    avg_f1 = np.mean([v['f1'] for v in results.values()])
    return avg_f1, results


def main():
    model_dir = "model/"
    val_txt = "dog/val.txt"
    gt_file = "dog/A_train/coco/annotations/instance_val.json"
    
    # 先用默认阈值验证 F1 计算是否正确
    print("验证 F1 计算 (默认阈值 0.3)...")
    pred_config = PredictConfig(model_dir)
    predictor = load_predictor(model_dir)
    
    preprocess_ops = []
    for op_info in pred_config.preprocess_infos:
        new_op_info = op_info.copy()
        op_type = new_op_info.pop('type')
        preprocess_ops.append(eval(op_type)(**new_op_info))
    
    with open(val_txt, 'r') as f:
        image_list = [line.strip() for line in f if line.strip()]
    
    base_dir = os.path.dirname(os.path.abspath(val_txt))
    image_list = [os.path.join(base_dir, p) if not os.path.isabs(p) else p for p in image_list]
    
    # 测试默认阈值
    default_thresholds = {1: 0.3, 2: 0.3, 3: 0.3}
    pred_data = predict_with_thresholds(image_list, predictor, preprocess_ops, default_thresholds)
    test_f1, test_details = calculate_f1(pred_data, gt_file)
    print(f"默认阈值 F1: {test_f1:.4f}")
    for cat, d in test_details.items():
        print(f"  {cat}: TP={d['tp']} FP={d['fp']} FN={d['fn']} F1={d['f1']:.4f}")
    
    if test_f1 == 0:
        print("\n错误: F1 为 0，计算逻辑有问题，请检查!")
        return
    
    print(f"\n找到 {len(image_list)} 张图片")
    print("\n开始网格搜索最优阈值...")
    
    best_f1 = test_f1
    best_thresholds = default_thresholds.copy()
    best_details = test_details
    
    threshold_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    total = len(threshold_options) ** 3
    count = 0
    
    for t1 in threshold_options:
        for t2 in threshold_options:
            for t3 in threshold_options:
                count += 1
                class_thresholds = {1: t1, 2: t2, 3: t3}
                
                pred_data = predict_with_thresholds(image_list, predictor, preprocess_ops, class_thresholds)
                avg_f1, details = calculate_f1(pred_data, gt_file)
                
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_thresholds = class_thresholds.copy()
                    best_details = details
                    print(f"[{count}/{total}] 新最优: battery={t1} board={t2} fire={t3} -> F1={best_f1:.4f}")
    
    print("\n" + "=" * 60)
    print("最优阈值搜索结果")
    print("=" * 60)
    print(f"battery 阈值: {best_thresholds[1]}")
    print(f"board 阈值: {best_thresholds[2]}")
    print(f"fire 阈值: {best_thresholds[3]}")
    print(f"\n最优 F1: {best_f1:.4f}")
    print("\n各类别详情:")
    for cat, detail in best_details.items():
        print(f"  {cat}: P={detail['precision']:.4f}, R={detail['recall']:.4f}, F1={detail['f1']:.4f}")
    print("=" * 60)
    
    with open('best_thresholds.json', 'w') as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"\n最优阈值已保存到 best_thresholds.json")


if __name__ == '__main__':
    paddle.enable_static()
    main()
