# -*- coding: utf-8 -*-
"""
分析 FP (假阳性) 样本
找出误检的图片和类别，用于 hard negative mining
"""
import os
import sys
import json
import yaml
import cv2
import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor


# 预处理类
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
        self.labels = yml_conf['label_list']


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


def main():
    model_dir = "model/"
    val_txt = "dog/val.txt"
    gt_file = "dog/A_train/coco/annotations/instance_val.json"
    output_dir = "dog/fp_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    pred_config = PredictConfig(model_dir)
    predictor = load_predictor(model_dir)
    
    preprocess_ops = []
    for op_info in pred_config.preprocess_infos:
        new_op_info = op_info.copy()
        op_type = new_op_info.pop('type')
        preprocess_ops.append(eval(op_type)(**new_op_info))
    
    # 最优阈值
    class_thresholds = {1: 0.3, 2: 0.4, 3: 0.4}
    
    # 加载 GT
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_raw = json.load(f)
    
    filename_to_id = {}
    for img in gt_raw['images']:
        filename = os.path.splitext(img['file_name'])[0]
        filename_to_id[filename] = img['id']
    
    ground_truth = {}
    for ann in gt_raw['annotations']:
        img_id = ann['image_id']
        if img_id not in ground_truth:
            ground_truth[img_id] = []
        ground_truth[img_id].append(ann)
    
    # 获取图片列表
    with open(val_txt, 'r') as f:
        image_list = [line.strip() for line in f if line.strip()]
    
    base_dir = os.path.dirname(os.path.abspath(val_txt))
    image_list = [os.path.join(base_dir, p) if not os.path.isabs(p) else p for p in image_list]
    
    print(f"分析 {len(image_list)} 张图片...")
    
    # 分析每张图片
    fp_by_class = {1: [], 2: [], 3: []}  # 按类别统计 FP
    fp_by_image = {}  # 按图片统计 FP
    
    for im_path in image_list:
        im, im_info = preprocess(im_path, preprocess_ops)
        if im is None:
            continue
        
        inputs = create_inputs([im], [im_info])
        image_id = os.path.basename(im_path).split('.')[0]
        
        # 推理
        input_names = predictor.get_input_names()
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])
        
        predictor.run()
        
        output_names = predictor.get_output_names()
        num_outs = int(len(output_names) / 2)
        np_boxes = predictor.get_output_handle(output_names[0]).copy_to_cpu()
        np_boxes_num = predictor.get_output_handle(output_names[num_outs]).copy_to_cpu()
        
        # 获取 GT
        filename = os.path.splitext(image_id)[0]
        img_id = filename_to_id.get(filename)
        if img_id is None:
            continue
        
        gts = ground_truth.get(img_id, [])
        
        # 分析预测结果
        im_bboxes_num = np_boxes_num[0]
        if im_bboxes_num > 0:
            bbox_results = np_boxes[0:im_bboxes_num, 2:]
            id_results = np_boxes[0:im_bboxes_num, 0]
            score_results = np_boxes[0:im_bboxes_num, 1]
            
            matched_gt = set()
            image_fps = []
            
            for idx in range(im_bboxes_num):
                class_id = int(id_results[idx]) + 1
                score = float(score_results[idx])
                threshold = class_thresholds.get(class_id, 0.3)
                
                if score < threshold:
                    continue
                
                pred_bbox = [
                    float(bbox_results[idx][0]),
                    float(bbox_results[idx][1]),
                    float(bbox_results[idx][2]) - float(bbox_results[idx][0]),
                    float(bbox_results[idx][3]) - float(bbox_results[idx][1])
                ]
                
                # 检查是否匹配 GT
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gts):
                    gt_key = f"{img_id}_{gt_idx}"
                    if gt_key in matched_gt:
                        continue
                    if gt['category_id'] != class_id:
                        continue
                    iou = calculate_iou(pred_bbox, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou < 0.5:
                    # FP
                    fp_by_class[class_id].append({
                        'image_id': image_id,
                        'image_path': im_path,
                        'bbox': pred_bbox,
                        'score': score
                    })
                    image_fps.append({
                        'class_id': class_id,
                        'bbox': pred_bbox,
                        'score': score
                    })
                else:
                    matched_gt.add(f"{img_id}_{best_gt_idx}")
            
            if image_fps:
                fp_by_image[image_id] = image_fps
    
    # 输出统计
    print("\n" + "=" * 60)
    print("FP (假阳性) 分析结果")
    print("=" * 60)
    
    category_map = {1: 'battery', 2: 'board', 3: 'fire'}
    total_fp = 0
    
    for cat_id, cat_name in category_map.items():
        fp_count = len(fp_by_class[cat_id])
        total_fp += fp_count
        print(f"{cat_name}: {fp_count} 个 FP")
    
    print(f"\n总计: {total_fp} 个 FP")
    print(f"涉及图片: {len(fp_by_image)} 张")
    
    # 保存详细结果
    results = {
        'fp_by_class': {str(k): v for k, v in fp_by_class.items()},
        'fp_by_image': fp_by_image
    }
    
    with open(os.path.join(output_dir, 'fp_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成 hard negative 图片列表
    hard_negative_images = list(fp_by_image.keys())
    with open(os.path.join(output_dir, 'hard_negative_images.txt'), 'w', encoding='ascii') as f:
        for img_id in hard_negative_images:
            f.write(f"{img_id}\n")
    
    print(f"\n详细结果已保存到: {output_dir}/fp_analysis.json")
    print(f"Hard negative 图片列表: {output_dir}/hard_negative_images.txt")
    
    # 显示 FP 最多的图片
    if fp_by_image:
        print("\nFP 最多的图片 (Top 10):")
        sorted_fps = sorted(fp_by_image.items(), key=lambda x: len(x[1]), reverse=True)
        for img_id, fps in sorted_fps[:10]:
            print(f"  {img_id}: {len(fps)} 个 FP")


if __name__ == '__main__':
    paddle.enable_static()
    main()
