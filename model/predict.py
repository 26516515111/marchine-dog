"""
3 类目标检测推理脚本（battery / board / fire）
调用方式（评测系统自动调用）：
    python predict.py <data_txt> <result_json>
"""
import os
import time
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'PaddleDetection'))
sys.path.append(os.path.join(BASE_DIR, 'PaddleDetection', 'deploy', 'python'))
import json
import yaml

from PIL import Image
import cv2
import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

# 直接导入需要的模块，避免导入整个 preprocess 模块
def preprocess(image_path, preprocess_ops):
    """预处理图片"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    origin_shape = np.array([img.shape[0], img.shape[1]], dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_info = {
        'im_shape': np.array([img.shape[0], img.shape[1]], dtype=np.float32),
        'scale_factor': np.array([1.0, 1.0], dtype=np.float32),
        'origin_shape': origin_shape
    }
    
    for op in preprocess_ops:
        img, im_info = op(img, im_info)
    
    return img, im_info


class Resize:
    """调整图片大小"""
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
        if self.keep_ratio:
            im_scale = min(target_h / float(img_shape[0]),
                           target_w / float(img_shape[1]))
            resize_h = max(1, int(round(img_shape[0] * im_scale)))
            resize_w = max(1, int(round(img_shape[1] * im_scale)))
            im_scale_y = im_scale
            im_scale_x = im_scale
        else:
            resize_h = target_h
            resize_w = target_w
            im_scale_y = target_h / float(img_shape[0])
            im_scale_x = target_w / float(img_shape[1])
        
        img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interp)
        
        im_info['scale_factor'] = np.array([im_scale_y, im_scale_x], dtype=np.float32)
        im_info['im_shape'] = np.array([img.shape[0], img.shape[1]], dtype=np.float32)
        
        return img, im_info


class NormalizeImage:
    """图片归一化"""
    def __init__(self, mean=None, std=None, is_scale=True, norm_type=None, **kwargs):
        self.is_scale = is_scale
        self.norm_type = norm_type
        # 预计算为 shape=(1,1,3) 的数组，避免每帧重新 allocate 临时 numpy 数组（FPS 优化）
        _mean = mean or [0.485, 0.456, 0.406]
        _std  = std  or [0.229, 0.224, 0.225]
        self._mean = np.array(_mean, dtype=np.float32).reshape(1, 1, 3)
        self._std  = np.array(_std,  dtype=np.float32).reshape(1, 1, 3)
    
    def __call__(self, img, im_info):
        img = img.astype(np.float32)
        if self.norm_type == 'none' or self.is_scale:
            img /= 255.0
        img -= self._mean
        img /= self._std
        return img, im_info


class Permute:
    """通道顺序转换"""
    def __init__(self, to_bgr=False):
        self.to_bgr = to_bgr
    
    def __call__(self, img, im_info):
        img = img.transpose((2, 0, 1))
        if self.to_bgr:
            img = img[[2, 1, 0], :, :]
        return img, im_info


class PadStride:
    """填充图片"""
    def __init__(self, stride=32):
        self.stride = stride
    
    def __call__(self, img, im_info):
        s = self.stride
        pad_h = (s - img.shape[1] % s) % s
        pad_w = (s - img.shape[2] % s) % s
        if pad_h == 0 and pad_w == 0:
            return img, im_info
        # zeros+切片比 np.pad(mode='constant') 少一次全量内存拷贝（FPS 优化）
        out = np.zeros((img.shape[0], img.shape[1] + pad_h, img.shape[2] + pad_w), dtype=img.dtype)
        out[:, :img.shape[1], :img.shape[2]] = img
        return out, im_info


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
        self.tracker = yml_conf.get('tracker', None)
        self.nms = yml_conf.get('NMS', None)
        self.fpn_stride = yml_conf.get('fpn_stride', None)
        
        # Adaptive label mapping based on model's infer_cfg.yml
        target_class_mapping = {"battery": 1, "board": 2, "fire": 3}
        self.id_to_target_class = {}
        for i, label in enumerate(self.labels):
            self.id_to_target_class[i] = target_class_mapping.get(label, i + 1)
            
        self.print_config()

    def print_config(self):
        print('%s: %s' % ('Model Arch', self.arch))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))


class Timer:
    """计时器"""
    def __init__(self):
        self.total_time = 0.0
        self.call_count = 0
    
    def start(self):
        self.start_time = time.time()
    
    def end(self, average=True):
        self.end_time = time.time()
        self.total_time += self.end_time - self.start_time
        self.call_count += 1
        if average:
            return self.total_time / self.call_count
        return self.end_time - self.start_time


def get_test_images(infer_file):
    if not os.path.isabs(infer_file):
        candidates = [
            infer_file,
            os.path.join(os.getcwd(), infer_file),
            os.path.join(BASE_DIR, infer_file),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                infer_file = candidate
                break
    infer_file = os.path.abspath(infer_file)
    infer_dir = os.path.dirname(infer_file)
    with open(infer_file, 'r', encoding='utf-8') as f:
        dirs = f.readlines()
    images = []
    for line in dirs:
        line = line.strip()
        if line:
            line = line.replace('\\', '/')
            if not os.path.isabs(line):
                line = os.path.join(infer_dir, line)
            images.append(line)
    assert len(images) > 0, "no image found in {}".format(infer_file)
    return images


def load_predictor(model_dir):
    """加载推理模型"""
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams')
    )

    # 刻意硬检查：非 GPU 环境直接 crash，拒绝在 CPU 上以极低 FPS 提交
    # if not paddle.is_compiled_with_cuda():
    #     raise RuntimeError('CUDA Paddle is required for official FPS constraints.')
    # 显存池默认 2000 MB（原 200 MB）：给模型 + 激活值留足空间，避免 Paddle 动态增长
    # 导致推理途中显存重新分配拖慢 FPS。若评测机显存 < 3 GB，可设环境变量降低：
    #   PREDICT_GPU_POOL_MB=1000 python predict.py ...
    gpu_pool_mb = int(os.getenv('PREDICT_GPU_POOL_MB', '2000'))
    if gpu_pool_mb < 1000:
        raise ValueError('PREDICT_GPU_POOL_MB must be >= 1000 to keep FPS stable.')
    config.enable_use_gpu(gpu_pool_mb, 0)

    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(False)
    predictor = create_predictor(config)
    return predictor, config


def create_inputs(imgs, im_info):
    inputs = {}
    im_shape = []
    scale_factor = []
    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))
    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)
    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = np.array(img, dtype=np.float32)
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs


class Detector(object):
    def __init__(self, pred_config, model_dir):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(model_dir)
        self.det_times = Timer()
        self.preprocess_ops = self.get_ops()

    def get_ops(self):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        return preprocess_ops

    def predict(self, inputs):
        input_names = self.predictor.get_input_names()
        for name in input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        num_outs = int(len(output_names) / 2)
        np_boxes = self.predictor.get_output_handle(
            output_names[0]).copy_to_cpu()
        np_boxes_num = self.predictor.get_output_handle(
            output_names[num_outs]).copy_to_cpu()
        return dict(boxes=np_boxes, boxes_num=np_boxes_num)


def compute_iou(box1, box2):
    """计算两个框的 IoU
    box 格式: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def nms_per_class(detections, nms_threshold=0.55):
    """按类别执行 NMS
    detections: list of dict, 每个 dict 包含:
        - class_id: 类别ID
        - score: 置信度
        - bbox: [x1, y1, x2, y2]
    nms_threshold: IoU 阈值（float 或 dict {class_id: threshold}）
    返回: NMS 后的 detections
    """
    if not detections:
        return []
    
    # 按类别分组
    class_groups = {}
    for det in detections:
        cls = det['class_id']
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(det)
    
    result = []
    for cls, dets in class_groups.items():
        cls_nms_threshold = nms_threshold.get(cls, 0.55) if isinstance(nms_threshold, dict) else nms_threshold
        # 按置信度降序排序
        dets.sort(key=lambda x: x['score'], reverse=True)
        
        keep = []
        while dets:
            # 取置信度最高的
            best = dets.pop(0)
            keep.append(best)
            
            # 抑制与 best IoU >= threshold 的框
            remaining = []
            for det in dets:
                iou = compute_iou(best['bbox'], det['bbox'])
                if iou < cls_nms_threshold:
                    remaining.append(det)
            dets = remaining
        
        result.extend(keep)
    
    return result


def clip_bbox(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    return [x1, y1, x2, y2]


def scale_bbox(bbox, width_scale, height_scale, image_width, image_height):
    """Scale a detection box around its center and clip it to image bounds."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    new_w = max(0.0, (x2 - x1) * width_scale)
    new_h = max(0.0, (y2 - y1) * height_scale)
    scaled = [
        cx - new_w * 0.5,
        cy - new_h * 0.5,
        cx + new_w * 0.5,
        cy + new_h * 0.5,
    ]
    return clip_bbox(scaled, image_width, image_height)


def apply_bbox_adjustment(det, image_width, image_height, bbox_scales):
    width_scale, height_scale = bbox_scales.get(det['class_id'], (1.0, 1.0))
    det = det.copy()
    det['bbox'] = scale_bbox(det['bbox'], width_scale, height_scale, image_width, image_height)
    return det


def predict_image(detector, image_list, result_path, class_thresholds, nms_threshold=0.55, bbox_scales=None):
    bbox_scales = bbox_scales or {}
    c_results = {"result": []}
    for im_path in image_list:
        input_im_lst = []
        input_im_info_lst = []
        im, im_info = preprocess(im_path, detector.preprocess_ops)
        if im is None:
            continue
        image_height, image_width = im_info['origin_shape']
        input_im_lst.append(im)
        input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        image_id = os.path.basename(im_path).split('.')[0]
        det_results = detector.predict(inputs)
        im_bboxes_num = det_results['boxes_num'][0]
        
        # 收集该图片的所有检测结果
        image_detections = []
        if im_bboxes_num > 0:
            bbox_results  = det_results['boxes'][0:im_bboxes_num, 2:]
            id_results    = det_results['boxes'][0:im_bboxes_num, 0]
            score_results = det_results['boxes'][0:im_bboxes_num, 1]
            for idx in range(im_bboxes_num):
                model_class_idx = int(id_results[idx])
                class_id = detector.pred_config.id_to_target_class.get(model_class_idx, model_class_idx + 1)
                score = float(score_results[idx])
                threshold = class_thresholds.get(class_id, 0.3)
                if score >= threshold:
                    x1 = float(bbox_results[idx][0])
                    y1 = float(bbox_results[idx][1])
                    x2 = float(bbox_results[idx][2])
                    y2 = float(bbox_results[idx][3])
                    image_detections.append({
                        'class_id': class_id,
                        'score': score,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # 应用 NMS
        image_detections = nms_per_class(image_detections, nms_threshold)
        image_detections = [
            apply_bbox_adjustment(det, image_width, image_height, bbox_scales)
            for det in image_detections
        ]
        
        # 添加到结果
        for det in image_detections:
            x1, y1, x2, y2 = det['bbox']
            c_results["result"].append({
                "image_id": image_id,
                "type": det['class_id'],
                "x": x1,
                "y": y1,
                "width":  x2 - x1,
                "height": y2 - y1,
                "segmentation": []
            })
    
    with open(result_path, 'w') as ft:
        json.dump(c_results, ft)
    print("Results written to", result_path)


def predict_image(detector, image_list, result_path, class_thresholds, nms_threshold=0.55, bbox_scales=None):
    bbox_scales = bbox_scales or {}
    batch_size = max(1, int(os.getenv('PREDICT_BATCH_SIZE', '24')))
    c_results = {"result": []}

    for start in range(0, len(image_list), batch_size):
        batch_paths = image_list[start:start + batch_size]
        input_im_lst = []
        input_im_info_lst = []
        valid_paths = []
        origin_shapes = []

        for im_path in batch_paths:
            im, im_info = preprocess(im_path, detector.preprocess_ops)
            if im is None:
                continue
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
            valid_paths.append(im_path)
            origin_shapes.append(im_info['origin_shape'])

        if not input_im_lst:
            continue

        inputs = create_inputs(input_im_lst, input_im_info_lst)
        det_results = detector.predict(inputs)
        box_start = 0

        for batch_idx, im_path in enumerate(valid_paths):
            image_height, image_width = origin_shapes[batch_idx]
            image_id = os.path.basename(im_path).split('.')[0]
            im_bboxes_num = int(det_results['boxes_num'][batch_idx])
            image_detections = []

            if im_bboxes_num > 0:
                image_boxes = det_results['boxes'][box_start:box_start + im_bboxes_num]
                bbox_results = image_boxes[:, 2:]
                id_results = image_boxes[:, 0]
                score_results = image_boxes[:, 1]
                for idx in range(im_bboxes_num):
                    model_class_idx = int(id_results[idx])
                    class_id = detector.pred_config.id_to_target_class.get(model_class_idx, model_class_idx + 1)
                    score = float(score_results[idx])
                    threshold = class_thresholds.get(class_id, 0.3)
                    if score >= threshold:
                        image_detections.append({
                            'class_id': class_id,
                            'score': score,
                            'bbox': [float(x) for x in bbox_results[idx]]
                        })

            box_start += im_bboxes_num
            image_detections = nms_per_class(image_detections, nms_threshold)
            image_detections = [
                apply_bbox_adjustment(det, image_width, image_height, bbox_scales)
                for det in image_detections
            ]

            for det in image_detections:
                x1, y1, x2, y2 = det['bbox']
                c_results["result"].append({
                    "image_id": image_id,
                    "type": det['class_id'],
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "segmentation": []
                })

    with open(result_path, 'w') as ft:
        json.dump(c_results, ft)
    print("Results written to", result_path)


def main(infer_txt, result_path, det_model_path, class_thresholds, nms_threshold=0.55, bbox_scales=None):
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)
    img_list = get_test_images(infer_txt)
    predict_image(detector, img_list, result_path, class_thresholds, nms_threshold, bbox_scales)


if __name__ == '__main__':
    start_time = time.time()
    det_model_path = os.path.join(BASE_DIR, "model")

    # ── 置信度阈值（按类别）──────────────────────────────────────────────
    # class 1 = battery: 0.35（高于全局默认 0.3，去掉低置信度边缘框误检）
    # class 2 = board:   0.3
    # class 3 = fire:    0.4（fire 本身召回率优先，保持不变）
    #
    # 调参依据（raw_preds_with_score.json 扫参结果）：
    #   battery threshold=0.35, battery NMS=0.4 → battery F1=0.8750 (TP=7 FP=1 FN=1)
    #   对比 threshold=0.3 全局 NMS=0.5         → battery F1=0.7778 (TP=7 FP=3 FN=1)
    #   fire F1 在此参数下保持 0.8895，综合 F1=0.8864
    # keep_ratio=True retune on val:
    # battery threshold 0.50 removes one low-score FP (score 0.4529) while
    # the lowest retained battery TP is still around 0.6926.
    class_thresholds = {1: 0.50, 2: 0.3, 3: 0.4}

    # ── NMS IoU 阈值（按类别）────────────────────────────────────────────
    # battery 专项后处理：NMS 调严至 0.4，压掉重复框/低置信度边缘框误检。
    #   误检集中在少数帧，且多为重复框，严 NMS 是最快最稳的修复方式。
    # fire/board 维持 0.5，避免因调严 NMS 损失 fire 的召回率。
    #
    # !! 不建议对 battery 启用小框面积过滤 !!
    #   剩余误检帧（frame_01779）中 battery 框面积约 12680，并非小框，
    #   面积阈值无法区分误检与真实目标，反而可能误伤真实 battery。
    nms_threshold = {
        1: 0.4,   # battery — 专项严化，抑制重复/边缘框
        2: 0.5,   # board
        3: 0.5,   # fire
    }

    bbox_scales = {
        1: (1.00, 1.00),
        2: (1.00, 1.00),
        3: (1.00, 1.00),
    }

    paddle.enable_static()
    infer_txt   = sys.argv[1]
    result_path = sys.argv[2]
    main(infer_txt, result_path, det_model_path, class_thresholds, nms_threshold, bbox_scales)
    print('total time:', time.time() - start_time)
