# -*- coding: utf-8 -*-
"""
测试模型 FPS 的脚本 (基于 predict.py 的推理方式)
"""
import os
import sys
import time
import json
import yaml
import cv2
import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor


# ==================== 预处理类 (与 predict.py 一致) ====================
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
        im_scale_y = target_h / float(img_shape[0])
        im_scale_x = target_w / float(img_shape[1])
        
        img = cv2.resize(img, (target_w, target_h), interpolation=self.interp)
        
        im_info['scale_factor'] = np.array([im_scale_y, im_scale_x], dtype=np.float32)
        im_info['im_shape'] = np.array([img.shape[0], img.shape[1]], dtype=np.float32)
        
        return img, im_info


class NormalizeImage:
    """图片归一化"""
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
        img = np.pad(img, ((0, 0), (0, (self.stride - img.shape[1] % self.stride) % self.stride), (0, (self.stride - img.shape[2] % self.stride) % self.stride)), mode='constant', constant_values=0)
        return img, im_info


# ==================== 模型加载 ====================
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
    """加载推理模型 (GPU)"""
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
    """预处理图片"""
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
    """创建输入张量"""
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
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = np.array(img, dtype=np.float32)
        padding_imgs.append(padding_im)
        padding_imgs_shape.append(
            np.array([max_shape_h, max_shape_w]).astype('float32'))
    inputs['image'] = np.stack(padding_imgs, axis=0)
    inputs['im_shape'] = np.stack(padding_imgs_shape, axis=0)
    inputs['scale_factor'] = origin_scale_factor
    return inputs


def run_inference(predictor, inputs):
    """运行推理"""
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
        input_tensor.copy_from_cpu(inputs[name])
    
    predictor.run()
    
    output_names = predictor.get_output_names()
    num_outs = int(len(output_names) / 2)
    np_boxes = predictor.get_output_handle(output_names[0]).copy_to_cpu()
    np_boxes_num = predictor.get_output_handle(output_names[num_outs]).copy_to_cpu()
    
    return dict(boxes=np_boxes, boxes_num=np_boxes_num)


# ==================== FPS 测试 ====================
def test_fps(model_dir, image_dir, num_warmup=10, num_test=100):
    """测试 FPS"""
    print(f"加载模型: {model_dir}")
    print(f"使用 GPU 推理")
    
    # 加载配置和模型
    pred_config = PredictConfig(model_dir)
    predictor = load_predictor(model_dir)
    
    # 创建预处理操作
    preprocess_ops = []
    for op_info in pred_config.preprocess_infos:
        new_op_info = op_info.copy()
        op_type = new_op_info.pop('type')
        preprocess_ops.append(eval(op_type)(**new_op_info))
    
    # 获取测试图片
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
              if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"错误: 在 {image_dir} 中未找到图片")
        return
    
    print(f"找到 {len(images)} 张图片")
    print(f"预热 {num_warmup} 次，测试 {num_test} 次")
    
    # 预热
    print("预热中...")
    for i in range(num_warmup):
        img_path = images[i % len(images)]
        im, im_info = preprocess(img_path, preprocess_ops)
        if im is not None:
            input_im_lst = [im]
            input_im_info_lst = [im_info]
            inputs = create_inputs(input_im_lst, input_im_info_lst)
            run_inference(predictor, inputs)
    
    # 测试
    print("测试中...")
    times = []
    for i in range(num_test):
        img_path = images[i % len(images)]
        im, im_info = preprocess(img_path, preprocess_ops)
        if im is None:
            continue
        
        input_im_lst = [im]
        input_im_info_lst = [im_info]
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        
        start = time.perf_counter()
        run_inference(predictor, inputs)
        end = time.perf_counter()
        
        times.append(end - start)
    
    # 计算结果
    times = np.array(times)
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print("\n" + "=" * 50)
    print("FPS 测试结果")
    print("=" * 50)
    print(f"平均推理时间: {avg_time * 1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"最短时间: {np.min(times) * 1000:.2f} ms")
    print(f"最长时间: {np.max(times) * 1000:.2f} ms")
    print(f"标准差: {np.std(times) * 1000:.2f} ms")
    print("=" * 50)
    
    if fps >= 20:
        print(f"✓ FPS ({fps:.2f}) >= 20, 满足要求")
    else:
        print(f"✗ FPS ({fps:.2f}) < 20, 不满足要求")
    
    return fps


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python test_fps.py <模型目录> [图片目录] [预热次数] [测试次数]")
        print("示例: python test_fps.py model A_train/coco/val 10 100")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    image_dir = sys.argv[2] if len(sys.argv) > 2 else 'A_train/coco/val'
    num_warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    num_test = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    
    paddle.enable_static()
    test_fps(model_dir, image_dir, num_warmup, num_test)
