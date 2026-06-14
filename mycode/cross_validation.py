# -*- coding: utf-8 -*-
"""
多折交叉验证脚本
对每个划分运行 predict.py 并计算 F1，找出平均最优参数
"""
import json
import os
import subprocess
import numpy as np
from collections import defaultdict

CONFIG = {
    'split_dir': 'dog/A_train/coco/splits',
    'n_splits': 5,
    'model_dir': 'model',
    'temp_dir': 'temp_cv',
    'gt_path': 'dog/A_train/coco/annotations/instance_val.json',
}

def run_predict(image_list_file, result_file):
    """运行 predict.py"""
    # 使用绝对路径，加引号处理空格
    abs_image_list = os.path.abspath(image_list_file)
    abs_result = os.path.abspath(result_file)
    cmd = f'python predict.py "{abs_image_list}" "{abs_result}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # 检查是否成功（通过输出判断）
    return 'Results written to' in result.stdout

def calculate_f1(pred_file, gt_file):
    """计算 F1"""
    # 导入 calculate_f1 模块
    import sys
    sys.path.append('dog/mycode')
    from calculate_f1 import load_ground_truth, load_predictions, calculate_f1
    
    images, ground_truth, filename_to_id = load_ground_truth(gt_file)
    predictions = load_predictions(pred_file)
    metrics, overall_score = calculate_f1(ground_truth, predictions, filename_to_id)
    
    return {
        'battery_f1': metrics['battery']['f1'],
        'board_f1': metrics['board']['f1'],
        'fire_f1': metrics['fire']['f1'],
        'overall_f1': overall_score,
        'battery_tp': metrics['battery']['true_positives'],
        'battery_fp': metrics['battery']['false_positives'],
        'battery_fn': metrics['battery']['false_negatives'],
    }

def create_val_list_from_split(split_file, output_file):
    """从划分文件创建验证集图片列表"""
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    val_images = split_data['val']['images']
    
    # 获取图片路径（使用绝对路径）
    img_dir = os.path.abspath('dog/A_train/coco/train')
    
    with open(output_file, 'w') as f:
        for img in val_images:
            img_path = os.path.join(img_dir, img['file_name'])
            f.write(img_path + '\n')
    
    return len(val_images)

def create_gt_from_split(split_file, output_file):
    """从划分文件创建验证集 GT"""
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    gt_data = {
        'images': split_data['val']['images'],
        'annotations': split_data['val']['annotations'],
        'categories': [
            {'id': 1, 'name': 'battery'},
            {'id': 2, 'name': 'board'},
            {'id': 3, 'name': 'fire'}
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, ensure_ascii=False)
    
    return len(gt_data['annotations'])

def main():
    print("=" * 70)
    print("多折交叉验证")
    print("=" * 70)
    
    os.makedirs(CONFIG['temp_dir'], exist_ok=True)
    
    all_results = []
    
    for i in range(CONFIG['n_splits']):
        split_file = os.path.join(CONFIG['split_dir'], f'split_{i}.json')
        
        if not os.path.exists(split_file):
            print(f"Split {i} 不存在，跳过")
            continue
        
        print(f"\n--- Split {i} ---")
        
        # 创建临时文件
        val_list_file = os.path.join(CONFIG['temp_dir'], f'val_list_{i}.txt')
        gt_file = os.path.join(CONFIG['temp_dir'], f'gt_{i}.json')
        pred_file = os.path.join(CONFIG['temp_dir'], f'pred_{i}.json')
        
        # 创建验证集列表和 GT
        n_images = create_val_list_from_split(split_file, val_list_file)
        n_anns = create_gt_from_split(split_file, gt_file)
        print(f"  验证图片数: {n_images}, 标注数: {n_anns}")
        
        # 运行预测
        print(f"  运行预测...")
        success = run_predict(val_list_file, pred_file)
        
        if not success:
            print(f"  预测失败，跳过")
            continue
        
        # 计算 F1
        print(f"  计算 F1...")
        metrics = calculate_f1(pred_file, gt_file)
        all_results.append(metrics)
        
        print(f"  Battery F1: {metrics['battery_f1']:.4f}")
        print(f"  Board F1: {metrics['board_f1']:.4f}")
        print(f"  Fire F1: {metrics['fire_f1']:.4f}")
        print(f"  Overall F1: {metrics['overall_f1']:.4f}")
    
    # 汇总统计
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)
    
    if len(all_results) == 0:
        print("没有有效的验证结果")
        return
    
    # 计算均值和标准差
    battery_f1s = [r['battery_f1'] for r in all_results]
    board_f1s = [r['board_f1'] for r in all_results]
    fire_f1s = [r['fire_f1'] for r in all_results]
    overall_f1s = [r['overall_f1'] for r in all_results]
    
    print(f"\nBattery F1: {np.mean(battery_f1s):.4f} ± {np.std(battery_f1s):.4f}")
    print(f"Board F1: {np.mean(board_f1s):.4f} ± {np.std(board_f1s):.4f}")
    print(f"Fire F1: {np.mean(fire_f1s):.4f} ± {np.std(fire_f1s):.4f}")
    print(f"Overall F1: {np.mean(overall_f1s):.4f} ± {np.std(overall_f1s):.4f}")
    
    # 找出最稳定的划分（接近均值的）
    mean_overall = np.mean(overall_f1s)
    print(f"\n各 Split 与均值差距:")
    for i, r in enumerate(all_results):
        diff = r['overall_f1'] - mean_overall
        print(f"  Split {i}: {r['overall_f1']:.4f} ({'+' if diff >= 0 else ''}{diff:.4f})")
    
    # 保存结果
    result_file = os.path.join(CONFIG['temp_dir'], 'cv_results.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': all_results,
            'summary': {
                'battery_f1_mean': float(np.mean(battery_f1s)),
                'battery_f1_std': float(np.std(battery_f1s)),
                'board_f1_mean': float(np.mean(board_f1s)),
                'board_f1_std': float(np.std(board_f1s)),
                'fire_f1_mean': float(np.mean(fire_f1s)),
                'fire_f1_std': float(np.std(fire_f1s)),
                'overall_f1_mean': float(np.mean(overall_f1s)),
                'overall_f1_std': float(np.std(overall_f1s)),
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {result_file}")

if __name__ == '__main__':
    main()
