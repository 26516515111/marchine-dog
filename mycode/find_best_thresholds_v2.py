# -*- coding: utf-8 -*-
"""
Per-Class 置信度阈值优化脚本 (改进版)

功能：
  - 两阶段搜索：粗粒度(step=0.05)定位区间 -> 细粒度(step=0.01)精化
  - 先独立优化每类阈值，再联合微调，大幅减少推理次数
  - 支持从已有预测 JSON 文件进行搜索（无需重复推理）

用法：
  # 模式1: 直接从模型推理 + 搜索
  python mycode/find_best_thresholds_v2.py --model model/ --val-txt val.txt --gt A_train/coco/annotations/instance_val.json

  # 模式2: 从已有低阈值预测结果搜索（推荐，速度更快）
  # 先用极低阈值(0.01)生成预测：
  #   python mycode/predict.py val.txt raw_preds.json  (修改 score_threshold=0.01 导出模型)
  # 然后搜索：
  python mycode/find_best_thresholds_v2.py --pred-json raw_preds.json --gt A_train/coco/annotations/instance_val.json
"""
import os
import sys
import json
import argparse
import itertools
import numpy as np
from collections import defaultdict

CATEGORY_MAP = {1: 'battery', 2: 'board', 3: 'fire'}
CATEGORY_IDS = [1, 2, 3]  # battery=1, board=2, fire=3


# ─────────────────────────────────────────────────────────────────
# 核心 F1 计算（复用 calculate_f1.py 逻辑）
# ─────────────────────────────────────────────────────────────────

def load_gt(gt_file):
    with open(gt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    filename_to_id = {os.path.splitext(img['file_name'])[0]: img['id'] for img in data['images']}
    gt_by_image = defaultdict(list)
    for ann in data['annotations']:
        gt_by_image[ann['image_id']].append(ann)
    return gt_by_image, filename_to_id


def load_raw_preds(pred_file, filename_to_id):
    """加载原始预测（保留所有低置信度框），按 image_id 组织"""
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    predictions = data['result'] if isinstance(data, dict) and 'result' in data else data

    preds_by_image = defaultdict(list)
    for pred in predictions:
        img_filename = os.path.splitext(pred.get('image_id', ''))[0]
        img_id = filename_to_id.get(img_filename)
        if img_id is None:
            continue
        cat_id = pred.get('type') or pred.get('category_id')
        if cat_id not in CATEGORY_IDS:
            continue
        score = float(pred.get('score', 1.0))
        if 'bbox' in pred:
            bbox = pred['bbox']
        else:
            bbox = [pred['x'], pred['y'], pred['width'], pred['height']]
        preds_by_image[img_id].append({'bbox': bbox, 'cat_id': cat_id, 'score': score})
    return preds_by_image


def calc_iou(b1, b2):
    """b1, b2: [x, y, w, h]"""
    ax1, ay1 = b1[0], b1[1]
    ax2, ay2 = b1[0] + b1[2], b1[1] + b1[3]
    bx1, by1 = b2[0], b2[1]
    bx2, by2 = b2[0] + b2[2], b2[1] + b2[3]
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    if inter == 0:
        return 0.0
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


def eval_thresholds(preds_by_image, gt_by_image, thresholds, iou_thr=0.5):
    """
    thresholds: dict {cat_id: score_threshold}
    返回: avg_f1, per_class_details
    """
    stats = {cid: {'tp': 0, 'fp': 0, 'fn': 0} for cid in CATEGORY_IDS}

    all_img_ids = set(gt_by_image.keys()) | set(preds_by_image.keys())

    for img_id in all_img_ids:
        gts = gt_by_image.get(img_id, [])
        all_preds = preds_by_image.get(img_id, [])

        # 按阈值过滤
        preds = [p for p in all_preds if p['score'] >= thresholds.get(p['cat_id'], 0.3)]
        preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)

        matched_gt = set()
        for pred in preds_sorted:
            best_iou, best_idx = 0.0, -1
            for gi, gt in enumerate(gts):
                if gi in matched_gt or gt['category_id'] != pred['cat_id']:
                    continue
                iou = calc_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            cat = pred['cat_id']
            if best_iou >= iou_thr and best_idx >= 0:
                matched_gt.add(best_idx)
                stats[cat]['tp'] += 1
            else:
                stats[cat]['fp'] += 1

        for gi, gt in enumerate(gts):
            if gi not in matched_gt:
                stats[gt['category_id']]['fn'] += 1

    details = {}
    f1s = []
    for cid in CATEGORY_IDS:
        tp, fp, fn = stats[cid]['tp'], stats[cid]['fp'], stats[cid]['fn']
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        details[CATEGORY_MAP[cid]] = {'tp': tp, 'fp': fp, 'fn': fn, 'p': p, 'r': r, 'f1': f1}
        f1s.append(f1)

    return float(np.mean(f1s)), details


# ─────────────────────────────────────────────────────────────────
# 两阶段搜索
# ─────────────────────────────────────────────────────────────────

def independent_search(preds_by_image, gt_by_image, candidates, init_thresholds):
    """
    阶段1：每个类别独立搜索最优阈值（其他类保持不变）
    大幅减少搜索空间：O(N) 而非 O(N^3)
    """
    best = dict(init_thresholds)
    best_f1, _ = eval_thresholds(preds_by_image, gt_by_image, best)

    for cid in CATEGORY_IDS:
        cat_best_f1 = best_f1
        cat_best_t = best[cid]
        for t in candidates:
            trial = dict(best)
            trial[cid] = t
            f1, _ = eval_thresholds(preds_by_image, gt_by_image, trial)
            if f1 > cat_best_f1:
                cat_best_f1 = f1
                cat_best_t = t
        best[cid] = cat_best_t
        best_f1, _ = eval_thresholds(preds_by_image, gt_by_image, best)
        print(f"  {CATEGORY_MAP[cid]}: best_t={cat_best_t:.3f}")

    return best, best_f1


def joint_refine(preds_by_image, gt_by_image, center, radius, step):
    """
    阶段2：以 center 为中心，在 ±radius 范围内联合网格搜索（细粒度）
    """
    grids = {}
    for cid in CATEGORY_IDS:
        lo = max(0.01, center[cid] - radius)
        hi = min(0.99, center[cid] + radius)
        grids[cid] = np.arange(lo, hi + step * 0.5, step).tolist()
        grids[cid] = [round(v, 3) for v in grids[cid]]

    print(f"\n阶段2 细粒度联合搜索（step={step}）")
    print(f"  battery: {grids[1]}")
    print(f"  board:   {grids[2]}")
    print(f"  fire:    {grids[3]}")

    best_thresholds = dict(center)
    best_f1, _ = eval_thresholds(preds_by_image, gt_by_image, best_thresholds)
    total = len(grids[1]) * len(grids[2]) * len(grids[3])
    count = 0

    for t1, t2, t3 in itertools.product(grids[1], grids[2], grids[3]):
        trial = {1: t1, 2: t2, 3: t3}
        f1, _ = eval_thresholds(preds_by_image, gt_by_image, trial)
        count += 1
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = trial
            print(f"  [{count}/{total}] 新最优 battery={t1} board={t2} fire={t3} -> F1={best_f1:.5f}")

    return best_thresholds, best_f1


# ─────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Per-Class 阈值优化（两阶段搜索）')
    parser.add_argument('--pred-json', required=True,
                        help='原始预测 JSON（使用极低 score_threshold=0.01 生成）')
    parser.add_argument('--gt', required=True,
                        help='GT 标注文件，例如 A_train/coco/annotations/instance_val.json')
    parser.add_argument('--coarse-step', type=float, default=0.05,
                        help='阶段1 粗搜索步长（默认 0.05）')
    parser.add_argument('--fine-step', type=float, default=0.01,
                        help='阶段2 细搜索步长（默认 0.01）')
    parser.add_argument('--fine-radius', type=float, default=0.07,
                        help='阶段2 在最优点附近的搜索半径（默认 0.07）')
    parser.add_argument('--init-threshold', type=float, default=0.05,
                        help='初始统一阈值（默认 0.05，与 A1 配置一致）')
    parser.add_argument('--out', default='best_thresholds_v2.json',
                        help='输出文件（默认 best_thresholds_v2.json）')
    args = parser.parse_args()

    print("=" * 65)
    print("Per-Class 置信度阈值优化（两阶段搜索）")
    print("=" * 65)

    # 加载数据
    print(f"\n加载 GT: {args.gt}")
    gt_by_image, filename_to_id = load_gt(args.gt)
    print(f"  GT 图片数: {len(gt_by_image)}")

    print(f"加载预测: {args.pred_json}")
    preds_by_image = load_raw_preds(args.pred_json, filename_to_id)
    total_preds = sum(len(v) for v in preds_by_image.values())
    print(f"  预测框总数（含低置信度）: {total_preds}")

    # 基线评估
    init_t = {1: args.init_threshold, 2: args.init_threshold, 3: args.init_threshold}
    baseline_f1, baseline_details = eval_thresholds(preds_by_image, gt_by_image, init_t)
    print(f"\n基线（统一阈值 {args.init_threshold}）F1: {baseline_f1:.5f}")
    for cat, d in baseline_details.items():
        print(f"  {cat:8s}: TP={d['tp']:3d} FP={d['fp']:3d} FN={d['fn']:3d} "
              f"P={d['p']:.3f} R={d['r']:.3f} F1={d['f1']:.4f}")

    # ── 阶段1：粗粒度独立搜索 ──
    coarse_candidates = [round(v, 3) for v in np.arange(0.01, 0.96, args.coarse_step)]
    print(f"\n阶段1 粗搜索（step={args.coarse_step}，{len(coarse_candidates)} 个候选值）")
    print(f"  搜索空间：{len(coarse_candidates)*3} 次推理（独立搜索）")

    coarse_best, coarse_f1 = independent_search(
        preds_by_image, gt_by_image, coarse_candidates, init_t)
    print(f"\n阶段1 结果: battery={coarse_best[1]} board={coarse_best[2]} fire={coarse_best[3]} F1={coarse_f1:.5f}")

    # ── 阶段2：细粒度联合搜索 ──
    fine_best, fine_f1 = joint_refine(
        preds_by_image, gt_by_image,
        center=coarse_best,
        radius=args.fine_radius,
        step=args.fine_step
    )

    # ── 最终结果 ──
    final_f1, final_details = eval_thresholds(preds_by_image, gt_by_image, fine_best)

    print("\n" + "=" * 65)
    print("最优阈值搜索完成")
    print("=" * 65)
    print(f"battery 阈值: {fine_best[1]}")
    print(f"board   阈值: {fine_best[2]}")
    print(f"fire    阈值: {fine_best[3]}")
    print(f"\n基线 F1:  {baseline_f1:.5f}")
    print(f"最优 F1:  {final_f1:.5f}  (+{final_f1 - baseline_f1:.5f})")
    print("\n各类别详情:")
    for cat, d in final_details.items():
        print(f"  {cat:8s}: TP={d['tp']:3d} FP={d['fp']:3d} FN={d['fn']:3d} "
              f"P={d['p']:.4f} R={d['r']:.4f} F1={d['f1']:.4f}")
    print("=" * 65)

    # 保存结果
    output = {
        'thresholds': {CATEGORY_MAP[k]: v for k, v in fine_best.items()},
        'thresholds_by_id': fine_best,
        'best_f1': final_f1,
        'baseline_f1': baseline_f1,
        'improvement': final_f1 - baseline_f1,
        'per_class': final_details
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至 {args.out}")
    print("\n使用方法：在 predict.py 中读取 best_thresholds_v2.json 并应用 per-class 阈值")


if __name__ == '__main__':
    main()
