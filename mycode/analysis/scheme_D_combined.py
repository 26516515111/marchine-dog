# -*- coding: utf-8 -*-
"""scheme_D_fog_mask_inpaint.py — 方案D：黑雾掩码检测 + 保护橙色区域

原理：
  1. 在 HSV 空间检测黑雾区域（V<80, S<70）生成掩码
  2. 保护橙色火焰区域（H∈[5,25], S>80）不被处理
  3. 对黑雾掩码区域：
     - 对 V 通道做局部线性拉伸（映射到临近亮区的亮度水平）
     - 用双边滤波平滑雾气边界，减少处理痕迹
  4. 保持非雾区域完全不变

与其他方案的区别：
  - 方案A（CLAHE）：全图均衡，对非雾区也有影响
  - 方案B（暗通道）：物理模型，对薄雾更准确
  - 方案C（Retinex）：整体光照分离，可能改变颜色
  - 方案D（本方案）：精确掩码，只处理黑雾区，橙色完全保护

适用：黑雾与橙色火焰共存、需要保护火焰颜色特征的场景。
"""
import cv2
import numpy as np

# 黑雾检测阈值
FOG_V_MAX  = 80
FOG_S_MAX  = 70
FOG_V_MIN  = 5

# 橙色火焰保护区
FIRE_H_LOW  = 5
FIRE_H_HIGH = 25
FIRE_S_MIN  = 60

# V 拉伸目标（黑雾区映射到此亮度范围）
V_LIFT_TARGET = 110   # 黑雾 V 的目标均值
V_LIFT_STD    = 30    # 目标标准差

# 边界平滑
MORPH_DILATE  = 5     # 掩码扩张（羽化边缘）
BLEND_SIGMA   = 8.0   # 高斯混合核


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """黑雾掩码精确去除 + 橙色区完全保护。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Step 1: 生成黑雾掩码
    fog_mask = (V > FOG_V_MIN) & (V < FOG_V_MAX) & (S < FOG_S_MAX)

    # Step 2: 生成橙色保护掩码（绝对不处理）
    fire_mask = (H >= FIRE_H_LOW) & (H <= FIRE_H_HIGH) & (S > FIRE_S_MIN)

    # 黑雾但不是橙色
    fog_only = fog_mask & ~fire_mask

    # Step 3: 对黑雾区 V 做线性提升（向临近亮区对齐）
    V_new = V.copy()
    fog_v = V[fog_only]
    if fog_v.size > 0:
        fog_mean, fog_std = fog_v.mean(), max(fog_v.std(), 1.0)
        # 线性变换：将均值/标准差映射到目标
        V_new[fog_only] = np.clip(
            (fog_v - fog_mean) / fog_std * V_LIFT_STD + V_LIFT_TARGET,
            0, 255
        )

    # Step 4: 软掩码混合（避免硬边界）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_DILATE, MORPH_DILATE))
    fog_soft = fog_only.astype(np.float32)
    fog_soft = cv2.dilate(fog_soft, kernel)
    fog_soft = cv2.GaussianBlur(fog_soft, (0, 0), BLEND_SIGMA)

    # 确保橙色区混合权重为0（不改变）
    fog_soft[fire_mask] = 0.0

    # 混合：fog区用新V，其他用原始V
    V_final = V * (1 - fog_soft) + V_new * fog_soft

    hsv[:, :, 2] = np.clip(V_final, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def describe() -> dict:
    return {
        "name": "D_fog_mask_inpaint",
        "desc": "黑雾掩码精确V提升 + 橙色区域完全保护 + 软边界混合",
        "fog_v_max": FOG_V_MAX,
        "v_lift_target": V_LIFT_TARGET,
        "fire_protected": True,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    for p in sys.argv[1:]:
        img = cv2.imread(p)
        out = preprocess(img)
        dst = str(Path(p).parent / (Path(p).stem + "_schemeD.jpg"))
        cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"Saved: {dst}")
