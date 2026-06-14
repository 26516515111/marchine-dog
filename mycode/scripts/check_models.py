import os
import re

configs = {
    'PP-YOLOE+-s':   'PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml',
    'PP-YOLOE+-m':   'PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml',
    'RT-DETRv2-R18': 'PaddleDetection/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
    'RT-DETRv3-R18': 'PaddleDetection/configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml',
    'RT-DETRv3-R50': 'PaddleDetection/configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml',
    'DINO-R50':      'PaddleDetection/configs/dino/dino_r50_4scale_1x_coco.yml',
    'Co-DETR-R50':   'PaddleDetection/configs/co_detr/co_detr_r50_1x_coco.yml',
    'TOOD-R50':      'PaddleDetection/configs/tood/tood_r50_fpn_1x_coco.yml',
}

for name, path in configs.items():
    if not os.path.exists(path):
        print('%s: NOT FOUND' % name)
        continue
    with open(path, encoding='utf-8', errors='ignore') as f:
        content = f.read()
    epoch = re.search(r'^epoch\s*:\s*(\d+)', content, re.M)
    lr    = re.search(r'base_lr\s*:\s*([\d.e\-]+)', content)
    pw    = re.search(r'pretrain_weights\s*:\s*(\S+)', content)
    print('%s:' % name)
    print('  epoch=%s  lr=%s' % (
        epoch.group(1) if epoch else '?',
        lr.group(1) if lr else '?'))
    if pw:
        print('  pretrain=...%s' % pw.group(1)[-70:])
    print()
