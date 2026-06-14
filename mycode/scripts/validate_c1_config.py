import sys
sys.path.insert(0, 'PaddleDetection')
from ppdet.core.workspace import load_config
cfg = load_config('PaddleDetection/configs/custom/ppyoloe_plus_fire_c1.yml')
print('配置解析成功!')
print('num_classes  :', cfg.get('num_classes'))
print('epoch        :', cfg.get('epoch'))
print('depth_mult   :', cfg.get('depth_mult'))
print('width_mult   :', cfg.get('width_mult'))
print('use_ema      :', cfg.get('use_ema'))
print('ema_decay    :', cfg.get('ema_decay'))
pw = str(cfg.get('pretrain_weights', ''))
print('pretrain_weights (last 70):', pw[-70:])
