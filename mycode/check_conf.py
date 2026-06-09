import os
import json
import cv2
import numpy as np
from paddle.inference import Config, create_predictor

MODEL_DIR = r'D:\work\Marchine Dog\model'
TRAIN_IMAGE_DIR = r'D:\work\Marchine Dog\dog\A_train\Image'
TRAIN_ANNOTATION = r'D:\work\Marchine Dog\dog\A_train\coco\annotations\instance_train.json'

config = Config(os.path.join(MODEL_DIR, 'model.pdmodel'), os.path.join(MODEL_DIR, 'model.pdiparams'))
config.enable_use_gpu(1000, 0)
config.enable_memory_optim()
predictor = create_predictor(config)

with open(TRAIN_ANNOTATION, 'r', encoding='utf-8') as f:
    data = json.load(f)
img_info = {img['id']: img for img in data['images']}

img_id = 1
info = img_info[img_id]
img_path = os.path.join(TRAIN_IMAGE_DIR, info['file_name'])

img = cv2.imread(img_path)
h, w = img.shape[:2]
scale = 640 / max(h, w)
new_h, new_w = int(h * scale), int(w * scale)
img_resized = cv2.resize(img, (new_w, new_h))
padded = np.zeros((640, 640, 3), dtype=np.uint8)
padded[:new_h, :new_w, :] = img_resized
img_norm = padded.astype(np.float32) / 255.0
img_norm = img_norm.transpose((2, 0, 1))
img_batch = np.expand_dims(img_norm, axis=0).astype(np.float32)
sf = np.array([[scale, scale]], dtype=np.float32)

predictor.get_input_handle('image').reshape(img_batch.shape)
predictor.get_input_handle('image').copy_from_cpu(img_batch)
predictor.get_input_handle('scale_factor').reshape(sf.shape)
predictor.get_input_handle('scale_factor').copy_from_cpu(sf)

predictor.run()

dets = predictor.get_output_handle('multiclass_nms3_0.tmp_0').copy_to_cpu()
num = predictor.get_output_handle('multiclass_nms3_0.tmp_2').copy_to_cpu()[0]

print(f'Image: {info["file_name"]}')
print(f'Total detections: {num}')
print(f'Conf distribution:')
for det in dets[:num]:
    print(f'  class={int(det[0])}, conf={det[1]:.3f}, bbox={det[2:6].tolist()}')
