# check_images.py
import cv2
import os

# 请将下面的路径改为你数据集图片所在的目录
image_dir = "/home/alpha/Desktop/w-yolo/marchine-dog/A_train/coco/train"

for img_name in os.listdir(image_dir):
    if not img_name.endswith(('.jpg', '.png', '.jpeg')):
        continue
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"发现坏图，建议删除: {img_path}")
        # os.remove(img_path)  # 确认无误后，可以取消注释来直接删除
