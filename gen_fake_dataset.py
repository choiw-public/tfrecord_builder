import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

size = [28, 64]
num_classes = 5
num_images_per_class = 10

for cls in range(num_classes):
    for img in range(num_images_per_class):
        rnd_h = np.random.randint(size[0], size[1], [], int)
        rnd_w = np.random.randint(size[0], size[1], [], int)
        rnd_img = np.random.randint(0, 256, [rnd_h, rnd_w, 3], np.uint8)
        dst_dir = "./classification_dataset/%s" % cls
        os.makedirs(dst_dir, exist_ok=True)
        dst_name = os.path.join(dst_dir, "%s.jpg" % img)
        cv.imwrite(dst_name, rnd_img)

size = [28, 64]
num_classes = 5
subfolders = ["subfolder1", "subfolder2", "subfolder3", "subfolder4", "subfolder5"]
num_images_per_subfolder = 10
for subfolder in subfolders:
    for img in range(num_images_per_subfolder):
        rnd_h = np.random.randint(size[0], size[1], [], int)
        rnd_w = np.random.randint(size[0], size[1], [], int)
        rnd_img = np.random.randint(0, 256, [rnd_h, rnd_w, 3], np.uint8)
        dst_img_dir = "./segmentation_dataset/img/%s" % subfolder
        dst_seg_dir = "./segmentation_dataset/seg/%s" % subfolder
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_seg_dir, exist_ok=True)
        dst_img_name = os.path.join(dst_img_dir, "%s.jpg" % img)
        dst_seg_name = os.path.join(dst_seg_dir, "%s.png" % img)
        cv.imwrite(dst_img_name, rnd_img)
        cv.imwrite(dst_seg_name, rnd_img[:, :, 0])
