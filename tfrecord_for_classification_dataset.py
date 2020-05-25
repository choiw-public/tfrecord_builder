import math
import os
import tensorflow as tf
import time
from random import shuffle
import utils

"""
This script does not perform any preprocessing.
Images should be devided by folders, each folder should represent classes.
src_image_dir: the root folder of images, where there are subfolders representing each class.
               assumed all images are in jpg format, otherwise the code will return error.
               assumed all images have rgb channels.
dst_dir: the folder where tfrecord will be placed.
prefix: any string you want.
num_split: any number you want.
"""

src_image_dir = '/media/wooram/data_hdd/00.DL_datasets/MNIST/training'
dst_dir = '/media/wooram/data_hdd/00.DL_datasets/MNIST/training'
prefix = 'train'
num_split = 1

img_list = utils.list_getter(src_image_dir, "jpg")
img_ext = utils.get_extensions(img_list)
if len(img_ext) != 1:
    raise ValueError("'jpg' format for all images is expected. Check the extensions of your images")
img_reader = utils.ImageReader(img_ext, channels=3)
class_vs_label = utils.get_class_vs_list(img_list)
img_list_split = utils.divide_list(img_list, num_split)
os.makedirs(dst_dir, exist_ok=True)


def write_tfrecord(file_list, gt_dict, split_idx):
    if num_split == 1:
        dst_file_name = os.path.join(dst_dir, "%s.tfrecord")
    else:
        dst_file_name = os.path.join(dst_dir, "%s_%3d-of-%03d.tfrecord" % (prefix, split_idx, num_split))
    with tf.python_io.TFRecordWriter(dst_file_name) as tfrecord_writer:
        for filename in file_list:
            image_data = tf.gfile.FastGfile(filename, 'r').read()
            h, w, c = img_reader.read_image_dims(image_data)
            class_name = filename.split("/")[-2]
            label = gt_dict[class_name]
            instance = utils.tfrecord_for_classification(image_data,
                                                         os.path.basename(filename),
                                                         h,
                                                         w,
                                                         int(label),
                                                         class_name)
