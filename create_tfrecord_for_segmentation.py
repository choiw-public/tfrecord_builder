import os
import tensorflow as tf
import multiprocessing
from random import shuffle
import utils

"""
This script does not perform any preprocessing.
Image names and segmentation names are assumed to be equal. If not, this script will return error. 
src_image_dir: the root folder of images
               assumed all images are in a single type of image format (jpg or png), if not the code will return error.
               assumed all images are rgb images.
               
src_segmentation_dir: the root folder of ground truths.
                      the folder structure should be equal to <src_image_dir>.
                      assumed all segmentations are single channel and png.

dst_dir: the folder where tfrecord will be placed.

prefix: any string you want.

num_split: any integer you want.
"""

src_image_dir = './segmentation_dataset/img'
src_segmentation_dir = './segmentation_dataset/seg'
dst_dir = './segmentation_dataset'
prefix = 'train'
num_split = 4

img_list = utils.list_getter(src_image_dir)
shuffle(img_list)
img_ext = utils.get_extensions(img_list)
img_reader = utils.ImageReader(img_ext, channels=3)
seg_reader = utils.ImageReader("png", channels=1)
img_list_split = utils.divide_list(img_list, num_split)
os.makedirs(dst_dir, exist_ok=True)


def write_tfrecord(file_list, split_idx):
    if num_split == 1:
        dst_file_name = os.path.join(dst_dir, "%s.tfrecord" % prefix)
    else:
        dst_file_name = os.path.join(dst_dir, "%s-%03d-of-%03d.tfrecord" % (prefix, split_idx, num_split))
    with tf.python_io.TFRecordWriter(dst_file_name) as tfrecord_writer:
        for image_filename in file_list:
            seg_filename = image_filename.replace(src_image_dir, src_segmentation_dir)[:-4] + ".png"
            image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
            seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
            img_h, img_w, _ = img_reader.read_image_dims(image_data)
            seg_h, seg_w, _ = seg_reader.read_image_dims(seg_data)

            if not img_h == seg_h and img_w == seg_w:
                raise ValueError("the sizes of image and segmentation should be equall but: %s[%d x %d], seg: %s[%d x %d]"
                                 % (image_filename, img_h, img_w, seg_filename, seg_h, seg_w))
            instance = utils.tfrecord_for_segmentation(image_data,
                                                       os.path.basename(image_filename).encode("utf-8"),
                                                       img_h,
                                                       img_w,
                                                       seg_data)
            tfrecord_writer.write(instance.SerializeToString())


num_cpus = multiprocessing.cpu_count()
for idx, sublist in enumerate(img_list_split):
    write_tfrecord(sublist, idx + 1)
