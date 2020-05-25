import os
import tensorflow as tf
import multiprocessing
from random import shuffle
import utils

"""
This script does not perform any preprocessing.
src_image_dir: the root folder of images, where there are subfolders representing each class.
               ex) root_folder/class_name1/img1.jpg
                   root_folder/class_name1/img2.jpg
                   root_folder/class_name2/img1.jpg
                   root_folder/class_name2/img2.jpg               
               assumed all images are in a single type of image format (jpg or png), if not the code will return error.
               assumed all images are rgb images.
dst_dir: the folder where tfrecord will be placed.
prefix: any string you want.
num_split: any integer you want.
"""

src_image_dir = './classification_dataset'
dst_dir = './classification_dataset'
prefix = 'train'
num_split = 4

img_list = utils.list_getter(src_image_dir)
shuffle(img_list)
img_ext = utils.get_extensions(img_list)
img_reader = utils.ImageReader(img_ext, channels=3)
class_vs_label = utils.get_class_vs_list(img_list)
img_list_split = utils.divide_list(img_list, num_split)
os.makedirs(dst_dir, exist_ok=True)

with open(os.path.join(dst_dir, 'class_vs_label_info.csv'), 'w') as writer:
    writer.write('name,label\n')
    for k, v in class_vs_label.items():
        writer.write('%s, %d\n' % (k, v))


def write_tfrecord(file_list, gt_dict, split_idx):
    if num_split == 1:
        dst_file_name = os.path.join(dst_dir, "%s.tfrecord" % prefix)
    else:
        dst_file_name = os.path.join(dst_dir, "%s-%03d-of-%03d.tfrecord" % (prefix, split_idx, num_split))
    with tf.python_io.TFRecordWriter(dst_file_name) as tfrecord_writer:
        for filename in file_list:
            image_data = tf.gfile.FastGFile(filename, 'rb').read()
            h, w, _ = img_reader.read_image_dims(image_data)
            class_name = filename.split("/")[-2]
            label = gt_dict[class_name]
            instance = utils.tfrecord_for_classification(image_data,
                                                         os.path.basename(filename).encode("utf-8"),
                                                         h,
                                                         w,
                                                         int(label),
                                                         class_name.encode("utf-8"))
            tfrecord_writer.write(instance.SerializeToString())


num_cpus = multiprocessing.cpu_count()
for idx, sublist in enumerate(img_list_split):
    write_tfrecord(sublist, class_vs_label, idx + 1)
