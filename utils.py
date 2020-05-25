import collections
import tensorflow as tf
import os
import re
import math


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, image_format='jpeg', channels=3):
        """Class constructor.

        Args:
          image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
          channels: Image channels.
        """
        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session()
            if self._image_format in ('jpeg', 'jpg'):
                self._decode = tf.image.decode_jpeg(self._decode_data,
                                                    channels=channels)
            elif self._image_format == 'png':
                self._decode = tf.image.decode_png(self._decode_data,
                                                   channels=channels)

    def read_image_dims(self, image_data):
        image = self.decode_image(image_data)
        return image.shape

    def decode_image(self, image_data):
        image = self._session.run(self._decode,
                                  feed_dict={self._decode_data: image_data})
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('The image channels not supported.')

        return image


def convert_to_int64(values):
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def convert_to_bytes(values):
    # value = string
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def tfrecord_for_classification(image_data, filename, height, width, label, clsname):
    feature = {
        'image': convert_to_bytes(image_data),
        'filename': convert_to_bytes(filename),
        'height': convert_to_int64(height),
        'width': convert_to_int64(width),
        'channels': convert_to_int64(3),
        'label': convert_to_int64(label),
        'class': convert_to_bytes(clsname)
    }
    feature = tf.train.Features(feature=feature)
    return tf.train.Example(features=feature)


def get_extensions(file_list):
    file_extension = []
    for entry in file_list:
        basename = os.path.basename(entry)
        file_extension.append(os.path.basename(basename).split('.')[-1])
    file_extension = list(set(file_extension))
    if len(file_extension) == 1:
        return file_extension[0]
    else:
        raise ValueError('standardization of image file type is required')


def list_getter(dir_name, extension, must_include=None):
    def sort_nicely(a_list):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        a_list.sort(key=alphanum_key)

    file_list = []
    if dir_name:
        for path, subdirs, files in os.walk(dir_name):
            for name in files:
                if name.lower().endswith(tuple(extension)):
                    if must_include:
                        if must_include in name:
                            file_list.append(os.path.join(path, name))
                    else:
                        file_list.append(os.path.join(path, name))
        sort_nicely(file_list)
    return file_list


def divide_list(filelist, division):
    # todo: make it simpler. index based
    num = int(math.ceil(float(len(filelist)) / division))
    new_filelist = []
    while True:
        try:
            tmp_list = []
            for i in range(num):
                tmp_list.append(filelist.pop(0))
            new_filelist.append(tmp_list)
        except:
            new_filelist.append(tmp_list)
            break
    return new_filelist


def get_class_vs_list(file_list):
    cls_n_lbl = collections.OrderedDict()
    classes = sorted(list(set([filename.split('/')[-2] for filename in file_list])))
    for i, cls in enumerate(classes):
        cls_n_lbl[cls] = i
    return cls_n_lbl
