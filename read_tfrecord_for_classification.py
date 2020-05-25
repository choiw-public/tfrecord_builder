import tensorflow as tf
import utils

tfrecord_dir = "classification_dataset"
repeat = True
batch = 4
crop_size = [16, 16]  # becasue raw images have different sizes

tfrecord_feature = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                    "filename": tf.FixedLenFeature((), tf.string, default_value=""),
                    "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                    "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                    "channels": tf.FixedLenFeature((), tf.int64, default_value=0),
                    "label": tf.FixedLenFeature((), tf.int64, default_value=0),
                    "class": tf.FixedLenFeature((), tf.string, default_value="")}


def preprocessing(image):
    # do whatever you want
    return tf.image.random_crop(image, crop_size + [3])


def parser(data):
    parsed = tf.parse_single_example(data, tfrecord_feature)
    image = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image"], channels=3))
    filename = tf.convert_to_tensor(parsed["filename"])
    height = tf.convert_to_tensor(parsed["height"])
    width = tf.convert_to_tensor(parsed["width"])
    channel = tf.convert_to_tensor(parsed["channels"])
    label = tf.convert_to_tensor(parsed["label"])
    cls_number = tf.convert_to_tensor(parsed["class"])
    image = preprocessing(image)
    return {"image": image, "filename": filename, "height": height, "width": width, "channel": channel, "label": label, "class": cls_number}


tfrecord_list = utils.list_getter(tfrecord_dir, "tfrecord")
data = tf.data.TFRecordDataset(tfrecord_list)
if repeat:
    data = data.repeat()
data = data.shuffle(batch * 10)
data = data.map(parser, num_parallel_calls=4).batch(batch, drop_remainder=True)
data = data.prefetch(4)
data = data.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    while True:
        data_from_tfrecord = sess.run(data)
