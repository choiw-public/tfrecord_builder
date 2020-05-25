import tensorflow as tf
import utils

tfrecord_dir = "segmentation_dataset"
repeat = True
batch = 4
crop_size = [16, 16]  # becasue raw images have different sizes

tfrecord_feature = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                    "filename": tf.FixedLenFeature((), tf.string, default_value=""),
                    "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                    "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                    "channels": tf.FixedLenFeature((), tf.int64, default_value=0),
                    "segmentation": tf.FixedLenFeature((), tf.string, default_value="")}


def preprocessing(image, segmentation):
    # do whatever you want
    paired = tf.concat([image, segmentation], 2)
    cropped_paired = tf.image.random_crop(paired, crop_size + [4])
    return cropped_paired[:, :, :3], cropped_paired[:, :, 3]


def parser(data):
    parsed = tf.parse_single_example(data, tfrecord_feature)
    image = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image"], channels=3))
    segmentation = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["segmentation"], channels=1))
    filename = tf.convert_to_tensor(parsed["filename"])
    height = tf.convert_to_tensor(parsed["height"])
    width = tf.convert_to_tensor(parsed["width"])
    channel = tf.convert_to_tensor(parsed["channels"])
    image, segmentation = preprocessing(image, segmentation)
    return {"image": image, "filename": filename, "height": height, "width": width, "channel": channel, "segmentation": segmentation}


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
