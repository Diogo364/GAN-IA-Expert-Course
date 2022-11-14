import tensorflow as tf
from dotenv import load_dotenv

def tf_normalize(data, feature_range=(0, 1)):
    flatten_data = tf.experimental.numpy.ravel(data)
    min_value = tf.math.reduce_min(flatten_data)
    max_value = tf.math.reduce_max(flatten_data)
    real_scale = max_value - min_value
    target_scale = feature_range[1] - feature_range[0]
    x_std = (data - min_value) / (real_scale)
    return x_std * target_scale + feature_range[0]

def normalize(target_img, label_img):
    target_img = tf_normalize(target_img, feature_range=(-1, 1))
    label_img = tf_normalize(label_img, feature_range=(-1, 1))
    return target_img, label_img

def random_crop(target_img, label_img):
  stacked_img = tf.stack([target_img, label_img], axis = 0)
  cropped_img = tf.image.random_crop(stacked_img, size = [2, target_img.shape[0], target_img.shape[1], 3])
  return cropped_img[0], cropped_img[1]

def random_flip(target_img, label_img):
  if tf.random.uniform(()) > 0.5:
    target_img = tf.image.flip_left_right(target_img)
    label_img = tf.image.flip_left_right(label_img)
  return target_img, label_img

def remove_extension_names(filename):
    return filename.split('.')[0]

def load_env():
    load_dotenv('.env')