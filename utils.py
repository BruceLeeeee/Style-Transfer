
import tensorflow as tf
from functools import reduce
import numpy as np
import skimage
import skimage.io
import skimage.transform
from scipy.misc import toimage


def get_content_loss(input_img, content_img, layer):
    with tf.variable_scope('get_content_loss'):
        input_feature_maps = getattr(input_img, layer)
        content_feature_maps = getattr(content_img, layer)

        return get_l2_norm_loss(input_feature_maps - content_feature_maps)

def get_style_loss(input_model, style_model, layers):
    with tf.name_scope('get_style_loss'):
        #layers = {'conv1_2': .25, 'conv2_2': .25, 'conv3_3': .25, 'conv4_3': .25}
        style_layer_loss = [get_style_loss_for_layer(input_model, style_model, layer) for layer in layers.keys()]
        style_layer_loss = tf.convert_to_tensor(style_layer_loss)
        style_weights = tf.constant(list(layers.values()))
        style_loss = tf.multiply(style_weights, style_layer_loss)

        return tf.reduce_sum(style_loss)

def get_style_loss_for_layer(input_img, style_img, layer):
    with tf.name_scope('get_style_loss_for_layer'):
        input_feature_maps = getattr(input_img, layer)
        style_feature_maps = getattr(style_img, layer)
        input_gram_matrix = get_gram_matrix(input_feature_maps)
        style_gram_matrix = get_gram_matrix(style_feature_maps)

        shape = input_gram_matrix.get_shape().as_list()
        size = reduce(lambda x, y: x * y, shape) ** 2
        style_loss = get_l2_norm_loss(input_gram_matrix - style_gram_matrix)

        return style_loss / size

def get_gram_matrix(feature_maps):
    dim = feature_maps.get_shape().as_list()
    vectorized_maps = tf.reshape(feature_maps, [dim[1] * dim[2], dim[3]])

    return tf.matmul(vectorized_maps, vectorized_maps, transpose_a=True)

def get_l2_norm_loss(difference):
    shape = difference.get_shape().as_list()
    size = reduce(lambda x, y: x * y, shape) ** 2
    sum_of_squared_difference = tf.reduce_sum(tf.square(difference))

    return sum_of_squared_difference

def get_total_variation(input_img, shape, smoothing=1.5):
    with tf.name_scope('get_total_variation'):
        height = shape[1]
        width = shape[2]
        size = reduce(lambda x, y: x * y, shape) ** 2

        x_cropped = input_img[:, :height - 1, :width - 1, :]
        left_term = tf.square(input_img[:, 1:, :width - 1, :] - x_cropped)
        right_term = tf.square(input_img[:, :height - 1, 1:, :] - x_cropped)
        smoothed_terms = tf.pow(left_term + right_term, smoothing / 2.)

        return tf.reduce_sum(smoothed_terms) / size

def load_img(path, height, width):
    img = io.imread(path) / 255.0
    ny = height
    nx = width
    if len(img.shape) < 3:
      img = np.dstack((img, img, img))

    return transform.resize(img, (ny, nx)), [ny, nx, 3]

def render(img, path_out):
    clipped_img = np.clip(img, 0., 1.)
    toimage(np.reshape(clipped_img, img.shape[1:])).save(path_out)