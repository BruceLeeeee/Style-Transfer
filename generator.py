
import tensorflow as tf

class Generator:
    def __init__(self, is_training=True):
        self.training = is_training

    def build(self, img):
        self.padded = self.__pad(img, 40)

        self.conv1 = self.__conv_block(self.padded, filters_shape=[9, 9, 3, 32], stride=1, name='conv1')
        self.conv2 = self.__conv_block(self.conv1, filters_shape=[3, 3, 32, 64], stride=2, name='conv2')
        self.conv3 = self.__conv_block(self.conv2, filters_shape=[3, 3, 64, 128], stride=2, name='conv3')

        self.resid1 = self.__residual_block(self.conv3, filters_shape=[3, 3, 128, 128], stride=1, name='resid1')
        self.resid2 = self.__residual_block(self.resid1, filters_shape=[3, 3, 128, 128], stride=1, name='resid2')
        self.resid3 = self.__residual_block(self.resid2, filters_shape=[3, 3, 128, 128], stride=1, name='resid3')
        self.resid4 = self.__residual_block(self.resid3, filters_shape=[3, 3, 128, 128], stride=1, name='resid4')
        self.resid5 = self.__residual_block(self.resid4, filters_shape=[3, 3, 128, 128], stride=1, name='resid5')

        self.conv4 = self.__upsample_block(self.resid5, filters_shape=[3, 3, 64, 128], stride=2, name='conv4')
        self.conv5 = self.__upsample_block(self.conv4, filters_shape=[3, 3, 32, 64], stride=2, name='conv5')
        self.conv6 = self.__conv_block(self.conv5, filters_shape=[9, 9, 32, 3], stride=1, name='conv6', activation=None)

        self.output = tf.nn.sigmoid(self.conv6)

    def __get_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0, stddev=.1), dtype=tf.float32)

    def __instance_normalize(self, inputs):
        with tf.variable_scope('instance_normalize'):
            batch, height, width, channels = inputs.get_shape().as_list()
            mu, sigma_sq = tf.nn.moments(inputs, [1, 2], keep_dims=True)

            epsilon = 1e-3
            shift = tf.Variable(tf.constant(.1, shape=[channels]))
            scale = tf.Variable(tf.ones([channels]))
            normalized = (inputs - mu) / (sigma_sq + epsilon) ** .5

            return scale * normalized + shift

    def __pad(self, inputs, size):
        return tf.pad(inputs, [[0, 0], [size, size], [size, size], [0, 0]], "REFLECT")

    def __conv_block(self, inputs, filters_shape, stride, name, normalize=True, padding='SAME', activation=tf.nn.relu):
        with tf.variable_scope(name):
            filters = self.__get_weights(filters_shape)
            feature_maps = tf.nn.conv2d(inputs, filters, [1, stride, stride, 1], padding=padding)
            num_filters = filters_shape[3]
            bias = tf.Variable(tf.constant(.1, shape=[num_filters]))
            feature_maps = tf.nn.bias_add(feature_maps, bias)

            if normalize:
                feature_maps = self.__instance_normalize(feature_maps)

            if activation:
                return activation(feature_maps)
            else:
                return feature_maps

    def __upsample_block(self, inputs, filters_shape, stride, name):
        with tf.variable_scope(name):
            filters = self.__get_weights(filters_shape)

            batch, height, width, channels = inputs.get_shape().as_list()
            out_height = height * stride
            out_width = width * stride
            out_size = filters_shape[2]
            out_shape = tf.stack([batch, out_height, out_width, out_size])
            stride = [1, stride, stride, 1]

            upsample = tf.nn.conv2d_transpose(inputs, filters, output_shape=out_shape, strides=stride)
            bias = tf.Variable(tf.constant(.1, shape=[out_size]))
            upsample = tf.nn.bias_add(upsample, bias)
            upsample = self.__instance_normalize(upsample)

            return tf.nn.relu(upsample)

    # simple version of residual block
    def __residual_block(self, inputs, filters_shape, stride, name):
        with tf.variable_scope(name):
            conv1 = self.__conv_block(inputs, filters_shape, stride=stride, name='c1', padding='VALID')
            conv2 = self.__conv_block(conv1, filters_shape, stride=stride, name='c2', padding='VALID', activation=None)

            batch = inputs.get_shape().as_list()[0]
            patch_height, patch_width, num_filters = conv2.get_shape().as_list()[1:]
            out_shape = tf.stack([batch, patch_height, patch_width, num_filters])
            cropped_inputs = tf.slice(inputs, [0, 1, 1, 0], out_shape)

            return conv2 + cropped_inputs

