
import tensorflow as tf
from vgg16 import *
import numpy as np
from utils import *

class slover(object):
    def __init__(self, generator, image_dimension):
        self.input_dir =""
        self.out_dir =""
        self.style_dir =""
        self.image_height = image_dimension['height']
        self.image_width = image_dimension['width']
        self.net = generator


    def train(self, epochs, learning_rate, content_layer, content_weights, style_layers, style_weights, tv_weights):
        style_img, style_img_shape = load_img(self.style_dir, height=self.image_height, width=self.image_width)
        style_img_shape = [1] + style_img_shape
        style_img = style_img.reshape(style_img_shape).astype(np.float32)

        input_img_placeholder = tf.placeholder(dtype=tf.float32, shape=style_img_shape)
        self.net.build(input_img_placeholder)
        output_img = self.net.output

        with tf.name_scope('vgg16_style'):
            style_model = vgg16.Vgg16()
            style_model.build(style_img, shape=style_img_shape[1:])

        with tf.name_scope('vgg16_content'):
            content_placeholder = tf.placeholder(dtype=tf.float32, shape=style_img_shape)
            content_model = vgg16.Vgg16()
            content_model.build(input_img_placeholder, shape=style_img_shape[1:])

        with tf.name_scope('vgg16_input'):
            input_model = vgg16.Vgg16()
            input_model.build(output_img, shape=style_img_shape[1:])

        with tf.name_scope('loss'):
            content_loss = get_content_loss(input_model, content_model, content_layer) * content_weights
            style_loss = get_style_loss(input_model, style_model, style_layers) * style_weights
            tv_loss = get_total_variation(output_img, style_img_shape) * tv_weights
            loss = content_loss + style_loss + tv_loss

        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gradients = optimizer.compute_gradients(loss, trainable_variables)
            updated_weights = optimizer.apply_gradients(gradients)

        example = self.__next_examples(height=style_img_shape[1], width=style_img_shape[2])
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in range(epochs):
            self.net.is_training = True

            input_img = sess.run(example) / 255.
            input_img = input_img.reshape([1] + list(input_img.shape)).astype(np.float32)

            _, loss = sess.run([updated_weights, loss],
                                       feed_dict={input_img_placeholder:input_img, content_placeholder:input_img})
            if step % 100 ==0:
                print("[epoch %2.4f] loss %.4f\t " % (step, loss))

        coord.request_stop()
        coord.join(threads)

        saver = tf.train.Saver(trainable_variables)
        saver.save(sess, self.out_dir)

    def __next_examples(self, height, width):
        filenames = tf.train.match_filenames_once(self.input_dir + '*.jpg')
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        _, files = reader.read(filename_queue)
        input_img = tf.image.decode_jpeg(files, channels=3)
        input_img = tf.image.resize_images(input_img, [height, width])

        return input_img