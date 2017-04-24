import tensorflow as tf
import generator
from solver import *
from utils import *

flags = tf.app.flags

INPUT_DIR = ""
OUTPUT_DIR = ""
CKPT_DIR = ""


def main(_):

    input_img, _ = load_img(INPUT_DIR)
    input_img = tf.convert_to_tensor(input_img, dtype=tf.float32)
    input_img = tf.expand_dims(input_img, axis=0)

    with tf.Session() as sess:
        with tf.variable_scope('generator'):
            gen = generator.Generator()
            gen.build(tf.convert_to_tensor(input_img))
            sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, CKPT_DIR)

        img = sess.run(gen.output)
        render(img, OUTPUT_DIR)


if __name__ == '__main__':
    main()