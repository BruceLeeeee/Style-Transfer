import tensorflow as tf
import generator
from solver import *

flags = tf.app.flags

#solver
flags.DEFINE_string("input_dir", "", "")
flags.DEFINE_string("out_dir", "", "")
flags.DEFINE_string("style_dir", "", "")
flags.DEFINE_integer("num_epoch", 20000, "train epoch num")
flags.DEFINE_integer("image_height", 256, "image_height")
flags.DEFINE_integer("image_width", 256, "image_width")
flags.DEFINE_float("learning_rate", 4e-4, "learning rate")
flags.DEFINE_float("content_weights", 1., "content weights")
flags.DEFINE_float("style_weights", 3., "style weights")
flags.DEFINE_float("tv_weights", .1, "tv weights")

conf = flags.FLAGS

def main(_):

    with tf.variable_scope('generator'):
        gen = generator.Generator()
        slover = slover.Trainer(gen)
        slover.train()

if __name__ == '__main__':
    tf.app.run()