import tensorflow as tf
import pprint

from os import makedirs
from model import ESPCN

flags = tf.app.flags
flags.DEFINE_integer("epoch", 150000, "Number of epoch [150000]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("hdf5_path", "train.h5", "Path to hdf5 data file [train.h5]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    makedirs(FLAGS.checkpoint_dir, exist_ok=True)
    
    with tf.Session() as session:
        espcn = ESPCN(session)
        espcn.train(FLAGS)

if __name__=='__main__':
    tf.app.run()
