import tensorflow as tf
import argparse

from model import ESPCN

parser = argparse.ArgumentParser(description='Super resolution training')
parser.add_argument('--checkpoint_dir',
                    type=str,
                    required=False,
                    default='checkpoint',
                    help="path to checkpoint folder")
parser.add_argument('--load_hdf5',
                    type=str,
                    required=False,
                    default='train.h5',
                    help="load hdf5 file")
parser.add_argument('--epochs',
                    type=int,
                    required=False,
                    default=15000,
                    help="downsampling factor")
parser.add_argument('--batch_size',
                    type=int,
                    required=False,
                    default=128,
                    help="size of batch images")
parser.add_argument('--learning_rate',
                    type=float,
                    required=False,
                    default=1e-4,
                    help="The learning rate of gradient descent algorithm")
                    
def main():
    opt = parser.parse_args()
    
    with tf.Session() as session:
        espcn = ESPCN(session)
        espcn.train(hdf5_path=opt.load_hdf5,
                    checkpoint_dir=opt.checkpoint_dir,
                    learning_rate=opt.learning_rate,
                    batch_size=opt.batch_size,
                    epochs=opt.epochs)

if __name__=='__main__':
    main()
