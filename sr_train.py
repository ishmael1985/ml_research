import tensorflow as tf
import argparse

from fsrcnn import FSRCNN

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
                    help="numbers of epochs")
                    
def main():
    opt = parser.parse_args()
    
    with tf.Session() as session:
        fsrcnn = FSRCNN(session, opt.checkpoint_dir)
        fsrcnn.train(opt.load_hdf5, opt.epochs)

if __name__=='__main__':
    main()

