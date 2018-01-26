import tensorflow as tf
import numpy as np
import time
import os
import json

from utils import read_hdf5


class ESPCN():
    def __init__(self, session, batch_size):
        self.session = session
        self.batch_size = batch_size
        
        with open("config.json", "r") as config:
            params = json.load(config)

        self.patch_size = params["block_size"]
        self.channels = params["channels"]
        self.scale = params["scale"]
        
        self._build_model()

    def _build_model(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, self.channels], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, self.patch_size * self.scale , self.patch_size * self.scale, self.channels], name='labels')
        
        self.weights = {
            'w1': tf.Variable(tf.random_normal([5, 5, self.channels, 64], stddev=np.sqrt(2.0/25/3)), name='w1'),
            'w2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=np.sqrt(2.0/9/64)), name='w2'),
            'w3': tf.Variable(tf.random_normal([3, 3, 32, self.channels * self.scale * self.scale ], stddev=np.sqrt(2.0/9/32)), name='w3')
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([64], name='b1')),
            'b2': tf.Variable(tf.zeros([32], name='b2')),
            'b3': tf.Variable(tf.zeros([self.channels * self.scale * self.scale ], name='b3'))
        }
        
        self.pred = self.model()
        
        # Loss function (MSE)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver()

    def model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.inputs, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3']

        ps = self.PS(conv3, self.scale)
        
        return tf.nn.tanh(ps)

    #def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        #bsize, a, b, c = I.get_shape().as_list()
        #X = tf.reshape(I, (self.batch_size, a, b, r, r))
        #X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        #X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        #X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        #X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        #return tf.reshape(X, (self.batch_size, a*r, b*r, 1))
        
    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (self.batch_size, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (self.batch_size, a*r, b*r, 1))
        
    def PS(self, X, r):
        # Main OP that you can arbitrarily use in you tensorflow code
        if self.channels == 3:
            Xc = tf.split(X, 3, 3)
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3)
        else:
            X = self._phase_shift(X, r)
        return X
        
    def _load(self, checkpoint_dir):
        print("[*] Reading Checkpoints...")
        model_dir = "%s_%s_%s" % ("espcn", self.patch_size, self.scale) # give the model name by label_size
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        # Test if checkpoint exists 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Load SUCCESS! %s" % chkpt_path)
        else:
            print(" [!] Load failed...")
            
    def _save(self, checkpoint_dir, step):
        model_name = "ESPCN.model"
        model_dir = "%s_%s_%s" % ("espcn", self.patch_size, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.saver.save(self.session,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def train(self, config):
        input_, label_ = read_hdf5(config.hdf5_path)

        # Stochastic gradient descent with the standard backpropagation
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        
        tf.initialize_all_variables().run()

        counter = 0
        time_ = time.time()

        self._load(config.checkpoint_dir)
        
        for ep in range(config.epoch):
            # Run by batch images
            batch_idxs = len(input_)
            for idx in range(0, batch_idxs):
                batch_inputs = input_[idx * self.batch_size : (idx + 1) * self.batch_size]
                batch_labels = label_[idx * self.batch_size : (idx + 1) * self.batch_size]
                
                counter += 1
                _, err = self.session.run([optimizer, self.loss], feed_dict={self.inputs: batch_inputs, self.labels: batch_labels})

                if counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), counter, time.time()-time_, err))
                if counter % 500 == 0:
                    self._save(config.checkpoint_dir, counter)
            
