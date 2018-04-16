import tensorflow as tf
import numpy as np
import sys
import time
import os
import json

from utils import read_hdf5

def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


class ESPCN:
    def __init__(self, session, checkpoint_dir):
        self.session = session
        self.checkpoint_dir = checkpoint_dir
        
        with open("hdf5.json", "r") as config:
            params = json.load(config)

        self.patch_size = params["input_size"]
        self.channels = params["channels"]
        self.scale = params["upscale_factor"]

        self.test_image = None
        self.saver = None
        self.is_loaded = False

    def _build_model(self):
        if not self.test_image:
            self.inputs = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, self.channels], name='inputs')
            self.labels = tf.placeholder(tf.float32, [None, self.patch_size * self.scale , self.patch_size * self.scale, self.channels], name='labels')
        else:
            self.width, self.height = self.test_image.size
            self.inputs = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='inputs')    

        if not self.saver:
            self.weights = {
                'w1': create_variable('w1', [5, 5, self.channels, 64]),
                'w2': create_variable('w2', [3, 3, 64, 32]),
                'w3': create_variable('w3', [3, 3, 32, self.channels * self.scale * self.scale ])
            }

            self.biases = {
                'b1': create_bias_variable('b1', [64]),
                'b2': create_bias_variable('b2', [32]),
                'b3': create_bias_variable('b3', [self.channels * self.scale * self.scale ])
            }

            self.saver = tf.train.Saver()
        
        self.pred = self._model()
        
        # Loss function (MSE)
        if not self.test_image:
            self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

    def _model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.inputs, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3']

        ps = self.PS(conv3, self.scale)
        
        return tf.nn.tanh(ps)

    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (bsize, a*r, b*r, 1))
        
    def PS(self, X, r):
        # Main OP that you can arbitrarily use in you tensorflow code
        if self.channels == 3:
            Xc = tf.split(X, 3, 3)
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3)
        else:
            X = self._phase_shift(X, r)
        return X
        
    def _load(self):
        print("[*] Reading Checkpoints...")
        model_dir = "%s_%s_%s" % ("espcn", self.patch_size, self.scale) # give the model name by label_size
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        # Test if checkpoint exists 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, ckpt_name))
            print("[*] Load SUCCESS!")
            self.is_loaded = True
        else:
            print("[!] Load failed...")
            self.is_loaded = False

        return self.is_loaded
            
    def _save(self, step):
        model_name = "ESPCN.model"
        model_dir = "%s_%s_%s" % ("espcn", self.patch_size, self.scale)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.saver.save(self.session,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def train(self, hdf5_path, learning_rate, batch_size, epochs):
        self._build_model()
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        tf.global_variables_initializer().run()

        self._load()

        input_, label_ = read_hdf5(hdf5_path)

        counter = 0
        time_ = time.time()
        
        for epoch in range(epochs):
            # Run by batch images
            batch_idxs = len(input_) // batch_size
            for idx in range(0, batch_idxs):
                batch_inputs = input_[idx * batch_size : (idx + 1) * batch_size]
                batch_labels = label_[idx * batch_size : (idx + 1) * batch_size]
                
                counter += 1
                _, err = self.session.run([optimizer, self.loss], feed_dict={self.inputs: batch_inputs, self.labels: batch_labels})

                if counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((epoch+1), counter, time.time()-time_, err))
                if counter % 500 == 0:
                    self._save(counter)

    def test(self, test_image):
        if test_image and self.channels == 1:
            self.test_image = test_image.split()[0]
        else:
            self.test_image = test_image
            
        self._build_model()

        if not self.is_loaded:
            if not self._load():
                return False

        img = np.asarray(self.test_image)
        img = img / 255.0
        
        result = self.pred.eval({self.inputs: img.reshape(1,
                                                          self.height,
                                                          self.width,
                                                          self.channels)})
        return np.squeeze(result)


