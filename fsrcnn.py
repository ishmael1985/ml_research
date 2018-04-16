import tensorflow as tf
import numpy as np
import sys
import time
import os
import json

from utils import read_hdf5


class FSRCNN:
    def __init__(self, session, checkpoint_dir):
        self.session = session
        self.checkpoint_dir = checkpoint_dir
        
        with open("hdf5.json", "r") as config:
            params = json.load(config)

        self.scale = params["upscale_factor"]
        self.input_size = params["input_size"][self.scale - 2]
        self.label_size = params["label_size"][self.scale - 2]
        self.channels = params["channels"]

        with open("fsrcnn.json", "r") as config:
            self.model_params = json.load(config)

        # Different model layer counts and filter sizes for FSRCNN vs FSRCNN-s (fast), (s, d, m) in paper
        if not self.model_params["fast"]:
            self.model_config = (56, 12, 4)
        else:
            self.model_config = (32, 5, 1)

        self.test_image = None
        self.saver = None
        self.is_loaded = False

    def prelu(self, _x, i):
        """
        PreLU tensorflow implementation
        """
        alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg
        
    def _model(self):
        # Feature Extraction
        conv_feature = self.prelu(tf.nn.conv2d(self.inputs, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'], 1)

        # Shrinking
        conv_shrink = self.prelu(tf.nn.conv2d(conv_feature, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'], 2)

        # Mapping (# mapping layers = m)
        prev_layer, m = conv_shrink, self.model_config[2]
        for i in range(3, m + 3):
            weights, biases = self.weights['w{}'.format(i)], self.biases['b{}'.format(i)]
            prev_layer = self.prelu(tf.nn.conv2d(prev_layer, weights, strides=[1,1,1,1], padding='SAME') + biases, i)

        # Expanding
        expand_weights, expand_biases = self.weights['w{}'.format(m + 3)], self.biases['b{}'.format(m + 3)]
        conv_expand = self.prelu(tf.nn.conv2d(prev_layer, expand_weights, strides=[1,1,1,1], padding='SAME') + expand_biases, m + 3)

        # Deconvolution
        if not self.test_image:
            deconv_output = [self.batch, self.label_size, self.label_size, self.channels]
        else:
            deconv_output = [1, (self.height - 4) * self.scale, (self.width - 4) * self.scale, self.channels]
        deconv_stride = [1, self.scale, self.scale, 1]
        deconv_weights, deconv_biases = self.weights['w{}'.format(m + 4)], self.biases['b{}'.format(m + 4)]
        conv_deconv = tf.nn.conv2d_transpose(conv_expand, deconv_weights, output_shape=deconv_output, strides=deconv_stride, padding='SAME') + deconv_biases

        return conv_deconv

    def _build_model(self):
        if not self.test_image:
            self.inputs = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, self.channels], name='inputs')
            self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.channels], name='labels')
        else:
            self.width, self.height = self.test_image.size
            self.inputs = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='inputs')

        # Batch size differs in training vs testing
        self.batch = tf.placeholder(tf.int32, shape=[], name='batch')

        # FSCRNN-s (fast) has smaller filters and less layers but can achieve faster performance
        s, d, m = self.model_config

        expand_weight, deconv_weight = 'w{}'.format(m + 3), 'w{}'.format(m + 4)
        self.weights = {
            'w1': tf.Variable(tf.random_normal([5, 5, 1, s], stddev=0.0378, dtype=tf.float32), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, s, d], stddev=0.3536, dtype=tf.float32), name='w2'),
            expand_weight: tf.Variable(tf.random_normal([1, 1, d, s], stddev=0.189, dtype=tf.float32), name=expand_weight),
            deconv_weight: tf.Variable(tf.random_normal([9, 9, 1, s], stddev=0.0001, dtype=tf.float32), name=deconv_weight)
        }

        expand_bias, deconv_bias = 'b{}'.format(m + 3), 'b{}'.format(m + 4)
        self.biases = {
            'b1': tf.Variable(tf.zeros([s]), name='b1'),
            'b2': tf.Variable(tf.zeros([d]), name='b2'),
            expand_bias: tf.Variable(tf.zeros([s]), name=expand_bias),
            deconv_bias: tf.Variable(tf.zeros([1]), name=deconv_bias)
        }

        # Create the m mapping layers weights/biases
        for i in range(3, m + 3):
            weight_name, bias_name = 'w{}'.format(i), 'b{}'.format(i)
            self.weights[weight_name] = tf.Variable(tf.random_normal([3, 3, d, d], stddev=0.1179, dtype=tf.float32), name=weight_name)
            self.biases[bias_name] = tf.Variable(tf.zeros([d]), name=bias_name)

        self.saver = tf.train.Saver()

        self.pred = self._model()

        # Loss function (MSE)
        if not self.test_image:
            self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        
    def _load(self):
        print("[*] Reading Checkpoints...")
        model_dir = "%s_%s" % ("fsrcnn", self.scale) # give the model name by label_size
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
        model_name = "FSRCNN.model"
        model_dir = "%s_%s" % ("fsrcnn", self.scale)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.saver.save(self.session,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def train(self, hdf5_path, epochs):
        self._build_model()
        
        learning_rate = self.model_params["learning_rate"]
        batch_size = self.model_params["batch_size"]
        momentum = self.model_params["momentum"]

        # SGD with momentum
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.loss)

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
                _, err = self.session.run([optimizer, self.loss],
                                          feed_dict={self.inputs: batch_inputs,
                                                     self.labels: batch_labels,
                                                     self.batch: batch_size})

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

        tf.global_variables_initializer().run()

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







