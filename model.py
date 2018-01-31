import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as tflayers
import numpy as np

def leaky_relu(input, slope=0.2):
    return tf.nn.relu(input) - slope * tf.nn.relu(-input)

class Model:
    def __init__(self):
        pass

    def residual_block(self, input, name):
        print("Residual Block", name)
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d], # 3, 1
                                num_outputs=256, kernel_size=3, stride=1, padding='SAME',
                                activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
                                weights_initializer=tflayers.xavier_initializer()):

                #conv = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'
                conv = slim.conv2d(input)
                print(conv.shape)
                #conv = tf.pad(conv, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
                conv = slim.conv2d(conv)
                print(conv.shape)

            output = tf.nn.relu(input + conv)
            return output

    def refiner(self, input, reuse=False):
        with tf.variable_scope("refiner", reuse=reuse):
            with slim.arg_scope([slim.conv2d], # 3, 2
                                kernel_size=3, stride=2, padding='SAME',
                                activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
                                weights_initializer=tflayers.xavier_initializer()):
                conv = slim.conv2d(input, num_outputs=128)       # 32 x 32 x 128
                conv = slim.conv2d(conv, num_outputs=256)        # 16 x 16 x 256
                # conv = slim.conv2d(conv, num_outputs=512)        # 8 x 8 x 512

            # conv = self.residual_block(conv, "ResBlock1")
            # conv = self.residual_block(conv, "ResBlock2")
            # conv = self.residual_block(conv, "ResBlock3")
            # conv = self.residual_block(conv, "ResBlock4")
            # conv = self.residual_block(conv, "ResBlock5")
            # conv = self.residual_block(conv, "ResBlock6")


            # 3, 2
            # conv = tf.image.resize_images(conv, [16, 16])
            # conv = slim.conv2d(conv, num_outputs=512, kernel_size=3, stride=1, padding='SAME',
            #                     activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
            #                     weights_initializer=tflayers.xavier_initializer(),
            #                     biases_initializer=tflayers.xavier_initializer())

            conv = tf.image.resize_images(conv, [32, 32])
            conv = slim.conv2d_transpose(conv, num_outputs=256, kernel_size=3, stride=1, padding='SAME',
                               activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
                               weights_initializer=tflayers.xavier_initializer(),
                               biases_initializer=tflayers.xavier_initializer())

            # conv = tf.image.resize_images(conv, [32, 32])
            # conv = slim.conv2d(conv, num_outputs=256, kernel_size=3, stride=1, padding='SAME',
            #                    activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
            #                    weights_initializer=tflayers.xavier_initializer(),
            #                    biases_initializer=tflayers.xavier_initializer())

            # conv = slim.conv2d_transpose(conv, num_outputs=512, kernel_size=1, stride=1, padding='SAME',
            #                              activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
            #                              weights_initializer=tflayers.xavier_initializer())
            #
            # conv = slim.conv2d_transpose(conv, num_outputs=256, kernel_size=1, stride=1, padding='SAME',
            #                              activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
            #                              weights_initializer=tflayers.xavier_initializer())

            # 3, 2
            conv = tf.image.resize_images(conv, [64, 64])
            output = slim.conv2d_transpose(conv, num_outputs=3, kernel_size=3, stride=1, padding='SAME',
                                activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
                                weights_initializer=tflayers.xavier_initializer(),
                                biases_initializer=tflayers.xavier_initializer())

            return output

    def discriminator(self, input, reuse=False):
        with tf.variable_scope("disctiminator", reuse=reuse):
            with slim.arg_scope([slim.conv2d], # 3x 2
                                kernel_size=2, stride=2, padding='SAME',
                                activation_fn=leaky_relu, normalizer_fn=tflayers.batch_norm,
                                weights_initializer=tflayers.xavier_initializer(),
                                biases_initializer=tflayers.xavier_initializer()):

                conv = slim.conv2d(input, num_outputs=128)      # 32 x 32 x 128
                conv = slim.conv2d(conv, num_outputs=256)       # 16 x 16 x 256
                conv = slim.conv2d(conv, num_outputs=512)       # 8 x 8 x 512
                conv = slim.conv2d(conv, num_outputs=256)       # 4 x 4 x 256

            # 2 x 2 x 1
            conv = slim.conv2d(conv, num_outputs=1, kernel_size=2, stride=1,
                               normalizer_fn=tflayers.batch_norm, weights_initializer=tflayers.xavier_initializer(),
                               biases_initializer=tflayers.xavier_initializer())

            return conv

    def build(self):
        self.synthetic = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.real = tf.placeholder(tf.float32, [None, 64, 64, 3])

        self.is_history = tf.placeholder(tf.bool)

        self.history_refined = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.refined = self.refiner(self.synthetic)
        # self.refined = tf.cond(self.is_history,
        #                        lambda: tf.placeholder(tf.float32, [None, 64, 64, 3]),
        #                        lambda: self.refiner(self.synthetic))
        # self.refined = tf.placeholder(tf.float32, [None, 64, 64, 3]) if self.is_history is True else self.refiner(self.synthetic)
        self.refined_prop_map = tf.cond(self.is_history,
                                        lambda: self.discriminator(self.history_refined),
                                        lambda: self.discriminator(self.refined, reuse=True))
        self.real_prop_map = self.discriminator(self.real, reuse=True)

        print('Refined Prop Map :', self.refined_prop_map.shape)
        print('Real Prop Map :', self.real_prop_map.shape)

        self.d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.real_prop_map), self.real_prop_map)
        self.d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.refined_prop_map), self.refined_prop_map)
        self.d_loss = tf.reduce_mean(self.d_loss_real + self.d_loss_fake)
        self.r_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(self.refined_prop_map), self.refined_prop_map))

        self.reg_loss = tf.reduce_mean(tf.abs(self.refined - self.real))

        self.optimizer_d = tf.train.AdamOptimizer(0.001).minimize(self.d_loss)
        self.optimizer_r = tf.train.AdamOptimizer(0.001).minimize(self.r_loss)
        self.optimizer_reg = tf.train.AdamOptimizer(0.001).minimize(self.reg_loss)
