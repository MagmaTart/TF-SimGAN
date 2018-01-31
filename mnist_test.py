import tensorflow as tf
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.slim as slim
import numpy as np
import cv2

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST', one_hot=True)

def generator(z):
    z = np.reshape(z, [len(z), 28, 28, 1])

    conv = slim.conv2d_transpose(z, num_outputs=128, kernel_size=3, stride=1, padding='SAME',
                                 activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
                                 weights_initializer=tflayers.xavier_initializer())
    conv = slim.conv2d_transpose(conv, num_outputs=128, kernel_size=3, stride=1, padding='SAME',
                                 activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
                                 weights_initializer=tflayers.xavier_initializer())

a, _ = mnist.train.next_batch(128)
generator(a)