import tensorflow as tf
import numpy as np
import cv2
from load_data import Loader
from model import Model

import random

batch_size = 128
history_size = int(batch_size/2)
history_buffer_size = 512
history_buffer = [0 for i in range(history_buffer_size)]
history_index = 0

history_initialize = True

buffer_ringed = False

loader = Loader()
loader.load_data()

model = Model()

model.build()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    synthetic, real = loader.get_next_batch(batch_size=batch_size), loader.get_next_batch(batch_size=batch_size)

    history_train_refined = np.random.rand(1, 64, 64, 3)
    history_train_real = np.random.rand(1, 64, 64, 3)

    # History training
    if history_index > history_buffer_size:
        history_train_refined = np.array([history_buffer[random.randrange(0, history_buffer_size)] for i in range(history_size)])
        history_train_real = loader.get_next_batch(history_size)
        sess.run(model.optimizer_d, feed_dict={model.is_history: True, model.synthetic: synthetic, model.history_refined: history_train_refined, model.real: history_train_real})

    feed_dict = {model.is_history: False, model.history_refined: history_train_refined, model.synthetic: synthetic, model.real: real}
    sess.run([model.optimizer_d], feed_dict=feed_dict)
    # sess.run([model.optimizer_d], feed_dict=feed_dict)
    # sess.run([model.optimizer_d], feed_dict=feed_dict)
    # sess.run([model.optimizer_d], feed_dict=feed_dict)
    # sess.run([model.optimizer_d], feed_dict=feed_dict)
    sess.run([model.optimizer_r], feed_dict=feed_dict)
    sess.run([model.optimizer_r], feed_dict=feed_dict)
    sess.run([model.optimizer_r], feed_dict=feed_dict)
    sess.run([model.optimizer_r], feed_dict=feed_dict)
    sess.run([model.optimizer_r], feed_dict=feed_dict)
    sess.run([model.optimizer_reg], feed_dict=feed_dict)

    losses = sess.run([model.d_loss, model.r_loss, model.reg_loss], feed_dict=feed_dict)
    print(i, losses[0], losses[1], losses[2])

    history_refines = sess.run(model.refined, feed_dict=feed_dict)

    for k in range(history_size):
        history_buffer[history_index % history_buffer_size] = history_refines[k]
        history_index += 1

    # Save sample
    if i % 10 == 0:
        print('Sample image saved')
        test = sess.run(model.refined, feed_dict={model.is_history: False, model.synthetic: np.array([real[0]])})
        cv2.imwrite('./Samples/'+str(i).zfill(5)+'R.jpg', real[0] * 127.5)
        cv2.imwrite('./Samples/'+str(i).zfill(5)+'S.jpg', test[0] * 127.5)
        # *127.5