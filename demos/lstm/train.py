# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trains and Evaluates a simple LSTM network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output_dir', '/tmp/simple_lstm',
                           'Directory to write checkpoint.')

def main(unused_argv):
  data = np.array(
      [[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4]])

  tf.reset_default_graph()

  x = tf.placeholder(dtype=tf.int32, shape=[1, data.shape[1] - 1])
  y = tf.placeholder(dtype=tf.int32, shape=[1, data.shape[1] - 1])

  NHIDDEN = 20
  NLABELS = 10

  lstm1 = tf.contrib.rnn.BasicLSTMCell(NHIDDEN)
  lstm2 = tf.contrib.rnn.BasicLSTMCell(NHIDDEN)
  lstm = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2])
  initial_state = lstm.zero_state(1, tf.float32)

  outputs, final_state = tf.nn.dynamic_rnn(
      cell=lstm, inputs=tf.one_hot(x, NLABELS), initial_state=initial_state)

  logits = tf.contrib.layers.linear(outputs, NLABELS)

  softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y, logits=logits)

  predictions = tf.argmax(logits, axis=-1)
  loss = tf.reduce_mean(softmax_cross_entropy)

  train_op = tf.train.AdamOptimizer().minimize(loss)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  print('Starting training...')
  NEPOCH = 1000
  for step in range(NEPOCH + 1):
    loss_out, _ = sess.run([loss, train_op],
                           feed_dict={
                               x: data[:, :-1],
                               y: data[:, 1:],
                           })
    if step % 100 == 0:
      print('Loss at step {}: {}'.format(step, loss_out))

  print('Expected data:')
  print(data[:, 1:])
  print('Results:')
  print(sess.run([predictions], feed_dict={x: data[:, :-1]}))

  saver = tf.train.Saver()
  path = saver.save(sess, FLAGS.output_dir, global_step=step)
  print('Saved checkpoint at {}'.format(path))


if __name__ == '__main__':
  tf.app.run(main)

