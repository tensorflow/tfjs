# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

if os.environ['TFJS2KERAS_TEST_USING_TF_KERAS'] == '1':
  print('Using tensorflow.keras.')
  from tensorflow import keras
else:
  print('Using keras-team/keras.')
  import keras


class Tfjs2KerasExportTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    cls._tmp_dir = os.path.join(curr_dir, 'test-data')

  def _loadAndTestModel(self, model_path):
    """Load a Keras Model from artifacts generated by tensorflow.js.

    This method tests:
      - Python Keras loading of the topology JSON file saved from TensorFlow.js.
      - Python Keras loading of the model's weight values.
      - The equality of the model.predict() output between Python Keras and
        TensorFlow.js (up to a certain numeric tolerance.)

    Args:
      model_path: Path to the model JSON file.
    """
    xs_shape_path = os.path.join(
        self._tmp_dir, model_path + '.xs-shapes.json')
    xs_data_path = os.path.join(
        self._tmp_dir, model_path + '.xs-data.json')
    with open(xs_shape_path, 'rt') as f:
      xs_shapes = json.load(f)
    with open(xs_data_path, 'rt') as f:
      xs_values = json.load(f)
    xs = [np.array(value, dtype=np.float32).reshape(shape)
          for value, shape in zip(xs_values, xs_shapes)]
    if len(xs) == 1:
      xs = xs[0]

    ys_shape_path = os.path.join(
        self._tmp_dir, model_path + '.ys-shapes.json')
    ys_data_path = os.path.join(
        self._tmp_dir, model_path + '.ys-data.json')
    with open(ys_shape_path, 'rt') as f:
      ys_shapes = json.load(f)
    with open(ys_data_path, 'rt') as f:
      ys_values = json.load(f)
    ys = [np.array(value, dtype=np.float32).reshape(shape)
          for value, shape in zip(ys_values, ys_shapes)]
    if len(ys) == 1:
      ys = ys[0]

    session = tf.Session() if hasattr(tf, 'Session') else tf.compat.v1.Session()
    with tf.Graph().as_default(), session:
      model_json_path = os.path.join(self._tmp_dir, model_path, 'model.json')
      print('Loading model from path %s' % model_json_path)
      model = tfjs.converters.load_keras_model(model_json_path)
      ys_new = model.predict(xs)
      if isinstance(ys, list):
        self.assertEqual(len(ys), len(ys_new))
        for i, y in enumerate(ys):
          self.assertAllClose(y, ys_new[i])
      else:
        self.assertAllClose(ys, ys_new)

  def testMLP(self):
    self._loadAndTestModel('mlp')

  def testCNN(self):
    self._loadAndTestModel('cnn')

  def testDepthwiseCNN(self):
    self._loadAndTestModel('depthwise_cnn')

  def testSimpleRNN(self):
    self._loadAndTestModel('simple_rnn')

  def testGRU(self):
    self._loadAndTestModel('gru')

  def testBidirectionalLSTM(self):
    self._loadAndTestModel('bidirectional_lstm')

  def testTimeDistributedLSTM(self):
    self._loadAndTestModel('time_distributed_lstm')

  def testOneDimensional(self):
    self._loadAndTestModel('one_dimensional')

  def testFunctionalMerge(self):
    self._loadAndTestModel('functional_merge.json')


if __name__ == '__main__':
  tf.test.main()
