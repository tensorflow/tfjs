# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Test for benchmarks.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import unittest

import keras

from scripts import benchmarks


class BenchmarksTest(unittest.TestCase):

  def _simpleModelFn(self, input_shape, target_shape):
    input_layer = keras.Input(input_shape)
    dense_layer = keras.layers.Dense(target_shape[0])
    output = dense_layer(input_layer)
    return  keras.Model(input_layer, output)

  def testBenchmakrAndSerializeModel(self):
    model_json, weights_json, train_time, predict_time = (
        benchmarks.benchmark_and_serialize_model(
            self._simpleModelFn, [10], [1], 'sgd', 'mean_squared_error', 8, 4))
    json.loads(model_json)
    self.assertEqual(2, len(weights_json['weights']))
    self.assertGreater(train_time, 0.0)
    self.assertGreater(predict_time, 0.0)

  def testDenseModelFns(self):
    model = benchmarks.dense_tiny_model_fn([8], [1])
    self.assertEqual(3, len(model.layers))
    self.assertEqual(1, len(model.outputs))
    self.assertEqual(1, model.outputs[0].shape[1])
    self.assertTrue(model.trainable)
    model = benchmarks.dense_large_model_fn([8], [2])
    self.assertEqual(5, len(model.layers))
    self.assertEqual(1, len(model.outputs))
    self.assertEqual(2, model.outputs[0].shape[1])
    self.assertTrue(model.trainable)

  def testConvolutionalModelFn(self):
    model = benchmarks.convolutional_model_fn(1, [28, 28, 1], [10])
    self.assertEqual(10, len(model.layers))
    self.assertEqual(1, len(model.outputs))
    self.assertEqual(10, model.outputs[0].shape[1])
    self.assertTrue(model.trainable)


if __name__ == '__main__':
  unittest.main()
