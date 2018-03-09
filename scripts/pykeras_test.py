# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Unit tests for artifact conversion to and from Python Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import unittest

import h5py
import keras

from scripts import pykeras


class ConvertH5WeightsTest(unittest.TestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    super(ConvertH5WeightsTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(ConvertH5WeightsTest, self).tearDown()

  def testWeightsOnly(self):
    input_tensor = keras.layers.Input((3,))
    dense1 = keras.layers.Dense(
        4, use_bias=True, kernel_initializer='ones', bias_initializer='zeros',
        name='MyDense1')(input_tensor)
    output = keras.layers.Dense(
        2, use_bias=False, kernel_initializer='ones', name='MyDense2')(dense1)
    model = keras.models.Model(inputs=[input_tensor], outputs=[output])
    h5_path = os.path.join(self._tmp_dir, 'MyModel.h5')
    model.save_weights(h5_path)

    # Load the saved weights as a JSON string.
    json, groups = pykeras.dispatch_pykeras_conversion(h5py.File(h5_path))
    self.assertTrue(len(json) == 0)

    # Check the loaded weights.
    weights1 = groups[0]
    self.assertEqual(2, len(weights1))
    # contents of weights are verified in tests of the library code

  def testConvertSavedModel(self):
    input_tensor = keras.layers.Input((3,))
    dense1 = keras.layers.Dense(
        4, use_bias=True, kernel_initializer='ones', bias_initializer='zeros',
        name='MergedDense1')(input_tensor)
    output = keras.layers.Dense(
        2, use_bias=False,
        kernel_initializer='ones', name='MergedDense2')(dense1)
    model = keras.models.Model(inputs=[input_tensor], outputs=[output])
    h5_path = os.path.join(self._tmp_dir, 'MyModelMerged.h5')
    model.save(h5_path)

    # Load the saved weights as a JSON string.
    json, groups = pykeras.dispatch_pykeras_conversion(h5py.File(h5_path))
    # check the model topology was stored
    self.assertTrue('layers' in json['model_config'])

    # Check the loaded weights.
    self.assertEqual(keras.__version__, json['keras_version'])
    self.assertEqual('tensorflow', json['backend'])
    weights1 = groups[0]
    self.assertEqual(2, len(weights1))
    # contents of weights are verified in tests of the library code

  def testConvertWeightsFromSequentialModel(self):
    sequential_model = keras.models.Sequential([
        keras.layers.Dense(
            3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
            name='Dense1'),
        keras.layers.Dense(
            1, use_bias=False, kernel_initializer='ones', name='Dense2')])
    h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
    sequential_model.save_weights(h5_path)

    # Load the saved weights as a JSON string.
    json, groups = pykeras.dispatch_pykeras_conversion(h5py.File(h5_path))
    self.assertTrue(len(json) == 0)

    # Check the loaded weights.
    weights1 = groups[0]
    self.assertEqual(2, len(weights1))
    # contents of weights are verified in tests of the library code


if __name__ == '__main__':
  unittest.main()
