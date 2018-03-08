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

import json
import os
import shutil
import tempfile
import unittest

import h5py
import keras
import numpy as np

from scripts import h5_conversion

class ConvertH5WeightsTest(unittest.TestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    self._converter = h5_conversion.HDF5Converter()
    super(ConvertH5WeightsTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(ConvertH5WeightsTest, self).tearDown()

  def testConvertWeightsFromSimpleModel(self):
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
    out = self._converter.h5_weights_to_json(h5py.File(h5_path))

    # Check the loaded weights.
    self.assertEqual(keras.__version__, out['keras_version'])
    self.assertEqual('tensorflow', out['backend'])
    weights = out['weights']
    weights1 = weights['MyDense1']
    self.assertEqual(2, len(weights1))
    kernel1 = weights1[0]
    self.assertEqual('MyDense1/kernel:0', kernel1['name'])
    self.assertEqual('float32', kernel1['dtype'])
    self.assertEqual([3, 4], kernel1['shape'])
    self.assertEqual(np.ones([3, 4]).tolist(), kernel1['value'])
    bias1 = weights1[1]
    self.assertEqual('MyDense1/bias:0', bias1['name'])
    self.assertEqual('float32', bias1['dtype'])
    self.assertEqual([4], bias1['shape'])
    self.assertEqual(np.zeros([4]).tolist(), bias1['value'])
    weights2 = weights['MyDense2']
    self.assertEqual(1, len(weights2))
    kernel2 = weights2[0]
    self.assertEqual('MyDense2/kernel:0', kernel2['name'])
    self.assertEqual('float32', kernel2['dtype'])
    self.assertEqual([4, 2], kernel2['shape'])
    self.assertEqual(np.ones([4, 2]).tolist(), kernel2['value'])

  def testConvertMergedModelFromSimpleModel(self):
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
    config_json = json.loads(model.to_json(), encoding='utf8')

    # Load the saved weights as a JSON string.
    out = self._converter.h5_merged_saved_model_to_json(h5py.File(h5_path))
    saved_topology = json.loads(out['model_config'])

    # check the model topology was stored
    self.assertEqual(config_json['class_name'], saved_topology['class_name'])
    self.assertEqual(config_json['config'], saved_topology['config'])

    # Check the loaded weights.
    self.assertEqual(keras.__version__, out['keras_version'])
    self.assertEqual('tensorflow', out['backend'])
    weights = out['model_weights']
    weights1 = weights['MergedDense1']
    self.assertEqual(2, len(weights1))
    kernel1 = weights1[0]
    self.assertEqual('MergedDense1/kernel:0', kernel1['name'])
    self.assertEqual('float32', kernel1['dtype'])
    self.assertEqual([3, 4], kernel1['shape'])
    self.assertEqual(np.ones([3, 4]).tolist(), kernel1['value'])
    bias1 = weights1[1]
    self.assertEqual('MergedDense1/bias:0', bias1['name'])
    self.assertEqual('float32', bias1['dtype'])
    self.assertEqual([4], bias1['shape'])
    self.assertEqual(np.zeros([4]).tolist(), bias1['value'])
    weights2 = weights['MergedDense2']
    self.assertEqual(1, len(weights2))
    kernel2 = weights2[0]
    self.assertEqual('MergedDense2/kernel:0', kernel2['name'])
    self.assertEqual('float32', kernel2['dtype'])
    self.assertEqual([4, 2], kernel2['shape'])
    self.assertEqual(np.ones([4, 2]).tolist(), kernel2['value'])


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
    out = self._converter.h5_weights_to_json(h5py.File(h5_path))

    # Check the loaded weights.
    self.assertEqual(keras.__version__, out['keras_version'])
    self.assertEqual('tensorflow', out['backend'])
    weights1 = out['weights']['Dense1']
    self.assertEqual(2, len(weights1))
    kernel1 = weights1[0]
    self.assertEqual('Dense1/kernel:0', kernel1['name'])
    self.assertEqual('float32', kernel1['dtype'])
    self.assertEqual([2, 3], kernel1['shape'])
    self.assertEqual(np.ones([2, 3]).tolist(), kernel1['value'])
    bias1 = weights1[1]
    self.assertEqual('Dense1/bias:0', bias1['name'])
    self.assertEqual('float32', bias1['dtype'])
    self.assertEqual([3], bias1['shape'])
    self.assertEqual(np.zeros([3]).tolist(), bias1['value'])
    weights2 = out['weights']['Dense2']
    self.assertEqual(1, len(weights2))
    kernel2 = weights2[0]
    self.assertEqual('Dense2/kernel:0', kernel2['name'])
    self.assertEqual('float32', kernel2['dtype'])
    self.assertEqual([3, 1], kernel2['shape'])
    self.assertEqual(np.ones([3, 1]).tolist(), kernel2['value'])

  def testNegativeDecimalPlacesRaisesException(self):
    with self.assertRaises(ValueError):
      h5_conversion.HDF5Converter(-1)


if __name__ == '__main__':
  unittest.main()
