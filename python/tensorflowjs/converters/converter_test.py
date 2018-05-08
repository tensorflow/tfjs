# Copyright 2018 Google LLC
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
"""Unit tests for artifact conversion to and from Python Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
import shutil
import tempfile
import unittest

import keras
import tensorflow as tf

from tensorflowjs.converters import converter

# TODO(adarob): Add tests for quantization option.


class ConvertH5WeightsTest(unittest.TestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    super(ConvertH5WeightsTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(ConvertH5WeightsTest, self).tearDown()

  def testWeightsOnly(self):
    with tf.Graph().as_default(), tf.Session():
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
    model_json, groups = (
        converter.dispatch_keras_h5_to_tensorflowjs_conversion(
            h5_path, output_dir=self._tmp_dir))
    self.assertIsNone(model_json)

    # Check the loaded weights.
    weights1 = groups[0]
    self.assertEqual(2, len(weights1))
    # contents of weights are verified in tests of the library code

    # Check the content of the output directory.
    output_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))
    self.assertEqual(model_json, output_json['modelTopology'])
    self.assertIsInstance(output_json['weightsManifest'], list)
    self.assertTrue(glob.glob(os.path.join(self._tmp_dir, 'group*-*')))

  def testConvertSavedModel(self):
    with tf.Graph().as_default(), tf.Session():
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
    model_json, groups = (
        converter.dispatch_keras_h5_to_tensorflowjs_conversion(
            h5_path, output_dir=self._tmp_dir))
    # check the model topology was stored
    self.assertIsInstance(model_json['model_config'], dict)
    self.assertIsInstance(model_json['model_config']['config'], dict)
    self.assertIn('layers', model_json['model_config']['config'])

    # Check the loaded weights.
    self.assertEqual(keras.__version__, model_json['keras_version'])
    self.assertEqual('tensorflow', model_json['backend'])
    weights1 = groups[0]
    self.assertEqual(2, len(weights1))
    # contents of weights are verified in tests of the library code

    # Check the content of the output directory.
    output_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))
    self.assertEqual(model_json, output_json['modelTopology'])
    self.assertIsInstance(output_json['weightsManifest'], list)
    self.assertTrue(glob.glob(os.path.join(self._tmp_dir, 'group*-*')))

  def testConvertWeightsFromSequentialModel(self):
    with tf.Graph().as_default(), tf.Session():
      sequential_model = keras.models.Sequential([
          keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1'),
          keras.layers.Dense(
              1, use_bias=False, kernel_initializer='ones', name='Dense2')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save_weights(h5_path)

    # Load the saved weights as a JSON string.
    model_json, groups = (
        converter.dispatch_keras_h5_to_tensorflowjs_conversion(
            h5_path, output_dir=self._tmp_dir))
    self.assertIsNone(model_json)

    # Check the loaded weights.
    weights1 = groups[0]
    self.assertEqual(2, len(weights1))
    # contents of weights are verified in tests of the library code

    # Check the content of the output directory.
    output_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))
    self.assertEqual(model_json, output_json['modelTopology'])
    self.assertIsInstance(output_json['weightsManifest'], list)
    self.assertTrue(glob.glob(os.path.join(self._tmp_dir, 'group*-*')))

  def testConvertModelForNonexistentDirCreatesDir(self):
    with tf.Graph().as_default(), tf.Session():
      output_dir = os.path.join(self._tmp_dir, 'foo_model')
      sequential_model = keras.models.Sequential([
          keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save_weights(h5_path)
      converter.dispatch_keras_h5_to_tensorflowjs_conversion(
          h5_path, output_dir=output_dir)

    # Check the content of the output directory.
    output_json = json.load(
        open(os.path.join(output_dir, 'model.json'), 'rt'))
    self.assertIsNone(output_json['modelTopology'])
    self.assertIsInstance(output_json['weightsManifest'], list)
    self.assertTrue(glob.glob(os.path.join(output_dir, 'group*-*')))

  def testOutpuDirAsAnExistingFileLeadsToValueError(self):
    output_path = os.path.join(self._tmp_dir, 'foo_model')
    with open(output_path, 'wt') as f:
      f.write('\n')

    with tf.Graph().as_default(), tf.Session():
      sequential_model = keras.models.Sequential([
          keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save_weights(h5_path)

    with self.assertRaisesRegexp(  # pylint: disable=deprecated-method
        ValueError, r'already exists as a file'):
      converter.dispatch_keras_h5_to_tensorflowjs_conversion(
          h5_path, output_dir=output_path)

  def testTensorflowjsToKerasConversionSucceeds(self):
    with tf.Graph().as_default(), tf.Session():
      sequential_model = keras.models.Sequential([
          keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1'),
          keras.layers.Dense(
              1, use_bias=False, kernel_initializer='ones', name='Dense2')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save(h5_path)
      converter.dispatch_keras_h5_to_tensorflowjs_conversion(
          h5_path, output_dir=self._tmp_dir)
      old_model_json = sequential_model.to_json()

    # Convert the tensorflowjs artifacts to a new H5 file.
    new_h5_path = os.path.join(self._tmp_dir, 'new.h5')
    converter.dispatch_tensorflowjs_to_keras_h5_conversion(
        os.path.join(self._tmp_dir, 'model.json'), new_h5_path)

    # Load the new H5 and compare the model JSONs.
    with tf.Graph().as_default(), tf.Session():
      new_model = keras.models.load_model(new_h5_path)
      self.assertEqual(old_model_json, new_model.to_json())

  def testTensorflowjsToKerasConversionFailsOnDirInputPath(self):
    with self.assertRaisesRegexp(  # pylint: disable=deprecated-method
        ValueError, r'input path should be a model\.json file'):
      converter.dispatch_tensorflowjs_to_keras_h5_conversion(
          self._tmp_dir, os.path.join(self._tmp_dir, 'new.h5'))

  def testTensorflowjsToKerasConversionFailsOnExistingDirOutputPath(self):
    with tf.Graph().as_default(), tf.Session():
      sequential_model = keras.models.Sequential([
          keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1'),
          keras.layers.Dense(
              1, use_bias=False, kernel_initializer='ones', name='Dense2')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save(h5_path)
      converter.dispatch_keras_h5_to_tensorflowjs_conversion(
          h5_path, output_dir=self._tmp_dir)

    with self.assertRaisesRegexp(  # pylint: disable=deprecated-method
        ValueError, r'but received an existing directory'):
      converter.dispatch_tensorflowjs_to_keras_h5_conversion(
          os.path.join(self._tmp_dir, 'model.json'), self._tmp_dir)

  def testTensorflowjsToKerasConversionFailsOnInvalidJsonFile(self):
    fake_json_path = os.path.join(self._tmp_dir, 'fake.json')
    with open(fake_json_path, 'wt') as f:
      f.write('__invalid_json_content__')

    with self.assertRaisesRegexp(  # pylint: disable=deprecated-method
        ValueError, r'cannot read valid JSON content from'):
      converter.dispatch_tensorflowjs_to_keras_h5_conversion(
          fake_json_path, os.path.join(self._tmp_dir, 'model.h5'))


if __name__ == '__main__':
  unittest.main()
