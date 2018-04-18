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
"""Test the Python API and shell binary of the tensorflowjs pip package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import keras
import tensorflow as tf

import tensorflowjs as tfjs


def _createKerasModel(layer_name_prefix, h5_path=None):
  """Create a Keras model for testing.

  Args:
    layer_name_prefix: A prefix string for layer names. This helps avoid
      clashes in layer names between different test methods.
    h5_path: Optional string path for a HDF5 (.h5) file to save the model
      in.

  Returns:
    An instance of keras.Model.
  """
  input_tensor = keras.layers.Input((3, ))
  dense1 = keras.layers.Dense(
      4,
      use_bias=True,
      kernel_initializer='ones',
      bias_initializer='zeros',
      name=layer_name_prefix + '1')(input_tensor)
  output = keras.layers.Dense(
      2,
      use_bias=False,
      kernel_initializer='ones',
      name=layer_name_prefix + '2')(dense1)
  model = keras.models.Model(inputs=[input_tensor], outputs=[output])
  if h5_path:
    model.save(h5_path)
  return model


def _createTensorFlowSavedModel(name_scope, save_path):
  """Create a TensorFlow SavedModel for testing.

  Args:
    name_scope: Name scope to create the model under. This helps avoid
      op and variable name clashes between different test methods.
    save_path: The directory path in which to save the model.
  """

  with tf.name_scope(name_scope):
    x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
    w = tf.get_variable('w', shape=[2, 2])
    y = tf.matmul(x, w)
    tf.nn.softmax(y)
    init_op = w.initializer

    # Create a builder.
    builder = tf.saved_model.builder.SavedModelBuilder(save_path)

    with tf.Session() as sess:
      # Run the initializer on `w`.
      sess.run(init_op)

      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map=None,
          assets_collection=None)

    builder.save()


class APIAndShellTest(tf.test.TestCase):
  """Tests for the Python API of the pip package."""

  @classmethod
  def setUpClass(cls):
    cls.class_tmp_dir = tempfile.mkdtemp()
    cls.tf_saved_model_dir = os.path.join(cls.class_tmp_dir,
                                          'tf_saved_model')
    _createTensorFlowSavedModel('a', cls.tf_saved_model_dir)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.class_tmp_dir)

  def setUp(self):
    # Make sure this file is not being run from the source directory, to
    # avoid picking up source files.
    if os.path.isdir(
        os.path.join(os.path.dirname(__file__), 'tensorflowjs')):
      self.fail('Do not run this test from the Python source directory. '
                'This file is intended to be run on pip install.')

    self._tmp_dir = tempfile.mkdtemp()
    super(APIAndShellTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(APIAndShellTest, self).tearDown()

  def testVersionString(self):
    self.assertEqual(2, tfjs.__version__.count('.'))

  def testSaveKerasModel(self):
    with self.test_session():
      # First create a toy keras model.
      model = _createKerasModel('MergedDense')

      tfjs.converters.save_keras_model(model, self._tmp_dir)

      # Briefly check the model topology.
      json_content = json.load(
          open(os.path.join(self._tmp_dir, 'model.json')))
      model_json = json_content['modelTopology']
      self.assertIsInstance(model_json['model_config'], dict)
      self.assertIsInstance(model_json['model_config']['config'], dict)
      self.assertIn('layers', model_json['model_config']['config'])

      weights_manifest = json_content['weightsManifest']
      self.assertIsInstance(weights_manifest, list)

      # Briefly check the weights manifest.
      weight_shapes = dict()
      weight_dtypes = dict()
      for manifest_item in weights_manifest:
        for weight in manifest_item['weights']:
          weight_name = weight['name']
          weight_shapes[weight_name] = weight['shape']
          weight_dtypes[weight_name] = weight['dtype']

      self.assertEqual(
          sorted(list(weight_shapes.keys())),
          sorted([
              'MergedDense1/kernel', 'MergedDense1/bias',
              'MergedDense2/kernel'
          ]))
      self.assertEqual(weight_shapes['MergedDense1/kernel'], [3, 4])
      self.assertEqual(weight_shapes['MergedDense1/bias'], [4])
      self.assertEqual(weight_shapes['MergedDense2/kernel'], [4, 2])
      self.assertEqual(weight_dtypes['MergedDense1/kernel'], 'float32')
      self.assertEqual(weight_dtypes['MergedDense1/bias'], 'float32')
      self.assertEqual(weight_dtypes['MergedDense2/kernel'], 'float32')

  def testLoadKerasModel(self):
    # Use separate tf.Graph and tf.Session contexts to prevent name collision.
    with tf.Graph().as_default(), tf.Session():
      # First create a toy keras model.
      model1 = _createKerasModel('MergedDense')
      tfjs.converters.save_keras_model(model1, self._tmp_dir)
      model1_weight_values = model1.get_weights()

    with tf.Graph().as_default(), tf.Session():
      # Load the model from saved artifacts.
      model2 = tfjs.converters.load_keras_model(
          os.path.join(self._tmp_dir, 'model.json'))

      # Compare the loaded model with the original one.
      model2_weight_values = model2.get_weights()
      self.assertEqual(len(model1_weight_values), len(model2_weight_values))
      for model1_weight_value, model2_weight_value in zip(
          model1_weight_values, model2_weight_values):
        self.assertAllClose(model1_weight_value, model2_weight_value)

  def testConvertTensorFlowSavedModel(self):
    output_dir = os.path.join(self._tmp_dir, 'tensorflowjs_model')
    tfjs.converters.convert_tf_saved_model(
        self.tf_saved_model_dir,
        'a/Softmax',
        output_dir,
        saved_model_tags='serve'
    )

    weights = [{
        'paths': ['group1-shard1of1'],
        'weights': [{
            'shape': [2, 2],
            'name': 'a/Softmax',
            'dtype': 'float32'
        }]
    }]
    # Load the saved weights as a JSON string.
    with open(os.path.join(output_dir, 'weights_manifest.json'),
              'rt') as f:
      output_json = json.load(f)
    self.assertEqual(output_json, weights)

    # Check the content of the output directory.
    self.assertTrue(
        glob.glob(os.path.join(output_dir, 'tensorflowjs_model.pb')))
    self.assertTrue(glob.glob(os.path.join(output_dir, 'group*-*')))

  def testUsageWithoutInputFormatErrors(self):
    process = subprocess.Popen(
        ['tensorflowjs_converter', self._tmp_dir, self._tmp_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    self.assertGreater(process.returncode, 0)
    self.assertIn(b'--input_format', tf.compat.as_bytes(stderr))

  def testInvalidInputFormatRaisesError(self):
    process = subprocess.Popen(
        [
            'tensorflowjs_converter', '--input_format',
            'nonsensical_format', self._tmp_dir, self._tmp_dir
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    self.assertGreater(process.returncode, 0)
    self.assertIn(b'--input_format', tf.compat.as_bytes(stderr))

  def testKerasH5ConversionWorksFromCLI(self):
    # First create a toy keras model.
    os.makedirs(os.path.join(self._tmp_dir, 'keras_h5'))
    h5_path = os.path.join(self._tmp_dir, 'keras_h5', 'model.h5')
    _createKerasModel('MergedDenseForCLI', h5_path)

    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'keras', h5_path,
        self._tmp_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    # Briefly check the model topology.
    with open(os.path.join(self._tmp_dir, 'model.json'), 'rt') as f:
      json_content = json.load(f)
    model_json = json_content['modelTopology']
    self.assertIsInstance(model_json['model_config'], dict)
    self.assertIsInstance(model_json['model_config']['config'], dict)
    self.assertIn('layers', model_json['model_config']['config'])

    weights_manifest = json_content['weightsManifest']
    self.assertIsInstance(weights_manifest, list)

    # Briefly check the weights manifest.
    weight_shapes = dict()
    weight_dtypes = dict()
    for manifest_item in weights_manifest:
      for weight in manifest_item['weights']:
        weight_name = weight['name']
        weight_shapes[weight_name] = weight['shape']
        weight_dtypes[weight_name] = weight['dtype']

    self.assertEqual(
        sorted(list(weight_shapes.keys())),
        sorted([
            'MergedDenseForCLI1/kernel', 'MergedDenseForCLI1/bias',
            'MergedDenseForCLI2/kernel'
        ]))
    self.assertEqual(weight_shapes['MergedDenseForCLI1/kernel'], [3, 4])
    self.assertEqual(weight_shapes['MergedDenseForCLI1/bias'], [4])
    self.assertEqual(weight_shapes['MergedDenseForCLI2/kernel'], [4, 2])
    self.assertEqual(weight_dtypes['MergedDenseForCLI1/kernel'], 'float32')
    self.assertEqual(weight_dtypes['MergedDenseForCLI1/bias'], 'float32')
    self.assertEqual(weight_dtypes['MergedDenseForCLI2/kernel'], 'float32')

  def testKerasH5ConversionWithOutputNodeNamesErrors(self):
    process = subprocess.Popen(
        [
            'tensorflowjs_converter', '--input_format', 'keras',
            '--output_node_names', 'foo,bar',
            os.path.join(self._tmp_dir, 'foo.h5'),
            os.path.join(self._tmp_dir, 'output')
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    self.assertGreater(process.returncode, 0)
    self.assertIn(
        b'The --output_node_names flag is applicable only to input format '
        b'"tensorflow"', tf.compat.as_bytes(stderr))

  def testConvertTFSavedModelWithCommandLineWorks(self):
    output_dir = os.path.join(self._tmp_dir)
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'tf_saved_model',
        '--output_node_names', 'a/Softmax', '--saved_model_tags', 'serve',
        self.tf_saved_model_dir, output_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    weights = [{
        'paths': ['group1-shard1of1'],
        'weights': [{
            'shape': [2, 2],
            'name': 'a/Softmax',
            'dtype': 'float32'
        }]
    }]
    # Load the saved weights as a JSON string.
    output_json = json.load(
        open(os.path.join(output_dir, 'weights_manifest.json'), 'rt'))
    self.assertEqual(output_json, weights)

    # Check the content of the output directory.
    self.assertTrue(
        glob.glob(os.path.join(output_dir, 'tensorflowjs_model.pb')))
    self.assertTrue(glob.glob(os.path.join(output_dir, 'group*-*')))


if __name__ == '__main__':
  unittest.main()
